r"""실험 드라이버 — 블록 캐시 위에서 후보를 조합만 해서 돌린다.

## `d_decomp.py` 와 무엇이 다른가

`d_decomp.py` 는 실행마다 `raw -> feature -> model` 을 통째로 다시 탄다.
`build_state_lag` 를 세 번(lag 0/1/2), 등가성 검증용 `build_state` 를 한 번 더,
`eb_shrunk` 5개, 상황 기준선까지 — `--only` 로 구성 하나만 재도 이 앞단이 붙는다.
게다가 `ADD` 딕셔너리가 **필터 전에** 모든 후보 행렬을 만들어 놓는다
(구성 25개 x 약 160MB = 수 GB). 관측된 RAM 3.3GB 의 상당 부분이 이것이다.

여기서는

    RAW -> 블록 캐시(디스크) -> 필요한 블록만 mmap -> 후보 = 블록 조합 -> 모델

로 바꾼다. 블록은 후보와 무관하게 값이 같으므로 한 번 만들어 재사용한다.
**모델 정의·폴드·시드·하이퍼파라미터는 그대로다.** 바뀌는 것은 같은 값을 몇 번
계산하느냐뿐이다.

## 사용

    .\.venv\Scripts\python.exe -u exp\run_exp.py --list
    .\.venv\Scripts\python.exe -u exp\run_exp.py --only "champ,NEW1" --out r1.json
    .\.venv\Scripts\python.exe -u exp\run_exp.py --only champ --fresh   # 블록 재생성

새 후보를 추가하려면 `BLOCKS` 에 빌더를, `CONFIGS` 에 블록 이름 목록을 넣는다.
후보가 늘어도 앞단 비용은 늘지 않는다.
"""
import gc
import io
import json
import os
import sys
import time

import numpy as np
from catboost import CatBoostClassifier

from asof_state import HP, RATE_COLS, build_state
from blocks import Blocks, free_gb, log_run, threads

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "exp", "cache")
RUNLOG = os.path.join(ROOT, "exp", "run_resources.jsonl")
FOLDS = (2022, 2023, 2024)
LBLS = [lb for _, _, lb in RATE_COLS]
NCOLS = ("asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n")
RT = ("succ", "mid", "ball", "rev", "str")


def argv(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


# ---------------------------------------------------------------- 후보 정의
# 이름 -> 블록 이름 목록. "prod" 는 기본 55p 이고 항상 앞에 붙는다.
CONFIGS = {
    "champ":     ["D", "CTX", "LVL"],          # 25회차 1049.9226 = 82p
    "champ_noL": ["D", "CTX"],                 # 24회차 1044.7656 = 76p
    "champ_noX": ["D"],                        # 22회차 1040.8656 = 68p
    # EB 축소 succ **1열만** Champion 에 얹은 판 (83p).
    # 주의 — 22-r 의 marginal(+20.1/+6.3/+0.7)은 5열 블록 안에서의 값이고
    # 이 단독 구성은 아직 측정된 적이 없다.
    "champ_ebsucc": ["D", "CTX", "LVL", "EBsucc"],
    # REGIME B — 아웃/주자수 압박. 현 Champion 의 맥락 4종(카운트우위·주자유무·
    # 같은손·볼-스트라이크)에 **아웃카운트가 없다.** 주자도 유무(0/1)로만 들어가
    # 있고 몇 명인지는 곱해진 적이 없다.
    #   WHY NEW  : outs_before 는 원시 피처로만 존재하고 cur_state 와 곱해진 적 없음
    #   WHY NOT X: X 의 onb 는 주자 유무 이진값. 여기는 아웃 3단 x 주자수 4단이고
    #              2아웃 지시자는 야구에서 가장 뚜렷한 체제 전환 중 하나다
    "champ_regB": ["D", "CTX", "LVL", "REGB"],
    # REGIME C — 경기 국면. inning / top_bottom / li(leverage) 는 원시 피처로만
    # 있고 cur_state 와 곱해진 적이 없다.
    #   WHY NEW  : 시간축(이닝)과 중요도(li)가 상태와 결합된 적 없음
    #   WHY NOT X: X 의 맥락 4종은 전부 타석 안(카운트·주자·손)이다. 여기는
    #              **경기 전체에서의 위치**이고 다른 축이다
    #   형태     : 22-f 대로 이산 전환만. 연속 li 곱은 G1~G3 와 같은 계열이라 제외
    "champ_regC": ["D", "CTX", "LVL", "REGC"],
    # 진단 기반 후보 — 2스트라이크 구간이 rho^2 0.37~0.85 배로 모델이 눈이 먼다.
    # 원인 가설: 유인구/승부 의도가 관측되지 않는다. 의도는 못 보지만
    # **"이 투수가 얼마나 자주 빼는가"** 는 볼 수 있다 (cur_ball / cur_rev).
    #   WHY NEW  : H2 는 cur_{succ,mid} x 카운트였고 기각됐다. 여기는 곱하는
    #              **수준이 다르다** — 유인구 성향을 담는 것은 ball/rev 다
    "champ_k2": ["D", "CTX", "LVL", "K2"],
    # champ 과 같은 구성. --only 가 부분문자열이라 "champ" 으로 거르면 전부
    # 걸리므로, 기준선만 따로 부를 때 쓰는 별칭이다.
    "base82": ["D", "CTX", "LVL"],
    # RISK — 정보 추가가 아니라 **압축**이다. cur_{mid,ball,rev} 세 실패 유형을
    # 학습 구간에서 추정한 가중으로 1열로 합친다.
    #   WHY NEW : 지금까지는 항상 열을 **늘렸다**. 처음으로 줄인다.
    #   근거    : 27 에서 depth 8 이 -5.4 였다. 병목은 표현력이 아니라 표본
    #             효율이고, 정보를 여러 열에 흩으면 트리가 조합을 찾느라
    #             표본을 소모한다. 미리 합치면 그 비용이 사라진다.
    #   모든 행에 값이 있다 (K2 처럼 절반을 0 으로 끄지 않는다).
    "champ_risk_add": ["D", "CTX", "LVL", "RISK"],       # 원시 유지 + 압축
    "champ_risk_rep": ["Dm3", "CTX", "LVL", "RISK"],     # 원시 3열을 압축으로 대체
    # TREND — prev{1,3,5} 의 시간 구조. **대수 감사 후 남은 것만** 넣는다.
    #   제외: prev1/3/5 level (이미 원시 6열), prev1-prev3 (= CAAFE cf_trend13),
    #         prev3-prev5 (= cf_trend35), prev1_mid-prev3_mid (= cf_midtrend13),
    #         prev5-통산 (= cf_form5 = F 계열)
    #   포함: 추세 x **현재 상태**, 추세 x **이산 맥락**, 그리고 CAAFE 에 없는
    #         prev3_mid - prev5_mid
    #   WHY NEW : CAAFE 는 추세를 **주효과**로만 줬고 전이 0 이었다(§15).
    #             여기는 "추세가 상황·현재상태에 따라 다르게 읽히는가"를 묻는다.
    #             X/H1 이 통한 원리(트리가 못 만드는 곱)와 같은 형태다.
    "champ_trend": ["D", "CTX", "LVL", "TREND"],
    # TR-VAR — 투수별 **릴리스/구질 반복성**. 지금 TrackMan 은 4개 물리량의
    # **평균 편차 8열**로만 쓰이고 extension·rel_height·rel_side 는 아예 안 쓴다.
    # 분산은 어떤 형태로도 안 쓴다.
    #   WHY NEW : 타깃이 "제구"이고 커맨드의 물리적 정의가 릴리스 반복성이다.
    #             같은 지점에서 같은 궤적으로 반복하는 투수가 제구가 좋다.
    #             모델은 그 투수의 릴리스 산포를 알 방법이 없다 (D 와 같은 구조).
    #   규정    : 학습 구간 TrackMan 으로 만든 선수별 상수. 17-e 와 같은 근거.
    #   구종내  : 구종을 섞으면 산포가 부풀므로 **구종 안에서** 재고 가중평균한다.
    "champ_trvar": ["D", "CTX", "LVL", "TRVAR"],
    # K2 내부 분해 — 4열을 1열 블록으로 쪼개 어떤 부분집합도 조합으로 만든다.
    # 목표는 기대값을 높이는 것이 아니라 **정보량을 유지하며 분산을 줄이는 것**.
    # ball×2S 와 rev×2S 의 상관이 0.93 이라 하나는 잉여일 수 있다.
    "k2_full":    ["D", "CTX", "LVL", "K2b2s", "K2b2l", "K2r2s", "K2r2l"],
    "k2_noB2S":   ["D", "CTX", "LVL", "K2b2l", "K2r2s", "K2r2l"],
    "k2_noR2S":   ["D", "CTX", "LVL", "K2b2s", "K2b2l", "K2r2l"],
    "k2_ballonly": ["D", "CTX", "LVL", "K2b2s", "K2b2l"],
    "k2_revonly":  ["D", "CTX", "LVL", "K2r2s", "K2r2l"],
}


def main():
    only = argv("--only", "")
    out = os.path.join(ROOT, "exp", argv("--out", "run_exp.json"))
    seeds = tuple(int(v) for v in argv("--seeds", "42,43").split(","))
    folds = tuple(int(v) for v in argv("--folds", "2022,2023,2024").split(","))
    # --train-from : 학습 행을 이 시즌 이후로 제한한다. 피처·하이퍼파라미터·
    # 폴드·시드는 그대로다. 25회차까지 아무도 이 줄을 건드리지 않았고 "전 시즌
    # 균등 학습"이 최적인지 검증된 적이 없다. `asof_prior` 상수는 바꾸지 않는다
    # (선수 이력은 길수록 정확하고, 학습 행 선택과는 별개 문제다).
    tfrom = int(argv("--train-from", "0"))
    # --depth / --border : 모델 용량 재탐색. 현행 6/32 는 **55피처 pre-D 시절**에
    # 튜닝된 값이고 82피처에서 재검토된 적이 없다. X/H1 을 만든 근거가
    # "depth 6 트리는 두 열의 곱을 못 만든다" 였는데 깊이를 안 올려봤다.
    # border_count 32 는 기본값 254 의 1/8 — 연속 cur_* 를 32구간으로만 쪼갠다.
    dep = int(argv("--depth", "0"))
    bor = int(argv("--border", "0"))
    names = [n for n in CONFIGS if not only or any(t in n for t in only.split(","))]
    if "--list" in sys.argv or not names:
        print("후보:", ", ".join(CONFIGS))
        return

    meta = json.load(io.open(f"{CACHE}/cols.json", encoding="utf-8"))
    ixc = {c: i for i, c in enumerate(meta["cols"])}
    X = np.load(f"{CACHE}/X.npy", mmap_mode="r")
    y = np.load(f"{CACHE}/y.npy").astype(np.float64)
    season = np.load(f"{CACHE}/season.npy")
    C = lambda n: np.asarray(X[:, ixc[n]], dtype=np.float64)
    B = Blocks(fresh="--fresh" in sys.argv)

    # --- 블록 빌더. 필요한 것만 불린다 (S0 도 지연 생성) ---
    _s0 = {}

    def S0():
        if not _s0:
            t = time.time()
            _s0["v"] = build_state(C, C("pitcher_id").astype(np.int64),
                                   C("batter_id").astype(np.int64), season)
            print(f"  [S0] as-of 분해 {time.time() - t:.0f}s", flush=True)
        return _s0["v"]

    L = lambda a: np.log1p(np.clip(a, 0, None)).astype(np.float32)
    F32 = lambda cs: np.column_stack(cs).astype(np.float32)
    mul = lambda: {
        "adv": (C("strikes_before") > C("balls_before")).astype(np.float64),
        "onb": (C("num_runners_on") > 0).astype(np.float64),
        "sh": (C("pitcher_hand") == C("batter_hand")).astype(np.float64),
        "bs": C("balls_before") - C("strikes_before")}

    def build_D():
        s = S0()
        return F32([s[f"cur_{lb}"] for lb in LBLS]
                   + [L(s[f"cur_n_{n}"]) for n in NCOLS])

    def build_CTX():
        s, m = S0(), mul()
        return F32([s[f"cur_{lb}"] * m[t]
                    for lb in ("succ", "mid") for t in ("adv", "onb", "sh", "bs")])

    def build_LVL():
        s, m = S0(), mul()
        return F32([s[f"cur_{r}"] * m[t]
                    for r in ("ball", "rev", "str") for t in ("sh", "bs")])

    def build_EBsucc():
        """EB 축소 cur_succ 1열. `d_decomp.eb_shrunk(r,"prior")` 와 같은 정의.

        시즌 g 의 k 와 부모는 시즌 <g 로만 만든다 (워크포워드).
        k = mu(1-mu)/sigma^2_between,  부모 = 선수 prior_succ (없으면 리그평균)
        """
        s = S0()
        cn = s["cur_n_asof_pitcher_n"]
        ev = np.nan_to_num(s["cur_ev_succ"]) if "cur_ev_succ" in s else None
        if ev is None:                      # build_state 는 cur_ev 를 안 준다
            ev = np.nan_to_num(s["cur_succ"]) * cn
        rate, out = s["cur_succ"], np.full(len(season), np.nan)
        for g in sorted(np.unique(season)):
            m, pr = season == g, season < g
            src = pr if pr.any() else m
            ok = src & (cn > 0) & ~np.isnan(rate)
            if not ok.any():
                continue
            mu = float(ev[ok].sum() / max(cn[ok].sum(), 1))
            v_tot = float(np.average((rate[ok] - mu) ** 2, weights=cn[ok]))
            v_bin = float(np.average(mu * (1 - mu) / np.maximum(cn[ok], 1),
                                     weights=cn[ok]))
            k = mu * (1 - mu) / max(v_tot - v_bin, 1e-8)
            par = np.where(np.isnan(s["prior_succ"]), mu, s["prior_succ"])
            out[m] = (ev[m] + k * par[m]) / (cn[m] + k)
        return out.astype(np.float32).reshape(-1, 1)

    def build_REGB():
        s, o = S0(), C("outs_before")
        nr = C("num_runners_on")
        return F32([s[f"cur_{r}"] * v
                    for r in ("succ", "mid")
                    for v in (o, nr, (o == 2).astype(np.float64))])

    def build_REGC():
        s = S0()
        late = (C("inning") >= 7).astype(np.float64)
        tb = C("top_bottom").astype(np.float64)
        hi = (C("li") >= 1.5).astype(np.float64)
        return F32([s[f"cur_{r}"] * v
                    for r in ("succ", "mid") for v in (late, tb, hi)])

    def build_K2():
        s = S0()
        k2 = (C("strikes_before") == 2).astype(np.float64)
        two = ((C("strikes_before") == 2) & (C("balls_before") <= 1)
               ).astype(np.float64)          # 유인구 여유가 큰 구간
        return F32([s[f"cur_{r}"] * v
                    for r in ("ball", "rev") for v in (k2, two)])

    def k2col(rate, kind):
        """K2 단일 열. `script.py` 의 K2_COLS 정의와 같다."""
        def f():
            s = S0()
            st, bl = C("strikes_before"), C("balls_before")
            m = ((st == 2) if kind == "2s"
                 else ((st == 2) & (bl <= 1))).astype(np.float64)
            return (s[f"cur_{rate}"] * m).astype(np.float32).reshape(-1, 1)
        return f

    def build_TRVAR():
        import pandas as pd
        Q = ["rel_height", "rel_side", "extension",
             "spin_rate", "induced_vert_break", "horz_break"]
        idm = pd.read_csv(os.path.join(ROOT, "pitcher_id_map.csv"))
        idm = idm[idm["conf"] >= 0.9]
        tm = pd.read_csv(os.path.join(ROOT, "data", "trackman_history.csv"),
                         encoding="utf-8-sig",
                         usecols=["season", "pitcher_trackman_id",
                                  "pitch_type_group"] + Q)
        tm["pid"] = tm["pitcher_trackman_id"].map(
            idm.set_index("pitcher_trackman_id")["pitcher_id"])
        tm = tm.dropna(subset=["pid"])
        tm["pid"] = tm["pid"].astype(np.int64)
        pid = C("pitcher_id").astype(np.int64)
        out = np.full((len(season), len(Q) + 1), np.nan, np.float32)
        for g in sorted(np.unique(season)):
            m = season == g
            src = tm[tm["season"] < g]
            if len(src) < 10000:                # 첫 시즌은 참조 구간이 없다
                continue
            # **구종 안에서** 산포를 재고 구종 표본수로 가중평균한다
            gp = src.groupby(["pid", "pitch_type_group"])
            sd = gp[Q].std()
            cnt = gp.size().rename("n")
            j = sd.join(cnt).dropna(subset=["n"])
            w = j["n"].to_numpy(np.float64)
            agg = {}
            for q in Q:
                v = j[q].to_numpy(np.float64)
                ok = np.isfinite(v)
                tmp = pd.DataFrame({"pid": j.index.get_level_values(0)[ok],
                                    "wv": (v[ok] * w[ok]), "w": w[ok]})
                r = tmp.groupby("pid").sum()
                agg[q] = r["wv"] / r["w"]
            npitch = src.groupby("pid").size()
            tab = pd.DataFrame(agg)
            tab["logn"] = np.log1p(npitch.reindex(tab.index).fillna(0))
            look = tab.reindex(pid[m])
            out[m] = look.to_numpy(np.float32)
            print(f"    {g}  참조 {len(src):,}구  커버리지 "
                  f"{100*look[Q[0]].notna().mean():.1f}%", flush=True)
        return out

    def build_TREND():
        s, m = S0(), mul()
        g = lambda c: C(c)
        p1 = g("asof_pitcher_prev1_game_success_rate")
        p3 = g("asof_pitcher_prev3_game_success_rate")
        p5 = g("asof_pitcher_prev5_game_success_rate")
        m3 = g("asof_pitcher_prev3_game_middle_rate")
        m5 = g("asof_pitcher_prev5_game_middle_rate")
        t13, t15 = p1 - p3, p1 - p5
        return F32([s["cur_succ"] * t13,      # 추세 x 현재 수준
                    s["cur_succ"] * t15,
                    t13 * m["sh"],            # 추세 x 이산 맥락
                    t13 * m["bs"],
                    m3 - m5])                 # CAAFE 에 없는 유일한 차분

    RISK_SRC = ("mid", "ball", "rev")

    def build_RISK():
        """세 실패 유형을 학습 구간 추정 가중으로 1열 압축.

        시즌 g 의 가중은 **시즌 <g** 행으로만 추정한다 (워크포워드).
        검증 폴드의 타깃은 보지 않는다.
        """
        s0 = S0()
        F = np.column_stack([s0[f"cur_{r}"] for r in RISK_SRC])
        out = np.full(len(season), np.nan)
        for g in sorted(np.unique(season)):
            m = season == g
            pr = season < g
            src = pr if pr.any() else m       # 첫 시즌은 자기 자신뿐
            ok = src & np.isfinite(F).all(1)
            if ok.sum() < 10000:
                continue
            A = np.hstack([F[ok], np.ones((ok.sum(), 1))])
            w, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
            mm = m & np.isfinite(F).all(1)
            out[mm] = F[mm] @ w[:-1] + w[-1]
            if g in (2022, 2024):
                print(f"    {g}  w = " + ", ".join(
                    f"{r}:{v:+.3f}" for r, v in zip(RISK_SRC, w[:-1])),
                    flush=True)
        return out.astype(np.float32).reshape(-1, 1)

    def build_Dm3():
        """D 에서 cur_{mid,ball,rev} 세 열을 뺀 10열."""
        keep = [i for i, lb in enumerate(LBLS) if lb not in RISK_SRC]
        s0 = S0()
        return F32([s0[f"cur_{LBLS[i]}"] for i in keep]
                   + [L(s0[f"cur_n_{n}"]) for n in NCOLS])

    BUILD = {"D": ("asof13_v1", build_D),
             "RISK": ("risk_mbr_wf_v1", build_RISK),
             "Dm3": ("asof10_drop_mbr_v1", build_Dm3),
             "CTX": ("dx8_v1", build_CTX),
             "LVL": ("lx6_v1", build_LVL),
             "EBsucc": ("ebsucc_prior_v1", build_EBsucc),
             "REGB": ("regb_outs_nrun_v1", build_REGB),
             "REGC": ("regc_game_v1", build_REGC),
             "K2": ("k2_ballrev_v1", build_K2),
             "K2b2s": ("k2_ball_2s_v1", k2col("ball", "2s")),
             "K2b2l": ("k2_ball_2slow_v1", k2col("ball", "2slow")),
             "K2r2s": ("k2_rev_2s_v1", k2col("rev", "2s")),
             "K2r2l": ("k2_rev_2slow_v1", k2col("rev", "2slow")),
             "TRVAR": ("trvar_wtype_v1", build_TRVAR),
             "TREND": ("trend_inter_v1", build_TREND)}

    prod = meta["prod"]
    need = sorted({b for n in names for b in CONFIGS[n]})
    print(f"후보 {len(names)}개, 블록 {len(need)}개 필요: {need}", flush=True)
    blk = {b: B.get(b, *BUILD[b]) for b in need}
    _s0.clear()
    gc.collect()
    print(f"  {B.report()}", flush=True)

    base = np.column_stack([np.asarray(X[:, ixc[c]], dtype=np.float32)
                            for c in prod])
    nthr = threads()
    if nthr == 0:
        print("여유 RAM 2GB 미만 — 학습을 시작하지 않는다 (8번 지시)")
        return
    hp = {**HP, "thread_count": nthr}
    tag = ""
    if dep:
        hp["depth"] = dep; tag += f"_d{dep}"
    if bor:
        hp["border_count"] = bor; tag += f"_b{bor}"
    if tag:
        print(f"용량 변경 depth={hp['depth']} border={hp['border_count']}",
              flush=True)
    print(f"스레드 {nthr}  여유 RAM {free_gb():.1f}GB", flush=True)

    R = {}
    if os.path.exists(out):
        R = {k: {int(a): b for a, b in v.items()}
             for k, v in json.load(io.open(out, encoding="utf-8")).items()}
    for f in folds:
        tr, va = season < f, season == f
        if tfrom:
            tr = tr & (season >= tfrom)
        yv, ytr = y[va], y[tr].astype(int)
        print(f"\n=== 폴드 {f}  학습 {int(tr.sum()):,}행"
              + (f"  (>= {tfrom})" if tfrom else "") + " ===", flush=True)
        for n in names:
            n_key = (f"{n}@{tfrom}" if tfrom else n) + tag
            if f in R.get(n_key, {}):
                print(f"  {n_key:<18}(캐시) {R[n_key][f]['rho2']:>9.1f}", flush=True)
                continue
            t0, c0 = time.time(), time.process_time()
            parts = [base[tr]] + [np.asarray(blk[b][tr]) for b in CONFIGS[n]]
            Mtr = np.hstack(parts)
            parts = [base[va]] + [np.asarray(blk[b][va]) for b in CONFIGS[n]]
            Mva = np.hstack(parts)
            del parts
            acc = np.zeros(int(va.sum()))
            for sd in seeds:                       # 시드는 순차 (7번 지시)
                m = CatBoostClassifier(random_seed=sd, **hp)
                m.fit(Mtr, ytr)
                acc += m.predict_proba(Mva)[:, 1]
                del m
                gc.collect()
            p = acc / len(seeds)
            if "--save-pred" in sys.argv:   # 잔차 진단용. 기본은 저장 안 함
                np.save(os.path.join(ROOT, "exp",
                                     f"pred_{n_key}_{f}.npy"), p)
            rec = {"rho2": 1e5 * np.corrcoef(p, yv)[0, 1] ** 2,
                   "brier": float(np.mean((p - yv) ** 2)), "p": Mtr.shape[1]}
            R.setdefault(n_key, {})[f] = rec
            wall, cpu = time.time() - t0, time.process_time() - c0
            log_run(RUNLOG, {"config": n_key, "fold": f, "wall_s": round(wall),
                             "cpu_s": round(cpu), "rows": int(tr.sum()),
                             "features": int(Mtr.shape[1]),
                             "seeds": list(seeds), "threads": nthr,
                             "free_gb_after": round(free_gb(), 1)})
            del Mtr, Mva
            gc.collect()
            print(f"  {n_key:<18}{rec['p']:>4}p{rec['rho2']:>10.1f}"
                  f"   Brier {rec['brier']:.6f}{wall:>7.0f}s", flush=True)
            json.dump({k: {str(a): b for a, b in v.items()} for k, v in R.items()},
                      io.open(out, "w", encoding="utf-8"), indent=1)

    keys = [k for k in R if all(f in R[k] for f in folds)]
    if not keys:
        return
    ref = keys[0]
    print(f"\n=== {ref} 대비 증분 ===")
    print(f"  {'후보':<18}" + "".join(f"{f:>10}" for f in folds))
    for k in keys:
        print(f"  {k:<18}"
              + "".join(f"{R[k][f]['rho2'] - R[ref][f]['rho2']:>+10.1f}"
                        for f in folds))


if __name__ == "__main__":
    main()
