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
}


def main():
    only = argv("--only", "")
    out = os.path.join(ROOT, "exp", argv("--out", "run_exp.json"))
    seeds = tuple(int(v) for v in argv("--seeds", "42,43").split(","))
    folds = tuple(int(v) for v in argv("--folds", "2022,2023,2024").split(","))
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

    BUILD = {"D": ("asof13_v1", build_D),
             "CTX": ("dx8_v1", build_CTX),
             "LVL": ("lx6_v1", build_LVL),
             "EBsucc": ("ebsucc_prior_v1", build_EBsucc)}

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
    print(f"스레드 {nthr}  여유 RAM {free_gb():.1f}GB", flush=True)

    R = {}
    if os.path.exists(out):
        R = {k: {int(a): b for a, b in v.items()}
             for k, v in json.load(io.open(out, encoding="utf-8")).items()}
    for f in folds:
        tr, va = season < f, season == f
        yv, ytr = y[va], y[tr].astype(int)
        print(f"\n=== 폴드 {f}  학습 {int(tr.sum()):,}행 ===", flush=True)
        for n in names:
            if f in R.get(n, {}):
                print(f"  {n:<14}(캐시) {R[n][f]['rho2']:>9.1f}", flush=True)
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
            rec = {"rho2": 1e5 * np.corrcoef(p, yv)[0, 1] ** 2,
                   "brier": float(np.mean((p - yv) ** 2)), "p": Mtr.shape[1]}
            R.setdefault(n, {})[f] = rec
            wall, cpu = time.time() - t0, time.process_time() - c0
            log_run(RUNLOG, {"config": n, "fold": f, "wall_s": round(wall),
                             "cpu_s": round(cpu), "rows": int(tr.sum()),
                             "features": int(Mtr.shape[1]),
                             "seeds": list(seeds), "threads": nthr,
                             "free_gb_after": round(free_gb(), 1)})
            del Mtr, Mva
            gc.collect()
            print(f"  {n:<14}{rec['p']:>4}p{rec['rho2']:>10.1f}"
                  f"   Brier {rec['brier']:.6f}{wall:>7.0f}s", flush=True)
            json.dump({k: {str(a): b for a, b in v.items()} for k, v in R.items()},
                      io.open(out, "w", encoding="utf-8"), indent=1)

    ref = names[0]
    print(f"\n=== {ref} 대비 증분 ===")
    print(f"  {'후보':<14}" + "".join(f"{f:>10}" for f in folds))
    for n in names:
        print(f"  {n:<14}"
              + "".join(f"{R[n][f]['rho2'] - R[ref][f]['rho2']:>+10.1f}"
                        for f in folds))


if __name__ == "__main__":
    main()
