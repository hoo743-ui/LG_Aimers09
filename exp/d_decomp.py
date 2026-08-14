r"""D 축 분해 — 1040 까지 전이된 유일한 축의 **내부**를 뜯는다.

## D 가 무엇인가 (코드에서 복원, `script.py` L138~)

D 는 "현재 − 잡음" 형태의 **차이가 아니다.** 대수적 역산으로 얻은 **수준**과
그 수준의 **신뢰도**다.

    asof_rate(통산) = (prior_events + cur_events) / (prior_n + cur_n)
                    = w * prior_rate + (1-w) * cur_rate,   w = prior_n / asof_n

    cur_rate = (asof_n * asof_rate - prior_events) / cur_n      <- 수준 10개
    cur_logn = log1p(cur_n)                                     <- 신뢰도 3개

`prior_*` 는 학습 데이터만으로 만든 선수별 상수다. 모델은 통산값만 보면
`w` 를 모르므로 두 성분을 **원리적으로 못 가른다.** D 가 그것을 갈라 준다.

**D 가 성공하고 F 가 실패한 이유가 여기 있다** (§19-c) — D 는 `cur_rate`
자체(수준)를 줬고 F 는 `prev - cur`(두 잡음 추정량의 차)를 줬다. 수준은 시즌을
넘고 잡음의 모양은 안 넘는다. 그래서 이 스크립트는 **전부 수준 단위**로만
만든다.

## 세 축 (사전 등록, 폴드를 보고 고르지 않는다)

### 1) 시간척도 — 창 길이 (`W2`, `W3`)

`prior` 를 언제까지로 잡느냐가 곧 창 길이다. `lag` 를 주면

    lag=0  prior < g      -> cur = 시즌 g            (D. 추론 시 2025)
    lag=1  prior < g-1    -> cur = 시즌 g-1 ~ g      (추론 시 2024~2025)
    lag=2  prior < g-2    -> cur = 시즌 g-2 ~ g      (추론 시 2023~2025)

전부 학습 데이터 상수만 쓰고 **D 와 같은 단위**다.

> **`recent 3/5/10/20 게임` · `EWMA` 는 만들 수 없다.** 2025 행에 대해 쓸 수 있는
> 시즌내 정보는 그 행 자신의 `asof_*` 컬럼뿐이고(다른 test 행 참조는 규칙 4)
> 위반), 경기 단위 창은 공식 컬럼 `prev{1,3,5}_game_*` 로 **이미 기본 47피처에
> 들어가 있다.** 그 위에 얹을 수 있는 시간척도는 시즌 단위 창뿐이다.

### 2) 신뢰도 가중 — 경험적 베이즈 축소 (`S500`, `S2000`)

    cur_eb(k) = (cur_events + k * prior_rate) / (cur_n + k)

`prior_rate` 가 없는 신인은 그 시점까지의 리그 평균으로 퇴화시킨다.
`cur_n` 중앙값이 526 이므로 `k=4000` 은 중앙에서 가중 0.11 로 **사실상 prior
로 붕괴**하고, `prior` 는 §17-d 에서 이미 1.0005(=0)로 측정됐다. 그래서
후보에서 뺀다. 500(중앙 가중 0.51)과 2000(0.21)만 쓴다.

### 3) 맥락 — D 가 강해지거나 약해지는 상황이 있는가 (`X`)

트리는 **단조 재파라미터화에 불변**이다 (logit 변환 등은 분할 순서를 안 바꾸므로
아무 일도 일어나지 않는다). 그래서 "비선형 보정"으로 의미가 있는 것은
(a) `n` 과 섞이는 축소(위 2번)와 (b) **곱항** 둘뿐이다. 상호작용은 명시적으로
준다 — `cur_succ`·`cur_mid` 를 카운트 우위 / 주자유무 / 같은손 / (볼−스트라이크)
와 곱한 8개.

## 판정 (§19-b 의 새 규칙)

Champion(=prod55 + D) **위에서의 증분**만 본다. 축 하나가 슬롯 하나를 받으려면

    min gain >= 1.01   AND   마진 >= 시드 노이즈의 5배

F 는 min 1.0026(노이즈의 2배)으로 통과시켰다가 전이 −60% 를 맞았다.

    .\.venv\Scripts\python.exe -u exp\d_decomp.py
    .\.venv\Scripts\python.exe -u exp\d_decomp.py --only C0,S500 --seeds 44,45
"""
import io
import json
import os
import sys
import time

import numpy as np
from catboost import CatBoostClassifier

from asof_state import HP, RATE_COLS, build_state

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "exp", "cache")


def argv(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


FOLDS = tuple(int(v) for v in argv("--folds", "2022,2023,2024").split(","))
SEEDS = tuple(int(v) for v in argv("--seeds", "42,43").split(","))
ONLY = argv("--only", "")
OUT = os.path.join(ROOT, "exp", argv("--out", "d_decomp.json"))

LBLS = [lb for _, _, lb in RATE_COLS]
NCOLS = ("asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n")
NOF = {"succ": "asof_pitcher_n", "mid": "asof_pitcher_n",
       "ball": "asof_pitcher_n", "rev": "asof_pitcher_n",
       "str": "asof_pitcher_n", "fb": "asof_pitcher_pitchmix_n",
       "bb": "asof_pitcher_pitchmix_n", "os": "asof_pitcher_pitchmix_n",
       "bsucc": "asof_batter_n", "bmid": "asof_batter_n"}


def build_state_lag(C, ID_P, ID_B, season, lag):
    """`build_state` 를 창 길이로 일반화한다 — prior 를 `< g-lag` 로 잡는다.

    lag=0 에서 `build_state` 와 **정확히 같은 값**을 내는지 아래 main 에서
    검증한다 (복붙 대신 등가성을 증명한다).
    """
    out = {}
    n_p, n_b = C("asof_pitcher_n"), C("asof_batter_n")
    n_mix = C("asof_pitcher_pitchmix_n")
    NCOL = {"asof_pitcher_n": n_p, "asof_batter_n": n_b,
            "asof_pitcher_pitchmix_n": n_mix}
    IDOF = {"asof_pitcher_n": ID_P, "asof_batter_n": ID_B,
            "asof_pitcher_pitchmix_n": ID_P}
    for nc in NCOL:
        out[f"prior_n_{nc}"] = np.zeros(len(season))
        out[f"cur_n_{nc}"] = np.zeros(len(season))
    for rc, nc, lb in RATE_COLS:
        out[f"prior_{lb}"] = np.full(len(season), np.nan)
        out[f"cur_{lb}"] = np.full(len(season), np.nan)
        out[f"cur_ev_{lb}"] = np.full(len(season), np.nan)

    for g in sorted(np.unique(season)):
        m = season == g
        pr = season < g - lag
        for nc in NCOL:
            ids, cnt = IDOF[nc], NCOL[nc]
            u, c = np.unique(ids[pr], return_counts=True)
            look = np.zeros(len(season))
            ix = np.clip(np.searchsorted(u, ids[m]), 0, max(len(u) - 1, 0))
            ok = (u[ix] == ids[m]) if len(u) else np.zeros(m.sum(), bool)
            look[np.where(m)[0][ok]] = c[ix[ok]]
            out[f"prior_n_{nc}"][m] = look[m]
            out[f"cur_n_{nc}"][m] = np.maximum(cnt[m] - look[m], 0)
        for rc, nc, lb in RATE_COLS:
            ids, cnt, rate = IDOF[nc], NCOL[nc], C(rc)
            tot = cnt * np.nan_to_num(rate)
            pn = out[f"prior_n_{nc}"][m]
            o = np.argsort(ids[pr], kind="stable")
            k, v = ids[pr][o], tot[pr][o]
            uu, s0 = np.unique(k, return_index=True)
            mx = np.maximum.reduceat(v, s0) if len(uu) else np.zeros(0)
            ix = np.clip(np.searchsorted(uu, ids[m]), 0, max(len(uu) - 1, 0))
            ok = (uu[ix] == ids[m]) if len(uu) else np.zeros(m.sum(), bool)
            ps = np.zeros(m.sum())
            ps[ok] = mx[ix[ok]]
            cn = out[f"cur_n_{nc}"][m]
            out[f"prior_{lb}"][m] = np.where(pn > 0, ps / np.maximum(pn, 1), np.nan)
            out[f"cur_ev_{lb}"][m] = tot[m] - ps
            out[f"cur_{lb}"][m] = np.where(cn > 0,
                                           (tot[m] - ps) / np.maximum(cn, 1), np.nan)
    return out


def eb(S, lb, k, lg_mean):
    """경험적 베이즈 축소 — 그 선수의 prior_rate 로 당긴다.

    prior 가 없는 신인은 그 시점까지의 리그 평균으로 퇴화한다 (학습 데이터
    상수이고 그 행 자신 외의 평가 행을 보지 않는다).
    """
    base = np.where(np.isnan(S[f"prior_{lb}"]), lg_mean, S[f"prior_{lb}"])
    cn = S[f"cur_n_{NOF[lb]}"]
    ev = np.nan_to_num(S[f"cur_ev_{lb}"])
    return (ev + k * base) / (cn + k)


def main():
    meta = json.load(io.open(f"{CACHE}/cols.json", encoding="utf-8"))
    ixc = {c: i for i, c in enumerate(meta["cols"])}
    prod = meta["prod"]
    X = np.load(f"{CACHE}/X.npy", mmap_mode="r")
    y = np.load(f"{CACHE}/y.npy").astype(np.float64)
    season = np.load(f"{CACHE}/season.npy")
    C = lambda n: np.asarray(X[:, ixc[n]], dtype=np.float64)
    ID_P = C("pitcher_id").astype(np.int64)
    ID_B = C("batter_id").astype(np.int64)

    t = time.time()
    S0 = build_state_lag(C, ID_P, ID_B, season, 0)
    ref = build_state(C, ID_P, ID_B, season)
    d = max(float(np.nanmax(np.abs(S0[k] - ref[k])))
            for k in ref if k in S0)
    print(f"lag 일반화 검증 — build_state 대비 최대차 {d:.3e}  "
          f"(0 이어야 등가)   {time.time() - t:.0f}s", flush=True)
    assert d == 0.0, "lag=0 이 원본과 다르다 — 측정 무효"
    del ref                          # 가용 3GB 다. 쓰고 나면 즉시 버린다

    L = lambda a: np.log1p(np.clip(a, 0, None)).astype(np.float32)
    F32 = lambda cols: np.column_stack(cols).astype(np.float32)
    # 리그 평균은 그 행의 시즌 **이전**까지로만 만든다
    lg = np.zeros(len(season))
    for g in sorted(np.unique(season)):
        pr = season < g
        lg[season == g] = y[pr].mean() if pr.any() else y.mean()

    D = F32([S0[f"cur_{lb}"] for lb in LBLS]
            + [L(S0[f"cur_n_{n}"]) for n in NCOLS])
    win = {}
    for tag, lag in (("W2", 1), ("W3", 2)):
        Sl = build_state_lag(C, ID_P, ID_B, season, lag)
        win[tag] = F32([Sl[f"cur_{lb}"] for lb in LBLS]
                       + [L(Sl["cur_n_asof_pitcher_n"])])
        print(f"  {tag} cur_n 중앙 {np.median(Sl['cur_n_asof_pitcher_n']):.0f}"
              f"  (D 는 {np.median(S0['cur_n_asof_pitcher_n']):.0f})", flush=True)
        del Sl
    S500 = F32([eb(S0, lb, 500, lg) for lb in LBLS])
    S2000 = F32([eb(S0, lb, 2000, lg) for lb in LBLS])
    # 후속 (--only 로만 돈다) — 1차에서 k 방향이 500 > 2000 으로 잡힌 뒤의
    # 괄호치기다. 무작정 격자가 아니다.
    S150 = F32([eb(S0, lb, 150, lg) for lb in LBLS])
    # 교체판 — 지금은 원시 cur_rate 와 축소본을 **둘 다** 준다. 원시 쪽이
    # 잡음을 도로 들여오는지 보려면 빼고 재야 한다.
    S500R = F32([eb(S0, lb, 500, lg) for lb in LBLS]
                + [L(S0[f"cur_n_{n}"]) for n in NCOLS])

    adv = (C("strikes_before") > C("balls_before")).astype(np.float64)
    onb = (C("num_runners_on") > 0).astype(np.float64)
    same = (C("pitcher_hand") == C("batter_hand")).astype(np.float64)
    bs = C("balls_before") - C("strikes_before")
    CTX = F32([S0["cur_succ"] * v for v in (adv, onb, same, bs)]
              + [S0["cur_mid"] * v for v in (adv, onb, same, bs)])
    del S0

    H = lambda *b: np.hstack(b)
    ADD = {"C0 Champion(D)": D,
           "W2 2시즌창": H(D, win["W2"]),
           "W3 3시즌창": H(D, win["W3"]),
           "S500 축소k500": H(D, S500),
           "S2000 축소k2000": H(D, S2000),
           "X D×맥락": H(D, CTX),
           # --- 후속 (--only 로만) ---
           "S150 축소k150": H(D, S150),
           "S500R 축소교체": S500R,
           "XS 맥락+축소": H(D, CTX, S500)}
    if ONLY:
        keep = ONLY.split(",")
        ADD = {k: v for k, v in ADD.items() if any(s in k for s in keep)}
    for k, v in ADD.items():
        print(f"  {k:<18}{55 + v.shape[1]:>4}p", flush=True)

    base = np.column_stack([np.asarray(X[:, ixc[c]], dtype=np.float32)
                            for c in prod])
    EXTRA = ADD
    del win, S500, S2000, S150, S500R, CTX

    R, done = {}, {}
    if os.path.exists(OUT):
        done = {k: {int(a): b for a, b in v.items()}
                for k, v in json.load(io.open(OUT, encoding="utf-8")).items()}
    for f in FOLDS:
        tr, va = season < f, season == f
        yv, ytr = y[va], y[tr].astype(int)
        print(f"\n=== 폴드 {f}  학습 {int(tr.sum()):,}행 ===", flush=True)
        for nm, ex in EXTRA.items():
            if f in done.get(nm, {}):
                R.setdefault(nm, {})[f] = done[nm][f]
                print(f"  {nm:<18}(캐시) {done[nm][f]['rho2']:>9.1f}", flush=True)
                continue
            t = time.time()
            Mtr, Mva = np.hstack([base[tr], ex[tr]]), np.hstack([base[va], ex[va]])
            acc = np.zeros(int(va.sum()))
            for sd in SEEDS:
                m = CatBoostClassifier(random_seed=sd, **HP)
                m.fit(Mtr, ytr)
                acc += m.predict_proba(Mva)[:, 1]
                del m
            p = acc / len(SEEDS)
            rec = {"rho2": 1e5 * np.corrcoef(p, yv)[0, 1] ** 2,
                   "brier": float(np.mean((p - yv) ** 2)), "p": Mtr.shape[1]}
            R.setdefault(nm, {})[f] = rec
            del Mtr, Mva
            c0 = R.get("C0 Champion(D)", {}).get(f)
            print(f"  {nm:<18}{rec['p']:>5}p{rec['rho2']:>10.1f}"
                  + (f"{rec['rho2'] - c0['rho2']:>+9.1f}" if c0 else " " * 9)
                  + f"   Brier {rec['brier']:.6f}"
                  + f"{time.time() - t:>7.0f}s", flush=True)
            json.dump({k: {str(a): b for a, b in v.items()} for k, v in R.items()},
                      io.open(OUT, "w", encoding="utf-8"), indent=1)

    den = R.get("C0 Champion(D)")
    if not den:
        return
    print(f"\n=== Champion 대비 증분 (시드 {SEEDS} 평균) ===")
    print(f"  {'구성':<18}" + "".join(f"{f:>10}" for f in FOLDS)
          + f"{'min':>9}{'worst Δ':>10}{'Brier Δ':>10}")
    for nm in R:
        r = [R[nm][f]["rho2"] / den[f]["rho2"] for f in FOLDS]
        dd = [R[nm][f]["rho2"] - den[f]["rho2"] for f in FOLDS]
        db = np.mean([R[nm][f]["brier"] - den[f]["brier"] for f in FOLDS])
        print(f"  {nm:<18}" + "".join(f"{v:>10.4f}" for v in r)
              + f"{min(r):>9.4f}{min(dd):>+10.1f}{db:>+10.6f}")
    print("\n  통과 기준 (19-b) — min gain >= 1.01 AND 마진 >= 시드노이즈 5배.")
    print("  참고: 시드 노이즈는 폴드 2022 에서 D 수준의 0.13% 로 실측됐다.")


if __name__ == "__main__":
    main()
