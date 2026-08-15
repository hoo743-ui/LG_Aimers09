r"""편차 4축 가중을 **현 Champion 위에서** 다시 푼다 — 재학습 없이 되는 개선.

## 왜 이것이 남아 있는가

`WPOST = [0.20, 0.825, 0.280, 0.45]` 와 축소 `k = [300, 2000, 800, 2000]` 는
**19회차 모델(D 이전)에서 최적화된 값**이다. 22회차는 그 가중을 **그대로 들고**
D 를 모델 안에 넣었다 (`build_asof.py` 주석: "편차 4축 (19회차 가중 그대로)").

그런데 D 는 투수의 현재 시즌 수준을 모델에 직접 준다. 편차 4축의 1번(투수x타자손,
부모=투수)과 2번(플래툰x투수유리)은 **같은 투수 수준 정보를 후처리로 더하던
것**이다. 모델이 그 일부를 이미 흡수했다면 지금 가중은 **과다 적용**이고,
가중 하나 낮추는 것만으로 점수가 오른다. 재학습이 0회다.

## 프로토콜 — 워크포워드로 가중을 고른다

폴드 안에서 최적화한 값을 그 폴드에서 평가하면 낙관 편향이다. 두 가지를 다 낸다.

    in-fold 최적   그 폴드에서 직접 최적화 -> **상한**(달성 불가, 참고용)
    walk-forward   가중을 **이전 폴드들**에서 고르고 다음 폴드에 적용 -> **정직한 값**
                     2022 에서 고른 w -> 2023 평가
                     2022+2023 에서 고른 w -> 2024 평가

판정은 walk-forward 쪽으로 한다. 평가셋(2025)에 적용될 때와 같은 형태이기 때문이다.

## 규정

편차표는 **학습 구간 행들로만** 만들고(`nested_dev(... y[tr] ...)`), 평가 행에는
조회만 한다. 평가셋의 다른 행을 보지 않는다.

    .\.venv\Scripts\python.exe -u exp\post_reopt.py
"""
import io
import json
import os
import sys
import time

import numpy as np
from catboost import CatBoostClassifier

from asof_state import HP, RATE_COLS, build_state
from build_asof import KSH, WPOST, look, nested_dev

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "exp", "cache")
FOLDS = (2022, 2023, 2024)
SEEDS = (42, 43)
# --ctx : 현 Champion(24회차, 1044.7656)은 X 8개를 포함한 76p 다. 68p 로 재면
# 다른 모델 위에서 가중을 고르는 셈이라 측정이 무효다.
CTX = "--ctx" in sys.argv
PRED = os.path.join(ROOT, "exp",
                    "champ_oof_x.npz" if CTX else "champ_oof.npz")
OUT = os.path.join(ROOT, "exp",
                   "post_reopt_x.json" if CTX else "post_reopt.json")

LBLS = [lb for _, _, lb in RATE_COLS]
NCOLS = ("asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n")


def axes(C):
    """편차 4축의 (부모, 자식) 키. `build_asof.main` 과 같은 정의다."""
    P = C("pitcher_id").astype(np.int64)
    BH = C("batter_hand").astype(np.int64)
    BB = C("balls_before").astype(np.int64)
    SS = C("strikes_before").astype(np.int64)
    OB = (C("num_runners_on").astype(np.int64) > 0).astype(np.int64)
    PH = P * 10 + BH
    PHA = PH * 10 + (SS > BB).astype(np.int64)
    return [(P, PH), (PH, PHA), (PHA, PHA * 100 + (BB * 4 + SS)), (PH, PH * 10 + OB)]


def rho2(p, y):
    return 1e5 * np.corrcoef(p, y)[0, 1] ** 2


BAND = 0.40          # 각 축을 현행 가중의 ±40% 안에서만 움직인다


def optimize(p0, dev, y, w_init, rounds=4):
    """좌표하강. 축마다 격자를 훑고 최선을 취한다 (자유도 4개뿐).

    **범위를 현행 가중의 ±40% 로 묶는다.** 이전 실행에서 열린 격자(0~2.0)를 주자
    2023 이 `[2, 2, 2, 2]` 로 상단에 붙었고 그 가중을 2024 에 쓰면 −138 이었다.
    퇴화 폴드가 격자 끝으로 달아나는 것을 구조적으로 막는다.
    """
    w = np.array(w_init, dtype=float)
    grids = [np.linspace(v * (1 - BAND), v * (1 + BAND), 17) for v in w_init]
    best = rho2(p0 + dev @ w, y)
    for _ in range(rounds):
        moved = False
        for j in range(len(w)):
            cur = w[j]
            vals = []
            for g in grids[j]:
                w[j] = g
                vals.append(rho2(p0 + dev @ w, y))
            k = int(np.argmax(vals))
            if vals[k] > best + 1e-9:
                best, w[j], moved = vals[k], grids[j][k], True
            else:
                w[j] = cur
        if not moved:
            break
    return w, best


def main():
    meta = json.load(io.open(f"{CACHE}/cols.json", encoding="utf-8"))
    ixc = {c: i for i, c in enumerate(meta["cols"])}
    prod = meta["prod"]
    X = np.load(f"{CACHE}/X.npy", mmap_mode="r")
    y = np.load(f"{CACHE}/y.npy").astype(np.float64)
    season = np.load(f"{CACHE}/season.npy")
    C = lambda n: np.asarray(X[:, ixc[n]], dtype=np.float64)

    S = build_state(C, C("pitcher_id").astype(np.int64),
                    C("batter_id").astype(np.int64), season)
    L = lambda a: np.log1p(np.clip(a, 0, None)).astype(np.float32)
    cols = [S[f"cur_{lb}"] for lb in LBLS] + [L(S[f"cur_n_{n}"]) for n in NCOLS]
    if CTX:
        # `script.py` 의 CTX_COLS 와 같은 정의 — cur_{succ,mid} x 상황 4종
        adv = (C("strikes_before") > C("balls_before")).astype(np.float64)
        onb = (C("num_runners_on") > 0).astype(np.float64)
        same = (C("pitcher_hand") == C("batter_hand")).astype(np.float64)
        bs = C("balls_before") - C("strikes_before")
        cols += [S["cur_succ"] * v for v in (adv, onb, same, bs)]
        cols += [S["cur_mid"] * v for v in (adv, onb, same, bs)]
    D = np.column_stack(cols).astype(np.float32)
    del S, cols
    M = np.hstack([np.column_stack([np.asarray(X[:, ixc[c]], dtype=np.float32)
                                    for c in prod]), D])
    print(f"Champion 피처 {M.shape[1]}p", flush=True)
    AX = axes(C)

    # --- 폴드별 Champion 예측 (재사용 위해 저장) ---
    cache = dict(np.load(PRED)) if os.path.exists(PRED) else {}
    P, DEV = {}, {}
    for f in FOLDS:
        tr, va = season < f, season == f
        key = f"p{f}"
        if key in cache:
            P[f] = cache[key]
            print(f"  폴드 {f} 예측 (캐시)", flush=True)
        else:
            t = time.time()
            acc = np.zeros(int(va.sum()))
            for sd in SEEDS:
                m = CatBoostClassifier(random_seed=sd, **HP)
                m.fit(M[tr], y[tr].astype(int))
                acc += m.predict_proba(M[va])[:, 1]
                del m
            P[f] = acc / len(SEEDS)
            cache[key] = P[f]
            np.savez_compressed(PRED, **cache)
            print(f"  폴드 {f} 예측 {time.time() - t:.0f}s", flush=True)
        DEV[f] = np.column_stack([
            look(*nested_dev(p[tr], c[tr], y[tr], k), c[va])
            for (p, c), k in zip(AX, KSH)])

    # --- 현행 / 무보정 / in-fold 최적 ---
    print(f"\n{'폴드':<8}{'후처리없음':>12}{'현행 WPOST':>12}{'in-fold 최적':>14}"
          f"   최적 가중", flush=True)
    R, WOPT = {}, {}
    for f in FOLDS:
        yv = y[season == f]
        r_no = rho2(P[f], yv)
        r_cur = rho2(P[f] + DEV[f] @ WPOST, yv)
        w, r_opt = optimize(P[f], DEV[f], yv, WPOST)
        WOPT[f] = w
        R[f] = {"none": r_no, "cur": r_cur, "opt": r_opt, "w": w.tolist()}
        print(f"{f:<8}{r_no:>12.1f}{r_cur:>12.1f}{r_opt:>14.1f}"
              f"   [{', '.join(f'{v:.2f}' for v in w)}]", flush=True)

    # --- 워크포워드: 이전 폴드에서 고른 가중을 다음 폴드에 적용 ---
    print(f"\n=== 워크포워드 (정직한 값) ===", flush=True)
    print(f"{'평가폴드':<10}{'가중 출처':<14}{'현행':>10}{'재최적':>10}{'증분':>10}")
    inc = []
    for i, f in enumerate(FOLDS[1:], start=1):
        # **가중 출처는 건강한 폴드만.** 2023 은 Champion rho^2 가 2022 의 1/14 인
        # 퇴화 폴드이고, 그 in-fold 최적을 2024 에 쓰면 -138 이었다 (지시 7).
        src = tuple(s for s in FOLDS[:i] if s != 2023) or FOLDS[:i]
        w = np.mean([WOPT[s] for s in src], axis=0)
        yv = y[season == f]
        r_cur = rho2(P[f] + DEV[f] @ WPOST, yv)
        r_new = rho2(P[f] + DEV[f] @ w, yv)
        inc.append(r_new - r_cur)
        print(f"{f:<10}{'+'.join(str(s) for s in src):<14}"
              f"{r_cur:>10.1f}{r_new:>10.1f}{r_new - r_cur:>+10.1f}", flush=True)
        R[f]["wf_w"] = w.tolist()
        R[f]["wf_gain"] = float(r_new - r_cur)

    print(f"\n현행 WPOST = [{', '.join(f'{v:.3f}' for v in WPOST)}]  k = {KSH}")
    print(f"워크포워드 증분 평균 {np.mean(inc):+.1f}  최소 {min(inc):+.1f}")
    print(f"\n  주의 — in-fold 최적은 상한이고 달성 불가다. 판정은 워크포워드로 한다.")
    json.dump({str(k): v for k, v in R.items()},
              io.open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
