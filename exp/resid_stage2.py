r"""잔차에 2단 모델을 직접 학습해 **남은 구조의 상한**을 잰다. 워크포워드.

## 왜 개별 스캔 대신 이걸 하는가

"batter x count, pitcher x count, recent_form x hand, threshold, piecewise, ..." 를
하나씩 재는 것은 느리고 빠뜨리기 쉽다. **잔차를 타깃으로 규제된 모델을 학습시키면
그 모델이 형태를 가리지 않고 잡는다** — 선형이든 임계값이든 상호작용이든.

    residual = y - q          q = 현행 최고 구성 (모델 + 플래툰 + 플래툰x카운트)
    stage2   : 55개 피처 + 편차항 -> residual
    평가     : q + w*stage2(x) 의 rho^2 증분

**이전 폴드의 OOF 잔차로만 학습하고 대상 폴드에서 평가한다.** 여기서 0 이 나오면
잔차에 남은 구조가 없다는 뜻이고, 개별 후보를 더 뒤질 이유가 없다.

규제를 세게 건 작은 모델부터 쓴다 (4-1: 이 문제는 표현력이 아니라 과신하지 않기).

    .\.venv\Scripts\python.exe exp\resid_stage2.py
"""
import json
import os
import sys
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import level_probe as L                                   # noqa: E402
import make_final as MF                                   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "exp", "cache")
FOLDS = [2021, 2022, 2023, 2024]
VAL = [2022, 2023, 2024]


def main():
    meta = json.load(open(f"{CACHE}/cols.json"))
    ix = {c: i for i, c in enumerate(meta["cols"])}
    prod = meta["prod"]
    X = np.load(f"{CACHE}/X.npy", mmap_mode="r")
    y = np.load(f"{CACHE}/y.npy").astype(np.float64)
    season = np.load(f"{CACHE}/season.npy")

    def col(n):
        return np.asarray(X[:, ix[n]], dtype=np.float64).astype(np.int64)

    P, BH = col("pitcher_id"), col("batter_hand")
    CNT = col("balls_before") * 4 + col("strikes_before")
    PH = P * 10 + BH

    def mk(par, ch, k, tr, m):
        u, d, _ = MF.nested_dev(par[tr], ch[tr], y[tr], k)
        q = ch[m]
        pos = np.searchsorted(u, q)
        ok = (pos < len(u))
        pos = np.clip(pos, 0, len(u) - 1)
        return np.where(ok & (u[pos] == q), d[pos], 0.0)

    # 각 폴드의 OOF 예측과 잔차 (편차항 포함한 현행 최고 구성)
    Q, R, DEV = {}, {}, {}
    for f in FOLDS:
        m = season == f
        tr = season < f
        p = L.model_preds(f)
        d1 = mk(P, PH, 300, tr, m)
        d2 = mk(PH, PH * 100 + CNT, 800, tr, m)
        q = p + 0.20 * d1 + 0.5785 * d2
        Q[f], R[f], DEV[f] = q, y[m] - q, np.column_stack([d1, d2])

    cols = [ix[c] for c in prod]
    print("2단 모델 — 이전 폴드의 OOF 잔차로만 학습, 대상 폴드에서 평가\n")
    hdr = (f"{'val':>5s} {'모델':16s} {'학습행':>10s} {'기준 rho^2':>11s} "
           f"{'w':>5s} {'증분':>8s} {'상관(잔차)':>10s}")
    print(hdr)
    print("-" * len(hdr))
    summary = {}
    for i, f in enumerate(VAL, start=1):
        prior = FOLDS[:FOLDS.index(f)]
        mtr = np.isin(season, prior)
        mev = season == f
        Xtr = np.column_stack([np.asarray(X[mtr][:, cols], dtype=np.float32),
                               np.vstack([DEV[g] for g in prior])])
        rtr = np.concatenate([R[g] for g in prior])
        Xev = np.column_stack([np.asarray(X[mev][:, cols], dtype=np.float32),
                               DEV[f]])
        q, yv = Q[f], y[mev]
        base = 1e5 * np.corrcoef(q, yv)[0, 1] ** 2
        for name, mdl in (
            ("HGB leaf8 리지", HistGradientBoostingRegressor(
                max_leaf_nodes=8, max_iter=150, learning_rate=0.05,
                l2_regularization=100.0, min_samples_leaf=2000,
                random_state=42)),
            ("HGB leaf31", HistGradientBoostingRegressor(
                max_leaf_nodes=31, max_iter=300, learning_rate=0.05,
                l2_regularization=10.0, min_samples_leaf=500,
                random_state=42)),
            ("Ridge", Ridge(alpha=1000.0)),
        ):
            t = time.time()
            Xf = np.nan_to_num(Xtr, nan=0.0) if name == "Ridge" else Xtr
            Xg = np.nan_to_num(Xev, nan=0.0) if name == "Ridge" else Xev
            mdl.fit(Xf, rtr)
            dhat = mdl.predict(Xg)
            cc = np.corrcoef(dhat, R[f])[0, 1] if dhat.std() > 1e-12 else 0.0
            best = (0.0, 0.0)
            for w in (0.25, 0.5, 1.0, 1.5):
                v = 1e5 * np.corrcoef(q + w * dhat, yv)[0, 1] ** 2 - base
                if v > best[1]:
                    best = (w, v)
            print(f"{f:5d} {name:16s} {mtr.sum():10,} {base:11.2f} "
                  f"{best[0]:5.2f} {best[1]:+8.2f} {cc:10.4f}  [{time.time()-t:.0f}s]")
            summary.setdefault(name, []).append(best[1])
        print("-" * len(hdr))

    print(f"\n{'모델':16s} {'평균 증분':>10s} {'부호':>6s}   " +
          "  ".join(str(f).rjust(8) for f in VAL))
    for name, v in summary.items():
        print(f"{name:16s} {np.mean(v):10.2f} {sum(x>0 for x in v)}/{len(v)}   "
              + "  ".join(f"{x:+8.2f}" for x in v))
    print("\n주의 — w 를 대상 폴드에서 고른 값이라 **낙관 상한**이다.")
    print("     이 상한이 작으면 잔차에 남은 구조가 없다는 뜻이다.")


if __name__ == "__main__":
    main()
