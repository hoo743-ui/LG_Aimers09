r"""U자 곡률 보정 — 모양은 고정, 크기(lambda)만 강하게 축소해서 워크포워드 검증.

## 설계

잔차 탐색에서 `E[y|q]` 가 U자로 휘어 있고 3폴드 부호가 일치했다 (6-d). 그런데 2차항
계수를 폴드마다 자유롭게 맞추면 2021 +0.1467 / 2022 +0.0048 로 30배 흔들려
워크포워드가 무너졌다. 그래서 **모양을 고정하고 크기만 본다.**

    z      = (q - c) / s          c, s 는 **이전 폴드의 q** 에서만 (타깃 미사용)
    shape  = z^2 - 1              평균 0 으로 맞춰 중심 이동을 만들지 않는다
    q'     = q + lambda * s * shape

`lambda` 는 무차원이고 폴드 간 의미가 같다. 후보는 0 / 0.10 / 0.25 / 0.50 / 0.75 / 1.00.

## 지표

LB 는 아핀을 자유롭게 고를 수 있으므로(4-25) **원 Brier 는 중심·퍼짐 어긋남을 섞는다.**
두 가지를 같이 찍는다.

  - `dBrier`   : 원 Brier 변화 (요청 형식)
  - `d(1e5*rho^2)` : 아핀 최적에서의 변화 = **실제 LB 스케일의 해상도 변화**

판정은 뒤쪽이다. 앞쪽만 좋아지면 그건 캘리브레이션이지 해상도가 아니다.

    .\.venv\Scripts\python.exe exp\curv_test.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import level_probe as L                                   # noqa: E402
import make_final as MF                                   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "exp", "cache")
FOLDS = [2021, 2022, 2023, 2024]
LAMBDAS = [0.0, 0.10, 0.25, 0.50, 0.75, 1.00]


def build_base(w2):
    meta = json.load(open(f"{CACHE}/cols.json"))
    ix = {c: i for i, c in enumerate(meta["cols"])}
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

    out = {}
    for f in FOLDS:
        m = season == f
        tr = season < f
        p = L.model_preds(f)
        q = (p + 0.20 * mk(P, PH, 300, tr, m)
             + w2 * mk(PH, PH * 100 + CNT, 800, tr, m))
        out[f] = (q, y[m])
    return out


def rho2(yv, q):
    return float(1e5 * np.corrcoef(q, yv)[0, 1] ** 2)


def brier(yv, q):
    return float(((np.clip(q, 0, 1) - yv) ** 2).mean())


def run(base, label):
    print(f"\n{'='*78}\n{label}\n{'='*78}")
    hdr = (f"{'val':>5s} {'lam':>5s} {'Brier':>10s} {'dBrier':>10s} "
           f"{'1e5rho^2':>10s} {'d(rho^2)':>10s} {'보정폭':>8s}")
    print(hdr)
    print("-" * len(hdr))
    table = {lam: {} for lam in LAMBDAS}
    for i, f in enumerate(FOLDS[1:], start=1):
        prior = FOLDS[:i]
        qp = np.concatenate([base[g][0] for g in prior])   # 타깃 미사용
        c, s = float(qp.mean()), float(qp.std())
        q, yv = base[f]
        b0, r0 = brier(yv, q), rho2(yv, q)
        for lam in LAMBDAS:
            z = (q - c) / s
            qc = q + lam * s * (z ** 2 - 1.0)
            b, r = brier(yv, qc), rho2(yv, qc)
            table[lam][f] = (b - b0, r - r0)
            print(f"{f:5d} {lam:5.2f} {b:10.6f} {b-b0:+10.6f} "
                  f"{r:10.2f} {r-r0:+10.2f} {np.abs(qc-q).mean():8.5f}")
        print("-" * len(hdr))
    print(f"\n{'lam':>5s} {'평균 dBrier':>12s} {'평균 d(rho^2)':>14s} "
          f"{'부호':>6s} {'최악 d(rho^2)':>14s}")
    best = None
    for lam in LAMBDAS:
        db = np.mean([v[0] for v in table[lam].values()])
        dr = np.mean([v[1] for v in table[lam].values()])
        pos = sum(1 for v in table[lam].values() if v[1] > 0)
        worst = min(v[1] for v in table[lam].values())
        print(f"{lam:5.2f} {db:+12.6f} {dr:+14.2f} {pos}/{len(table[lam])} "
              f"{worst:+14.2f}")
        if lam > 0 and (best is None or (pos, dr) > best[0]):
            best = ((pos, dr), lam, worst)
    return best, table


def main():
    for w2, label in ((0.20, "A = 현재 LB 기준선 (플래툰 0.20 + 카운트 0.20) = 950.0112"),
                      (0.5785, "C = 대기 후보 (카운트 0.5785 재최적화) + 곡률")):
        base = build_base(w2)
        best, _ = run(base, label)
        if best:
            (pos, dr), lam, worst = best
            print(f"\n  -> 최선 lambda {lam:.2f}: 평균 d(rho^2) {dr:+.2f}, "
                  f"부호 {pos}/3, 최악 {worst:+.2f}")


if __name__ == "__main__":
    main()
