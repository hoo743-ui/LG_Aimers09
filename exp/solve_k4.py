r"""19회차(955.2193198652)로 중첩 2층의 전이율을 푼다. 재학습 0회.

## 무엇을 알 수 있고 무엇을 못 아는가

19회차 구성에는 **빌려온 전이율이 둘**이다.

    k_c  거친축(플래툰x투수유리, 3단)    <- 0.9726 차용
    k_n  카운트|거친(4단)              <- 0.9726 차용

관측은 하나뿐이라 둘을 동시에 못 푼다. 그래서 **일관 궤적**을 낸다 — 관측과
맞는 `(k_c, k_n)` 쌍의 자취. 그 위에서 어느 조합이 그럴듯한지 판단한다.

예상 957.4367 실측 955.2193 오차 **−2.22**. 지금까지 '풀린 파라미터' 범주의
최대 오차(−0.30)보다 훨씬 크다 — **차용이 빗나갔다는 뜻이고, 그 자체가 정보다.**

    .\.venv\Scripts\python.exe exp\solve_k4.py
"""
import json
import os

import numpy as np
from scipy.optimize import brentq, fsolve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "exp", "cache")
R_EVAL = 0.460900
BASE = R_EVAL * (1 - R_EVAL)
A13, C13, S13 = 1.105030, 0.598664, 942.4577639361
OBS4 = [(0.3904, 0.0, 1.083674, 0.598664, 940.1357117095),
        (0.1990, 0.0, 1.089306, 0.620389, 946.3826029949),
        (0.2000, 0.2000, 1.090437, 0.620268, 950.0112119476),
        (0.2000, 0.5785, 1.089294, 0.622802, 952.4231549068)]
OBS18 = (0.2000, 0.5470, 0.3000, 1.089163, 0.622907, 953.7373675006)
# 19회차: w=(플래툰 .20, 거친 .825, 카운트|거친 .28, 주자유무 .45)
OBS19 = (0.20, 0.825, 0.280, 0.45, 1.089079, 0.620698, 955.2193198652)


def nested_dev(parent, child, y, k):
    o = np.argsort(child, kind="stable")
    Ys, Ps, Cs = y[o], parent[o], child[o]
    u, s = np.unique(Cs, return_index=True)
    cnt = np.diff(np.append(s, len(Cs)))
    cell = np.add.reduceat(Ys, s) / cnt
    par = Ps[s]
    op = np.argsort(parent, kind="stable")
    Yp, Pp = y[op], parent[op]
    pu, ps = np.unique(Pp, return_index=True)
    pc = np.diff(np.append(ps, len(Pp)))
    pmean = np.add.reduceat(Yp, ps) / pc
    return u, cnt * (cell - pmean[np.searchsorted(pu, par)]) / (cnt + k)


def lookup(u, dev, keys):
    out = np.zeros(len(keys))
    ix = np.clip(np.searchsorted(u, keys), 0, len(u) - 1)
    ok = u[ix] == keys
    out[ok] = dev[ix[ok]]
    return out


def main():
    meta = json.load(open(f"{CACHE}/cols.json"))
    ixc = {c: i for i, c in enumerate(meta["cols"])}
    X = np.load(f"{CACHE}/X.npy", mmap_mode="r")
    y = np.load(f"{CACHE}/y.npy").astype(np.float64)
    season = np.load(f"{CACHE}/season.npy")
    col = lambda n: np.asarray(X[:, ixc[n]], dtype=np.float64)

    P = col("pitcher_id").astype(np.int64)
    BH = col("batter_hand").astype(np.int64)
    BB, SS = col("balls_before").astype(np.int64), col("strikes_before").astype(np.int64)
    OB = (col("num_runners_on") > 0).astype(np.int64)
    PH, CNT = P * 10 + BH, BB * 4 + SS
    ADV = (SS > BB).astype(np.int64)
    PHA = PH * 10 + ADV
    tr, va = season <= 2023, season == 2024
    yv = y[va]
    pm = np.load(os.path.join(ROOT, "exp",
                              "valpred_cat_s3.npz"))["p"].astype(np.float64)
    dv = lambda par, ch, k: lookup(*nested_dev(par[tr], ch[tr], y[tr], k),
                                   ch[va])

    C_M = S13 * BASE / (1e5 * A13)
    S2_M, M_M = C_M / A13, (R_EVAL - C13 * (1 - A13)) / A13
    d1 = dv(P, PH, 300)
    d2 = dv(PH, PH * 100 + CNT, 800)
    d3 = dv(PH, PH * 10 + OB, 2000)
    dC = dv(PH, PHA, 2000)
    dN = dv(PHA, PHA * 100 + CNT, 800)

    def make(D, L):
        v = [float(np.var(x)) for x in D]
        cm = [float(np.cov(pm, x, ddof=0)[0, 1]) for x in D]
        cy = [float(np.cov(x, yv, ddof=0)[0, 1]) for x in D]
        mu = [float(np.mean(x)) for x in D]
        cc = [[float(np.cov(a, b, ddof=0)[0, 1]) for b in D] for a in D]

        def VC(W, K):
            V = S2_M
            for i in range(len(D)):
                V += 2 * W[i] * L[i] * cm[i] + W[i] ** 2 * L[i] * v[i]
                for j in range(i + 1, len(D)):
                    V += 2 * W[i] * W[j] * np.sqrt(L[i] * L[j]) * cc[i][j]
            C = C_M + sum(W[i] * K[i] * cy[i] for i in range(len(D)))
            return V, C, M_M + sum(W[i] * mu[i] for i in range(len(D)))
        return VC

    def real(VC, W, K, A, B):
        V, C, m = VC(W, K)
        return 1e5 * (1 - (A * A * V - 2 * A * C + BASE
                           + (A * m + B - R_EVAL) ** 2) / BASE)

    k1, l1, k2, l2 = fsolve(
        lambda t: [real(make([d1, d2, d3], [t[1], t[3], 1]), [a, b, 0],
                        [t[0], t[2], 0], A, c * (1 - A)) - s
                   for a, b, A, c, s in OBS4], [0.7685, .9725, .9340, .9725])
    w1c, w2c, w3c, A18, c18, S18 = OBS18
    k3 = brentq(lambda k: real(make([d1, d2, d3], [l1, l2, l2]),
                               [w1c, w2c, w3c], [k1, k2, k], A18,
                               c18 * (1 - A18)) - S18, -2, 4)
    KD = 0.5 * (k2 + k3)
    print(f"기존 실측  2단 k1={k1:.4f} | 3단 k2={k2:.4f} k3={k3:.4f} "
          f"-> KD={KD:.4f}   lam1={l1:.4f} lam2={l2:.4f}\n")

    w1, wc, wn, w3, A19, c19, S19 = OBS19
    W = [w1, wc, wn, w3]
    VC = make([d1, dC, dN, d3], [l1, l2, l2, l2])
    B19 = c19 * (1 - A19)
    pred = real(VC, W, [k1, KD, KD, k3], A19, B19)
    print(f"=== 19회차 ===")
    print(f"  예상(차용 kc=kn={KD:.4f})  {pred:.4f}")
    print(f"  실측                     {S19:.4f}   오차 {S19 - pred:+.4f}")
    print(f"  18회차 대비 {S19 - S18:+.4f}")

    print(f"\n=== 일관 궤적 — 관측과 맞는 (k_c, k_n) 쌍 ===")
    print(f"  {'k_n 고정':>10}{'-> k_c':>10}      {'k_c 고정':>10}{'-> k_n':>10}")
    for fix in (0.0, 0.5, k1, KD, 1.2):
        try:
            kc = brentq(lambda k: real(VC, W, [k1, k, fix, k3], A19, B19) - S19,
                        -3, 6)
        except ValueError:
            kc = float("nan")
        try:
            kn = brentq(lambda k: real(VC, W, [k1, fix, k, k3], A19, B19) - S19,
                        -6, 9)
        except ValueError:
            kn = float("nan")
        print(f"  {fix:>10.4f}{kc:>10.4f}      {fix:>10.4f}{kn:>10.4f}")

    kc_at_KD = brentq(lambda k: real(VC, W, [k1, k, KD, k3], A19, B19) - S19,
                      -3, 6)
    kn_at_KD = brentq(lambda k: real(VC, W, [k1, KD, k, k3], A19, B19) - S19,
                      -6, 9)
    print(f"\n  가장 자연스러운 두 읽기")
    print(f"   (a) 4단이 3단과 같다(k_n={KD:.4f}) 면  k_c = {kc_at_KD:.4f}")
    print(f"   (b) 거친축이 3단 실측대로다(k_c={KD:.4f}) 면  k_n = {kn_at_KD:.4f}")

    print(f"\n=== 재최적화 — 두 읽기 각각에서 ===")
    for lbl, K in [(f"(a) kc={kc_at_KD:.4f}, kn={KD:.4f}",
                    [k1, kc_at_KD, KD, k3]),
                   (f"(b) kc={KD:.4f}, kn={kn_at_KD:.4f}",
                    [k1, KD, kn_at_KD, k3])]:
        bb = None
        for a in np.linspace(0.1, 0.3, 5):
            for b in np.linspace(0, 1.6, 65):
                for c in np.linspace(0, 0.8, 33):
                    for d in np.linspace(0.2, 0.7, 11):
                        V, C, m = VC([a, b, c, d], K)
                        s = 1e5 * C * C / (V * BASE)
                        if bb is None or s > bb[0]:
                            bb = (s, a, b, c, d, C / V, m)
        s, a, b, c, d, A, m = bb
        print(f"  {lbl}")
        print(f"    최적 w=({a:.2f}, {b:.3f}, {c:.3f}, {d:.2f})  {s:.4f}  "
              f"({s - S19:+.3f} vs 19회차)")
        print(f"    alpha={A:.6f} center={(R_EVAL - A * m) / (1 - A):.6f}")

    print(f"\n=== 참고: 18회차 구성(중첩 없음)으로 되돌리면 ===")
    VC3 = make([d1, d2, d3], [l1, l2, l2])
    bb = None
    for a in np.linspace(0.1, 0.3, 5):
        for b in np.linspace(0, 1.2, 61):
            for c in np.linspace(0, 0.8, 33):
                V, C, m = VC3([a, b, c], [k1, k2, k3])
                s = 1e5 * C * C / (V * BASE)
                if bb is None or s > bb[0]:
                    bb = (s, a, b, c, C / V, m)
    s, a, b, c, A, m = bb
    print(f"  최적 w=({a:.2f}, {b:.3f}, {c:.3f})  {s:.4f}  "
          f"({s - S19:+.3f} vs 19회차)")


if __name__ == "__main__":
    main()
