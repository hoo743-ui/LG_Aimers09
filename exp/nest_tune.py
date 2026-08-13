r"""중첩 구조 정밀화 — 축소 수준 재탐색 + 2차원 위험표. 재학습 0회.

## 왜 축소를 다시 푸는가

`nest_refine.py` 의 최적 가중이 거친축에서 **1.150** 이다. 1 을 넘는다는 것은
`n/(n+2000)` 축소가 **너무 셌다**는 뜻이다 (중앙 173행이면 축소 계수 0.08).
가중으로 되돌리는 것보다 축소 자체를 맞추는 편이 낫다 — 가중이 1 근처면 위험표의
해석도 안정된다.

## 그리고 위험을 두 축으로 본다

이 구성은 빌려온 전이율이 **둘**이다.

    거친축 (3단)      k_c  <- 0.9726 차용. 3단 두 축에서 0.9651 / 0.9800 실측
    카운트|거친 (4단)  k_n  <- 미측정. 깊이 추세상 >= 3단일 것으로 기대되나 근거 없음

가중의 대부분이 거친축에 실리므로 `k_c` 쪽 하방이 실제 위험이다.

    .\.venv\Scripts\python.exe exp\nest_tune.py
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
    return u, cnt * (cell - pmean[np.searchsorted(pu, par)]) / (cnt + k), cnt


def lookup(u, dev, keys):
    out = np.zeros(len(keys), dtype=np.float64)
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

    def dev24(parent, child, k):
        u, d, c = nested_dev(parent[tr], child[tr], y[tr], k)
        return lookup(u, d, child[va]), int(np.median(c))

    C_M = S13 * BASE / (1e5 * A13)
    S2_M, M_M = C_M / A13, (R_EVAL - C13 * (1 - A13)) / A13
    d1, _ = dev24(P, PH, 300)
    d2, _ = dev24(PH, PH * 100 + CNT, 800)
    d3, _ = dev24(PH, PH * 10 + OB, 2000)

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
            m = M_M + sum(W[i] * mu[i] for i in range(len(D)))
            return V, C, m
        return VC

    def realized(W, K, A, B, VC):
        V, C, m = VC(W, K)
        return 1e5 * (1 - (A * A * V - 2 * A * C + BASE
                           + (A * m + B - R_EVAL) ** 2) / BASE)

    k1, l1, k2, l2 = fsolve(
        lambda t: [realized([a, b, 0], [t[0], t[2], 0], A, c * (1 - A),
                            make([d1, d2, d3], [t[1], t[3], 1])) - s
                   for a, b, A, c, s in OBS4], [0.7685, .9725, .9340, .9725])
    w1c, w2c, w3c, A18, c18, S18 = OBS18
    k3 = brentq(lambda k: realized([w1c, w2c, w3c], [k1, k2, k], A18,
                                   c18 * (1 - A18),
                                   make([d1, d2, d3], [l1, l2, l2])) - S18,
                -2, 4)
    KD = 0.5 * (k2 + k3)
    print(f"k1={k1:.4f}  k2={k2:.4f}  k3={k3:.4f}  -> 3단 평균 KD={KD:.4f}\n")

    def opt(D, K, grids):
        VC = make(D, [l1, l2, l2, l2])
        best = None
        for W in np.stack(np.meshgrid(*grids, indexing="ij"), -1).reshape(
                -1, len(D)):
            V, C, m = VC(list(W), K)
            s = 1e5 * C * C / (V * BASE)
            if best is None or s > best[0]:
                best = (s, list(W), C / V, m)
        s, W, A, m = best
        return s, W, A, R_EVAL - A * m

    g1 = np.linspace(0.1, 0.3, 5)
    g3 = np.linspace(0.3, 0.6, 7)
    gw = np.linspace(0.0, 1.6, 65)

    print("=== 축소 수준 재탐색 (거친축 x 카운트|거친축) ===")
    print(f"  {'거친 k':>8}{'카운트 k':>10}{'중앙셀':>14}"
          f"{'w거친':>8}{'w카운트':>9}{'점수':>11}{'18회차대비':>11}")
    rows = []
    for kc in (300, 800, 2000, 5000):
        dC, medC = dev24(PH, PHA, kc)
        for kn in (300, 800, 2000):
            dN, medN = dev24(PHA, PHA * 100 + CNT, kn)
            s, W, A, B = opt([d1, dC, dN, d3], [k1, KD, KD, k3],
                             [g1, gw, gw, g3])
            rows.append((s, kc, kn, W, A, B, medC, medN, dC, dN))
            print(f"  {kc:>8}{kn:>10}{f'{medC},{medN}':>14}"
                  f"{W[1]:>8.3f}{W[2]:>9.3f}{s:>11.4f}{s - S18:>+11.3f}")

    rows.sort(reverse=True, key=lambda r: r[0])
    s, kc, kn, W, A, B, medC, medN, dC, dN = rows[0]
    print(f"\n=== 최적 구성 ===")
    print(f"  거친축 축소 n/(n+{kc})   카운트|거친 축소 n/(n+{kn})")
    print(f"  w = (플래툰 {W[0]:.2f}, 거친 {W[1]:.3f}, "
          f"카운트|거친 {W[2]:.3f}, 주자유무 {W[3]:.3f})")
    print(f"  기대 {s:.4f}   18회차 {S18:.4f} 대비 {s - S18:+.4f}")
    print(f"  alpha={A:.6f}  center={B / (1 - A):.6f}")
    print(f"  격자 끝에 붙었는가: 거친 {W[1] >= gw[-1] - 1e-9}  "
          f"카운트 {W[2] >= gw[-1] - 1e-9}")
    print(f"  d2(현행 카운트축) 와의 상관: 거친 "
          f"{np.corrcoef(d2, dC)[0, 1]:+.3f}  "
          f"카운트|거친 {np.corrcoef(d2, dN)[0, 1]:+.3f}   "
          f"거친-카운트|거친 {np.corrcoef(dC, dN)[0, 1]:+.3f}")

    VC = make([d1, dC, dN, d3], [l1, l2, l2, l2])
    print(f"\n=== 2차원 위험표 — 빌려온 전이율 둘을 동시에 흔든다 ===")
    print(f"    (w 고정, 18회차 953.7374 대비 증분. 음수면 손해)")
    KCS = [0.0, 0.5, 0.7689, KD, 1.2]
    print(f"  {'k_c \\ k_n':>10}" + "".join(f"{k:>10.3f}" for k in KCS))
    for kn_ in KCS:
        row = ""
        for kc_ in KCS:
            V, C, m = VC(W, [k1, kc_, kn_, k3])
            row += f"{1e5 * C * C / (V * BASE) - S18:>+10.3f}"
        print(f"  {kn_:>10.4f}{row}")
    print(f"    행 = k_n (카운트|거친, 4단, 미측정)")
    print(f"    열 = k_c (거친축, 3단, 두 축에서 0.9651/0.9800 실측)")

    print(f"\n=== 폴드 검사 ===")
    for lbl, ky, par, ksh in [("거친축(3단)", PHA, PH, kc),
                              ("카운트|거친(4단)", PHA * 100 + CNT, PHA, kn)]:
        cs = []
        for f in (2021, 2022, 2023, 2024):
            mt, mv = season < f, season == f
            u, d, _ = nested_dev(par[mt], ky[mt], y[mt], ksh)
            cs.append(float(np.cov(lookup(u, d, ky[mv]), y[mv],
                                   ddof=0)[0, 1]))
        print(f"  {lbl:<18}" + " ".join(f"{c:+.2e}" for c in cs)
              + f"   {sum(c > 0 for c in cs)}/4")


if __name__ == "__main__":
    main()
