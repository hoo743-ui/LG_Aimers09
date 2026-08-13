r"""거친 축을 **병렬로** 얹을 것인가 **중첩으로** 끼울 것인가. 재학습 0회.

## 문제

`count_refine.py` 에서 `투수유리(S>B)` 를 병렬로 얹으면 +4.78 이 나온다. 그런데
`투수유리` 는 `카운트` 의 결정적 함수다 — 같은 정보를 두 번 넣는 것이고, `w2` 가
0.540 -> 0.320 으로 밀리는 것이 그 증거다. `k4`=0 에서 −17.4 라는 큰 하방도
**공선성**의 증상이다.

16회차가 같은 문제를 만났다. "부모를 투수로 두면 플래툰과 겹쳐서 증분이 줄어든다"
-> 부모를 플래툰 셀로 바꿔 해결했다. 편차를 **자기 부모에서** 재면 층끼리
직교하고, 깊을수록 전이도 좋아진다 (k1=0.769 -> k2=0.965 / k3=0.980).

## 그래서 비교한다

    A 현행     dev(플래툰x카운트 | 부모=플래툰, 800)
    B 병렬     A + dev(플래툰x거친축 | 부모=플래툰, 2000)          <- 겹친다
    C 중첩     dev(플래툰x거친축 | 부모=플래툰, 2000)
             + dev(플래툰x카운트  | 부모=플래툰x거친축, 800)      <- 직교

C 는 같은 정보를 층으로 분해한다. 2층은 4단이므로 전이가 3단(0.9726) 이상일 것으로
기대되지만 **미측정이므로 위험표로 판단한다.**

    .\.venv\Scripts\python.exe exp\nest_refine.py
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
    PH = P * 10 + BH
    CNT = BB * 4 + SS
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
    d2, n2 = dev24(PH, PH * 100 + CNT, 800)
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
    print(f"k1={k1:.4f} | k2={k2:.4f} | k3={k3:.4f} -> 3단 평균 {KD:.4f}"
          f"   lam1={l1:.4f} lam2={l2:.4f}\n")

    def opt(D, K, L, grids):
        VC = make(D, L)
        best = None
        for W in np.stack(np.meshgrid(*grids, indexing="ij"), -1).reshape(
                -1, len(D)):
            V, C, m = VC(list(W), K)
            s = 1e5 * C * C / (V * BASE)
            if best is None or s > best[0]:
                best = (s, list(W), C / V, m)
        s, W, A, m = best
        return s, W, A, R_EVAL - A * m, make(D, L)

    g1 = np.linspace(0.1, 0.3, 5)
    g3 = np.linspace(0.2, 0.7, 11)
    gw = np.linspace(0.0, 1.3, 53)

    sA, WA, _, _, _ = opt([d1, d2, d3], [k1, k2, k3], [l1, l2, l2],
                          [g1, gw, g3])
    print(f"A 현행 3항        w=({WA[0]:.2f}, {WA[1]:.3f}, {WA[2]:.3f})"
          f"   {sA:.4f}")

    AXES = [("투수유리 S>B", (SS > BB).astype(np.int64)),
            ("2스트라이크", (SS == 2).astype(np.int64)),
            ("스트라이크 3단", SS)]

    print(f"\n{'축':<16}{'구조':<6}{'w(거친)':>9}{'w(카운트)':>11}"
          f"{'점수':>11}{'증분':>9}{'k=0 손실':>11}{'상관':>8}")
    res = []
    for nm, ax in AXES:
        key = PH * 10 + ax if ax.max() <= 3 else PH * 100 + ax
        dC, medC = dev24(PH, key, 2000)
        rho = float(np.corrcoef(d2, dC)[0, 1])

        # B 병렬
        sB, WB, _, _, VCB = opt([d1, d2, d3, dC], [k1, k2, k3, KD],
                                [l1, l2, l2, l2], [g1, gw, g3, gw])
        V, C, m = VCB(WB, [k1, k2, k3, 0.0])
        b0 = 1e5 * C * C / (V * BASE) - sA
        print(f"{nm:<16}{'병렬':<6}{WB[3]:>9.3f}{WB[1]:>11.3f}"
              f"{sB:>11.4f}{sB - sA:>+9.3f}{b0:>+11.3f}{rho:>8.3f}")

        # C 중첩 — 카운트를 거친축 셀에서 다시 잰다
        dN, medN = dev24(key, key * 100 + CNT, 800)
        sC, WC, AC, BC, VCC = opt([d1, dC, dN, d3], [k1, KD, KD, k3],
                                  [l1, l2, l2, l2], [g1, gw, gw, g3])
        V, C, m = VCC(WC, [k1, KD, 0.0, k3])
        c0 = 1e5 * C * C / (V * BASE) - sA
        rho2 = float(np.corrcoef(dC, dN)[0, 1])
        print(f"{'':<16}{'중첩':<6}{WC[1]:>9.3f}{WC[2]:>11.3f}"
              f"{sC:>11.4f}{sC - sA:>+9.3f}{c0:>+11.3f}{rho2:>8.3f}"
              f"   중앙셀 {medC:,}/{medN:,}")
        res.append((sC - sA, nm, WC, sC, AC, BC, dC, dN, key, medC, medN))

    res.sort(reverse=True)
    gain, nm, W, s, A, B, dC, dN, key, medC, medN = res[0]
    print(f"\n=== 최상위 중첩 '{nm}' ===")
    print(f"  w=(플래툰 {W[0]:.2f}, 거친 {W[1]:.3f}, 카운트|거친 {W[2]:.3f}, "
          f"주자유무 {W[3]:.3f})")
    print(f"  기대 {s:.4f}   18회차 실측 {S18:.4f} 대비 {s - S18:+.4f}")
    print(f"  alpha={A:.6f}  center={B / (1 - A):.6f}")
    print(f"  중앙 셀 표본  거친 {medC:,}행 / 카운트|거친 {medN:,}행 "
          f"(현행 카운트 축 {n2}행)")

    VC = make([d1, dC, dN, d3], [l1, l2, l2, l2])
    print(f"\n  위험표 — 2층(4단) 전이율 k_n 이 미측정이다. w 고정, 18회차 대비")
    print(f"  {'k_n':>8}" + "".join(f"{f'w={w:.2f}':>11}"
                                    for w in (0.3, 0.5, 0.7, 0.9)))
    for kn in (0.0, 0.25, 0.50, k1, KD, 1.2):
        row = ""
        for wn in (0.3, 0.5, 0.7, 0.9):
            V, C, m = VC([W[0], W[1], wn, W[3]], [k1, KD, kn, k3])
            row += f"{1e5 * C * C / (V * BASE) - S18:>+11.3f}"
        tag = ("  <- 2단" if abs(kn - k1) < 1e-9 else
               "  <- 3단 실측평균" if abs(kn - KD) < 1e-9 else "")
        print(f"  {kn:>8.4f}{row}{tag}")

    print(f"\n  폴드 검사 — cov(d,y) 가 시즌을 넘는가")
    for lbl, ky, par, ksh in [("거친축", key, PH, 2000),
                              ("카운트|거친축", key * 100 + CNT, key, 800)]:
        cs = []
        for f in (2021, 2022, 2023, 2024):
            mt, mv = season < f, season == f
            u, d, _ = nested_dev(par[mt], ky[mt], y[mt], ksh)
            cs.append(float(np.cov(lookup(u, d, ky[mv]), y[mv],
                                   ddof=0)[0, 1]))
        print(f"    {lbl:<14}" + " ".join(f"{c:+.2e}" for c in cs)
              + f"   {sum(c > 0 for c in cs)}/4")


if __name__ == "__main__":
    main()
