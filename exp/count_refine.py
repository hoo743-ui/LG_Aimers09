r"""`2스트라이크` 가 +3.03 으로 나왔다 — 중복인가 다해상도 이득인가. 재학습 0회.

## 왜 의심해야 하는가

`strikes_before` 는 **이미** `플래툰x카운트` 축(`CNT = balls*4 + strikes`) 안에 있다.
`플래툰x2스트라이크` 는 그 축의 **거친 판**이다. 둘을 같이 넣는 것은 같은 신호를
서로 다른 축소 수준으로 두 번 넣는 것이다.

그게 이득일 수 있다 — 셀이 잘면(중앙 29행) 추정 잡음이 크고, 거친 판(중앙 수백행)은
잡음이 작다. **다해상도 추정**은 실제로 통하는 기법이다. 그러나 단순 중복이면
`w2` 를 깎는 것으로 끝나야 한다.

판별법: `w2` 와 `w4` 를 **같이** 최적화한다. `w2` 가 크게 줄고 합이 비슷하면 중복이고,
둘 다 살아남으면 다해상도다.

## 그리고 위험

`k=0` 일 때 손실이 **−14.98** 이다. 지금까지 후보 중 가장 크다. `w4*`=0.601 은
큰 가중이고, 이건 14회차(`w`=0.3904 로 −23.3)와 같은 형태의 위험이다.

    .\.venv\Scripts\python.exe exp\count_refine.py
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
    BB = col("balls_before").astype(np.int64)
    SS = col("strikes_before").astype(np.int64)
    OB = (col("num_runners_on") > 0).astype(np.int64)
    PH = P * 10 + BH
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
    d2, n2 = dev24(PH, PH * 100 + (BB * 4 + SS), 800)
    d3, _ = dev24(PH, PH * 10 + OB, 2000)

    def moments(D):
        return dict(v=[float(np.var(x)) for x in D],
                    cm=[float(np.cov(pm, x, ddof=0)[0, 1]) for x in D],
                    cy=[float(np.cov(x, yv, ddof=0)[0, 1]) for x in D],
                    mu=[float(np.mean(x)) for x in D],
                    cc=[[float(np.cov(a, b, ddof=0)[0, 1]) for b in D]
                        for a in D])

    def make(D, L):
        M = moments(D)

        def VC(W, K):
            V = S2_M
            for i in range(len(D)):
                V += 2 * W[i] * L[i] * M["cm"][i] + W[i] ** 2 * L[i] * M["v"][i]
                for j in range(i + 1, len(D)):
                    V += (2 * W[i] * W[j] * np.sqrt(L[i] * L[j])
                          * M["cc"][i][j])
            C = C_M + sum(W[i] * K[i] * M["cy"][i] for i in range(len(D)))
            m = M_M + sum(W[i] * M["mu"][i] for i in range(len(D)))
            return V, C, m
        return VC

    # --- 전이율 재확인 ---
    VC3 = make([d1, d2, d3], [0.9731, 1.0422, 1.0422])

    def realized(W, K, A, B, VC):
        V, C, m = VC(W, K)
        return 1e5 * (1 - (A * A * V - 2 * A * C + BASE
                           + (A * m + B - R_EVAL) ** 2) / BASE)

    k1, l1, k2, l2 = fsolve(
        lambda t: [realized([a, b, 0], [t[0], t[2], 0], A, c * (1 - A),
                            make([d1, d2, d3], [t[1], t[3], 1])) - s
                   for a, b, A, c, s in OBS4], [0.7685, .9725, .9340, .9725])
    w1c, w2c, w3c, A18, c18, S18 = OBS18
    VCm = make([d1, d2, d3], [l1, l2, l2])
    k3 = brentq(lambda k: realized([w1c, w2c, w3c], [k1, k2, k], A18,
                                   c18 * (1 - A18), VCm) - S18, -2, 4)
    KD = 0.5 * (k2 + k3)
    print(f"k1={k1:.4f} lam1={l1:.4f} | k2={k2:.4f} lam2={l2:.4f} | "
          f"k3={k3:.4f}  -> 3단 평균 {KD:.4f}")

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
        return s, W, A, R_EVAL - A * m

    g1 = np.linspace(0.1, 0.3, 5)
    g2 = np.linspace(0.0, 1.0, 51)
    g3 = np.linspace(0.2, 0.7, 26)
    g4 = np.linspace(0.0, 1.2, 61)

    print(f"\n=== 기준: 3항 최적 (2스트라이크 없음) ===")
    s0, W0, A0, B0 = opt([d1, d2, d3], [k1, k2, k3], [l1, l2, l2],
                         [g1, g2, g3])
    print(f"  w=({W0[0]:.2f}, {W0[1]:.3f}, {W0[2]:.3f})   {s0:.4f}")

    print(f"\n=== 중복인가 다해상도인가 — 후보별 4항 동시 최적화 ===")
    print(f"  {'추가 축':<20}{'w2':>8}{'w4':>8}{'점수':>11}{'증분':>9}"
          f"{'k=0 손실':>11}")
    CAND = [
        ("2스트라이크", (SS == 2).astype(np.int64), 2000),
        ("스트라이크 3단", SS, 2000),
        ("3볼", (BB == 3).astype(np.int64), 2000),
        ("볼 4단", BB, 2000),
        ("투수유리 S>B", (SS > BB).astype(np.int64), 2000),
        ("카운트 거친축소", (BB * 4 + SS), 8000),
    ]
    out = []
    for nm, ax, ksh in CAND:
        child = PH * 100 + ax if ax.max() > 3 else PH * 10 + ax
        d4, med = dev24(PH, child, ksh)
        D = [d1, d2, d3, d4]
        s, W, A, B = opt(D, [k1, k2, k3, KD], [l1, l2, l2, l2],
                         [g1, g2, g3, g4])
        VC0 = make(D, [l1, l2, l2, l2])
        V, C, m = VC0(W, [k1, k2, k3, 0.0])
        s_k0 = 1e5 * C * C / (V * BASE)
        out.append((s - s0, nm, W, s, s_k0 - s0, med, A, B, d4))
        print(f"  {nm:<20}{W[1]:>8.3f}{W[3]:>8.3f}{s:>11.4f}"
              f"{s - s0:>+9.3f}{s_k0 - s0:>+11.3f}")
    print(f"  {'(없음)':<20}{W0[1]:>8.3f}{'—':>8}{s0:>11.4f}"
          f"{0.0:>+9.3f}{'—':>11}")

    out.sort(reverse=True, key=lambda r: r[0])
    gain, nm, W, s, l0, med, A, B, d4 = out[0]
    print(f"\n=== 최상위 '{nm}' 정밀 ===")
    print(f"  w2 가 {W0[1]:.3f} -> {W[1]:.3f} 로 "
          f"{'크게 줄었다 (중복 성격)' if W[1] < W0[1] * 0.7 else '거의 유지됐다 (다해상도)'}")
    print(f"  중앙 셀 표본 {med:,}행 (카운트 축 {n2}행)")
    D = [d1, d2, d3, d4]
    VC = make(D, [l1, l2, l2, l2])
    print(f"\n  위험표 — w=({W[0]:.2f}, {W[1]:.3f}, {W[2]:.3f}, w4) 고정, "
          f"진짜 k4 별 18회차 대비 증분")
    print(f"  {'k4':>8}" + "".join(f"{f'w4={w:.2f}':>11}"
                                   for w in (0.2, 0.4, 0.6, 0.8)))
    for k4 in (0.0, 0.25, 0.50, k1, KD, 1.2):
        row = ""
        for w4 in (0.2, 0.4, 0.6, 0.8):
            V, C, m = VC([W[0], W[1], W[2], w4], [k1, k2, k3, k4])
            row += f"{1e5 * C * C / (V * BASE) - S18:>+11.3f}"
        tag = ("  <- 2단" if abs(k4 - k1) < 1e-9 else
               "  <- 3단 실측평균" if abs(k4 - KD) < 1e-9 else "")
        print(f"  {k4:>8.4f}{row}{tag}")


if __name__ == "__main__":
    main()
