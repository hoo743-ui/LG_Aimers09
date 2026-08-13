r"""모순 확인 — nest_refine 은 k_n=0 손실을 -0.81, nest_tune 은 -23.22 로 냈다.

같은 구성 w=(0.20, 1.150, 0.275, 0.450) 인데 값이 다르다. 하나는 틀렸다.
깨끗하게 다시 계산해서 어느 쪽이 맞는지 가린다.

    .\.venv\Scripts\python.exe exp\nest_check.py
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
pm = np.load(os.path.join(ROOT, "exp", "valpred_cat_s3.npz"))["p"].astype(np.float64)


def dev24(parent, child, k):
    u, d, c = nested_dev(parent[tr], child[tr], y[tr], k)
    return lookup(u, d, child[va])


C_M = S13 * BASE / (1e5 * A13)
S2_M, M_M = C_M / A13, (R_EVAL - C13 * (1 - A13)) / A13
d1 = dev24(P, PH, 300)
d2 = dev24(PH, PH * 100 + CNT, 800)
d3 = dev24(PH, PH * 10 + OB, 2000)
dC = dev24(PH, PHA, 2000)
dN = dev24(PHA, PHA * 100 + CNT, 800)


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


k1, l1, k2, l2 = fsolve(
    lambda t: [(lambda V, C, m: 1e5 * (1 - (A * A * V - 2 * A * C + BASE
               + (A * m + c * (1 - A) - R_EVAL) ** 2) / BASE))(
        *make([d1, d2, d3], [t[1], t[3], 1])([a, b, 0], [t[0], t[2], 0])) - s
        for a, b, A, c, s in OBS4], [0.7685, .9725, .9340, .9725])
w1c, w2c, w3c, A18, c18, S18 = OBS18
VC3 = make([d1, d2, d3], [l1, l2, l2])
k3 = brentq(lambda k: (lambda V, C, m: 1e5 * (1 - (
    A18 * A18 * V - 2 * A18 * C + BASE
    + (A18 * m + c18 * (1 - A18) - R_EVAL) ** 2) / BASE))(
        *VC3([w1c, w2c, w3c], [k1, k2, k])) - S18, -2, 4)
KD = 0.5 * (k2 + k3)
print(f"k1={k1:.4f} k2={k2:.4f} k3={k3:.4f} KD={KD:.4f}\n")

D = [d1, dC, dN, d3]
VC = make(D, [l1, l2, l2, l2])
W = [0.20, 1.150, 0.275, 0.450]

print("=== 로컬 적률 ===")
for nm, d in [("d1 플래툰", d1), ("dC 거친(투수유리)", dC),
              ("dN 카운트|거친", dN), ("d3 주자유무", d3),
              ("d2 (현행 카운트축)", d2)]:
    print(f"  {nm:<18} var {np.var(d):.3e}  cov(.,y) {np.cov(d, yv, ddof=0)[0, 1]:+.3e}"
          f"  cov(pm,.) {np.cov(pm, d, ddof=0)[0, 1]:+.3e}")

print(f"\n=== w={W} 에서 각 k 를 따로 0 으로 ===")
Kfull = [k1, KD, KD, k3]
V, C, m = VC(W, Kfull)
s_full = 1e5 * C * C / (V * BASE)
print(f"  전부 실측/차용        {s_full:9.4f}   (18회차 {S18:.4f} 대비 {s_full - S18:+.3f})")
for i, nm in enumerate(["k1 플래툰", "k_c 거친", "k_n 카운트|거친", "k3 주자유무"]):
    K = list(Kfull)
    K[i] = 0.0
    V, C, m = VC(W, K)
    s = 1e5 * C * C / (V * BASE)
    print(f"  {nm:<18} 만 0 -> {s:9.4f}   ({s - S18:+.3f} vs 18회차, "
          f"{s - s_full:+.3f} vs 위)")

print(f"\n=== 기여도 분해 — 각 항의 cov 기여 (C = C_M + sum w*k*cy) ===")
cy = [float(np.cov(d, yv, ddof=0)[0, 1]) for d in D]
print(f"  C_M {C_M:.6e}")
for i, nm in enumerate(["d1", "dC", "dN", "d3"]):
    print(f"  {nm}: w={W[i]:.3f} k={Kfull[i]:.4f} cy={cy[i]:+.3e}  "
          f"기여 {W[i] * Kfull[i] * cy[i]:+.3e}  "
          f"({100 * W[i] * Kfull[i] * cy[i] / C_M:+.2f}% of C_M)")
