r"""19회차 일관 궤적 위에서 **위험표로** 20회차 가중을 고른다. 재학습 0회.

## 왜 기대값으로 고르면 안 되는가

`exp/solve_k4.py` 가 낸 결론은 "관측 하나 = 미지수 둘" 이다. `(k_c, k_n)` 은
궤적으로만 잡히고 두 읽기가 **정반대 처방**을 낸다.

    (a) k_n=0.9726 -> k_c=0.8651   최적 w=(0.20, 0.975, 0.275, 0.45)
    (b) k_c=0.9726 -> k_n=0.5582   최적 w=(0.20, 1.150, 0.050, 0.45)

(b) 의 기대 이득(+1.68)이 (a)(+0.13)보다 크다고 (b) 를 고르면 그것이 14회차다 —
빌려온 값 위의 기대값으로 가중을 고르고 −23.26 을 맞았다. 근거는 기대값이 아니라
**궤적 전체에 걸친 최악값**이어야 한다.

## 이 스크립트가 하는 일

1. 궤적을 `k_n` 으로 매개화해 `k_c(k_n)` 을 푼다 (19회차 관측과 일치하는 쌍의 자취).
2. 후보 가중마다 궤적 전 구간에서 실현 점수를 계산한다.
   `alpha`/`center` 도 가정한 `k` 에서 유도되므로 틀리면 그 비용까지 같이 문다.
3. 최악값을 최대화하는 가중(minimax)을 찾는다.

## 🚩 `wn`=0 의 특별한 지위

4단 항을 빼면 점수가 `k_n` 과 무관해진다. 그러면 20회차 관측은 `k_c` **하나만의**
함수가 되어 궤적이 점으로 붕괴한다 — 점수 시도이면서 동시에 **식별 실험**이다.
남은 슬롯이 2장이므로 20회차가 `k_c` 를 확정하면 21회차는 가정 없이 짤 수 있다.

    .\.venv\Scripts\python.exe exp\robust_w.py
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
OBS19 = (0.20, 0.825, 0.280, 0.45, 1.089079, 0.620698, 955.2193198652)

# 궤적 매개화 구간. 4단 전이율이 취할 수 있다고 보는 범위.
KN_FULL = (0.00, 1.30)      # 넓게 — 신호 전무(0) 부터 3단 초과(1.3) 까지
KN_PLAUS = (0.50, 1.20)     # 그럴듯한 구간 — 깊이가 늘면 전이는 좋아졌다


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


def build():
    """평가셋 모형 + 편차항 5종을 만든다. solve_k4.py 와 동일한 경로."""
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
    dv = lambda par, ch, k: lookup(*nested_dev(par[tr], ch[tr], y[tr], k), ch[va])

    C_M = S13 * BASE / (1e5 * A13)
    S2_M, M_M = C_M / A13, (R_EVAL - C13 * (1 - A13)) / A13
    D = dict(d1=dv(P, PH, 300),
             d2=dv(PH, PH * 100 + CNT, 800),
             d3=dv(PH, PH * 10 + OB, 2000),
             dC=dv(PH, PHA, 2000),
             dN=dv(PHA, PHA * 100 + CNT, 800))

    def make(keys, L):
        Ds = [D[k] for k in keys]
        v = [float(np.var(x)) for x in Ds]
        cm = [float(np.cov(pm, x, ddof=0)[0, 1]) for x in Ds]
        cy = [float(np.cov(x, yv, ddof=0)[0, 1]) for x in Ds]
        mu = [float(np.mean(x)) for x in Ds]
        cc = [[float(np.cov(a, b, ddof=0)[0, 1]) for b in Ds] for a in Ds]

        def VC(W, K):
            V = S2_M
            for i in range(len(Ds)):
                V += 2 * W[i] * L[i] * cm[i] + W[i] ** 2 * L[i] * v[i]
                for j in range(i + 1, len(Ds)):
                    V += 2 * W[i] * W[j] * np.sqrt(L[i] * L[j]) * cc[i][j]
            C = C_M + sum(W[i] * K[i] * cy[i] for i in range(len(Ds)))
            return V, C, M_M + sum(W[i] * mu[i] for i in range(len(Ds)))
        return VC
    return make


def real(VC, W, K, A, B):
    V, C, m = VC(W, K)
    return 1e5 * (1 - (A * A * V - 2 * A * C + BASE
                       + (A * m + B - R_EVAL) ** 2) / BASE)


def affine(VC, W, K):
    """가정한 K 에서 유도되는 최적 아핀. 틀린 K 면 이 상수도 같이 틀린다."""
    V, C, m = VC(W, K)
    A = C / V
    return A, (R_EVAL - A * m) / (1 - A)


def main():
    make = build()
    k1, l1, k2, l2 = fsolve(
        lambda t: [real(make(["d1", "d2", "d3"], [t[1], t[3], 1]), [a, b, 0],
                        [t[0], t[2], 0], A, c * (1 - A)) - s
                   for a, b, A, c, s in OBS4], [0.7685, .9725, .9340, .9725])
    w1c, w2c, w3c, A18, c18, S18 = OBS18
    k3 = brentq(lambda k: real(make(["d1", "d2", "d3"], [l1, l2, l2]),
                               [w1c, w2c, w3c], [k1, k2, k], A18,
                               c18 * (1 - A18)) - S18, -2, 4)
    KD = 0.5 * (k2 + k3)
    print(f"실측 전이율  k1={k1:.4f}  k2={k2:.4f}  k3={k3:.4f}  KD={KD:.4f}"
          f"   lam1={l1:.4f} lam2={l2:.4f}")

    w1, wc, wn, w3, A19, c19, S19 = OBS19
    VC = make(["d1", "dC", "dN", "d3"], [l1, l2, l2, l2])
    B19 = c19 * (1 - A19)

    # --- 1. 궤적을 k_n 으로 매개화 -------------------------------------
    def kc_of(kn):
        return brentq(lambda k: real(VC, [w1, wc, wn, w3], [k1, k, kn, k3],
                                     A19, B19) - S19, -3, 6)

    grid_full = np.linspace(*KN_FULL, 27)
    locus = [(kn, kc_of(kn)) for kn in grid_full]
    plaus = [(kn, kc) for kn, kc in locus if KN_PLAUS[0] <= kn <= KN_PLAUS[1]]
    print(f"\n=== 궤적 (19회차 관측과 일치하는 쌍) ===")
    print(f"  전 구간   k_n {KN_FULL[0]:.2f}~{KN_FULL[1]:.2f}"
          f"  ->  k_c {locus[-1][1]:.4f}~{locus[0][1]:.4f}")
    print(f"  그럴듯   k_n {KN_PLAUS[0]:.2f}~{KN_PLAUS[1]:.2f}"
          f"  ->  k_c {plaus[-1][1]:.4f}~{plaus[0][1]:.4f}")

    def audit(W, K_assume, label, pts):
        """가정 K 로 아핀을 굳히고, 궤적 전체에서 실현 점수를 잰다."""
        A, c = affine(VC, W, K_assume)
        B = c * (1 - A)
        ss = [real(VC, W, [k1, kc, kn, k3], A, B) for kn, kc in pts]
        return dict(label=label, W=W, A=A, c=c, lo=min(ss), hi=max(ss),
                    mean=float(np.mean(ss)), ss=ss)

    # --- 2. 후보 가중 ---------------------------------------------------
    cands = []
    # 19회차 그대로 (참고 — 이미 제출됨)
    cands.append(([w1, wc, wn, w3], [k1, KD, KD, k3], "19회차 그대로"))
    # 두 읽기의 재최적화 (solve_k4.py 결과)
    cands.append(([0.20, 0.975, 0.275, 0.45], [k1, 0.8651, KD, k3],
                  "(a) 읽기 최적"))
    cands.append(([0.20, 1.150, 0.050, 0.45], [k1, KD, 0.5582, k3],
                  "(b) 읽기 최적"))

    # wn=0 계열 — k_n 과 무관해지므로 궤적이 k_c 축으로만 남는다
    kc_lo, kc_hi = plaus[-1][1], plaus[0][1]
    best0 = None
    for wcv in np.linspace(0.4, 1.8, 141):
        for w3v in (0.40, 0.45, 0.50):
            W = [0.20, wcv, 0.0, w3v]
            A, c = affine(VC, W, [k1, 0.5 * (kc_lo + kc_hi), 0, k3])
            B = c * (1 - A)
            lo = min(real(VC, W, [k1, kc, 0, k3], A, B)
                     for kc in np.linspace(kc_lo, kc_hi, 21))
            if best0 is None or lo > best0[0]:
                best0 = (lo, W)
    cands.append((best0[1], [k1, 0.5 * (kc_lo + kc_hi), 0, k3],
                  "wn=0 최악값 최대"))

    # 궤적 전체 minimax (wn 자유)
    bestm = None
    for wcv in np.linspace(0.6, 1.6, 51):
        for wnv in np.linspace(0.0, 0.45, 46):
            W = [0.20, wcv, wnv, 0.45]
            mid = plaus[len(plaus) // 2]
            A, c = affine(VC, W, [k1, mid[1], mid[0], k3])
            B = c * (1 - A)
            lo = min(real(VC, W, [k1, kc, kn, k3], A, B) for kn, kc in plaus)
            if bestm is None or lo > bestm[0]:
                bestm = (lo, W, [k1, mid[1], mid[0], k3])
    cands.append((bestm[1], bestm[2], "궤적 minimax"))

    # --- 3. 위험표 ------------------------------------------------------
    for scope, pts in (("그럴듯 구간", plaus), ("전 구간", locus)):
        print(f"\n=== 위험표 / {scope} (k_n {pts[0][0]:.2f}~{pts[-1][0]:.2f}) ===")
        print(f"  {'구성':<20}{'w':<28}{'최악':>10}{'최선':>10}{'평균':>10}"
              f"{'vs 19회차':>11}")
        for W, K, lab in cands:
            r = audit(W, K, lab, pts)
            ws = "(" + ", ".join(f"{x:.3f}" for x in W) + ")"
            print(f"  {lab:<20}{ws:<28}{r['lo']:>10.3f}{r['hi']:>10.3f}"
                  f"{r['mean']:>10.3f}{r['lo'] - S19:>+11.3f}")

    # --- 4. 채택 후보 상세 ----------------------------------------------
    print(f"\n=== 그럴듯 구간 최악값 기준 채택 ===")
    ranked = sorted((audit(W, K, lab, plaus) for W, K, lab in cands),
                    key=lambda r: -r["lo"])
    top = ranked[0]
    print(f"  {top['label']}   w={tuple(round(x, 4) for x in top['W'])}")
    print(f"  alpha={top['A']:.6f}  center={top['c']:.6f}")
    print(f"  최악 {top['lo']:.4f} ({top['lo'] - S19:+.3f})   "
          f"최선 {top['hi']:.4f} ({top['hi'] - S19:+.3f})")

    print(f"\n=== 채택안의 k_c 민감도 (wn=0 이면 k_n 무관) ===")
    W = top["W"]
    A, B = top["A"], top["c"] * (1 - top["A"])
    print(f"  {'k_c':>8}{'점수':>12}{'vs 19회차':>11}")
    for kc in np.linspace(0.75, 1.20, 10):
        s = real(VC, W, [k1, kc, top["W"][2] and KD or 0.0, k3], A, B)
        print(f"  {kc:>8.4f}{s:>12.4f}{s - S19:>+11.3f}")


if __name__ == "__main__":
    main()
