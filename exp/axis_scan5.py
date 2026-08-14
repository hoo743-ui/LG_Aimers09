r"""현 챔피언(19회차) 위에 얹을 **5번째 항**을 찾는다. 재학습 0회, LB 0장.

## 왜 다시 재는가

`axis_scan.py` 의 기준선은 17회차(2항)였다. 그 뒤로 18회차가 주자유무를,
19회차가 카운트축 중첩 2층 분해를 넣어 구성이 4항으로 바뀌었다. 그때
"+0.97, 손익분기 k=0.057" 로 눈에 띄었던 **4단 초말**이 지금 구성 위에서도
살아있는지는 다시 재야만 안다 — 중첩 분해가 이미 같은 자리를 먹었을 수 있다.

## 🚩 기준선 자체가 불확실하다는 점을 어떻게 다루는가

19회차 관측 하나로 `(k_c, k_n)` 이 안 갈린다 (`solve_k4.py`). 그래서 궤적 위의
**두 읽기 각각에서** 같은 후보를 재고, **둘 다에서 살아남는 것만** 후보로 친다.

    (a) k_c=0.8651, k_n=0.9726   4단이 3단과 같다는 읽기
    (b) k_c=0.9726, k_n=0.5582   거친축이 3단 실측대로라는 읽기

두 읽기 모두 19회차 실측 955.2193 을 재현하므로 기준선은 동일하고, 달라지는 것은
**새 항이 기존 항과 겹치는 정도**다.

## 판단 기준은 손익분기 k5

새 축의 전이율은 미측정이라 빌려와야 한다. 14 · 16 · 19회차에서 차용은 세 번
빗나갔다. 따라서 기대 이득이 아니라 **"전이율이 얼마나 나빠도 본전인가"** 로 고른다.

    .\.venv\Scripts\python.exe exp\axis_scan5.py
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
READINGS = [("(a) kc=0.8651 kn=0.9726", 0.8651, 0.9726),
            ("(b) kc=0.9726 kn=0.5582", 0.9726, 0.5582)]


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
    PHO = PH * 10 + OB
    PHAC = PHA * 100 + CNT

    TB = col("top_bottom").astype(np.int64)
    OUT = col("outs_before").astype(np.int64)
    NR = col("num_runners_on").astype(np.int64)
    BST = col("base_state").astype(np.int64)
    INN3 = np.digitize(np.clip(col("inning"), 1, 9), [4, 7]).astype(np.int64)
    SD3 = np.digitize(col("score_diff_pitcher_team"), [-2, 3]).astype(np.int64)
    SCOR = (col("score_diff_pitcher_team") == 0).astype(np.int64)
    MON = np.digitize(col("game_month"), [6, 8]).astype(np.int64)
    LI3 = np.digitize(col("li"), [0.8, 1.5]).astype(np.int64)
    R1 = col("runner_on_1b").astype(np.int64)
    R23 = ((col("runner_on_2b") + col("runner_on_3b")) > 0).astype(np.int64)
    TWO = (SS == 2).astype(np.int64)
    THREE = (BB == 3).astype(np.int64)

    tr, va = season <= 2023, season == 2024
    yv = y[va]
    pm = np.load(os.path.join(ROOT, "exp",
                              "valpred_cat_s3.npz"))["p"].astype(np.float64)

    def dev_of(parent, child, k):
        u, d, c = nested_dev(parent[tr], child[tr], y[tr], k)
        return lookup(u, d, child[va]), int(np.median(c)), len(u)

    C_M = S13 * BASE / (1e5 * A13)
    S2_M, M_M = C_M / A13, (R_EVAL - C13 * (1 - A13)) / A13

    d1, _, _ = dev_of(P, PH, 300)
    d2, _, _ = dev_of(PH, PH * 100 + CNT, 800)
    d3, _, _ = dev_of(PH, PHO, 2000)
    dC, _, _ = dev_of(PH, PHA, 2000)
    dN, _, _ = dev_of(PHA, PHAC, 800)

    def moments(D):
        return (D, [float(np.var(x)) for x in D],
                [float(np.cov(pm, x, ddof=0)[0, 1]) for x in D],
                [float(np.cov(x, yv, ddof=0)[0, 1]) for x in D],
                [float(np.mean(x)) for x in D],
                [[float(np.cov(a, b, ddof=0)[0, 1]) for b in D] for a in D])

    def maker(D, L):
        _, v, cm, cy, mu, cc = moments(D)

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

    # --- 전이율 실측 (solve_k4.py 와 동일) ---
    k1, l1, k2, l2 = fsolve(
        lambda t: [real(maker([d1, d2, d3], [t[1], t[3], 1]), [a, b, 0],
                        [t[0], t[2], 0], A, c * (1 - A)) - s
                   for a, b, A, c, s in OBS4], [0.7685, .9725, .9340, .9725])
    w1c, w2c, w3c, A18, c18, S18 = OBS18
    k3 = brentq(lambda k: real(maker([d1, d2, d3], [l1, l2, l2]),
                               [w1c, w2c, w3c], [k1, k2, k], A18,
                               c18 * (1 - A18)) - S18, -2, 4)
    print(f"실측 전이율  k1={k1:.4f}  k2={k2:.4f}  k3={k3:.4f}  "
          f"lam1={l1:.4f} lam2={l2:.4f}")

    w1, wc, wn, w3, _, _, S19 = OBS19
    W4 = [w1, wc, wn, w3]

    # --- 후보 축 (부모별) ---
    CANDS = [
        ("초말",   PH,   3, TB),   ("아웃",   PH,   3, OUT),
        ("이닝3",  PH,   3, INN3), ("점수차3", PH,   3, SD3),
        ("동점",   PH,   3, SCOR), ("월3",    PH,   3, MON),
        ("LI3",   PH,   3, LI3),  ("주자수",  PH,   3, NR),
        ("주자상태", PH,  3, BST),  ("1루",    PH,   3, R1),
        ("득점권",  PH,   3, R23),
        ("초말",   PHA,  4, TB),   ("아웃",   PHA,  4, OUT),
        ("주자유무", PHA,  4, OB),  ("득점권",  PHA,  4, R23),
        ("초말",   PHO,  4, TB),   ("아웃",   PHO,  4, OUT),
        ("카운트",  PHO,  4, CNT), ("투수유리", PHO,  4, ADV),
        ("초말",   PHAC, 5, TB),   ("아웃",   PHAC, 5, OUT),
        ("주자유무", PHAC, 5, OB),
    ]
    PNAME = {id(PH): "플래툰", id(PHA): "플래툰x거친", id(PHO): "플래툰x주자",
             id(PHAC): "플래툰x거친x카운트"}
    MULT = {id(PH): 100, id(PHA): 10, id(PHO): 100, id(PHAC): 10}

    for lbl, kc, kn in READINGS:
        K4 = [k1, kc, kn, k3]
        VC4 = maker([d1, dC, dN, d3], [l1, l2, l2, l2])
        V0, C0, _ = VC4(W4, K4)
        S0 = 1e5 * C0 * C0 / (V0 * BASE)
        print(f"\n=== 읽기 {lbl}   기준선 {S0:.4f}  [19회차 실측 {S19:.4f}] ===")
        print(f"  {'축':<10}{'부모':<20}{'깊이':>4}{'축소k':>7}{'셀수':>9}"
              f"{'중앙n':>7}{'w5*':>7}{'점수':>10}{'증분':>9}{'손익분기k':>10}")
        rows = []
        for nm, par, dep, ax in CANDS:
            for ksh in (2000, 5000):
                d5, med, ncell = dev_of(par, par * MULT[id(par)] + ax, ksh)
                VC5 = maker([d1, dC, dN, d3, d5], [l1, l2, l2, l2, l2])

                def s_of(k5, w5):
                    V, C, _ = VC5(W4 + [w5], K4 + [k5])
                    return 1e5 * C * C / (V * BASE)

                g = np.linspace(0, 2.5, 251)
                vals = [s_of(k2, w) for w in g]
                j = int(np.argmax(vals))
                w5, s5 = float(g[j]), float(vals[j])
                try:
                    be = brentq(lambda k: max(s_of(k, w) for w in g) - S0 - 1e-9,
                                0.0, 8.0)
                except ValueError:
                    be = float("nan")
                rows.append((s5 - S0, nm, PNAME[id(par)], dep, ksh, ncell,
                             med, w5, s5, be))
        for g5, nm, pn, dep, ksh, ncell, med, w5, s5, be in sorted(
                rows, key=lambda r: -r[0])[:12]:
            print(f"  {nm:<10}{pn:<20}{dep:>4}{ksh:>7}{ncell:>9,}{med:>7}"
                  f"{w5:>7.3f}{s5:>10.4f}{g5:>+9.4f}{be:>10.3f}")


if __name__ == "__main__":
    main()
