r"""측정된 전이율로 잔여 편차 축을 재평가한다. 재학습 0회, LB 0장.

`solve_k2.py` 가 LB 5점에서 깊이별 전이율을 실측했다.

    2단 (투수x타자손)          k1 = 0.7689   lam1 = 0.9731
    3단 (플래툰x카운트)         k2 = 0.9651   lam2 = 1.0422

원장 5-b / 5-c 는 이 축들을 **전이율을 낮게 가정한 채** 접었다. 이제 깊이에 맞는
실측값으로 다시 잰다. 4단은 미측정이므로 **손익분기 k4** 를 함께 낸다.

각 후보는 현행 구성 `p = 모델 + 0.20 d1 + 0.5785 d2` 위에 얹는다.

    .\.venv\Scripts\python.exe exp\axis_scan.py
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
OBS = [(0.3904, 0.0, 1.083674, 0.598664, 940.1357117095),
       (0.1990, 0.0, 1.089306, 0.620389, 946.3826029949),
       (0.2000, 0.2000, 1.090437, 0.620268, 950.0112119476),
       (0.2000, 0.5785, 1.089294, 0.622802, 952.4231549068)]
W1C, W2C = 0.20, 0.5785


def nested_dev_table(parent, child, y, k):
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

    def col(n):
        return np.asarray(X[:, ixc[n]], dtype=np.float64)

    P = col("pitcher_id").astype(np.int64)
    BT = col("batter_id").astype(np.int64)
    BH = col("batter_hand").astype(np.int64)
    PHAND = col("pitcher_hand").astype(np.int64)
    CNT = (col("balls_before") * 4 + col("strikes_before")).astype(np.int64)
    PH = P * 10 + BH
    PHC = PH * 100 + CNT
    BP = BT * 10 + PHAND

    inn = np.clip(col("inning"), 1, 9).astype(np.int64)
    inn3 = np.digitize(inn, [4, 7])                    # 1-3 / 4-6 / 7+
    outs = col("outs_before").astype(np.int64)
    nrun = col("num_runners_on").astype(np.int64)
    onbase = (nrun > 0).astype(np.int64)
    tb = col("top_bottom").astype(np.int64)
    bst = col("base_state").astype(np.int64)
    sd3 = np.digitize(col("score_diff_pitcher_team"), [-2, 3]).astype(np.int64)

    tr, va = season <= 2023, season == 2024
    yv = y[va]
    pm = np.load(os.path.join(ROOT, "exp",
                              "valpred_cat_s3.npz"))["p"].astype(np.float64)

    def dev_of(parent, child, k):
        u, d, c = nested_dev_table(parent[tr], child[tr], y[tr], k)
        return lookup(u, d, child[va]), int(np.median(c)), len(u)

    d1, _, _ = dev_of(P, PH, 300)
    d2, _, _ = dev_of(PH, PHC, 800)

    # --- 앵커 + 동시 해 (solve_k2.py 와 동일) ---
    C_M = S13 * BASE / (1e5 * A13)
    S2_M = C_M / A13
    M_M = (R_EVAL - C13 * (1 - A13)) / A13
    v1, v2 = np.var(d1), np.var(d2)
    cm1, cm2 = np.cov(pm, d1, ddof=0)[0, 1], np.cov(pm, d2, ddof=0)[0, 1]
    c12 = np.cov(d1, d2, ddof=0)[0, 1]
    cy1, cy2 = np.cov(d1, yv, ddof=0)[0, 1], np.cov(d2, yv, ddof=0)[0, 1]
    mu1, mu2 = np.mean(d1), np.mean(d2)

    def sc(w1, w2, A, B, k1, l1, k2, l2):
        lx = np.sqrt(max(l1 * l2, 1e-12))
        V = (S2_M + 2 * w1 * l1 * cm1 + w1 * w1 * l1 * v1
             + 2 * w2 * l2 * cm2 + w2 * w2 * l2 * v2 + 2 * w1 * w2 * lx * c12)
        C = C_M + w1 * k1 * cy1 + w2 * k2 * cy2
        m = M_M + w1 * mu1 + w2 * mu2
        return 1e5 * (1 - (A * A * V - 2 * A * C + BASE
                           + (A * m + B - R_EVAL) ** 2) / BASE)

    k1, l1, k2, l2 = fsolve(
        lambda v: [sc(a, b, A, c * (1 - A), *v) - s for a, b, A, c, s in OBS],
        [0.7685, 0.9725, 0.9340, 0.9725])
    print(f"실측 전이율   2단 k1={k1:.4f} lam1={l1:.4f}   "
          f"3단 k2={k2:.4f} lam2={l2:.4f}")

    # --- 현행 구성의 평가셋 적률 ---
    lx = np.sqrt(l1 * l2)
    V_C = (S2_M + 2 * W1C * l1 * cm1 + W1C ** 2 * l1 * v1
           + 2 * W2C * l2 * cm2 + W2C ** 2 * l2 * v2 + 2 * W1C * W2C * lx * c12)
    C_C = C_M + W1C * k1 * cy1 + W2C * k2 * cy2
    M_C = M_M + W1C * mu1 + W2C * mu2
    p_cur = pm + W1C * d1 + W2C * d2
    S_CUR = 1e5 * C_C * C_C / (V_C * BASE)
    print(f"현행 기준선   (w1={W1C}, w2={W2C}) 아핀최적 {S_CUR:.4f}   "
          f"[17회차 실측 952.4232]\n")

    def eval_axis(d3, k3, l3):
        """추가 편차 d3 를 얹었을 때의 최적 w3 와 점수."""
        cc3 = np.cov(p_cur, d3, ddof=0)[0, 1]
        v3, cy3, mu3 = np.var(d3), np.cov(d3, yv, ddof=0)[0, 1], np.mean(d3)

        def s_of(w):
            V = V_C + 2 * w * l3 * cc3 + w * w * l3 * v3
            C = C_C + w * k3 * cy3
            return 1e5 * C * C / (V * BASE)

        g = np.linspace(0, 2.0, 2001)
        vals = np.array([s_of(w) for w in g])
        j = int(np.argmax(vals))
        return float(g[j]), float(vals[j]), (cc3, v3, cy3, mu3)

    def breakeven_k(d3, l3):
        """이득이 0 이 되는 최소 전이율."""
        try:
            return brentq(lambda k: eval_axis(d3, k, l3)[1] - S_CUR - 1e-9,
                          0.0, 6.0)
        except ValueError:
            return float("nan")

    CAND3 = [("이닝3구간", inn3), ("주자수", nrun), ("아웃", outs),
             ("주자유무", onbase), ("초말", tb), ("주자상태", bst),
             ("점수차3", sd3)]
    CAND4 = [("아웃", outs), ("주자유무", onbase), ("초말", tb),
             ("이닝3구간", inn3)]

    print("=== 3단 축 (부모=플래툰) — 실측 k=%.4f lam=%.4f 적용 ===" % (k2, l2))
    print(f"{'축':<12}{'축소k':>7}{'셀수':>8}{'중앙n':>7}{'w3*':>7}"
          f"{'점수':>10}{'증분':>9}{'손익분기k':>10}")
    rows3 = []
    for nm, ax in CAND3:
        for ksh in (800, 2000):
            d3, med, ncell = dev_of(PH, PH * 100 + ax, ksh)
            w3, s3, _ = eval_axis(d3, k2, l2)
            be = breakeven_k(d3, l2)
            rows3.append((s3 - S_CUR, nm, ksh, w3, s3, be, d3))
            print(f"{nm:<12}{ksh:>7}{ncell:>8,}{med:>7}{w3:>7.3f}"
                  f"{s3:>10.4f}{s3 - S_CUR:>+9.4f}{be:>10.3f}")

    print(f"\n=== 4단 축 (부모=플래툰x카운트) — 3단 k 를 빌려 씀. "
          f"손익분기 k4 가 판단 기준 ===")
    print(f"{'축':<12}{'축소k':>7}{'셀수':>8}{'중앙n':>7}{'w4*':>7}"
          f"{'점수':>10}{'증분':>9}{'손익분기k':>10}")
    for nm, ax in CAND4:
        for ksh in (2000, 5000):
            d4, med, ncell = dev_of(PHC, PHC * 10 + ax, ksh)
            w4, s4, _ = eval_axis(d4, k2, l2)
            be = breakeven_k(d4, l2)
            print(f"{nm:<12}{ksh:>7}{ncell:>8,}{med:>7}{w4:>7.3f}"
                  f"{s4:>10.4f}{s4 - S_CUR:>+9.4f}{be:>10.3f}")

    print(f"\n=== 타자 쪽 계층 — 깊이별 실측 k 적용 (5-c 재평가) ===")
    print(f"{'후보':<22}{'깊이':>5}{'w*':>7}{'점수':>10}{'증분':>9}{'손익분기k':>10}")
    for nm, par, ch, ksh, kk, ll, dep in [
            ("타자x투수손", BT, BP, 600, k1, l1, "2단"),
            ("타자x투수손", BT, BP, 300, k1, l1, "2단"),
            ("타자x투수손x카운트", BP, BP * 100 + CNT, 2000, k2, l2, "3단"),
            ("타자x카운트", BT, BT * 100 + CNT, 800, k1, l1, "2단")]:
        d3, med, ncell = dev_of(par, ch, ksh)
        w3, s3, _ = eval_axis(d3, kk, ll)
        be = breakeven_k(d3, ll)
        print(f"{nm + f' k={ksh}':<22}{dep:>5}{w3:>7.3f}{s3:>10.4f}"
              f"{s3 - S_CUR:>+9.4f}{be:>10.3f}")

    # --- 최상위 3단 후보를 w1,w2,w3 동시 최적화 ---
    rows3.sort(reverse=True, key=lambda r: r[0])
    gain, nm, ksh, _, _, _, dbest = rows3[0]
    print(f"\n=== 최상위 후보 '{nm}' (축소 {ksh}) 3축 동시 최적화 ===")
    cc3 = np.cov(pm, dbest, ddof=0)[0, 1]
    c13 = np.cov(d1, dbest, ddof=0)[0, 1]
    c23 = np.cov(d2, dbest, ddof=0)[0, 1]
    v3, cy3, mu3 = np.var(dbest), np.cov(dbest, yv, ddof=0)[0, 1], np.mean(dbest)

    def s3_of(w1, w2, w3):
        V = (S2_M + 2 * w1 * l1 * cm1 + w1 * w1 * l1 * v1
             + 2 * w2 * l2 * cm2 + w2 * w2 * l2 * v2
             + 2 * w3 * l2 * cc3 + w3 * w3 * l2 * v3
             + 2 * w1 * w2 * lx * c12 + 2 * w1 * w3 * lx * c13
             + 2 * w2 * w3 * l2 * c23)
        C = C_M + w1 * k1 * cy1 + w2 * k2 * cy2 + w3 * k2 * cy3
        m = M_M + w1 * mu1 + w2 * mu2 + w3 * mu3
        A = C / V
        return 1e5 * C * C / (V * BASE), A, R_EVAL - A * m

    g = np.linspace(0, 1.2, 121)
    best = max(((s3_of(a, b, c)[0], a, b, c) for a in g for b in g for c in g))
    s_b, a_b, b_b, c_b = best
    _, A_b, B_b = s3_of(a_b, b_b, c_b)
    print(f"  w1={a_b:.3f}  w2={b_b:.3f}  w3={c_b:.3f}  ->  {s_b:.4f}  "
          f"({s_b - S_CUR:+.4f} vs 현행)")
    print(f"  script.py 용: alpha={A_b:.6f}  center={B_b / (1 - A_b):.6f}")


if __name__ == "__main__":
    main()
