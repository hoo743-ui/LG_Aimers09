r"""5번째 항 후보의 **진짜 위험**을 잰다. 재학습 0회, LB 0장.

## `axis_scan5.py` 의 손익분기 k 는 못 쓴다

`risk_table.py` 가 이미 적어둔 함정을 그대로 밟았다 — `k` 마다 `w5` 를 재최적화하면
최악의 경우 `w5`=0 이 뽑혀 **증분이 절대 음수가 안 된다.** 실제 제출은 `w5` 를
하나로 고정하므로, 전이가 낮으면 분산만 늘어 손해다. 그래서 손익분기 0.059 는
"하방이 없다"는 뜻이 아니라 **질문이 틀렸다**는 뜻이다.

여기서는 두 검사를 한다.

  1) `cov(d5, y)` 가 시즌을 넘는가 — 표는 `<f` 로 만들고 `f` 에서 잰다.
     모델이 없어도 되고, 신호의 전이 여부가 직접 보인다. 18회차를 통과시킨 검사다.
  2) `w5` **고정** 위험표 — 진짜 `k5` 가 빗나갔을 때 얼마를 잃는가.

기준선은 19회차 챔피언(4항)이다. `(k_c, k_n)` 은 궤적으로만 잡히지만
`axis_scan5.py` 에서 두 읽기가 소수 3자리까지 같은 답을 냈으므로 (a) 로 고정한다.

    .\.venv\Scripts\python.exe exp\risk5.py
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
KC, KN = 0.8651, 0.9726          # 읽기 (a)
FOLDS = (2021, 2022, 2023, 2024)


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
    PHO = PH * 10 + OB
    PHAC = PHA * 100 + CNT
    TB = col("top_bottom").astype(np.int64)
    R1 = col("runner_on_1b").astype(np.int64)
    SCOR = (col("score_diff_pitcher_team") == 0).astype(np.int64)

    tr, va = season <= 2023, season == 2024
    yv = y[va]
    pm = np.load(os.path.join(ROOT, "exp",
                              "valpred_cat_s3.npz"))["p"].astype(np.float64)
    dev24 = lambda par, ch, k: lookup(*nested_dev(par[tr], ch[tr], y[tr], k),
                                      ch[va])

    C_M = S13 * BASE / (1e5 * A13)
    S2_M, M_M = C_M / A13, (R_EVAL - C13 * (1 - A13)) / A13
    d1 = dev24(P, PH, 300)
    d2 = dev24(PH, PH * 100 + CNT, 800)
    d3 = dev24(PH, PHO, 2000)
    dC = dev24(PH, PHA, 2000)
    dN = dev24(PHA, PHAC, 800)

    def maker(D, L):
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
        lambda t: [real(maker([d1, d2, d3], [t[1], t[3], 1]), [a, b, 0],
                        [t[0], t[2], 0], A, c * (1 - A)) - s
                   for a, b, A, c, s in OBS4], [0.7685, .9725, .9340, .9725])
    w1c, w2c, w3c, A18, c18, S18 = OBS18
    k3 = brentq(lambda k: real(maker([d1, d2, d3], [l1, l2, l2]),
                               [w1c, w2c, w3c], [k1, k2, k], A18,
                               c18 * (1 - A18)) - S18, -2, 4)
    print(f"실측 전이율  2단 k1={k1:.4f}  3단 k2={k2:.4f} k3={k3:.4f}  "
          f"lam2={l2:.4f}")

    w1, wc, wn, w3, _, _, S19 = OBS19
    W4, K4 = [w1, wc, wn, w3], [k1, KC, KN, k3]
    VC4 = maker([d1, dC, dN, d3], [l1, l2, l2, l2])
    V_C, C_C, _ = VC4(W4, K4)
    S_CUR = 1e5 * C_C * C_C / (V_C * BASE)
    p_cur = pm + w1 * d1 + wc * dC + wn * dN + w3 * d3
    print(f"기준선(19회차 챔피언) {S_CUR:.4f}   [실측 {S19:.4f}]\n")

    CANDS = [("초말|플래툰x거친x카운트", PHAC, TB, 10, 5000, 5),
             ("초말|플래툰x거친x카운트", PHAC, TB, 10, 2000, 5),
             ("카운트|플래툰x주자", PHO, CNT, 100, 2000, 4),
             ("1루|플래툰", PH, R1, 100, 5000, 3),
             ("동점|플래툰", PH, SCOR, 100, 5000, 3)]

    # ---------- 1) 폴드별 신호 전이 — 결정적 검사 ----------
    print("=== 1) cov(d5, y) 가 시즌을 넘는가 — 표는 <f, 측정은 f ===")
    print(f"  {'후보':<26}{'축소':>6}", end="")
    for f in FOLDS:
        print(f"{f:>12}", end="")
    print(f"{'부호':>7}{'단조':>6}")
    for nm, par, ax, mul, ksh, _ in CANDS:
        ch = par * mul + ax
        cs = []
        for f in FOLDS:
            m_tr, m_va = season < f, season == f
            u, d = nested_dev(par[m_tr], ch[m_tr], y[m_tr], ksh)
            cs.append(float(np.cov(lookup(u, d, ch[m_va]),
                                   y[m_va], ddof=0)[0, 1]))
        mono = all(b >= a for a, b in zip(cs, cs[1:]))
        print(f"  {nm:<26}{ksh:>6}", end="")
        for c in cs:
            print(f"{c:>+12.3e}", end="")
        print(f"{sum(c > 0 for c in cs):>5}/4{'  예' if mono else '  아니오':>6}")

    print("\n  [대조군] LB 로 전이가 확인된 축")
    for nm, par, ch, ksh in [("플래툰", P, PH, 300),
                             ("플래툰x카운트", PH, PH * 100 + CNT, 800),
                             ("플래툰x주자유무", PH, PHO, 2000)]:
        cs = []
        for f in FOLDS:
            m_tr, m_va = season < f, season == f
            u, d = nested_dev(par[m_tr], ch[m_tr], y[m_tr], ksh)
            cs.append(float(np.cov(lookup(u, d, ch[m_va]),
                                   y[m_va], ddof=0)[0, 1]))
        print(f"  {nm:<32}", end="")
        for c in cs:
            print(f"{c:>+12.3e}", end="")
        print(f"{sum(c > 0 for c in cs):>5}/4")

    # ---------- 2) w5 고정 위험표 ----------
    print("\n=== 2) w5 를 고정했을 때 진짜 k5 에 따른 증분 (음수면 손해) ===")
    for nm, par, ax, mul, ksh, dep in CANDS:
        d5 = dev24(par, par * mul + ax, ksh)
        cc5 = float(np.cov(p_cur, d5, ddof=0)[0, 1])
        v5, cy5 = float(np.var(d5)), float(np.cov(d5, yv, ddof=0)[0, 1])

        def s_of(w, k5):
            V = V_C + 2 * w * l2 * cc5 + w * w * l2 * v5
            return 1e5 * (C_C + w * k5 * cy5) ** 2 / (V * BASE)

        ws = ([0.20, 0.40, 0.79, 1.20] if dep == 5 else
              [0.20, 0.35, 0.50, 0.68] if dep == 4 else [0.15, 0.30, 0.50, 0.73])
        print(f"\n  [{nm} 축소{ksh}]  " + "".join(f"w5={w:<8.2f}" for w in ws))
        for k5 in (0.0, 0.25, 0.50, k1, k2, 1.20):
            tag = ("  <- 2단 실측" if abs(k5 - k1) < 1e-3 else
                   "  <- 3단 실측" if abs(k5 - k2) < 1e-3 else "")
            print(f"   k5={k5:<6.4f}" + "".join(
                f"{s_of(w, k5) - S_CUR:>+12.3f}" for w in ws) + tag)


if __name__ == "__main__":
    main()
