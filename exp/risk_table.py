r"""후보 축의 **진짜 위험**을 잰다. 재학습 0회, LB 0장.

## axis_scan.py 의 손익분기 k 는 틀린 질문이었다

거기서는 `k` 를 바꿀 때마다 `w3` 를 다시 최적화했다. 그러면 최악의 경우 `w3`=0 이
선택되므로 **증분이 절대 음수가 안 된다.** 실제 제출은 `w3` 를 하나로 고정하므로
전이가 낮으면 분산만 늘어 손해다. 16회차 표(`k2` 행 x `w2` 열)가 옳은 형식이다.

## 그리고 로컬 적률이 2024 한 폴드뿐이다

`axis_scan.py` 의 순위는 2024 폴드 하나로 매겨졌다. 이 프로젝트가 반복해서 데인
것이 **한 폴드 최적의 비전이**다 (워크포워드 `w`=0.595 vs 평가셋 최적 0.20).

모델 예측은 2024 폴드만 있어 다폴드 점수는 못 낸다. 그러나 `k` 가 곱해지는 대상인
`cov(d3, y)` 자체는 모델 없이 폴드별로 잴 수 있다. 표를 `<f` 로 만들고 `f` 에서
`cov` 를 재면 **신호가 시즌을 넘는지**가 직접 보인다. 이게 결정적 검사다.

    .\.venv\Scripts\python.exe exp\risk_table.py
"""
import json
import os

import numpy as np
from scipy.optimize import fsolve

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
    return u, cnt * (cell - pmean[np.searchsorted(pu, par)]) / (cnt + k)


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
    BH = col("batter_hand").astype(np.int64)
    CNT = (col("balls_before") * 4 + col("strikes_before")).astype(np.int64)
    PH, PHC = P * 10 + BH, (P * 10 + BH) * 100 + CNT
    nrun = col("num_runners_on").astype(np.int64)
    onbase = (nrun > 0).astype(np.int64)
    bst = col("base_state").astype(np.int64)
    inn3 = np.digitize(np.clip(col("inning"), 1, 9), [4, 7]).astype(np.int64)

    tr, va = season <= 2023, season == 2024
    yv = y[va]
    pm = np.load(os.path.join(ROOT, "exp",
                              "valpred_cat_s3.npz"))["p"].astype(np.float64)

    def dev24(parent, child, k):
        u, d = nested_dev_table(parent[tr], child[tr], y[tr], k)
        return lookup(u, d, child[va])

    d1, d2 = dev24(P, PH, 300), dev24(PH, PHC, 800)
    C_M = S13 * BASE / (1e5 * A13)
    S2_M, M_M = C_M / A13, (R_EVAL - C13 * (1 - A13)) / A13
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
    lx = np.sqrt(l1 * l2)
    V_C = (S2_M + 2 * W1C * l1 * cm1 + W1C ** 2 * l1 * v1
           + 2 * W2C * l2 * cm2 + W2C ** 2 * l2 * v2 + 2 * W1C * W2C * lx * c12)
    C_C = C_M + W1C * k1 * cy1 + W2C * k2 * cy2
    S_CUR = 1e5 * C_C * C_C / (V_C * BASE)
    p_cur = pm + W1C * d1 + W2C * d2
    print(f"실측 전이율 2단 k={k1:.4f} / 3단 k={k2:.4f} lam={l2:.4f}")
    print(f"현행 기준선 {S_CUR:.4f}\n")

    CANDS = [("주자유무", onbase, 2000), ("주자유무", onbase, 800),
             ("주자상태", bst, 2000), ("주자수", nrun, 2000),
             ("이닝3구간", inn3, 2000)]

    # ---------- 1) 폴드별 신호 전이 (모델 불필요, 결정적 검사) ----------
    print("=== 1) cov(d3, y) 가 시즌을 넘는가 — 표는 <f, 측정은 f ===")
    print(f"{'축':<10}{'축소':>6}", end="")
    for f in (2021, 2022, 2023, 2024):
        print(f"{f:>11}", end="")
    print(f"{'부호':>7}{'평균':>11}")
    stable = {}
    for nm, ax, ksh in CANDS:
        cs = []
        for f in (2021, 2022, 2023, 2024):
            m_tr, m_va = season < f, season == f
            u, d = nested_dev_table(PH[m_tr], (PH * 100 + ax)[m_tr],
                                    y[m_tr], ksh)
            dv = lookup(u, d, (PH * 100 + ax)[m_va])
            cs.append(float(np.cov(dv, y[m_va], ddof=0)[0, 1]))
        pos = sum(c > 0 for c in cs)
        stable[(nm, ksh)] = (pos, float(np.mean(cs)))
        print(f"{nm:<10}{ksh:>6}", end="")
        for c in cs:
            print(f"{c:>+11.3e}", end="")
        print(f"{pos:>5}/4{np.mean(cs):>+11.3e}")

    # 대조 — 이미 실측된 두 축이 같은 검사에서 어떻게 나오는지
    print("\n  [대조군] 이미 LB 로 전이가 확인된 축")
    for nm, par, ch, ksh in [("플래툰", P, PH, 300),
                             ("플래툰x카운트", PH, PHC, 800)]:
        cs = []
        for f in (2021, 2022, 2023, 2024):
            m_tr, m_va = season < f, season == f
            u, d = nested_dev_table(par[m_tr], ch[m_tr], y[m_tr], ksh)
            cs.append(float(np.cov(lookup(u, d, ch[m_va]),
                                   y[m_va], ddof=0)[0, 1]))
        print(f"  {nm:<14}", end="")
        for c in cs:
            print(f"{c:>+11.3e}", end="")
        print(f"  {sum(c > 0 for c in cs)}/4")

    # ---------- 2) w3 고정 위험표 ----------
    print("\n=== 2) w3 를 고정했을 때 진짜 전이율 k3 에 따른 점수 ===")
    print("    (16회차 표와 같은 형식. 음수면 현행보다 손해)")
    for nm, ax, ksh in CANDS[:3]:
        d3 = dev24(PH, PH * 100 + ax, ksh)
        cc3, v3 = np.cov(p_cur, d3, ddof=0)[0, 1], np.var(d3)
        cy3 = np.cov(d3, yv, ddof=0)[0, 1]

        def s_of(w, k3):
            V = V_C + 2 * w * l2 * cc3 + w * w * l2 * v3
            return 1e5 * (C_C + w * k3 * cy3) ** 2 / (V * BASE)

        ws = [0.15, 0.30, 0.45, 0.60]
        print(f"\n  [{nm} 축소{ksh}]   " + "".join(f"w3={w:<9.2f}" for w in ws))
        for k3 in (0.0, 0.25, 0.50, 0.7689, 0.9651, 1.20):
            tag = ("  <- 2단 실측" if abs(k3 - k1) < 1e-3 else
                   "  <- 3단 실측" if abs(k3 - k2) < 1e-3 else "")
            print(f"   k3={k3:<6.4f}" + "".join(
                f"{s_of(w, k3) - S_CUR:>+12.3f}" for w in ws) + tag)


if __name__ == "__main__":
    main()
