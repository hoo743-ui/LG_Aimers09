r"""18회차(953.7374)로 `k3` 를 실측하고 다음 축을 훑는다. 재학습 0회, LB 0장.

## 이번 점의 진짜 정보

`k3` 자체보다 **3단 전이율이 축을 바꿔도 같은가**가 중요하다.

    k2 = 0.9651   플래툰 x 카운트   (16/17회차로 실측)
    k3 = ?        플래툰 x 주자유무  (18회차로 실측)

둘이 비슷하면 "3단 중첩의 전이율 ~0.96" 이 **축에 무관한 상수**라는 뜻이고,
그때부터 새 축에 대한 차용이 근거를 얻는다. 다르면 축마다 재야 한다 — 14·16회차가
그랬듯이.

## 2진화가 이겼다는 것이 이번 라운드의 교훈

원장 5-b 는 `주자상태`(8단) −0.01 / `주자수`(4단) +0.43 을 보고 폭을 닫았다.
같은 정보를 `주자유무`(2진)로 묶으면 살아난다. 그래서 **남은 축들도 2진화해서**
다시 훑는다.

    .\.venv\Scripts\python.exe exp\solve_k3.py
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

# (w1, w2, w3, 적용 alpha, 적용 center, 실측 LB)
OBS4 = [(0.3904, 0.0, 0.0, 1.083674, 0.598664, 940.1357117095),
        (0.1990, 0.0, 0.0, 1.089306, 0.620389, 946.3826029949),
        (0.2000, 0.2000, 0.0, 1.090437, 0.620268, 950.0112119476),
        (0.2000, 0.5785, 0.0, 1.089294, 0.622802, 952.4231549068)]
OBS18 = (0.2000, 0.5470, 0.3000, 1.089163, 0.622907, 953.7373675006)
KSH1, KSH2, KSH3 = 300, 800, 2000


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
    CNT = (col("balls_before") * 4 + col("strikes_before")).astype(np.int64)
    NR = col("num_runners_on").astype(np.int64)
    OB = (NR > 0).astype(np.int64)
    PH = P * 10 + BH
    PHC, PHO = PH * 100 + CNT, PH * 10 + OB

    tr, va = season <= 2023, season == 2024
    yv = y[va]
    pm = np.load(os.path.join(ROOT, "exp",
                              "valpred_cat_s3.npz"))["p"].astype(np.float64)

    def dev24(parent, child, k):
        u, d = nested_dev(parent[tr], child[tr], y[tr], k)
        return lookup(u, d, child[va])

    D = [dev24(P, PH, KSH1), dev24(PH, PHC, KSH2), dev24(PH, PHO, KSH3)]
    C_M = S13 * BASE / (1e5 * A13)
    S2_M, M_M = C_M / A13, (R_EVAL - C13 * (1 - A13)) / A13
    v = [float(np.var(x)) for x in D]
    cm = [float(np.cov(pm, x, ddof=0)[0, 1]) for x in D]
    cy = [float(np.cov(x, yv, ddof=0)[0, 1]) for x in D]
    mu = [float(np.mean(x)) for x in D]
    cc = [[float(np.cov(a, b, ddof=0)[0, 1]) for b in D] for a in D]

    def realized(W, K, L, A, B):
        V = S2_M
        for i in range(3):
            V += 2 * W[i] * L[i] * cm[i] + W[i] ** 2 * L[i] * v[i]
            for j in range(i + 1, 3):
                V += 2 * W[i] * W[j] * np.sqrt(L[i] * L[j]) * cc[i][j]
        C = C_M + sum(W[i] * K[i] * cy[i] for i in range(3))
        m = M_M + sum(W[i] * mu[i] for i in range(3))
        return 1e5 * (1 - (A * A * V - 2 * A * C + BASE
                           + (A * m + B - R_EVAL) ** 2) / BASE)

    k1, l1, k2, l2 = fsolve(
        lambda t: [realized([a, b, 0], [t[0], t[2], 0], [t[1], t[3], 1],
                            A, c * (1 - A)) - s
                   for a, b, _, A, c, s in OBS4], [0.7685, .9725, .9340, .9725])
    w1, w2, w3, A18, c18, S18 = OBS18
    k3 = brentq(lambda k: realized([w1, w2, w3], [k1, k2, k], [l1, l2, l2],
                                   A18, c18 * (1 - A18)) - S18, -2.0, 4.0)

    print("=== 깊이별 전이율 — 3단이 두 축에서 각각 실측됐다 ===")
    print(f"  2단  플래툰            k1 = {k1:.4f}   lam1 = {l1:.4f}")
    print(f"  3단  플래툰x카운트      k2 = {k2:.4f}   lam2 = {l2:.4f}")
    print(f"  3단  플래툰x주자유무    k3 = {k3:.4f}   (lam3 = lam2 가정)")
    print(f"  두 3단 축의 차 {abs(k3 - k2):.4f}  "
          f"-> {'축에 무관한 상수로 볼 수 있다' if abs(k3 - k2) < 0.10 else '축마다 다르다'}")
    print(f"  차용값 0.9651 로 예측했고 실측 오차 "
          f"{S18 - realized([w1, w2, w3], [k1, k2, k2], [l1, l2, l2], A18, c18 * (1 - A18)):+.4f}")

    def best(W, K):
        L = [l1, l2, l2]
        V = S2_M
        for i in range(3):
            V += 2 * W[i] * L[i] * cm[i] + W[i] ** 2 * L[i] * v[i]
            for j in range(i + 1, 3):
                V += 2 * W[i] * W[j] * np.sqrt(L[i] * L[j]) * cc[i][j]
        C = C_M + sum(W[i] * K[i] * cy[i] for i in range(3))
        m = M_M + sum(W[i] * mu[i] for i in range(3))
        A = C / V
        return 1e5 * C * C / (V * BASE), A, R_EVAL - A * m

    K = [k1, k2, k3]
    g = np.linspace(0, 1.2, 121)
    s_b, a_b, b_b, c_b = max((best([a, b, c], K)[0], a, b, c)
                             for a in g for b in g for c in g)
    _, A_b, B_b = best([a_b, b_b, c_b], K)
    print(f"\n=== 실측 k3 로 3축 재최적화 ===")
    print(f"  현행 (0.20, 0.5470, 0.30)      {S18:.4f}  [18회차 실측]")
    print(f"  최적 ({a_b:.3f}, {b_b:.3f}, {c_b:.3f})      {s_b:.4f}  "
          f"({s_b - S18:+.4f})")
    print(f"    alpha={A_b:.6f}  center={B_b / (1 - A_b):.6f}")

    # ---------- 남은 축을 2진화해서 훑는다 ----------
    inn = col("inning")
    outs = col("outs_before")
    sd = col("score_diff_pitcher_team")
    r2 = col("runner_on_2b")
    r3 = col("runner_on_3b")
    r1 = col("runner_on_1b")
    li = col("li")
    tb = col("top_bottom").astype(np.int64)
    S2_ = (col("strikes_before") == 2).astype(np.int64)

    CANDS = [
        ("득점권(2·3루)", ((r2 + r3) > 0).astype(np.int64)),
        ("1루만", ((r1 > 0) & (r2 + r3 == 0)).astype(np.int64)),
        ("2아웃", (outs == 2).astype(np.int64)),
        ("이닝>=7", (inn >= 7).astype(np.int64)),
        ("이닝>=6", (inn >= 6).astype(np.int64)),
        ("접전|점수차|<=2", (np.abs(sd) <= 2).astype(np.int64)),
        ("리드중", (sd > 0).astype(np.int64)),
        ("초말", tb),
        ("li>=1", (li >= 1.0).astype(np.int64)),
        ("2스트라이크", S2_),
    ]

    p_cur = pm + w1 * D[0] + w2 * D[1] + w3 * D[2]
    V_C = S2_M
    L = [l1, l2, l2]
    W = [w1, w2, w3]
    for i in range(3):
        V_C += 2 * W[i] * L[i] * cm[i] + W[i] ** 2 * L[i] * v[i]
        for j in range(i + 1, 3):
            V_C += 2 * W[i] * W[j] * np.sqrt(L[i] * L[j]) * cc[i][j]
    C_C = C_M + sum(W[i] * K[i] * cy[i] for i in range(3))
    S_CUR = 1e5 * C_C * C_C / (V_C * BASE)
    k3d = 0.5 * (k2 + k3)          # 3단 실측 두 개의 평균
    print(f"\n=== 남은 축 2진화 훑기 (부모=플래툰, 축소 2000, "
          f"3단 실측 평균 k={k3d:.4f}) ===")
    print(f"  현행 아핀최적 기준선 {S_CUR:.4f}")
    print(f"  {'축':<16}{'w4*':>7}{'증분':>9}{'k=0 손실':>10}"
          f"{'폴드 cov 부호':>14}")
    rows = []
    for nm, ax in CANDS:
        d4 = dev24(PH, PH * 10 + ax, 2000)
        cc4 = float(np.cov(p_cur, d4, ddof=0)[0, 1])
        v4, cy4 = float(np.var(d4)), float(np.cov(d4, yv, ddof=0)[0, 1])

        def s_of(w, kk):
            return 1e5 * (C_C + w * kk * cy4) ** 2 / (
                (V_C + 2 * w * l2 * cc4 + w * w * l2 * v4) * BASE)

        gg = np.linspace(0, 1.2, 1201)
        vals = [s_of(w, k3d) for w in gg]
        j = int(np.argmax(vals))
        w4, s4 = float(gg[j]), float(vals[j])

        cs = []
        for f in (2021, 2022, 2023, 2024):
            mt, mv = season < f, season == f
            u, dd = nested_dev(PH[mt], (PH * 10 + ax)[mt], y[mt], 2000)
            cs.append(float(np.cov(lookup(u, dd, (PH * 10 + ax)[mv]),
                                   y[mv], ddof=0)[0, 1]))
        pos = sum(c > 0 for c in cs)
        rows.append((s4 - S_CUR, nm, w4, s_of(w4, 0.0) - S_CUR, pos, cs))
    for gain, nm, w4, loss0, pos, cs in sorted(rows, reverse=True):
        print(f"  {nm:<16}{w4:>7.3f}{gain:>+9.3f}{loss0:>+10.3f}"
              f"{pos:>10}/4   " + " ".join(f"{c:+.1e}" for c in cs))


if __name__ == "__main__":
    main()
