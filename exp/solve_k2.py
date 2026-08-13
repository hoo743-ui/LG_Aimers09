r"""LB 5점(13~17)으로 두 편차 축의 전이율 4개를 **동시에** 푼다. 재학습 0회, LB 0장.

## 왜 이렇게 짜야 하는가

첫 시도는 플래툰 전이율만 원장 값(`k1`=0.7685, `lam1`=0.9725)으로 고정하고
로컬 적률은 여기서 새로 계산해 **섞어 썼다.** 그러자 15회차(`w2`=0) 재현이
-0.47 어긋났고, 그 편의가 `lam2` 로 흘러들어 1.25(비물리적)가 나왔다.

로컬 적률은 폴드 구성·시드·`w1` 소수점에 따라 원래 분석과 다를 수 있다. 따라서
**전이율을 빌려오지 말고 내 적률 위에서 전부 다시 풀어야 한다.**

## 앵커 — 13회차는 편차가 없어 완전해다

13회차는 `w1`=`w2`=0 이고 적용 아핀이 그 시점의 최적이었다. 여기서 평가셋
모델 단독 적률을 반올림 없이 되뽑는다 (원장의 s=0.0438 은 4자리라 못 쓴다).

    s^2 = C / A13,   m = (r - center13 (1-A13)) / A13,
    C   = score13 * base / (1e5 * A13)

## 미지수 4 / 식 4

    V(w1,w2) = s^2 + 2 w1 lam1 cov(pm,d1) + w1^2 lam1 var(d1)
                   + 2 w2 lam2 cov(pm,d2) + w2^2 lam2 var(d2)
                   + 2 w1 w2 lam_x cov(d1,d2)
    C(w1,w2) = C   + w1 k1 cov(d1,y) + w2 k2 cov(d2,y)
    m(w1,w2) = m   + w1 mean(d1) + w2 mean(d2)
    MSE      = A^2 V - 2 A C + base + (A m + B - r)^2      <- clip 미접촉이면 항등식

식은 14/15/16/17 회차 실측 네 개. `lam_x` 는 sqrt(lam1 lam2) 로 두고 민감도를 뽑는다.

    .\.venv\Scripts\python.exe exp\solve_k2.py
"""
import json
import os

import numpy as np
from scipy.optimize import fsolve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "exp", "cache")

R_EVAL = 0.460900
BASE = R_EVAL * (1 - R_EVAL)

W1S, KSH1, KSH2 = 0.20, 300, 800

# (w1, w2, 적용 alpha, 적용 center, 실측 LB)
A13, C13, S13 = 1.105030, 0.598664, 942.4577639361
OBS = [
    (0.3904, 0.0000, 1.083674, 0.598664, 940.1357117095),   # 14 center 유지(의도적 오류)
    (0.1990, 0.0000, 1.089306, 0.620389, 946.3826029949),   # 15
    (0.2000, 0.2000, 1.090437, 0.620268, 950.0112119476),   # 16
    (0.2000, 0.5785, 1.089294, 0.622802, 952.4231549068),   # 17
]


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
    dev = cnt * (cell - pmean[np.searchsorted(pu, par)]) / (cnt + k)
    return u, dev


def lookup(u, dev, keys):
    out = np.zeros(len(keys), dtype=np.float64)
    ix = np.clip(np.searchsorted(u, keys), 0, len(u) - 1)
    ok = u[ix] == keys
    out[ok] = dev[ix[ok]]
    return out


def main():
    meta = json.load(open(f"{CACHE}/cols.json"))
    ix = {c: i for i, c in enumerate(meta["cols"])}
    X = np.load(f"{CACHE}/X.npy", mmap_mode="r")
    y = np.load(f"{CACHE}/y.npy").astype(np.float64)
    season = np.load(f"{CACHE}/season.npy")

    def col(n):
        return np.asarray(X[:, ix[n]], dtype=np.float64).astype(np.int64)

    P, BH = col("pitcher_id"), col("batter_hand")
    CNT = col("balls_before") * 4 + col("strikes_before")
    PH, = (P * 10 + BH,)
    PHC = PH * 100 + CNT

    tr, va = season <= 2023, season == 2024
    u1, d1 = nested_dev_table(P[tr], PH[tr], y[tr], KSH1)
    u2, d2 = nested_dev_table(PH[tr], PHC[tr], y[tr], KSH2)
    dv1, dv2 = lookup(u1, d1, PH[va]), lookup(u2, d2, PHC[va])

    pm = np.load(os.path.join(ROOT, "exp",
                              "valpred_cat_s3.npz"))["p"].astype(np.float64)
    yv = y[va]

    # --- 로컬 적률 ---
    v1, v2 = float(np.var(dv1)), float(np.var(dv2))
    cm1 = float(np.cov(pm, dv1, ddof=0)[0, 1])
    cm2 = float(np.cov(pm, dv2, ddof=0)[0, 1])
    c12 = float(np.cov(dv1, dv2, ddof=0)[0, 1])
    cy1 = float(np.cov(dv1, yv, ddof=0)[0, 1])
    cy2 = float(np.cov(dv2, yv, ddof=0)[0, 1])
    mu1, mu2 = float(np.mean(dv1)), float(np.mean(dv2))

    # --- 13회차 앵커에서 모델 단독 평가셋 적률 ---
    C_M = S13 * BASE / (1e5 * A13)
    S2_M = C_M / A13
    M_M = (R_EVAL - C13 * (1 - A13)) / A13

    print("=== 13회차 앵커 (반올림 없이 되뽑음) ===")
    print(f"  C={C_M:.8f}  s={np.sqrt(S2_M):.6f}  m={M_M:.6f}   "
          f"(원장 0.002119 / 0.0438 / 0.473979)")
    print("\n=== 로컬 적률 (표 2019~2023 -> 2024 폴드) ===")
    print(f"  d1 var {v1:.4e}  cov(pm,d1) {cm1:+.4e}  cov(d1,y) {cy1:+.4e}  "
          f"mean {mu1:+.5f}")
    print(f"  d2 var {v2:.4e}  cov(pm,d2) {cm2:+.4e}  cov(d2,y) {cy2:+.4e}  "
          f"mean {mu2:+.5f}")
    print(f"  cov(d1,d2) {c12:+.4e}   상관 {c12 / np.sqrt(v1 * v2):+.4f}")

    def score(w1, w2, A, B, k1, lam1, k2, lam2, lam_x=None):
        if lam_x is None:
            lam_x = np.sqrt(max(lam1 * lam2, 1e-12))
        V = (S2_M + 2 * w1 * lam1 * cm1 + w1 * w1 * lam1 * v1
             + 2 * w2 * lam2 * cm2 + w2 * w2 * lam2 * v2
             + 2 * w1 * w2 * lam_x * c12)
        C = C_M + w1 * k1 * cy1 + w2 * k2 * cy2
        m = M_M + w1 * mu1 + w2 * mu2
        mse = A * A * V - 2 * A * C + BASE + (A * m + B - R_EVAL) ** 2
        return 1e5 * (1 - mse / BASE)

    print(f"\n  앵커 검산) 13회차 재현 "
          f"{score(0, 0, A13, C13 * (1 - A13), 1, 1, 1, 1):.6f}  "
          f"실측 {S13:.6f}")

    def eqs(v):
        k1, lam1, k2, lam2 = v
        return [score(w1, w2, A, c * (1 - A), k1, lam1, k2, lam2) - s
                for w1, w2, A, c, s in OBS]

    sol, _, flag, msg = fsolve(eqs, [0.7685, 0.9725, 0.9340, 0.9725],
                               full_output=True)
    k1, lam1, k2, lam2 = sol
    print(f"\n=== 동시 해 (14/15/16/17 네 점, 미지수 4) ===")
    print(f"  k1   = {k1:.4f}   (원장 0.7685)      플래툰 cov(f,y) 전이율")
    print(f"  lam1 = {lam1:.4f}   (원장 0.9725)      플래툰 2차 적률 전이율")
    print(f"  k2   = {k2:.4f}   (원장 0.9340)      플래툰x카운트 cov(f,y) 전이율")
    print(f"  lam2 = {lam2:.4f}   (빌려온 값 0.9725) 플래툰x카운트 2차 적률")
    print(f"  잔차 {np.abs(eqs(sol)).max():.2e}   수렴 {flag == 1}")

    def best(w1, w2, k1, lam1, k2, lam2):
        lam_x = np.sqrt(max(lam1 * lam2, 1e-12))
        V = (S2_M + 2 * w1 * lam1 * cm1 + w1 * w1 * lam1 * v1
             + 2 * w2 * lam2 * cm2 + w2 * w2 * lam2 * v2
             + 2 * w1 * w2 * lam_x * c12)
        C = C_M + w1 * k1 * cy1 + w2 * k2 * cy2
        m = M_M + w1 * mu1 + w2 * mu2
        A = C / V
        return 1e5 * C * C / (V * BASE), A, R_EVAL - A * m, m

    g = np.linspace(0, 1.2, 241)
    grid = np.array([[best(a, b, k1, lam1, k2, lam2)[0] for b in g] for a in g])
    ia, ib = np.unravel_index(np.argmax(grid), grid.shape)
    w1o, w2o = float(g[ia]), float(g[ib])
    so, Ao, Bo, mo = best(w1o, w2o, k1, lam1, k2, lam2)
    cur = OBS[-1][-1]

    print(f"\n=== 재최적화 (재학습 0회) ===")
    print(f"  17회차 실측                        {cur:.4f}")
    print(f"  같은 (w1,w2) 아핀만 최적           "
          f"{best(0.20, 0.5785, k1, lam1, k2, lam2)[0]:.4f}  "
          f"({best(0.20, 0.5785, k1, lam1, k2, lam2)[0] - cur:+.4f})")
    print(f"  w1,w2 까지 최적 ({w1o:.3f}, {w2o:.3f})       {so:.4f}  "
          f"({so - cur:+.4f})")
    print(f"     script.py 용: alpha={Ao:.6f}  center={Bo / (1 - Ao):.6f}")

    print(f"\n  민감도 — lam_x 를 lam1/lam2 로 바꿔 다시 풀면")
    for nm, lx in [("lam1", 0), ("lam2", 1)]:
        def eqs2(v, lx=lx):
            a, b, c, d = v
            return [score(w1, w2, A, cc * (1 - A), a, b, c, d,
                          lam_x=(b if lx == 0 else d)) - s
                    for w1, w2, A, cc, s in OBS]
        s2 = fsolve(eqs2, sol)
        print(f"    lam_x={nm}: k1={s2[0]:.4f} lam1={s2[1]:.4f} "
              f"k2={s2[2]:.4f} lam2={s2[3]:.4f}")


if __name__ == "__main__":
    main()
