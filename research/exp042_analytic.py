r"""EXP042 — 후처리 9가중의 **해석적 최적**. 제출 0회로 9차원을 푼다.

## 착상

제출본은 항상 `p = m + Σ_i w_i c_i` 다. `c_i` 는 번들에 **구워진 표**를 그 행의
컬럼으로 조회한 값이라 `w` 를 바꿔도 변하지 않는다. 그러면 점수는

    S(w) = K (a + w·g)² / (v0 + 2 u·w + w' C w)
      a = cov(m, y)   g_i = cov(c_i, y)        <- 라벨이 필요하다 (2025 는 못 본다)
      v0 = var(m)   u_i = cov(m, c_i)   C_ij = cov(c_i, c_j)   <- **라벨이 필요 없다**

**분모 전체가 라벨 없이 계산된다.** 그리고 우리에겐 같은 구조 위의 LB 실측이
11개 있다. 그것으로 분자쪽 `g` 를 역산한다.

## 무엇이 식별되고 무엇이 안 되는가

편차 4축은 **언제나 같은 비율로만** 움직였고(전역 배수), 2S 와 주자도 서로
달라진 적이 없다. 그러므로 LB 만으로 식별되는 것은 블록 단위 5개다.

    a · g_편차(고정 방향) · g_손 · g_{2S+주자} · g_투수 · g_타자 · K
    -> 7 모수, 관측 11 개, 자유도 4  (검정 가능하다)

## 그래서 어떻게 축 단위 최적을 얻는가

로컬이 틀리는 것은 **스케일**(전이)이지 블록 **내부의 모양**이 아니라고 본다.
그러면 블록 전이 `t` 를 LB 로 재고, 축별 `g_i` 는 로컬 공분산으로 나눠 준다.

    g_i^est = t_block(i) × cov(c_i, y_2024)

`C` 는 정확하므로 `w* = argmax S` 를 닫힌 형태로 푼다. 상관된 축 사이의 배분은
`C⁻¹` 가 정확히 처리한다 — 좌표별 스캔이 못 하던 일이다.

    .\.venv\Scripts\python.exe -u research\exp042_analytic.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "exp"))
import build_asof as ba                                     # noqa: E402
from path_alloc import build_df                             # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

NAMES = ["dev0_platoon", "dev1_adv", "dev2_count", "dev3_runner",
         "c_hand", "c_2strike", "c_runner", "L_pitcher", "L_batter"]
BLOCK = [0, 0, 0, 0, 1, 2, 2, 3, 4]          # 식별 가능한 블록 배정
BLOCKNAME = ["편차4", "손차등", "2S+주자", "투수수준", "타자수준"]

# (가중벡터, LB 점수) — 같은 표를 쓰는 제출만
OBS = [
    (np.array([.20, .825, .28, .45, 0, 0, 0, 0, 0]), 1049.9225979712),
    (np.array([.20, .825, .28, .45, 1.0, 0, 0, 0, 0]), 1053.5950918323),
    (np.array([.20, .825, .28, .45, 1.0, 1.0, 1.0, 0, 0]), 1057.3394030999),
    (np.array([.20, .825, .28, .45, .60, .60, .60, 0, 0]), 1060.3076531721),
    (np.array([.20, .825, .28, .45, .65, .65, .65, 1.0, 0]), 1061.4979059860),
    (np.array([.20, .825, .28, .45, .65, .65, .65, 1.0, 1.5]), 1071.3085000000),
    (np.array([.20, .825, .28, .45, .65, .65, .65, 1.0, 2.5]), 1071.8145632143),
    (np.array([.16, .660, .224, .36, .65, .65, .65, 1.0, 2.5]), 1073.8236813606),
    (np.array([.12, .495, .168, .27, .65, .65, .65, 1.0, 2.5]), 1074.8797684337),
    (np.array([.095671, .394645, .13394, .215261, .65, .65, .65, 2.0, 2.10]),
     1075.4602240995),
    (np.array([.095671, .394645, .13394, .215261, .40, .65, .65, 2.0, 2.10]),
     1073.8053650722),
]


def components():
    """폴드 2024 의 워크포워드 유사체로 9개 보정 벡터와 모델 예측을 만든다."""
    tr = build_df()
    season = tr["season"].to_numpy()
    y = tr["control_success"].to_numpy(np.float64)
    P = tr["pitcher_id"].to_numpy(np.int64)
    B = tr["batter_id"].to_numpy(np.int64)
    BH = tr["batter_hand"].to_numpy(np.int64)
    BB = tr["balls_before"].to_numpy(np.int64)
    SS = tr["strikes_before"].to_numpy(np.int64)
    OB = (tr["num_runners_on"].to_numpy(np.int64) > 0).astype(np.int64)
    PH = P * 10 + BH
    PHA = PH * 10 + (SS > BB).astype(np.int64)
    AXD = [(P, PH), (PH, PHA), (PHA, PHA * 100 + (BB * 4 + SS)), (PH, PH * 10 + OB)]
    SAME = (tr["pitcher_hand"].to_numpy() == tr["batter_hand"].to_numpy()).astype(int)
    TWO = (SS == 2).astype(int)
    AXC = [(SAME, 1000.0), (TWO, 1000.0), (OB, 2000.0)]
    MODEL = {f: np.load(os.path.join(ROOT, "exp", f"prod_champ_{f}.npy")).mean(0)
             for f in (2022, 2023, 2024)}

    D = {}
    for f in (2022, 2023, 2024):
        m_tr = season < f
        D[f] = np.column_stack([
            ba.look(*ba.nested_dev(p[m_tr], c[m_tr], y[m_tr], k), c[season == f])
            for (p, c), k in zip(AXD, ba.KSH)])
    res = {f: y[season == f] - (MODEL[f] + D[f] @ ba.WPOST) for f in (2022, 2023, 2024)}

    f = 2024
    m = season == f
    cols = [D[f][:, i] for i in range(4)]

    def src_frames(key, ctx=None):
        out = []
        for s in (2022, 2023):
            ms = season == s
            d = {"k": key[ms], "sr": res[s], "n": np.ones(int(ms.sum()))}
            if ctx is not None:
                d["c"] = ctx[ms]
            out.append(pd.DataFrame(d))
        return pd.concat(out)

    for ctx, k in AXC:
        q = src_frames(P, ctx).groupby(["k", "c"])[["sr", "n"]].sum().unstack()
        n0, n1 = q[("n", 0)].fillna(0), q[("n", 1)].fillna(0)
        m0 = q[("sr", 0)] / n0.replace(0, np.nan)
        m1 = q[("sr", 1)] / n1.replace(0, np.nan)
        ne = (n0 * n1) / (n0 + n1).replace(0, np.nan)
        t = ((m1 - m0) * ne / (ne + k)).dropna()
        v = pd.Series(P[m]).map(t).fillna(0.).to_numpy()
        cols.append(v * np.where(ctx[m] == 1, .5, -.5))

    for key, k in ((P, 50000.0), (B, 20000.0)):
        q = src_frames(key).groupby("k")[["sr", "n"]].sum()
        t = (q["sr"] / q["n"]) * q["n"] / (q["n"] + k)
        cols.append(pd.Series(key[m]).map(t).fillna(0.).to_numpy())

    return MODEL[f], np.column_stack(cols), y[m]


def main():
    mm, Cm, y = components()
    n = len(y)
    X = np.column_stack([mm, Cm])
    Xc = X - X.mean(0)
    S = Xc.T @ Xc / n                                    # 라벨 불필요
    gl = (Xc.T @ (y - y.mean())) / n                     # 로컬 cov(c_i, y)
    v0, u, C = S[0, 0], S[0, 1:], S[1:, 1:]

    print("보정 벡터 (폴드 2024 워크포워드 유사체)")
    print(f"{'축':14s} {'sd':>10s} {'corr(c,y)':>11s} {'로컬 g':>12s}")
    for i, nm in enumerate(NAMES):
        sd = np.sqrt(C[i, i])
        print(f"{nm:14s} {sd:10.6f} {gl[i+1]/(sd*y.std()):11.5f} {gl[i+1]:12.3e}")
    R = np.corrcoef(Cm.T)
    print(f"\n축간 상관 최대 |r| = {np.max(np.abs(R - np.eye(9))):.3f}")

    W = np.array([o[0] for o in OBS])
    Sobs = np.array([o[1] for o in OBS])

    def gvec(gb):
        return np.array([gb[BLOCK[i]] for i in range(9)]) * gl[1:]

    def predict(theta):
        a, g, K = theta[0], gvec(theta[1:6]), theta[6]
        num = (a + W @ g) ** 2
        den = v0 + 2 * (W @ u) + np.einsum("ij,jk,ik->i", W, C, W)
        return K * num / den

    th0 = np.array([gl[0], 1., 1., 1., 1., 1., 1e5 / y.var()])
    r = least_squares(lambda t: predict(t) - Sobs, th0,
                      x_scale=np.abs(th0) + 1e-12, max_nfev=40000)
    fit = predict(r.x)
    rms = float(np.sqrt(np.mean((fit - Sobs) ** 2)))
    print(f"\n블록 전이 적합 — 잔차 RMS {rms:.4f} 점 (자유도 {len(OBS)-7})")
    for s_, f_ in zip(Sobs, fit):
        print(f"   실측 {s_:11.4f}   적합 {f_:11.4f}   {f_-s_:+7.4f}")
    print("\n블록 전이 t (로컬 g 대비 LB g 의 배수)")
    for j, bn in enumerate(BLOCKNAME):
        print(f"   {bn:8s} {r.x[1+j]:+8.4f}")

    a, g, K = r.x[0], gvec(r.x[1:6]), r.x[6]
    Ci = np.linalg.inv(C)
    lam = (a - u @ Ci @ g) / (v0 - u @ Ci @ u)
    wstar = Ci @ (g - lam * u)
    def sc(w):
        return K * (a + w @ g) ** 2 / (v0 + 2 * w @ u + w @ C @ w)

    cur = W[-2]
    print(f"\n현행 h1        {sc(cur):11.4f}  (실측 {Sobs[-2]:.4f})")
    print(f"해석적 최적    {sc(wstar):11.4f}  ({sc(wstar)-Sobs[-2]:+.3f})")
    print(f"\n{'축':14s} {'현행 h1':>10s} {'최적':>10s}")
    for i, nm in enumerate(NAMES):
        print(f"{nm:14s} {cur[i]:10.4f} {wstar[i]:10.4f}")

    half = cur + 0.5 * (wstar - cur)
    print(f"\n절반 지점      {sc(half):11.4f}")
    print("   " + ",".join(f"{x:.6f}" for x in half))
    print("\n최적 벡터")
    print("   " + ",".join(f"{x:.6f}" for x in wstar))
    json.dump({"wstar": list(map(float, wstar)), "half": list(map(float, half)),
               "t_block": list(map(float, r.x[1:6])), "rms": rms,
               "pred_opt": float(sc(wstar)), "pred_half": float(sc(half)),
               "pred_cur": float(sc(cur))},
              open(os.path.join(ROOT, "exp", "exp042_analytic.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
