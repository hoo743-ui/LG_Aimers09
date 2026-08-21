r"""EXP043 — 43회차(cand_opt1 = 1074.0991) 를 넣은 재적합과 귀속.

cand_opt1 은 **한 번도 LB 로 측정된 적 없던 4 dof**(편차 내부 3 + 2S/주자 분리 1)
를 처음으로 찍은 점이다. 예상 1076.76 -> 실측 1074.10, 어긋남 -2.66.

이 스크립트가 하는 일
  1  보정 벡터 9개와 모델 예측을 캐시한다 (exp/cache/exp043_comp.npz)
  2  블록 구조를 여러 개 놓고 12점을 적합해 어느 분해가 잔차를 흡수하는지 본다
  3  LOO 로 각 후보 벡터의 예측 산포를 낸다 (모형 오차가 잡음이므로)
  4  이미 빌드된 후보들과 좌표 스캔의 예상 점수를 표로 낸다

    .\.venv\Scripts\python.exe -u research\exp043_refit.py
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "exp"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

CACHE = os.path.join(ROOT, "exp", "cache", "exp043_comp.npz")
LEDGER = json.load(open(os.path.join(ROOT, "exp", "lb_obs.json"), encoding="utf-8"))
NAMES = LEDGER["names"]

STRUCTS = {
    "S0 현행 5블록":        [0, 0, 0, 0, 1, 2, 2, 3, 4],
    "S1 편차 {01|23}":      [0, 0, 5, 5, 1, 2, 2, 3, 4],
    "S2 편차 {0|123}":      [5, 0, 0, 0, 1, 2, 2, 3, 4],
    "S3 편차 {013|2}":      [0, 0, 5, 0, 1, 2, 2, 3, 4],
    "S4 대비 {2S|주자}":    [0, 0, 0, 0, 1, 2, 5, 3, 4],
    "S5 대비+편차 동시":    [0, 0, 6, 6, 1, 2, 5, 3, 4],
}


def components():
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        return z["mm"], z["Cm"], z["y"]
    import exp042_analytic as e42
    mm, Cm, y = e42.components()
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, mm=mm, Cm=Cm, y=y)
    return mm, Cm, y


def moments(mm, Cm, y):
    n = len(y)
    X = np.column_stack([mm, Cm])
    Xc = X - X.mean(0)
    S = Xc.T @ Xc / n
    gl = (Xc.T @ (y - y.mean())) / n
    return gl, S[0, 0], S[0, 1:], S[1:, 1:], y.var()


def fit(W, Sobs, block, gl, v0, u, C, yvar):
    nb = max(block) + 1
    idx = np.array(block)

    def gvec(gb):
        return gb[idx] * gl[1:]

    den = v0 + 2 * (W @ u) + np.einsum("ij,jk,ik->i", W, C, W)

    def predict(t):
        return t[-1] * (t[0] + W @ gvec(t[1:1 + nb])) ** 2 / den

    th0 = np.concatenate([[gl[0]], np.ones(nb), [1e5 / yvar]])
    r = least_squares(lambda t: predict(t) - Sobs, th0,
                      x_scale=np.abs(th0) + 1e-12, max_nfev=60000)
    res = predict(r.x) - Sobs
    rms = float(np.sqrt(np.mean(res ** 2)))

    a, g, K = r.x[0], gvec(r.x[1:1 + nb]), r.x[-1]

    def sc(w):
        w = np.asarray(w, float)
        return float(K * (a + w @ g) ** 2 / (v0 + 2 * w @ u + w @ C @ w))

    return r.x, rms, res, sc, (a, g, K)


def main():
    mm, Cm, y = components()
    gl, v0, u, C, yvar = moments(mm, Cm, y)

    R = np.corrcoef(Cm.T)
    print("=== 보정 벡터 상관 (폴드 2024) ===")
    print("        " + " ".join(f"{n[:6]:>7s}" for n in NAMES))
    for i, n in enumerate(NAMES):
        print(f"{n[:7]:8s}" + " ".join(f"{R[i, j]:7.3f}" for j in range(9)))
    off = np.abs(R - np.eye(9))
    i, j = np.unravel_index(off.argmax(), off.shape)
    print(f"\n최대 |r| = {off.max():.3f}  ({NAMES[i]} x {NAMES[j]})")
    print("\n축별 로컬 g 와 corr(c,y)")
    for i, n in enumerate(NAMES):
        sd = np.sqrt(C[i, i])
        print(f"  {n:14s} sd {sd:9.6f}  corr {gl[i+1]/(sd*np.sqrt(yvar)):+9.5f}  g {gl[i+1]:+11.4e}")

    W = np.array([o["w"] for o in LEDGER["obs"]], float)
    Sobs = np.array([o["lb"] for o in LEDGER["obs"]], float)
    tag = [f'{o["round"]:>2d} {o["cand"][:16]}' for o in LEDGER["obs"]]

    # --- 11점 모형이 43회차를 얼마나 빗나갔나 (재확인) ---
    _, rms11, res11, sc11, _ = fit(W[:-1], Sobs[:-1], STRUCTS["S0 현행 5블록"],
                                   gl, v0, u, C, yvar)
    print(f"\n=== 11점 적합(43회차 제외) RMS {rms11:.4f} ===")
    print(f"  opt1 예측 {sc11(W[-1]):.4f}   실측 {Sobs[-1]:.4f}   "
          f"어긋남 {Sobs[-1]-sc11(W[-1]):+.4f}")

    # --- 구조 비교 ---
    print("\n=== 12점 적합 — 블록 구조 비교 ===")
    print(f"{'구조':22s} {'모수':>4s} {'dof':>4s} {'RMS':>8s} {'opt1잔차':>9s} {'h1잔차':>8s}")
    fits = {}
    for name, blk in STRUCTS.items():
        x, rms, res, sc, agk = fit(W, Sobs, blk, gl, v0, u, C, yvar)
        fits[name] = (x, rms, res, sc, blk)
        npar = max(blk) + 2 + 1
        print(f"{name:22s} {npar:4d} {len(Sobs)-npar:4d} {rms:8.4f} "
              f"{res[-1]:+9.4f} {res[-3]:+8.4f}")

    best = min(fits, key=lambda k: fits[k][1])
    print(f"\n잔차를 가장 잘 흡수하는 구조: {best}")
    for name in ("S0 현행 5블록", best):
        x, rms, res, sc, blk = fits[name]
        nb = max(blk) + 1
        print(f"\n--- {name} ---  전이 t = " +
              " ".join(f"{v:+.3f}" for v in x[1:1 + nb]))
        for t_, s_, r_ in zip(tag, Sobs, res):
            print(f"   {t_:20s} 실측 {s_:11.4f}  잔차 {r_:+7.4f}")

    # --- LOO 로 후보 예측의 산포 ---
    cands = dict(LEDGER["built"])
    h1 = np.array(cands["cand_h1"], float)
    for wv in (0.75, 0.80, 0.85, 0.90, 1.00, 1.10):
        v = h1.copy(); v[4] = wv; cands[f"hand={wv:.2f}"] = v.tolist()
    for wv in (0.30, 0.40, 0.50, 0.80, 1.00):
        v = h1.copy(); v[5] = wv; cands[f"2S={wv:.2f}"] = v.tolist()
        v = h1.copy(); v[6] = wv; cands[f"runner={wv:.2f}"] = v.tolist()
    for wv in (1.30, 1.52, 1.75, 2.50):
        v = h1.copy(); v[7] = wv; cands[f"wp={wv:.2f}"] = v.tolist()
    for wv in (1.80, 2.30, 2.60):
        v = h1.copy(); v[8] = wv; cands[f"wb={wv:.2f}"] = v.tolist()

    print("\n=== 후보 예상 점수 (12점 적합, LOO 산포) ===")
    print(f"{'후보':18s} {'S0':>10s} {best[:8]:>10s} {'LOO평균':>10s} {'LOO sd':>8s} {'최악':>10s}")
    rows = []
    for cn, cv in cands.items():
        cv = np.array(cv, float)
        p0 = fits["S0 현행 5블록"][3](cv)
        pb = fits[best][3](cv)
        loo = []
        for d in range(len(Sobs)):
            m = np.ones(len(Sobs), bool); m[d] = False
            _, _, _, scl, _ = fit(W[m], Sobs[m], STRUCTS["S0 현행 5블록"],
                                  gl, v0, u, C, yvar)
            loo.append(scl(cv))
        loo = np.array(loo)
        rows.append((cn, p0, pb, loo.mean(), loo.std(), loo.min()))
    for cn, p0, pb, lm, ls, lw in sorted(rows, key=lambda r: -r[3]):
        star = " <-" if lw > 1075.4602 else ""
        print(f"{cn:18s} {p0:10.3f} {pb:10.3f} {lm:10.3f} {ls:8.3f} {lw:10.3f}{star}")

    json.dump({"best_struct": best,
               "rms": {k: v[1] for k, v in fits.items()},
               "t_best": list(map(float, fits[best][0][1:1 + max(fits[best][4]) + 1])),
               "cands": {r[0]: {"S0": r[1], "best": r[2], "loo_mean": r[3],
                                "loo_sd": r[4], "loo_min": r[5]} for r in rows}},
              open(os.path.join(ROOT, "exp", "exp043_refit.json"), "w"), indent=1)
    print("\n-> exp/exp043_refit.json")


if __name__ == "__main__":
    main()
