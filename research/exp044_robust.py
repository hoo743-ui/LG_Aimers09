r"""EXP044 — **측정된 방향에만 머무는** 강건 최적. opt1 실패의 교정.

## opt1 이 왜 실패했나

12점 적합에서 확인된 것:
  * 현행 5블록(S0)은 12점을 못 맞춘다 (RMS 0.629, h1 -0.85 / opt1 +1.41)
  * 블록을 하나 쪼개면 **어느 것을 쪼개든** RMS 0.2887 로 같다
    -> 관측 1개가 미식별 4 dof 중 무엇을 움직였는지 **구별 불가**

즉 opt1 은 "모형이 틀린 방향"을 찍었고 그 방향은 지금도 식별되지 않는다.

## 그래서 목적함수를 바꾼다

    기존   max_w  S_hat(w)            <- 단일 적합의 점추정. opt1 이 이걸로 나왔다
    지금   max_w  min_e S_e(w)        <- 구조 6개 x LOO 13개 앙상블의 **최악값**

그리고 **편차 4축의 내부 모양은 h1 에 고정**한다 (전역 배수 m 하나만 허용).
그 4 dof 가 LB 로 한 번도 식별된 적이 없고 opt1 이 정확히 거기서 무너졌다.

    .\.venv\Scripts\python.exe -u research\exp044_robust.py
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import least_squares, minimize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "exp"))
from exp043_refit import STRUCTS, components, moments, fit, LEDGER, NAMES  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

H1 = np.array(LEDGER["built"]["cand_h1"], float)
DEV = H1[:4].copy()                      # 편차 4축의 **모양** (고정)
LO = np.array([0.00, 0.30, 0.20, 0.20, 0.50, 1.00])
HI = np.array([1.60, 1.30, 1.30, 1.30, 3.50, 3.50])
PNAME = ["m_dev", "c_hand", "c_2strike", "c_runner", "L_pitcher", "L_batter"]


def expand(p):
    """6 자유모수 -> 9 가중.  편차는 h1 모양 x 배수."""
    return np.concatenate([DEV * p[0], p[1:]])


def build_ensemble(gl, v0, u, C, yvar):
    W = np.array([o["w"] for o in LEDGER["obs"]], float)
    S = np.array([o["lb"] for o in LEDGER["obs"]], float)
    ens = []
    for sname, blk in STRUCTS.items():
        for d in range(-1, len(S)):
            m = np.ones(len(S), bool)
            if d >= 0:
                m[d] = False
            try:
                _, rms, _, sc, _ = fit(W[m], S[m], blk, gl, v0, u, C, yvar)
            except Exception:
                continue
            if not np.isfinite(rms) or rms > 3.0:
                continue
            ens.append((sname, d, rms, sc))
    return ens, W, S


def main():
    mm, Cm, y = components()
    gl, v0, u, C, yvar = moments(mm, Cm, y)
    ens, W, Sobs = build_ensemble(gl, v0, u, C, yvar)
    print(f"앙상블 {len(ens)} 개 적합 (구조 {len(STRUCTS)} x LOO {len(Sobs)+1})")

    base = np.array([e[3](H1) for e in ens])
    print("")
    print(f"기준 h1  실측 1075.4602   모형 min {base.min():.3f}  "
          f"mean {base.mean():.3f}  max {base.max():.3f}")
    print("판정 = **적합마다 짝지어 뺀** 증분 d_e = S_e(w) - S_e(h1) 의 분포")

    def dvec(w):
        return np.array([e[3](np.asarray(w, float)) for e in ens]) - base

    def rel(w):
        d = dvec(w)
        return d.min(), d.mean(), d.max()

    # ---------- 1. 강건 최적 (최악값 최대화) ----------
    p0 = np.array([1.0, 0.65, 0.65, 0.65, 2.0, 2.10])
    r = minimize(lambda p: -dvec(expand(p)).min(), p0,
                 method="Nelder-Mead",
                 options={"maxiter": 20000, "xatol": 1e-5, "fatol": 1e-7})
    p_rob = np.clip(r.x, LO, HI)
    # ---------- 2. 평균 최적 ----------
    r2 = minimize(lambda p: -dvec(expand(p)).mean(), p0,
                  method="Nelder-Mead",
                  options={"maxiter": 20000, "xatol": 1e-5, "fatol": 1e-7})
    p_avg = np.clip(r2.x, LO, HI)

    print("=== 6 자유모수 최적 (편차 모양 h1 고정) ===")
    print(f"{'모수':12s} {'h1':>8s} {'강건(minmax)':>13s} {'평균최적':>10s}")
    for i, n in enumerate(PNAME):
        cur = 1.0 if i == 0 else H1[3 + i]
        print(f"{n:12s} {cur:8.4f} {p_rob[i]:13.4f} {p_avg[i]:10.4f}")
    for lbl, p in (("강건", p_rob), ("평균", p_avg)):
        d = rel(expand(p))
        print(f"  {lbl} 증분   최악 {d[0]:+.3f}   평균 {d[1]:+.3f}   최선 {d[2]:+.3f}")

    # ---------- 3. 후보 표 ----------
    cands = {"h1 (현행)": H1, "강건최적": expand(p_rob), "평균최적": expand(p_avg)}
    for k, v in LEDGER["built"].items():
        if k != "cand_h1":
            cands[k] = np.array(v, float)
    grid = {4: ("hand", [0.55, 0.70, 0.75, 0.80, 0.85, 0.95]),
            5: ("2S", [0.45, 0.55, 0.80, 0.95]),
            6: ("runner", [0.45, 0.80, 0.95, 1.15]),
            7: ("wp", [1.50, 2.40, 2.80]),
            8: ("wb", [1.85, 2.35, 2.60, 2.90])}
    for i, (nm, vals) in grid.items():
        for vv in vals:
            w = H1.copy(); w[i] = vv
            cands[f"{nm}={vv:.2f}"] = w
    for mm_ in (0.0, 0.5, 1.5):
        w = H1.copy(); w[:4] = DEV * mm_
        cands[f"dev x{mm_:.1f}"] = w
    w = H1.copy(); w[2] = 0.0; w[3] = 0.0
    cands["dev23=0 (opt1방향)"] = w
    w = H1.copy(); w[0] = 0.038; w[1] = 0.520
    cands["dev01=opt1"] = w

    print(f"\n=== 후보 — h1 대비 증분 (앙상블 {len(ens)}개) ===")
    print(f"{'후보':22s} {'최악':>8s} {'평균':>8s} {'최선':>8s}   판정")
    rows = [(k, *rel(np.asarray(v, float))) for k, v in cands.items()]
    for k, lo, me, hi in sorted(rows, key=lambda t: -t[1]):
        verdict = "제출 가치" if lo > 0.05 else ("식별용" if hi - lo > 0.8 else "")
        print(f"{k:22s} {lo:+8.3f} {me:+8.3f} {hi:+8.3f}   {verdict}")

    json.dump({"p_rob": p_rob.tolist(), "w_rob": expand(p_rob).tolist(),
               "p_avg": p_avg.tolist(), "w_avg": expand(p_avg).tolist(),
               "n_ens": len(ens),
               "cands": {k: {"lo": lo, "mean": me, "hi": hi}
                         for k, lo, me, hi in rows}},
              open(os.path.join(ROOT, "exp", "exp044_robust.json"), "w"), indent=1)
    print("\n-> exp/exp044_robust.json")


if __name__ == "__main__":
    main()
