r"""EXP048 — **가법 좌표 곡선**. 모수에 선형이므로 닫힌 형태로 푼다.

    S(w) = S0 + Σ_j [ 2 P_j w_j - Q_j w_j^2 ]      P_j = A_j b_j,  Q_j = A_j
    -> 설계행렬 [1, 2w_1, -w_1^2, ..., 2w_5, -w_5^2] 의 **보통최소제곱**

축 최적은 `w*_j = b_j = P_j / Q_j`. 표준오차와 상관까지 그대로 나온다.

EXP042 의 전역 비율 모형은 미식별 방향으로 외삽해 43회차 -2.66 을 냈다. 이 모형은
**측정된 좌표 안에만** 산다 — 43회차(`cand_opt1`)는 dev 가 전역 배수가 아니므로
정의역 밖이라 제외한다.

    .\.venv\Scripts\python.exe -u research\exp048_addcurve.py
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
L = json.load(open(os.path.join(ROOT, "exp", "lb_obs.json"), encoding="utf-8"))
BASE_DEV = np.array([0.20, 0.825, 0.28, 0.45])
BN = ["dev_m", "c_hand", "c_2S+run", "L_pitcher", "L_batter"]


def blocks(w):
    w = np.asarray(w, float)
    r = w[:4] / BASE_DEV
    if r.std() > 1e-4 * max(r.mean(), 1e-9) or abs(w[5] - w[6]) > 1e-4:
        return None
    return np.array([r.mean(), w[4], w[5], w[7], w[8]])


def design(W):
    cols = [np.ones(len(W))]
    for j in range(5):
        cols += [2 * W[:, j], -W[:, j] ** 2]
    return np.column_stack(cols)


def fit(W, S):
    X = design(W)
    th, *_ = np.linalg.lstsq(X, S, rcond=None)
    r = S - X @ th
    dof = len(S) - X.shape[1]
    s2 = r @ r / max(dof, 1)
    cov = s2 * np.linalg.pinv(X.T @ X)
    return th, r, np.sqrt(np.maximum(np.diag(cov), 0)), dof


def score(th, w):
    return th[0] + sum(2 * th[1 + 2 * j] * w[j] - th[2 + 2 * j] * w[j] ** 2
                       for j in range(5))


def main():
    obs, used, skip = [], [], []
    for o in L["obs"]:
        b = blocks(o["w"])
        (used if b is not None else skip).append(o["cand"])
        if b is not None:
            obs.append((b, o["lb"]))
    W = np.array([o[0] for o in obs])
    S = np.array([o[1] for o in obs])
    th, res, se, dof = fit(W, S)
    print(f"정의역 안 {len(used)}점 / 제외 {skip}")
    print(f"\n=== OLS 가법 곡선 — 잔차 RMS {np.sqrt((res**2).mean()):.4f} 점, "
          f"모수 11, dof {dof} ===")
    for nm, s_, e in zip(used, S, res):
        print(f"  {nm:14s} 실측 {s_:11.4f}  잔차 {e:+7.4f}")

    cur = blocks(L["built"]["cand_rob2"])
    print(f"\n{'블록':11s} {'A=Q':>9s} {'±':>7s} {'최적 b':>9s} {'±':>7s} "
          f"{'rob2':>8s} {'남은':>8s}")
    opt = cur.copy()
    for j, nm in enumerate(BN):
        P, Q = th[1 + 2 * j], th[2 + 2 * j]
        sP, sQ = se[1 + 2 * j], se[2 + 2 * j]
        b = P / Q if abs(Q) > 1e-9 else np.nan
        sb = abs(b) * np.hypot(sP / abs(P) if P else 9, sQ / abs(Q) if Q else 9)
        g = lambda w: 2 * P * w - Q * w * w
        gain = g(b) - g(cur[j]) if np.isfinite(b) else np.nan
        opt[j] = b if np.isfinite(b) else cur[j]
        print(f"{nm:11s} {Q:9.3f} {sQ:7.3f} {b:9.4f} {sb:7.4f} "
              f"{cur[j]:8.4f} {gain:+8.3f}")
    print(f"\nrob2 모형값 {score(th, cur):.4f}   실측 1075.9242")
    print(f"전 블록 최적 {score(th, opt):.4f}   증분 {score(th,opt)-score(th,cur):+.3f}")

    ens = [fit(np.delete(W, d, 0), np.delete(S, d))[0] for d in range(len(S))] + [th]
    base = np.array([score(t, cur) for t in ens])

    def rel(w):
        d = np.array([score(t, np.asarray(w, float)) for t in ens]) - base
        return d.min(), d.mean(), d.max()

    cands = {"rob2 (현행)": cur, "전블록 최적": opt}
    for f in (0.3, 0.5, 0.7):
        cands[f"최적으로 {f:.0%}"] = cur + f * (opt - cur)
    for j, nm in enumerate(BN):
        for f in (0.5, 1.0):
            v = cur.copy(); v[j] = cur[j] + f * (opt[j] - cur[j])
            if abs(v[j] - cur[j]) > 1e-6:
                cands[f"{nm} {f:.0%}"] = v
    print(f"\n=== 후보 — rob2 대비 증분 (LOO {len(ens)}적합) ===")
    print(f"{'후보':16s} {'최악':>8s} {'평균':>8s} {'최선':>8s}   "
          f"{'  '.join(f'{n:>9s}' for n in BN)}")
    rows = [(k, *rel(v), v) for k, v in cands.items()]
    for k, lo, me, hi, v in sorted(rows, key=lambda r: -r[1]):
        print(f"{k:16s} {lo:+8.3f} {me:+8.3f} {hi:+8.3f}   " +
              "  ".join(f"{x:9.4f}" for x in v))
    json.dump({"theta": th.tolist(), "se": se.tolist(), "dof": int(dof),
               "rms": float(np.sqrt((res ** 2).mean())),
               "cur": cur.tolist(), "opt": opt.tolist(),
               "cands": {k: {"lo": lo, "mean": me, "hi": hi, "w": list(v)}
                         for k, lo, me, hi, v in rows}},
              open(os.path.join(ROOT, "exp", "exp048_addcurve.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
