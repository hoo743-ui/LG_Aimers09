r"""왜 선형 모델이 rho^2 ~ 2 인가. 버그인가 데이터의 성질인가.

sklearn LogisticRegression 과 내 FM(d=0) 이 **똑같이** 실패했다. 구현이 아니라
입력 쪽이다. 가장 값싼 판별부터 한다 — 단일 피처의 상관을 직접 잰다.

강한 피처 하나가 2024 에서 상관 0.05 를 내는데 55피처 선형모델이 0.005 라면 버그다.
단일 피처들도 다 0.005 근처라면 **선형 구조가 실제로 없는 것**이고, 그건 FM 의
선형부가 쓸모없다는 뜻이라 결론에 직접 쓰인다.

    .\.venv\Scripts\python.exe exp\fm_diag2.py
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "exp", "cache")
CAT_FIELDS = ["pitcher_id", "batter_id", "pitcher_hand", "batter_hand",
              "pitcher_team_id", "batter_team_id", "balls_before",
              "strikes_before", "outs_before", "inning", "top_bottom",
              "num_runners_on", "game_month"]


def rho2(p, y):
    if np.std(p) < 1e-12:
        return 0.0
    return float(1e5 * np.corrcoef(p, y)[0, 1] ** 2)


def main():
    meta = json.load(open(f"{CACHE}/cols.json"))
    ixc = {c: i for i, c in enumerate(meta["cols"])}
    X = np.load(f"{CACHE}/X.npy", mmap_mode="r")
    y = np.load(f"{CACHE}/y.npy").astype(np.float64)
    season = np.load(f"{CACHE}/season.npy")
    import joblib
    feats = joblib.load(os.path.join(ROOT, "model_cand",
                                     "grid_affine_solved.pkl"))["features"]

    tr, va = season <= 2023, season == 2024
    yva = y[va]
    pch = np.load(os.path.join(ROOT, "exp",
                               "valpred_cat_s3.npz"))["p"].astype(np.float64)
    print(f"val 2024 {va.sum():,}행   y평균 {yva.mean():.4f}")
    print(f"Champion(CatBoost) rho^2 {rho2(pch, yva):.2f}  "
          f"(상관 {np.corrcoef(pch, yva)[0, 1]:+.4f})\n")

    print("=== 단일 피처의 2024 상관 (부호 무관, rho^2 로 환산) ===")
    rows = []
    for c in feats:
        v = np.asarray(X[va, ixc[c]], dtype=np.float64)
        m = ~np.isnan(v)
        if m.sum() < 1000 or np.nanstd(v) < 1e-12:
            continue
        r = float(np.corrcoef(v[m], yva[m])[0, 1])
        rows.append((1e5 * r * r, r, c, 100 * m.mean()))
    rows.sort(reverse=True)
    for s, r, c, cov in rows[:14]:
        print(f"  {c:<38} rho^2 {s:>8.2f}  상관 {r:+.4f}  결측아님 {cov:5.1f}%")
    print(f"  ... 하위 3개")
    for s, r, c, cov in rows[-3:]:
        print(f"  {c:<38} rho^2 {s:>8.2f}  상관 {r:+.4f}  결측아님 {cov:5.1f}%")

    print(f"\n=== 선형 결합의 상한 — 2024 에서 직접 최소제곱 (낙관 상한, 누설) ===")
    NUM = np.asarray(X[np.ix_(va, [ixc[c] for c in feats])], dtype=np.float64)
    mu = np.nanmean(NUM, 0)
    Z = np.where(np.isnan(NUM), mu, NUM)
    Z = (Z - Z.mean(0)) / np.where(Z.std(0) < 1e-12, 1, Z.std(0))
    Z = np.clip(Z, -8, 8)
    A = np.c_[np.ones(len(Z)), Z]
    coef, *_ = np.linalg.lstsq(A, yva, rcond=None)
    print(f"  55피처 선형(2024 자체 적합) rho^2 {rho2(A @ coef, yva):.2f}"
          f"   <- 누설된 상한이다")

    print(f"\n=== 정직한 선형 — 2019~2023 에서 적합해 2024 평가 ===")
    NT = np.asarray(X[np.ix_(tr, [ixc[c] for c in feats])], dtype=np.float64)
    mut = np.nanmean(NT, 0)
    sdt = np.nanstd(NT, 0)
    sdt[sdt < 1e-12] = 1.0
    Zt = np.clip((np.where(np.isnan(NT), mut, NT) - mut) / sdt, -8, 8)
    Zv = np.clip((np.where(np.isnan(NUM), mut, NUM) - mut) / sdt, -8, 8)
    At = np.c_[np.ones(len(Zt)), Zt]
    Av = np.c_[np.ones(len(Zv)), Zv]
    for lam in (0.0, 1e2, 1e4, 1e6):
        G = At.T @ At + lam * np.eye(At.shape[1])
        G[0, 0] -= lam
        cf = np.linalg.solve(G, At.T @ y[tr])
        print(f"  ridge lam={lam:<8.0f} rho^2 {rho2(Av @ cf, yva):>8.2f}")

    print(f"\n=== ID 원핫만 vs 수치만 (정직한 분할) ===")
    from scipy.sparse import csr_matrix, hstack
    from sklearn.linear_model import Ridge
    nums = [c for c in feats if c not in CAT_FIELDS]
    ni = [feats.index(c) for c in nums]
    for nm, use in [("수치 42개만", ni),
                    ("전체 55개", list(range(len(feats))))]:
        cf = np.linalg.solve(
            At[:, [0] + [i + 1 for i in use]].T @ At[:, [0] + [i + 1 for i in use]]
            + 1e4 * np.eye(len(use) + 1),
            At[:, [0] + [i + 1 for i in use]].T @ y[tr])
        print(f"  {nm:<14} rho^2 "
              f"{rho2(Av[:, [0] + [i + 1 for i in use]] @ cf, yva):>8.2f}")

    P = np.asarray(X[:, ixc['pitcher_id']], dtype=np.int64)
    print(f"\n=== 참고: 투수 원핫만 (2019~2023 평균) ===")
    uq = np.unique(P[tr])
    mean_p = np.zeros(len(uq))
    for i, p in enumerate(uq):
        mean_p[i] = y[tr][P[tr] == p].mean() if (P[tr] == p).sum() else y[tr].mean()
    pos = np.clip(np.searchsorted(uq, P[va]), 0, len(uq) - 1)
    pv = np.where(uq[pos] == P[va], mean_p[pos], y[tr].mean())
    print(f"  투수 평균 성공률만으로 예측  rho^2 {rho2(pv, yva):.2f}")


if __name__ == "__main__":
    main()
