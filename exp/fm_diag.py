r"""FM 진단 — 1차 실행이 무효였다. 구현 문제인가 최적화 문제인가.

## 1차 실행에서 무엇이 잘못됐나

세 폴드 모두 `ep=0~1` 에서 멈췄고, 2024 Brier 0.2552 는 **상수 예측(0.2498)보다
나쁘다.** 학습이 안 된 것이지 FM 이 약한 것이 아니다. 이걸 "FM 기각" 으로 보고하면
안 된다.

의심 지점 두 개.

  (1) 배치 8192 에서 에폭당 스텝이 89 개뿐 -> 사실상 학습 전에 조기중단
  (2) 구현 자체의 버그

가리는 방법 — **대조군을 세운다.**

  d=0 FM (상호작용 없음)  == 로지스틱 회귀여야 한다
  sklearn LogisticRegression 을 같은 인코딩으로 적합해 비교

둘이 맞으면 구현은 옳고 (1) 이 원인이다. 안 맞으면 구현 버그다.

2024 폴드만 쓴다 — Champion 예측이 `valpred_cat_s3.npz` 에 이미 있어 재학습이 없다.

    .\.venv\Scripts\python.exe exp\fm_diag.py
"""
import json
import os
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "exp", "cache")
CAT_FIELDS = ["pitcher_id", "batter_id", "pitcher_hand", "batter_hand",
              "pitcher_team_id", "batter_team_id", "balls_before",
              "strikes_before", "outs_before", "inning", "top_bottom",
              "num_runners_on", "game_month"]


class FM:
    def __init__(self, n, d=8, lr=0.01, l2_w=1e-5, l2_v=1e-4, seed=0):
        r = np.random.default_rng(seed)
        self.w0, self.w = 0.0, np.zeros(n)
        self.V = r.normal(0, 0.01, (n, d)) if d else np.zeros((n, 1))
        self.d, self.lr, self.l2_w, self.l2_v = d, lr, l2_w, l2_v
        self.mw = np.zeros(n); self.vw = np.zeros(n)
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mw0 = self.vw0 = 0.0
        self.t = 0

    def _z(self, I, Xv):
        lin = self.w0 + (self.w[I] * Xv).sum(1)
        if not self.d:
            return lin, None
        xv = Xv[:, :, None] * self.V[I]
        S1 = xv.sum(1)
        return lin + 0.5 * ((S1 * S1) - (xv * xv).sum(1)).sum(1), S1

    def predict(self, I, Xv, bs=200_000):
        o = np.empty(len(I))
        for i in range(0, len(I), bs):
            z, _ = self._z(I[i:i + bs], Xv[i:i + bs])
            o[i:i + bs] = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        return o

    def _ad(self, p, m, v, g):
        m *= 0.9; m += 0.1 * g
        v *= 0.999; v += 0.001 * g * g
        p -= self.lr * (m / (1 - 0.9 ** self.t)) / (
            np.sqrt(v / (1 - 0.999 ** self.t)) + 1e-8)

    def epoch(self, I, Xv, y, rng, bs):
        for s in range(0, len(I), bs):
            b = rng.integers(0, len(I), bs)
            Ib, Xb, Yb = I[b], Xv[b], y[b]
            z, S1 = self._z(Ib, Xb)
            g = (1 / (1 + np.exp(-np.clip(z, -30, 30))) - Yb) / bs
            self.t += 1
            gw = np.zeros_like(self.w)
            np.add.at(gw, Ib, g[:, None] * Xb)
            gw += self.l2_w * self.w
            g0 = g.sum()
            self.mw0 = .9 * self.mw0 + .1 * g0
            self.vw0 = .999 * self.vw0 + .001 * g0 * g0
            self.w0 -= self.lr * (self.mw0 / (1 - .9 ** self.t)) / (
                np.sqrt(self.vw0 / (1 - .999 ** self.t)) + 1e-8)
            self._ad(self.w, self.mw, self.vw, gw)
            if self.d:
                Vf = self.V[Ib]
                coef = Xb[:, :, None] * (S1[:, None, :] - Xb[:, :, None] * Vf)
                gV = np.zeros_like(self.V)
                np.add.at(gV, Ib, g[:, None, None] * coef)
                gV += self.l2_v * self.V
                self._ad(self.V, self.mV, self.vV, gV)


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def rho2(p, y):
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
    col = lambda n: np.asarray(X[:, ixc[n]], dtype=np.float64)

    tr, va = season <= 2023, season == 2024
    inner = season == 2023
    core = tr & ~inner
    yv, yva = y[core], y[va]

    cats = {c: col(c).astype(np.int64) for c in CAT_FIELDS}
    cats["runner_presence"] = (col("num_runners_on") > 0).astype(np.int64)
    nums = [c for c in feats if c not in CAT_FIELDS]
    NUM = np.asarray(X[:, [ixc[c] for c in nums]], dtype=np.float64)

    parts_i, parts_v, base = [], [], 0
    for nm, a in cats.items():
        uq = np.unique(a[core])
        pos = np.clip(np.searchsorted(uq, a), 0, len(uq) - 1)
        parts_i.append(base + np.where(uq[pos] == a, pos, len(uq)))
        parts_v.append(np.ones(len(a)))
        base += len(uq) + 1
    mu, sd = np.nanmean(NUM[core], 0), np.nanstd(NUM[core], 0)
    sd[sd < 1e-9] = 1
    Z = np.clip((np.where(np.isnan(NUM), mu, NUM) - mu) / sd, -5, 5)
    for j in range(Z.shape[1]):
        parts_i.append(np.full(len(y), base + j, dtype=np.int64))
        parts_v.append(Z[:, j])
    base += Z.shape[1]
    I = np.stack(parts_i, 1).astype(np.int64)
    V = np.stack(parts_v, 1)
    print(f"피처공간 {base:,}  행당 활성 {I.shape[1]}  "
          f"core {core.sum():,} / inner {inner.sum():,} / val {va.sum():,}")

    pch = np.load(os.path.join(ROOT, "exp",
                               "valpred_cat_s3.npz"))["p"].astype(np.float64)
    print(f"\n기준선")
    print(f"  상수(core 평균 {yv.mean():.4f})   "
          f"Brier {brier(np.full(len(yva), yv.mean()), yva):.6f}")
    print(f"  Champion(CatBoost 3시드) Brier {brier(pch, yva):.6f}   "
          f"rho^2 {rho2(pch, yva):.2f}")

    print(f"\n=== 대조군 A) sklearn LogisticRegression (같은 인코딩, 희소) ===")
    t0 = time.time()
    from scipy.sparse import csr_matrix
    from sklearn.linear_model import LogisticRegression
    rows = np.repeat(np.arange(len(y)), I.shape[1])
    S = csr_matrix((V.ravel(), (rows, I.ravel())), shape=(len(y), base))
    lr = LogisticRegression(max_iter=200, C=1.0, solver="lbfgs")
    lr.fit(S[core], yv)
    plr = lr.predict_proba(S[va])[:, 1]
    print(f"  Brier {brier(plr, yva):.6f}  rho^2 {rho2(plr, yva):.2f}  "
          f"[{time.time() - t0:.0f}s]")

    print(f"\n=== 대조군 B) FM d=0 (상호작용 없음) — A 와 같아야 한다 ===")
    for bs, ep_max in [(1024, 8)]:
        fm = FM(base, d=0, lr=0.02, seed=1)
        rng = np.random.default_rng(1)
        for ep in range(ep_max):
            fm.epoch(I[core], V[core], yv, rng, bs)
            p = fm.predict(I[va], V[va])
            print(f"  bs={bs} ep={ep}  val Brier {brier(p, yva):.6f}  "
                  f"rho^2 {rho2(p, yva):.2f}   "
                  f"inner {brier(fm.predict(I[inner], V[inner]), y[inner]):.6f}")

    print(f"\n=== 본체) FM d=8 — 학습곡선 전체를 본다 (조기중단 없이) ===")
    for bs, lrate in [(1024, 0.02), (1024, 0.005)]:
        fm = FM(base, d=8, lr=lrate, l2_v=1e-4, seed=1)
        rng = np.random.default_rng(1)
        print(f"  --- bs={bs} lr={lrate} ---")
        for ep in range(12):
            t1 = time.time()
            fm.epoch(I[core], V[core], yv, rng, bs)
            pi = fm.predict(I[inner], V[inner])
            pv = fm.predict(I[va], V[va])
            print(f"   ep={ep:<2} inner {brier(pi, y[inner]):.6f}  "
                  f"val Brier {brier(pv, yva):.6f}  rho^2 {rho2(pv, yva):>7.2f}"
                  f"  [{time.time() - t1:.0f}s]")


if __name__ == "__main__":
    main()
