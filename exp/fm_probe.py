r"""Factorization Machine — 미검증 계열 직접 검증. 제출물 만들지 않는다.

## 기록 상태 (2026-08-13 확인)

`DEVIATION_LEDGER.md` §6-c 제목은 "잠재 투수-타자 상호작용 (MF / FM) — 기각" 이지만
**실제로 실행된 것은 가중 ALS(rank 2/4/8/16)뿐이다.** FM/DeepFM 은 한 번도 안 돌렸다.
278행 "그래서 FM / Wide&Deep / tabular NN 도 같은 운명이다" 는 측정이 아니라 논증이다.
문서가 측정보다 넓게 주장하고 있어 이 스크립트로 그 빈칸을 메운다.

사전 확률은 낮게 잡는다 — §6-g 가 잔차에 대해 **형태를 가리지 않는 상한 0** 을 쟀다
(트리 3폴드 모두 최적 가중 0). FM 이 그 상한을 넘을 수 있는 유일한 경로는 트리가
표현 못 하는 **고카디널리티 쌍의 잠재 인수분해**인데, 그 축은 §6-c 가 직접 쟀다.

## 프로토콜 — 누설 차단

    2019~2021 -> 2022      2019~2022 -> 2023      2019~2023 -> 2024

- 범주 인코딩은 **폴드 내부 train 에서만** fit. 미등장 수준은 OOV 인덱스로.
- 수치 표준화도 train 에서만 fit.
- 조기중단은 **train 의 마지막 시즌**을 안쪽 검증으로 쓴다. val 폴드는 안 본다.
- Champion 도 같은 split 으로 재학습한다 (CatBoost 3시드 + 편차항).
- 아핀은 train 에서 적합한다. `rho^2` 는 아핀 불변이라 누설과 무관하다.

    .\.venv\Scripts\python.exe exp\fm_probe.py
"""
import json
import os
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "exp", "cache")
OUT = os.path.join(ROOT, "exp", "fm_oof.npz")

FOLDS = [2022, 2023, 2024]
SEEDS = [42, 43, 44]
CB = dict(iterations=1200, learning_rate=0.02, depth=6, l2_leaf_reg=100.0,
          border_count=32, min_data_in_leaf=1000)

CAT_FIELDS = ["pitcher_id", "batter_id", "pitcher_hand", "batter_hand",
              "pitcher_team_id", "batter_team_id", "balls_before",
              "strikes_before", "outs_before", "inning", "top_bottom",
              "num_runners_on", "game_month"]      # + runner_presence 파생


# ---------------------------------------------------------------- 편차항
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


# ---------------------------------------------------------------- FM
class FM:
    """이진 FM. 모든 필드가 (인덱스, 값) 쌍이라 행마다 활성 수가 고정이다.

    y_hat = w0 + sum_j w[i_j] x_j
                + 0.5 sum_f [ (sum_j V[i_j,f] x_j)^2 - sum_j (V[i_j,f] x_j)^2 ]

    피처 공간이 작아(~1,700) 밀집 Adam 으로 충분하다.
    """

    def __init__(self, n_feat, d=8, lr=0.01, l2_w=1e-5, l2_v=1e-4, seed=0):
        r = np.random.default_rng(seed)
        self.w0 = 0.0
        self.w = np.zeros(n_feat)
        self.V = r.normal(0, 0.01, (n_feat, d))
        self.d, self.lr, self.l2_w, self.l2_v = d, lr, l2_w, l2_v
        self.mw = np.zeros_like(self.w); self.vw = np.zeros_like(self.w)
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mw0 = self.vw0 = 0.0
        self.t = 0

    def logit(self, idx, val):
        Vf = self.V[idx]                              # (B,F,d)
        xv = val[:, :, None] * Vf
        S1 = xv.sum(1)
        S2 = (xv * xv).sum(1)
        return (self.w0 + (self.w[idx] * val).sum(1)
                + 0.5 * (S1 * S1 - S2).sum(1)), S1

    def predict(self, idx, val, bs=200_000):
        out = np.empty(len(idx))
        for i in range(0, len(idx), bs):
            z, _ = self.logit(idx[i:i + bs], val[i:i + bs])
            out[i:i + bs] = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        return out

    def _adam(self, p, m, v, g, lr):
        m *= 0.9; m += 0.1 * g
        v *= 0.999; v += 0.001 * g * g
        p -= lr * (m / (1 - 0.9 ** self.t)) / (
            np.sqrt(v / (1 - 0.999 ** self.t)) + 1e-8)

    def fit_epoch(self, idx, val, y, rng, bs=8192):
        order = rng.permutation(len(idx))
        for s in range(0, len(order), bs):
            b = order[s:s + bs]
            I, Xv, Y = idx[b], val[b], y[b]
            z, S1 = self.logit(I, Xv)
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            g = (p - Y) / len(b)                       # (B,)
            self.t += 1
            gw = np.zeros_like(self.w)
            np.add.at(gw, I, g[:, None] * Xv)
            gw += self.l2_w * self.w
            Vf = self.V[I]
            # d(inter)/dV[i,f] = x_i (S1_f - x_i V[i,f])
            coef = Xv[:, :, None] * (S1[:, None, :] - Xv[:, :, None] * Vf)
            gV = np.zeros_like(self.V)
            np.add.at(gV, I, g[:, None, None] * coef)
            gV += self.l2_v * self.V
            gw0 = g.sum()
            self.t and None
            self.mw0 = 0.9 * self.mw0 + 0.1 * gw0
            self.vw0 = 0.999 * self.vw0 + 0.001 * gw0 * gw0
            self.w0 -= self.lr * (self.mw0 / (1 - 0.9 ** self.t)) / (
                np.sqrt(self.vw0 / (1 - 0.999 ** self.t)) + 1e-8)
            self._adam(self.w, self.mw, self.vw, gw, self.lr)
            self._adam(self.V, self.mV, self.vV, gV, self.lr)


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def rho2(p, y):
    return float(1e5 * np.corrcoef(p, y)[0, 1] ** 2)


def main():
    meta = json.load(open(f"{CACHE}/cols.json"))
    cols = meta["cols"]
    ixc = {c: i for i, c in enumerate(cols)}
    X = np.load(f"{CACHE}/X.npy", mmap_mode="r")
    y = np.load(f"{CACHE}/y.npy").astype(np.float64)
    season = np.load(f"{CACHE}/season.npy")

    import joblib
    feats = joblib.load(os.path.join(ROOT, "model_cand",
                                     "grid_affine_solved.pkl"))["features"]
    fidx = [ixc[c] for c in feats]
    print(f"Champion 피처 {len(feats)}개, 캐시 {len(cols)}컬럼, "
          f"행 {len(y):,}\n")

    Xf = np.asarray(X[:, fidx], dtype=np.float32)      # CatBoost 입력
    col = lambda n: np.asarray(X[:, ixc[n]], dtype=np.float64)
    P = col("pitcher_id").astype(np.int64)
    BH = col("batter_hand").astype(np.int64)
    BB = col("balls_before").astype(np.int64)
    SS = col("strikes_before").astype(np.int64)
    OB = (col("num_runners_on") > 0).astype(np.int64)
    PH, CNT = P * 10 + BH, BB * 4 + SS

    # FM 입력 원자료
    cat_raw = {c: col(c).astype(np.int64) for c in CAT_FIELDS}
    cat_raw["runner_presence"] = OB
    num_names = [c for c in feats if c not in CAT_FIELDS]
    NUM = np.asarray(X[:, [ixc[c] for c in num_names]], dtype=np.float64)
    print(f"FM 필드 — 범주 {len(cat_raw)}개, 수치 {len(num_names)}개")

    from catboost import CatBoostClassifier

    res = {}
    store = {}
    for f in FOLDS:
        tr, va = season < f, season == f
        ytr, yva = y[tr], y[va]
        print(f"\n{'=' * 62}\n=== fold {f}  train {tr.sum():,}  "
              f"val {va.sum():,} ===")

        # ---------- Champion ----------
        t0 = time.time()
        acc = np.zeros(va.sum())
        for sd in SEEDS:
            m = CatBoostClassifier(**CB, random_seed=sd, verbose=0,
                                   allow_writing_files=False, thread_count=20)
            m.fit(Xf[tr], ytr)
            acc += m.predict_proba(Xf[va])[:, 1]
        pc = acc / len(SEEDS)
        # 편차항 (18회차 구성). 표는 fold 내부 train 에서만 만든다.
        u1, d1 = nested_dev(P[tr], PH[tr], ytr, 300)
        u2, d2 = nested_dev(PH[tr], (PH * 100 + CNT)[tr], ytr, 800)
        u3, d3 = nested_dev(PH[tr], (PH * 10 + OB)[tr], ytr, 2000)
        pch = (pc + 0.20 * lookup(u1, d1, PH[va])
               + 0.5470 * lookup(u2, d2, (PH * 100 + CNT)[va])
               + 0.30 * lookup(u3, d3, (PH * 10 + OB)[va]))
        # 아핀은 train 에서 적합 (누설 없음). rho^2 는 어차피 아핀 불변이다.
        ctr = float(ytr.mean())
        pch_c = ctr + 1.09 * (pch - pch.mean() + pch.mean() - ctr)
        pch_c = np.clip(ctr + 1.09 * (pch - ctr), 0, 1)
        print(f"  Champion  [{time.time() - t0:.0f}s]  "
              f"Brier {brier(pch_c, yva):.6f}  rho^2 {rho2(pch_c, yva):.2f}"
              f"   (raw CatBoost rho^2 {rho2(pc, yva):.2f})")

        # ---------- FM ----------
        # 인코딩: train 에서만 fit, 미등장은 OOV
        idx_parts, val_parts, base = [], [], 0
        for nm, arr in cat_raw.items():
            uq = np.unique(arr[tr])
            pos = np.searchsorted(uq, arr)
            pos = np.clip(pos, 0, len(uq) - 1)
            hit = uq[pos] == arr
            code = np.where(hit, pos, len(uq))          # OOV = len(uq)
            idx_parts.append(base + code)
            val_parts.append(np.ones(len(arr)))
            base += len(uq) + 1
        mu = np.nanmean(NUM[tr], 0)
        sd_ = np.nanstd(NUM[tr], 0)
        sd_[sd_ < 1e-9] = 1.0
        Z = (np.where(np.isnan(NUM), mu, NUM) - mu) / sd_
        Z = np.clip(Z, -5, 5)
        for j in range(Z.shape[1]):
            idx_parts.append(np.full(len(y), base + j, dtype=np.int64))
            val_parts.append(Z[:, j])
        base += Z.shape[1]
        IDX = np.stack(idx_parts, 1).astype(np.int64)
        VAL = np.stack(val_parts, 1).astype(np.float64)
        print(f"  FM 피처공간 {base:,}   행당 활성 {IDX.shape[1]}")

        inner = season == (f - 1)                       # 안쪽 검증 = train 막시즌
        core = tr & ~inner
        best = None
        for d, l2v, lr in [(4, 1e-3, 0.01), (8, 1e-3, 0.01), (8, 1e-4, 0.01),
                           (16, 1e-3, 0.005)]:
            fm = FM(base, d=d, lr=lr, l2_v=l2v, seed=7)
            rng = np.random.default_rng(7)
            bi, bb, bad = None, 1e9, 0
            for ep in range(25):
                fm.fit_epoch(IDX[core], VAL[core], y[core], rng)
                b = brier(fm.predict(IDX[inner], VAL[inner]), y[inner])
                if b < bb - 1e-7:
                    bb, bad = b, 0
                    bi = (fm.w0, fm.w.copy(), fm.V.copy(), ep)
                else:
                    bad += 1
                    if bad >= 3:
                        break
            fm.w0, fm.w, fm.V, ep_b = bi
            pv = fm.predict(IDX[va], VAL[va])
            print(f"    d={d:<3} l2v={l2v:<6} lr={lr:<6} ep={ep_b:<3} "
                  f"inner {bb:.6f}  val Brier {brier(pv, yva):.6f}  "
                  f"rho^2 {rho2(pv, yva):.2f}")
            if best is None or bb < best[0]:
                best = (bb, pv, (d, l2v, lr))
        _, pfm, hp = best
        print(f"  FM 선택 (안쪽 기준) d={hp[0]} l2v={hp[1]} lr={hp[2]}")
        print(f"    Brier {brier(pfm, yva):.6f}  rho^2 {rho2(pfm, yva):.2f}")

        r = yva - pch_c
        res[f] = dict(
            brier_ch=brier(pch_c, yva), brier_fm=brier(pfm, yva),
            rho_ch=rho2(pch_c, yva), rho_fm=rho2(pfm, yva),
            corr=float(np.corrcoef(pfm, pch_c)[0, 1]),
            resid_corr=float(np.corrcoef(pfm, r)[0, 1]))
        store[f"fm_{f}"] = pfm
        store[f"ch_{f}"] = pch_c
        store[f"y_{f}"] = yva
        print(f"  corr(FM, Champion) {res[f]['corr']:.4f}   "
              f"corr(FM, Champion 잔차) {res[f]['resid_corr']:+.4f}")

        wg = np.linspace(0, 0.5, 51)
        sc = [rho2((1 - w) * pch_c + w * pfm, yva) for w in wg]
        j = int(np.argmax(sc))
        res[f]["blend_w"] = float(wg[j])
        res[f]["blend_gain"] = float(sc[j] - res[f]["rho_ch"])
        print(f"  혼합 최적(대상폴드 낙관상한) w={wg[j]:.2f}  "
              f"rho^2 {sc[j]:.2f} ({sc[j] - res[f]['rho_ch']:+.2f})")

    np.savez_compressed(OUT, **store)
    print(f"\n{'=' * 62}\n=== 요약 ===")
    print(f"{'fold':>6}{'Brier Ch':>11}{'Brier FM':>11}{'dBrier':>10}"
          f"{'rho2 Ch':>10}{'rho2 FM':>10}{'d rho2':>9}{'corr':>8}"
          f"{'resid':>8}{'blend':>9}")
    for f in FOLDS:
        r = res[f]
        print(f"{f:>6}{r['brier_ch']:>11.6f}{r['brier_fm']:>11.6f}"
              f"{r['brier_fm'] - r['brier_ch']:>+10.6f}"
              f"{r['rho_ch']:>10.2f}{r['rho_fm']:>10.2f}"
              f"{r['rho_fm'] - r['rho_ch']:>+9.2f}{r['corr']:>8.4f}"
              f"{r['resid_corr']:>+8.4f}{r['blend_gain']:>+9.2f}")
    db = [res[f]["brier_fm"] - res[f]["brier_ch"] for f in FOLDS]
    dr = [res[f]["rho_fm"] - res[f]["rho_ch"] for f in FOLDS]
    bl = [res[f]["blend_gain"] for f in FOLDS]
    print(f"\n  MEAN dBrier {np.mean(db):+.6f}   WORST {max(db):+.6f}")
    print(f"  MEAN d rho2 {np.mean(dr):+.2f}   WORST {min(dr):+.2f}")
    print(f"  MEAN 혼합이득 {np.mean(bl):+.2f}  (낙관상한)  "
          f"부호 {sum(b > 0 for b in bl)}/3")
    json.dump({str(k): v for k, v in res.items()},
              open(os.path.join(ROOT, "exp", "fm_result.json"), "w"),
              indent=1)


if __name__ == "__main__":
    main()
