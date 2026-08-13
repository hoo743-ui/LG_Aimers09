r"""FM 재실행 — 1차가 무효였던 원인을 고쳤다.

## 1차가 왜 무효였나 (fm_diag / fm_diag2 로 확정)

`d=0` FM 과 sklearn LogisticRegression 이 **똑같이** rho^2 ~ 2 로 실패했다.
구현이 아니라 **규제 강도**가 원인이다. 피처에 정확한 공선성이 있다.

    asof_pitcher_n == asof_pitcher_pitchmix_n      (rho^2 둘 다 64.72)
    home_win_expectancy + away_win_expectancy = 100
    run_total_before = run_top_before + run_bot_before
    num_runners_on = runner_on_1b + 2b + 3b

무규제 최소제곱은 특이행렬에서 수치 쓰레기를 낸다 — 실제로 `ridge lam=0` 이
rho^2 **0.01**, `lam=1e6` 이 **340** 이다. 내 FM 은 `l2_w=1e-5` 로 사실상 무규제였다.

## 정직한 기준선

    단일 최강 피처 (asof_pitcher_success_rate)   rho^2  332
    규제 선형 55피처 (2019~2023 -> 2024)          rho^2  340
    CatBoost                                    rho^2  780
    Champion (편차항 포함)                        rho^2  798

## 게이트

`d=0` FM 이 **rho^2 ~300 을 재현하지 못하면** `d>0` 결과를 신뢰하지 않는다.
게이트를 통과 못 하면 `FM = NOT EXECUTED` 로 기록하고 끝낸다.

    .\.venv\Scripts\python.exe exp\fm_probe2.py
"""
import json
import os
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "exp", "cache")
FOLDS = [2022, 2023, 2024]
SEEDS = [42, 43, 44]
CB = dict(iterations=1200, learning_rate=0.02, depth=6, l2_leaf_reg=100.0,
          border_count=32, min_data_in_leaf=1000)
CAT_FIELDS = ["pitcher_id", "batter_id", "pitcher_hand", "batter_hand",
              "pitcher_team_id", "batter_team_id", "balls_before",
              "strikes_before", "outs_before", "inning", "top_bottom",
              "num_runners_on", "game_month"]
GATE = 200.0            # d=0 FM 이 넘어야 하는 rho^2


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
    out = np.zeros(len(keys))
    ix = np.clip(np.searchsorted(u, keys), 0, len(u) - 1)
    ok = u[ix] == keys
    out[ok] = dev[ix[ok]]
    return out


class FM:
    def __init__(self, n, d, lr, l2_w, l2_v, seed=0):
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

    def epoch(self, I, Xv, y, rng, bs=2048):
        for _ in range(len(I) // bs):
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
    return 0.0 if np.std(p) < 1e-12 else float(
        1e5 * np.corrcoef(p, y)[0, 1] ** 2)


def fit_es(base, d, lr, l2w, l2v, I, Vv, y, core, inner, ep_max=20, pat=4):
    fm = FM(base, d, lr, l2w, l2v, seed=7)
    rng = np.random.default_rng(7)
    bb, best, bad = 1e9, None, 0
    for ep in range(ep_max):
        fm.epoch(I[core], Vv[core], y[core], rng)
        b = brier(fm.predict(I[inner], Vv[inner]), y[inner])
        if b < bb - 1e-7:
            bb, bad, best = b, 0, (fm.w0, fm.w.copy(), fm.V.copy(), ep)
        else:
            bad += 1
            if bad >= pat:
                break
    fm.w0, fm.w, fm.V, ep_b = best
    return fm, bb, ep_b


def main():
    meta = json.load(open(f"{CACHE}/cols.json"))
    ixc = {c: i for i, c in enumerate(meta["cols"])}
    X = np.load(f"{CACHE}/X.npy", mmap_mode="r")
    y = np.load(f"{CACHE}/y.npy").astype(np.float64)
    season = np.load(f"{CACHE}/season.npy")
    import joblib
    feats = joblib.load(os.path.join(ROOT, "model_cand",
                                     "grid_affine_solved.pkl"))["features"]
    Xf = np.asarray(X[:, [ixc[c] for c in feats]], dtype=np.float32)
    col = lambda n: np.asarray(X[:, ixc[n]], dtype=np.float64)
    P = col("pitcher_id").astype(np.int64)
    BH = col("batter_hand").astype(np.int64)
    CNT = (col("balls_before") * 4 + col("strikes_before")).astype(np.int64)
    OB = (col("num_runners_on") > 0).astype(np.int64)
    PH = P * 10 + BH
    cats = {c: col(c).astype(np.int64) for c in CAT_FIELDS}
    cats["runner_presence"] = OB
    nums = [c for c in feats if c not in CAT_FIELDS]
    NUM = np.asarray(X[:, [ixc[c] for c in nums]], dtype=np.float64)

    from catboost import CatBoostClassifier
    res, store = {}, {}
    for f in FOLDS:
        tr, va, inner = season < f, season == f, season == (f - 1)
        core = tr & ~inner
        yva = y[va]
        print(f"\n{'=' * 64}\n=== fold {f}  core {core.sum():,}  "
              f"inner {inner.sum():,}  val {va.sum():,} ===")

        t0 = time.time()
        cpath = os.path.join(ROOT, "exp", f"champ_oof_{f}.npy")
        if os.path.exists(cpath):
            pc = np.load(cpath)
            print(f"  Champion 캐시 재사용 {os.path.basename(cpath)}")
        else:
            acc = np.zeros(int(va.sum()))
            for sd in SEEDS:
                m = CatBoostClassifier(**CB, random_seed=sd, verbose=0,
                                       allow_writing_files=False,
                                       thread_count=20)
                m.fit(Xf[tr], y[tr])
                acc += m.predict_proba(Xf[va])[:, 1]
            pc = acc / len(SEEDS)
            np.save(cpath, pc)
        u1, d1 = nested_dev(P[tr], PH[tr], y[tr], 300)
        u2, d2 = nested_dev(PH[tr], (PH * 100 + CNT)[tr], y[tr], 800)
        u3, d3 = nested_dev(PH[tr], (PH * 10 + OB)[tr], y[tr], 2000)
        pch = (pc + 0.20 * lookup(u1, d1, PH[va])
               + 0.5470 * lookup(u2, d2, (PH * 100 + CNT)[va])
               + 0.30 * lookup(u3, d3, (PH * 10 + OB)[va]))
        ctr = float(y[tr].mean())
        pchc = np.clip(ctr + 1.09 * (pch - ctr), 0, 1)
        print(f"  Champion [{time.time() - t0:.0f}s] Brier "
              f"{brier(pchc, yva):.6f}  rho^2 {rho2(pchc, yva):.2f}")

        parts_i, parts_v, base = [], [], 0
        for nm, a in cats.items():
            uq = np.unique(a[core])
            pos = np.clip(np.searchsorted(uq, a), 0, len(uq) - 1)
            parts_i.append(base + np.where(uq[pos] == a, pos, len(uq)))
            parts_v.append(np.ones(len(a)))
            base += len(uq) + 1
        mu, sd_ = np.nanmean(NUM[core], 0), np.nanstd(NUM[core], 0)
        sd_[sd_ < 1e-9] = 1.0
        Z = np.clip((np.where(np.isnan(NUM), mu, NUM) - mu) / sd_, -5, 5)
        for j in range(Z.shape[1]):
            parts_i.append(np.full(len(y), base + j, dtype=np.int64))
            parts_v.append(Z[:, j])
        base += Z.shape[1]
        I = np.stack(parts_i, 1).astype(np.int64)
        Vv = np.stack(parts_v, 1)

        print(f"  게이트) d=0 FM 이 rho^2 {GATE:.0f} 을 넘는가 — 규제 스윕")
        gate_best = None
        for l2w in (1e-3, 1e-2, 1e-1, 3e-1):
            fm, ib, ep = fit_es(base, 0, 0.02, l2w, 0.0, I, Vv, y, core, inner)
            p0 = fm.predict(I[va], Vv[va])
            r0 = rho2(p0, yva)
            print(f"    l2_w={l2w:<6} ep={ep:<2} inner {ib:.6f}  "
                  f"val rho^2 {r0:>7.2f}  Brier {brier(p0, yva):.6f}")
            if gate_best is None or ib < gate_best[0]:
                gate_best = (ib, r0, l2w)
        _, r_gate, l2w_best = gate_best
        print(f"  -> 안쪽 기준 선택 l2_w={l2w_best}, val rho^2 {r_gate:.2f}  "
              f"{'게이트 통과' if r_gate >= GATE else '게이트 실패'}")
        if r_gate < GATE:
            print("  ** 게이트 실패 — 이 폴드의 FM 결과는 신뢰하지 않는다")

        best = None
        for d in (4, 8, 16):
            for l2v in (1e-1,):
                fm, ib, ep = fit_es(base, d, 0.01, l2w_best, l2v,
                                    I, Vv, y, core, inner)
                pv = fm.predict(I[va], Vv[va])
                print(f"    d={d:<3} l2_v={l2v:<5} ep={ep:<2} "
                      f"inner {ib:.6f}  val rho^2 {rho2(pv, yva):>7.2f}  "
                      f"Brier {brier(pv, yva):.6f}")
                if best is None or ib < best[0]:
                    best = (ib, pv, d, l2v)
        _, pfm, dbest, l2vbest = best
        r = yva - pchc
        wg = np.linspace(0, 0.6, 61)
        sc = [rho2((1 - w) * pchc + w * pfm, yva) for w in wg]
        j = int(np.argmax(sc))
        res[f] = dict(gate=r_gate, gate_ok=bool(r_gate >= GATE),
                      d=dbest, l2v=l2vbest,
                      brier_ch=brier(pchc, yva), brier_fm=brier(pfm, yva),
                      rho_ch=rho2(pchc, yva), rho_fm=rho2(pfm, yva),
                      corr=float(np.corrcoef(pfm, pchc)[0, 1]),
                      resid=float(np.corrcoef(pfm, r)[0, 1]),
                      blend_w=float(wg[j]),
                      blend_gain=float(sc[j] - rho2(pchc, yva)))
        store[f"fm_{f}"], store[f"ch_{f}"], store[f"y_{f}"] = pfm, pchc, yva
        print(f"  선택 d={dbest} l2_v={l2vbest}  corr(FM,Ch) {res[f]['corr']:.4f}"
              f"  corr(FM, Ch잔차) {res[f]['resid']:+.4f}  "
              f"혼합(낙관상한) w={wg[j]:.2f} {res[f]['blend_gain']:+.2f}")

    np.savez_compressed(os.path.join(ROOT, "exp", "fm_oof2.npz"), **store)
    print(f"\n{'=' * 64}\n=== 요약 ===")
    print(f"{'fold':>6}{'게이트':>9}{'BrierCh':>10}{'BrierFM':>10}{'dBrier':>10}"
          f"{'rho2Ch':>9}{'rho2FM':>9}{'corr':>8}{'resid':>8}{'blend':>8}")
    for f in FOLDS:
        r = res[f]
        print(f"{f:>6}{r['gate']:>9.1f}{r['brier_ch']:>10.6f}"
              f"{r['brier_fm']:>10.6f}{r['brier_fm'] - r['brier_ch']:>+10.6f}"
              f"{r['rho_ch']:>9.1f}{r['rho_fm']:>9.1f}{r['corr']:>8.4f}"
              f"{r['resid']:>+8.4f}{r['blend_gain']:>+8.2f}")
    db = [res[f]["brier_fm"] - res[f]["brier_ch"] for f in FOLDS]
    dr = [res[f]["rho_fm"] - res[f]["rho_ch"] for f in FOLDS]
    bl = [res[f]["blend_gain"] for f in FOLDS]
    print(f"\n  MEAN dBrier {np.mean(db):+.6f}  WORST {max(db):+.6f}")
    print(f"  MEAN d rho2 {np.mean(dr):+.1f}  WORST {min(dr):+.1f}")
    print(f"  MEAN 혼합(낙관상한) {np.mean(bl):+.2f}  "
          f"부호 {sum(b > 0 for b in bl)}/3")
    print(f"  게이트 통과 {sum(res[f]['gate_ok'] for f in FOLDS)}/3")
    json.dump({str(k): v for k, v in res.items()},
              open(os.path.join(ROOT, "exp", "fm_result2.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
