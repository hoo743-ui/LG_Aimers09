r"""EXP037 — 수준 축의 추정을 세 방향에서 개선한다. 학습 0회.

## 기제에서 나온 세 가설

수준 축이 통하는 이유는 **모델이 고차수 범주 효과를 과소적합**하기 때문이다
(l2_leaf_reg=100, border_count=32, 개체 500~800개). LB 전이율이 1.49 / 2.34 로
1을 넘은 것이 그 증거다.

### A. 개체 표본수에 대한 가중이 뒤집혀 있을 수 있다

    현재   가중 = n/(n+k)          n 이 클수록 **크게** 싣는다 (추정 정확도)
    그런데 모델의 과소적합은 그 개체의 학습 데이터가 **적을수록** 심하다
           -> n 이 작을수록 보정이 **더** 필요하다

두 힘이 정반대다. 우리는 앞의 것만 담고 있다. 개체를 n 구간으로 나눠 구간별
최적 가중 배수를 재면, 그것이 n 에 대해 체계적으로 감소하는지 볼 수 있다.

### B. 하드 창 대신 지수 감쇠

표는 "직전 2시즌"이라는 하드 창으로 만든다. 3시즌은 2024 +2.9 / 2023 −2.6 으로
엇갈렸다. 그런데 **수준은 대비보다 훨씬 지속적**이므로, 오래된 시즌을 버리는
대신 **감쇠 가중**으로 쓰면 표본을 늘리면서 최신성도 지킬 수 있다.

    d_e = Σ_s γ^(T−s) n_es r̄_es / Σ_s γ^(T−s) n_es     (그 뒤 축소)

### C. 0 이 아니라 **예측된 값**으로 축소

표본이 적은 개체의 `d_e` 는 0 으로 축소된다. 그런데 그 개체의 관측 가능한
속성(통산 비율·구종 믹스·TrackMan 성향)으로 `d_e` 를 예측할 수 있다면,
0 대신 그 예측값으로 축소하는 편이 낫다 (계층 모형의 표준적 개선).

    .\.venv\Scripts\python.exe -u research\exp037_level.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "exp"))
import engine as E                                          # noqa: E402
from path_alloc import build_df                             # noqa: E402
from resid_table import post_for                            # noqa: E402
from traj_probe import r2                                   # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

EXP = "EXP037"
PREV2 = {2022: (2020, 2021), 2023: (2021, 2022), 2024: (2022, 2023)}
ALLSRC = {2022: (2020, 2021), 2023: (2020, 2021, 2022),
          2024: (2020, 2021, 2022, 2023)}


def main():
    E.start_experiment(EXP, "L-EST", "python research/exp037_level.py", "load")
    tr = build_df()
    season = tr["season"].to_numpy()
    y = tr["control_success"].to_numpy(np.float64)
    P = tr["pitcher_id"].to_numpy(np.int64)
    B = tr["batter_id"].to_numpy(np.int64)

    def g(c):
        return tr[c].to_numpy(np.float64)

    SAME = (g("pitcher_hand") == g("batter_hand")).astype(int)
    TWO = (g("strikes_before").astype(int) == 2).astype(int)
    RUN = (g("num_runners_on") > 0).astype(int)
    AX = {"hand": (SAME, 1000), "2S": (TWO, 1000), "runner": (RUN, 2000)}

    pv0, res0 = {}, {}
    for f in (2020, 2021, 2022, 2023, 2024):
        m = season == f
        pv0[f] = (np.load(os.path.join(ROOT, "exp", f"prod_champ_{f}.npy"))[:2].mean(0)
                  + post_for(tr, y, season < f, m))
        res0[f] = y[m] - pv0[f]

    def ctab(ctx, src, k):
        p = np.concatenate([P[season == f] for f in src])
        c = np.concatenate([ctx[season == f] for f in src])
        r = np.concatenate([res0[f] for f in src])
        gg = pd.DataFrame({"p": p, "c": c, "r": r}).groupby(["p", "c"])["r"].agg(
            ["mean", "size"]).unstack()
        n0, n1 = gg[("size", 0)].fillna(0), gg[("size", 1)].fillna(0)
        ne = (n0 * n1) / (n0 + n1).replace(0, np.nan)
        return ((gg[("mean", 1)] - gg[("mean", 0)]) * ne / (ne + k)).dropna()

    CORE = {}
    for f in (2022, 2023, 2024):
        m = season == f
        v = pv0[f].copy()
        for a, (ctx, k) in AX.items():
            v += 0.65 * (pd.Series(P[m]).map(ctab(ctx, PREV2[f], k)).fillna(0.).to_numpy()
                         * np.where(ctx[m] == 1, .5, -.5))
        CORE[f] = v

    def raw(key, src, gamma=None):
        """개체별 (가중 잔차합, 가중 표본수). gamma 가 있으면 시즌 감쇠."""
        parts = []
        T = max(src)
        for s in src:
            m = season == s
            wt = 1.0 if gamma is None else gamma ** (T - s)
            parts.append(pd.DataFrame({"k": key[m], "sr": res0[s] * wt,
                                       "n": np.full(int(m.sum()), wt)}))
        d = pd.concat(parts).groupby("k")[["sr", "n"]].sum()
        return d

    def table(key, src, k, gamma=None, prior=None):
        d = raw(key, src, gamma)
        mu = d["sr"] / d["n"]
        pr = 0.0 if prior is None else prior.reindex(d.index).fillna(0.0)
        return (mu * d["n"] + pr * k) / (d["n"] + k), d["n"]

    def sc(f, ptab, btab, wp=1.0, wb=2.5):
        m = season == f
        return r2(CORE[f]
                  + wp * pd.Series(P[m]).map(ptab).fillna(0.).to_numpy()
                  + wb * pd.Series(B[m]).map(btab).fillna(0.).to_numpy(), y[m])

    def cur(f):
        return (table(P, PREV2[f], 50000)[0], table(B, PREV2[f], 20000)[0])

    b0 = {f: sc(f, *cur(f)) for f in (2022, 2023, 2024)}
    print(f"현행  2022 {b0[2022]:.1f}  2023 {b0[2023]:.1f}  2024 {b0[2024]:.1f}\n")
    E.beat("기준선")

    # ---- A. 개체 표본수 구간별 최적 가중 배수 ----
    print("=" * 84)
    print("A. 개체 표본수 n 구간별 최적 가중 배수 — n 이 작을수록 크게 실어야 하는가")
    print("=" * 84)
    for nm, key, wbase, kk in (("투수", P, 1.0, 50000), ("타자", B, 2.5, 20000)):
        print(f"\n{nm} (기준 가중 {wbase})")
        print(f"{'n 사분위':>10}{'개체':>7}" + "".join(f"{x:>8.2f}" for x in
                                                  (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)))
        for f in (2022, 2024):
            t, n = table(key, PREV2[f], kk)
            q = np.percentile(n, [25, 50, 75])
            m = season == f
            row = []
            kv = pd.Series(key[m]).map(t).fillna(0.).to_numpy()
            nv = pd.Series(key[m]).map(n).fillna(0.).to_numpy()
            bid = np.digitize(nv, q)
            pt, bt = cur(f)
            base = sc(f, pt, bt)
            for qi in range(4):
                sel = bid == qi
                gs = []
                for mul in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
                    adj = np.zeros(int(m.sum()))
                    adj[sel] = (mul - 1.0) * wbase * kv[sel]
                    gs.append(r2(CORE[f] + 1.0 * pd.Series(P[m]).map(pt).fillna(0.).to_numpy()
                                 + 2.5 * pd.Series(B[m]).map(bt).fillna(0.).to_numpy()
                                 + adj, y[m]) - base)
                row.append(gs)
            print(f"  {f} 폴드")
            for qi, gs in enumerate(row):
                print(f"{'Q'+str(qi+1):>10}{int((bid==qi).sum()/1000):>6}k"
                      + "".join(f"{v:>+8.1f}" for v in gs))

    # ---- B. 지수 감쇠 다중 시즌 ----
    print("\n" + "=" * 84)
    print("B. 시즌 지수 감쇠 (하드 2시즌 창 대신)")
    print("=" * 84)
    print(f"{'gamma':>8}{'21→22':>10}{'22→23':>10}{'23→24':>10}{'2022&2024':>12}")
    Bres = {}
    for gamma in (None, 0.3, 0.5, 0.7, 1.0):
        gs = []
        for f in (2022, 2023, 2024):
            src = PREV2[f] if gamma is None else ALLSRC[f]
            pt = table(P, src, 50000, gamma)[0]
            bt = table(B, src, 20000, gamma)[0]
            gs.append(sc(f, pt, bt) - b0[f])
        Bres[str(gamma)] = gs
        lab = "하드2" if gamma is None else f"{gamma:.1f}"
        print(f"{lab:>8}" + "".join(f"{v:>+10.1f}" for v in gs)
              + f"{np.mean([gs[0], gs[2]]):>12.1f}"
              + ("  ★" if all(x > 0 for x in gs) else ""))

    # ---- C. 예측값으로 축소 ----
    print("\n" + "=" * 84)
    print("C. 0 대신 **속성 예측값**으로 축소 (계층 모형)")
    print("=" * 84)
    pcols = [c for c in tr.columns if c.startswith(("asof_pitcher_", "tmc_", "tmh_"))
             and pd.api.types.is_numeric_dtype(tr[c])]
    bcols = [c for c in tr.columns if c.startswith("asof_batter_")
             and pd.api.types.is_numeric_dtype(tr[c])]
    print(f"  투수 속성 {len(pcols)}열 · 타자 속성 {len(bcols)}열")
    print(f"{'lam':>8}{'21→22':>10}{'22→23':>10}{'23→24':>10}{'2022&2024':>12}")
    Cres = {}
    for lam in (0.0, 0.3, 0.6, 1.0):
        gs = []
        for f in (2022, 2023, 2024):
            src = PREV2[f]
            out = []
            for key, cols, kk in ((P, pcols, 50000), (B, bcols, 20000)):
                t, n = table(key, src, kk)
                msrc = np.isin(season, src)
                F = pd.DataFrame({c: tr[c].to_numpy(np.float64)[msrc] for c in cols})
                F["id"] = key[msrc]
                PF = F.groupby("id").mean()
                idx = t.index.intersection(PF.index)
                X = np.nan_to_num(PF.loc[idx].to_numpy(np.float64), nan=0.)
                X = (X - X.mean(0)) / (X.std(0) + 1e-12)
                X = np.column_stack([np.ones(len(X)), X])
                w_ = n.reindex(idx).to_numpy()
                tg = t.reindex(idx).to_numpy()
                A = X.T @ (X * w_[:, None]) + 1e3 * np.eye(X.shape[1])
                beta = np.linalg.solve(A, X.T @ (tg * w_))
                pred = pd.Series(X @ beta, index=idx) * lam
                out.append(table(key, src, kk, prior=pred)[0])
            gs.append(sc(f, out[0], out[1]) - b0[f])
        Cres[str(lam)] = gs
        print(f"{lam:>8.1f}" + "".join(f"{v:>+10.1f}" for v in gs)
              + f"{np.mean([gs[0], gs[2]]):>12.1f}"
              + ("  ★" if all(x > 0 for x in gs) else ""))

    E.finish_experiment(dict(
        experiment_id=EXP, hypothesis_id="L-EST", type="B", level=3,
        started_at=E.read(E.CKPT)["start_time"], decay=Bres, prior=Cres,
        decision="측정",
        what_we_learned="수준 축 추정의 세 개선 — n 의존 가중 · 지수 감쇠 · 예측값 축소"))
    json.dump({"decay": Bres, "prior": Cres},
              open(os.path.join(ROOT, "exp", "exp037_level.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
