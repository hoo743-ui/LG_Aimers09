r"""EXP004 — LEVEL 1 표현 공간. 같은 정보, 다른 형태. 학습 0회.

## LEVEL 0 과 무엇이 다른가

LEVEL 0 은 "어떤 관계에 신호가 있는가"를 물었다 (답: 타자손 하나).
LEVEL 1 은 **그 신호를 지금 형태로 싣는 것이 최선인가**를 묻는다.

현행 표현은 세 가정을 동시에 깔고 있다.

    보정 = ±0.5 · d_p        강제 대칭 · 기저확률과 무관한 상수 · 축별 독립

각 가정이 하나씩 공격면이다.

    L1-A 공통 잠재 인자    축별 독립을 깬다. 세 d 가 상관하면 3배 표본으로 추정
    L1-B 분산 정규화       상수 가산을 깬다. sqrt(p(1-p)) 에 비례시킨다
    L1-C 비대칭 분해       강제 대칭을 깬다. 두 셀 편차를 자유 모수로
    L1-D 특성 사영         자기 이력 대신 투수 속성으로 회귀 (매끄럽고 신인 포함)
    L1-E 로짓 기하         확률 가산 대신 오즈 곱 (경계에서 밖으로 밀지 않는다)

## 규약

정보는 **완전히 동일**하다 — 직전 2시즌 strictly OOF 잔차, k=1000/1000/2000.
바뀌는 것은 형태뿐이다. 그래야 표현 효과만 분리된다.

혼합 계수 lambda 같은 자유 모수는 **2022·2023 에서만** 고르고 2024 는 숨긴다.

    .\.venv\Scripts\python.exe -u research\exp004_repr.py
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

EXP = "EXP004"
PREV2 = {2022: (2020, 2021), 2023: (2021, 2022), 2024: (2022, 2023)}
LAM = [0.0, 0.2, 0.4, 0.6, 0.8]


def main():
    E.start_experiment(EXP, "L1-A..E", "python research/exp004_repr.py", "load")
    tr = build_df()
    season = tr["season"].to_numpy()
    y = tr["control_success"].to_numpy(np.float64)
    P = tr["pitcher_id"].to_numpy(np.int64)

    def g(c):
        return tr[c].to_numpy(np.float64)

    AX = {"hand": ((g("pitcher_hand") == g("batter_hand")).astype(int), 1000),
          "2S": ((g("strikes_before").astype(int) == 2).astype(int), 1000),
          "runner": ((g("num_runners_on") > 0).astype(int), 2000)}

    pv0, res0 = {}, {}
    for f in (2020, 2021, 2022, 2023, 2024):
        m = season == f
        pv0[f] = (np.load(os.path.join(ROOT, "exp", f"prod_champ_{f}.npy"))[:2].mean(0)
                  + post_for(tr, y, season < f, m))
        res0[f] = y[m] - pv0[f]
    E.beat("잔차 준비")

    def cellstat(ctx, src):
        """투수 x 셀의 (평균, 개수). 원시값 — 축소 전."""
        p = np.concatenate([P[season == f] for f in src])
        c = np.concatenate([ctx[season == f] for f in src])
        r = np.concatenate([res0[f] for f in src])
        gg = pd.DataFrame({"p": p, "c": c, "r": r}).groupby(["p", "c"])["r"].agg(
            ["mean", "size"]).unstack()
        n0, n1 = gg[("size", 0)].fillna(0), gg[("size", 1)].fillna(0)
        return (gg[("mean", 0)], gg[("mean", 1)], n0, n1)

    def dvec(ctx, src, k):
        """현행 표현의 축소된 d_p 와 유효표본."""
        m0, m1, n0, n1 = cellstat(ctx, src)
        ne = (n0 * n1) / (n0 + n1).replace(0, np.nan)
        return ((m1 - m0) * ne / (ne + k)).dropna(), ne.dropna()

    def apply_sym(t, ctx, m):
        """현행 — 강제 대칭 상수 가산."""
        return (pd.Series(P[m]).map(t).fillna(0.).to_numpy()
                * np.where(ctx[m] == 1, .5, -.5))

    # ---- 진단: 세 축의 투수 수준 상관구조 (한 번도 잰 적 없다) ----
    print("=" * 78)
    print("진단 — 세 차등축이 투수 수준에서 상관하는가 (L1-A 의 전제)")
    print("=" * 78)
    D23 = {}
    for a, (ctx, k) in AX.items():
        D23[a] = dvec(ctx, (2022, 2023), k)[0]
    M = pd.DataFrame(D23).dropna()
    print(f"  세 축을 모두 가진 투수 {len(M):,}명")
    C = M.corr()
    print(f"  {'':<10}" + "".join(f"{c:>10}" for c in C.columns))
    for i in C.index:
        print(f"  {i:<10}" + "".join(f"{C.loc[i, j]:>+10.3f}" for j in C.columns))
    ev = np.linalg.eigvalsh(C.to_numpy())[::-1]
    print(f"  고유값 {ev.round(3)}   제1성분 설명력 {ev[0] / 3:.1%}"
          f"   (무상관이면 33.3%)")
    E.beat("상관구조 진단 완료")

    # ---- 표현별 보정 벡터 ----
    def corr_vectors(fold, lam):
        """폴드 하나에 대해 표현별 보정 벡터를 만든다."""
        m = season == fold
        src = PREV2[fold]
        base = pv0[fold]
        out = {}
        sym = np.zeros(int(m.sum()))
        Ds, NEs = {}, {}
        for a, (ctx, k) in AX.items():
            d, ne = dvec(ctx, src, k)
            Ds[a], NEs[a] = d, ne
            sym += apply_sym(d, ctx, m)
        out["V0 현행(대칭 상수)"] = sym

        # L1-B 분산 정규화 — sqrt(p(1-p)) 에 비례, 평균 크기 보존
        s = np.sqrt(np.clip(base, 1e-6, 1 - 1e-6) * (1 - np.clip(base, 1e-6, 1 - 1e-6)))
        out["V1 분산 정규화"] = sym * (s / s.mean())

        # L1-E 로짓 기하 — 오즈 곱으로 싣고 되돌린다
        pb = np.clip(base, 1e-6, 1 - 1e-6)
        z = np.log(pb / (1 - pb))
        dz = sym / (pb * (1 - pb))
        out["V2 로짓 기하"] = 1.0 / (1.0 + np.exp(-(z + dz))) - base

        # L1-C 비대칭 분해 — 두 셀 편차를 자유 모수로 (셀별 개별 축소)
        asym = np.zeros(int(m.sum()))
        for a, (ctx, k) in AX.items():
            m0, m1, n0, n1 = cellstat(ctx, src)
            tot = (n0 + n1).replace(0, np.nan)
            mu = (n0 * m0 + n1 * m1) / tot
            a1 = ((m1 - mu) * n1 / (n1 + k)).dropna()
            a0 = ((m0 - mu) * n0 / (n0 + k)).dropna()
            pm = pd.Series(P[m])
            asym += np.where(ctx[m] == 1, pm.map(a1).fillna(0.).to_numpy(),
                             pm.map(a0).fillna(0.).to_numpy())
        out["V3 비대칭 분해"] = asym

        # L1-A 공통 잠재 인자 — 세 축의 제1주성분으로 부분 대체
        Md = pd.DataFrame(Ds)
        sd = Md.std()
        Z = (Md / sd)
        f_p = Z.mean(axis=1, skipna=True)                    # 공통 인자 (z 평균)
        common = np.zeros(int(m.sum()))
        for a, (ctx, k) in AX.items():
            z = Z[a].dropna()
            lo = float(np.corrcoef(z, f_p.reindex(z.index))[0, 1])
            dhat = (f_p * lo * sd[a]).reindex(Ds[a].index)
            blend = lam * dhat.fillna(Ds[a]) + (1 - lam) * Ds[a]
            common += apply_sym(blend, ctx, m)
        out[f"V4 공통인자 lam={lam:g}"] = common

        # L1-D 특성 사영 — d_p 를 투수 속성으로 능형회귀
        pcols = [c for c in tr.columns
                 if c.startswith("asof_pitcher_") and "prev1" not in c
                 and pd.api.types.is_numeric_dtype(tr[c])]
        msrc = np.isin(season, src)
        FT = pd.DataFrame({c: tr[c].to_numpy(np.float64)[msrc] for c in pcols})
        FT["hand"] = g("pitcher_hand")[msrc]
        FT["pid"] = P[msrc]
        PF = FT.groupby("pid").mean()
        proj = np.zeros(int(m.sum()))
        for a, (ctx, k) in AX.items():
            idx = Ds[a].index.intersection(PF.index)
            Xr = PF.loc[idx].to_numpy(np.float64)
            Xr = np.nan_to_num(Xr, nan=0.0, posinf=0.0, neginf=0.0)
            Xr = (Xr - Xr.mean(0)) / (Xr.std(0) + 1e-12)
            Xr = np.column_stack([np.ones(len(Xr)), Xr])
            w = NEs[a].reindex(idx).to_numpy()
            tgt = Ds[a].reindex(idx).to_numpy()
            A = Xr.T @ (Xr * w[:, None]) + 1e3 * np.eye(Xr.shape[1])
            beta = np.linalg.solve(A, Xr.T @ (tgt * w))
            dhat = pd.Series(Xr @ beta, index=idx)
            blend = lam * dhat.reindex(Ds[a].index).fillna(Ds[a]) + (1 - lam) * Ds[a]
            proj += apply_sym(blend, ctx, m)
        out[f"V5 특성 사영 lam={lam:g}"] = proj
        return out, base

    print("\n" + "=" * 88)
    print("표현별 폴드 점수 (현행 V0 대비 증분)")
    print("=" * 88)
    rows = {}
    for fold in (2022, 2023, 2024):
        E.beat(f"fold {fold}")
        m = season == fold
        for lam in LAM:
            cv, base = corr_vectors(fold, lam)
            b0 = r2(base + cv["V0 현행(대칭 상수)"], y[m])
            for nm, vec in cv.items():
                if nm.endswith("lam=0") and lam != 0:
                    continue
                key = nm if "lam" not in nm else nm
                rows.setdefault(key, {})[fold] = r2(base + vec, y[m]) - b0
            rows.setdefault("__base__", {})[fold] = b0
    b0s = rows.pop("__base__")
    print(f"  C3 기준선   2022 {b0s[2022]:.1f}   2023 {b0s[2023]:.1f}"
          f"   2024 {b0s[2024]:.1f}\n")
    print(f"{'표현':<26}{'2022':>10}{'2023':>10}{'2024(숨김)':>13}{'과거평균':>10}")
    print("-" * 88)
    order = sorted(rows, key=lambda n: -np.mean([rows[n][2022], rows[n][2023]]))
    for nm in order:
        v = rows[nm]
        past = np.mean([v[2022], v[2023]])
        print(f"{nm:<26}{v[2022]:>+10.1f}{v[2023]:>+10.1f}{v[2024]:>+13.1f}"
              f"{past:>+10.1f}")

    cand = {n: v for n, v in rows.items() if not n.startswith("V0")}
    best = max(cand, key=lambda n: np.mean([cand[n][2022], cand[n][2023]]))
    bv = cand[best]
    ok = all(bv[f] > 0 for f in (2022, 2023, 2024))
    dec = "PROMISING" if (ok and bv[2024] >= 4.8) else "REJECTED"
    E.finish_experiment(dict(
        experiment_id=EXP, hypothesis_id="L1-A..E", type="REPRESENTATION", level=1,
        started_at=E.read(E.CKPT)["start_time"],
        corr_matrix=C.round(3).to_dict(), pc1=float(ev[0] / 3),
        results={n: {str(f): round(x, 2) for f, x in v.items()} for n, v in rows.items()},
        decision=dec, artifact=None,
        what_we_learned=(f"과거 최선 표현 = {best}; 2022 {bv[2022]:+.1f} "
                         f"2023 {bv[2023]:+.1f} 2024 {bv[2024]:+.1f}")))
    for h in ("L1-A", "L1-B", "L1-C", "L1-D", "L1-E"):
        E.set_hypothesis_status(h, "TESTED", level=1)
    json.dump({"corr": C.round(4).to_dict(), "rows": {n: {str(k): v for k, v in d.items()}
               for n, d in rows.items()}},
              open(os.path.join(ROOT, "exp", "exp004_repr.json"), "w"),
              indent=1, default=float)
    print(f"\n과거(2022·2023)로 고른 최선 = {best}")
    print(f"그 표현의 2024(숨김) = {bv[2024]:+.1f}")
    print(f"DECISION = {dec}")


if __name__ == "__main__":
    main()
