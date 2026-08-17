r"""TRANSFER-AWARE SHRINKAGE 검증 — `θ̂_future = ρ x θ̂_current` 가 성립하는가.

새 후보를 만들지 않는다. 이론 검증 전용. 학습 0회 (폴드 2020~2024 예측 재사용).

## 이론

관측: `m_{g,s} = θ_{g,s} + e`,  `Var(e) = v / n_{g,s}`
가정: `θ_{g,s+1} = ρ θ_{g,s} + η`

다음 시즌을 예측하는 최선의 선형 추정량은

    θ̂_{s+1} = ρ x [ n/(n + k_EB) x m_s ]        k_EB = v / sigma_θ^2

즉 **EB 추정값에 ρ 를 곱한다.** 이를 `n/(n+k_T)` 로 억지로 쓰면
`k_T = n(1/ρ − 1) + k_EB/ρ` 로 **n 에 의존**하므로 단일 k 로는 표현되지 않는다.
그래서 여기서는 단일 k 를 맞추지 않고 **곱셈 계수 ρ** 를 검증한다.

## 규약 (의사미래)

각 전이 `s -> s+1` 에서

    k_EB   시즌 s 에서만 적률법으로 추정
    ρ̂      **그 이전 전이들에서만** 적합 (미래 라벨 사용 금지)
           = m_{t+1} 를 eb_t 에 회귀한 가중 기울기 (t+1 <= s 인 t 만)
    평가    시즌 s+1 에서 MSE · 상관 · 부호보존 · 실제 rho^2 증분

경기 단위 오라클은 쓰지 않는다 — 평가는 **다음 시즌에 실제로 적용**해서 한다.

    .\.venv\Scripts\python.exe -u exp\ta_shrink.py
"""
import io
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "exp"))
from path_alloc import build_df                            # noqa: E402
from resid_table import post_for                           # noqa: E402
from traj_probe import r2                                  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

FOLDS = (2020, 2021, 2022, 2023, 2024)
TRANS = [(2020, 2021), (2021, 2022), (2022, 2023), (2023, 2024)]


def moment_k(m, n, v):
    """적률법 k = v / sigma_b^2.  가중 분산에서 잡음을 뺀다."""
    w = n / n.sum()
    var_m = float(np.sum(w * (m - np.sum(w * m)) ** 2))
    noise = float(np.sum(w * (v / n)))
    sb = max(var_m - noise, 1e-9)
    return v / sb, sb


def main():
    tr = build_df()
    season = tr["season"].to_numpy()
    y = tr["control_success"].to_numpy(np.float64)
    PID = tr["pitcher_id"].to_numpy(np.int64)
    pv, res, M = {}, {}, {}
    for f in FOLDS:
        m = season == f
        P = np.load(os.path.join(ROOT, "exp", f"prod_champ_{f}.npy"))
        pv[f] = P[:2].mean(0) + post_for(tr, y, season < f, m)
        res[f] = y[m] - pv[f]
        M[f] = pd.DataFrame({"p": PID[m], "r": res[f]}).groupby("p")["r"].agg(
            ["mean", "size"])
        print(f"  폴드 {f}: {int(m.sum()):,}행  투수 {len(M[f]):,}  "
              f"rho^2 {r2(pv[f], y[m]):.1f}", flush=True)

    EB = {}
    for f in FOLDS:
        v = float(np.var(res[f]))
        k, sb = moment_k(M[f]["mean"].to_numpy(), M[f]["size"].to_numpy(), v)
        EB[f] = dict(k=k, sigma_b2=sb, v=v)
        M[f]["eb"] = M[f]["mean"] * M[f]["size"] / (M[f]["size"] + k)
        print(f"  {f} 적률 k = {k:>8.0f}   sigma_b^2 = {sb:.2e}", flush=True)

    def fit_rho(upto):
        """t+1 <= upto 인 전이들에서 rho 를 가중 회귀로 적합한다."""
        X, Y, W = [], [], []
        for a, b in TRANS:
            if b > upto:
                continue
            J = M[a][["eb"]].join(M[b][["mean", "size"]], how="inner",
                                  rsuffix="_b").dropna()
            X.append(J["eb"].to_numpy())
            Y.append(J["mean"].to_numpy())
            W.append(J["size"].to_numpy())
        if not X:
            return None, 0
        X, Y, W = np.concatenate(X), np.concatenate(Y), np.concatenate(W)
        return float(np.sum(W * X * Y) / np.sum(W * X * X)), len(X)

    print("\n" + "=" * 96)
    print("전이별 검증 — 세 추정량을 다음 시즌에 실제로 적용한다")
    print("=" * 96)
    out = {}
    for a, b in TRANS:
        mb = season == b
        base = r2(pv[b], y[mb])
        rho, npair = fit_rho(a)          # a 까지의 전이만 사용
        rho_txt = f"{rho:+.3f} (과거 {npair}쌍)" if rho is not None else "추정 불가"
        # 사후 최적 ρ (참고용, 선택에 쓰지 않음)
        J = M[a][["eb"]].join(M[b][["mean", "size"]], how="inner",
                              rsuffix="_b").dropna()
        rho_post = float(np.sum(J["size"] * J["eb"] * J["mean"])
                         / np.sum(J["size"] * J["eb"] ** 2))
        print(f"\n[{a} -> {b}]  기준선 {base:.1f}   적률 k={EB[a]['k']:.0f}   "
              f"과거 ρ̂ {rho_txt}   사후최적 ρ {rho_post:+.3f}")
        print(f"  {'추정량':<22}{'다음시즌 이득':>13}{'가중MSE':>12}"
              f"{'상관':>8}{'부호보존':>9}{'평균 축소율':>11}")
        est = {"RAW (k=0)": M[a]["mean"],
               "STANDARD EB": M[a]["eb"]}
        if rho is not None:
            est["TRANSFER-AWARE (ρ̂ x EB)"] = M[a]["eb"] * rho
        est["참고: 사후최적 ρ x EB"] = M[a]["eb"] * rho_post
        row = {}
        for nm, s in est.items():
            add = pd.Series(PID[mb]).map(s).fillna(0.0).to_numpy()
            gain = r2(pv[b] + add, y[mb]) - base
            J2 = pd.DataFrame({"hat": s}).join(M[b][["mean", "size"]],
                                               how="inner").dropna()
            w = J2["size"].to_numpy()
            mse = float(np.sum(w * (J2["hat"] - J2["mean"]) ** 2) / w.sum())
            cr = float(np.corrcoef(J2["hat"], J2["mean"])[0, 1])
            sgn = float(np.mean(np.sign(J2["hat"]) == np.sign(J2["mean"])))
            shr = float(np.mean(np.abs(s) / np.abs(M[a]["mean"]).replace(0, np.nan)))
            row[nm] = dict(gain=gain, mse=mse, corr=cr, sign=sgn)
            print(f"  {nm:<22}{gain:>+13.1f}{mse:>12.2e}{cr:>+8.3f}"
                  f"{sgn:>9.0%}{shr:>11.3f}")
        out[f"{a}->{b}"] = dict(rho_hat=rho, rho_post=rho_post,
                                k_eb=EB[a]["k"], base=base, est=row)

    print("\n" + "=" * 96)
    print("ρ 안정성")
    print("=" * 96)
    print(f"  {'전이':<14}{'사후최적 ρ':>12}{'과거로 적합한 ρ̂':>18}")
    for a, b in TRANS:
        d = out[f"{a}->{b}"]
        rh = f"{d['rho_hat']:+.3f}" if d["rho_hat"] is not None else "-"
        print(f"  {a}->{b:<9}{d['rho_post']:>+12.3f}{rh:>18}")

    json.dump(out, io.open(os.path.join(ROOT, "exp", "ta_shrink.json"), "w",
                           encoding="utf-8"), indent=1, ensure_ascii=False,
              default=float)


if __name__ == "__main__":
    main()
