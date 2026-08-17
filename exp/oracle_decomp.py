r"""`+17.9` 분해 — 오라클이 쓴 정보 중 무엇이 합법적으로 관측 가능한가. 학습 0회.

## 오라클의 정확한 정체 (코드 재구성)

    resid = y2024 - Champion예측
    2024 행을 무작위 반으로 가르고, 한쪽 절반의 라벨로 **투수별 잔차 평균**을
    구해 n/(n+k) 로 축소한 뒤 반대쪽 절반에 조회해 더한다. k 를 훑어 최댓값.

즉 오라클이 아는 것은 **"그 투수의 그 시즌 나머지 절반에서의 평균 잔차"** 다.
행 자신의 라벨도, 시즌 전체 평균도 아니다. 최적 `k=2000` 이라는 사실 자체가
셀 평균의 대부분이 잡음임을 시사한다 (650투구 투수도 24%만 반영).

## 분해 축

    1 신뢰도    2024 안에서 절반 A vs B 의 투수별 잔차평균 상관.
                이항 잡음 하한과 비교해 **진짜 시즌내 효과의 크기**를 뽑는다.
    2 시즌 간   corr(2022/2023 잔차평균, 2024). 감쇠 보정도 함께.
    3 합법 대리 학습 시즌에서 [투수 수준 합법 피처 -> 잔차평균] 사상을 적합해
                2024 에 적용. **라벨 없이** 같은 보정을 만들 수 있는가.

    .\.venv\Scripts\python.exe -u exp\oracle_decomp.py
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

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

FEAT = ["cur_succ", "cur_mid", "cur_ball", "cur_rev", "cur_str",
        "cur_logn_pitch", "cur_bsucc", "cur_bmid",
        "asof_pitcher_success_rate", "asof_pitcher_middle_rate",
        "asof_pitcher_n", "asof_pitcher_prev5_game_success_rate",
        "asof_pitcher_prev5_game_middle_rate"]


def r2(p, y):
    return 1e5 * np.corrcoef(p, y)[0, 1] ** 2


def main():
    tr = build_df()
    season = tr["season"].to_numpy()
    y = tr["control_success"].to_numpy(np.float64)
    PID = tr["pitcher_id"].to_numpy(np.int64)
    pv, res = {}, {}
    for f in (2022, 2023, 2024):
        m = season == f
        P = np.load(os.path.join(ROOT, "exp", f"prod_champ_{f}.npy"))
        pv[f] = P[:3].mean(0) + post_for(tr, y, season < f, m)
        res[f] = y[m] - pv[f]
    m24 = season == 2024
    y24, p24, r24, pid24 = y[m24], pv[2024], res[2024], PID[m24]
    base = r2(p24, y24)
    print(f"기준선 {base:.1f}   2024 {len(y24):,}행   투수 {len(set(pid24)):,}명")

    print("\n" + "=" * 84)
    print("1. 신뢰도 — 투수별 잔차평균이 시즌 안에서 재현되는가")
    print("=" * 84)
    rng = np.random.default_rng(0)
    half = rng.random(len(y24)) < 0.5
    d = pd.DataFrame({"p": pid24, "r": r24, "h": half})
    A = d[d.h].groupby("p")["r"].agg(["mean", "size"])
    B = d[~d.h].groupby("p")["r"].agg(["mean", "size"])
    J = A.join(B, lsuffix="_a", rsuffix="_b", how="inner")
    out = {}
    for nmin in (50, 100, 300):
        S = J[(J["size_a"] >= nmin) & (J["size_b"] >= nmin)]
        if len(S) < 10:
            continue
        c = float(np.corrcoef(S["mean_a"], S["mean_b"])[0, 1])
        sd = float(np.std(np.concatenate([S["mean_a"], S["mean_b"]])))
        nse = float(np.sqrt(np.mean(0.25 / S[["size_a", "size_b"]].to_numpy())))
        sig = max(sd ** 2 - nse ** 2, 0) ** 0.5
        print(f"  절반당 {nmin:>3}투구 이상 투수 {len(S):>4}명"
              f"   절반 A~B 상관 {c:>+6.3f}"
              f"   관측 sd {sd:.4f}  잡음 sd {nse:.4f}  신호 sd {sig:.4f}"
              f"   신호분산비 {(sig/sd)**2 if sd > 0 else 0:.2f}")
        out[f"rel_n{nmin}"] = dict(n=len(S), corr=c, sd=sd, noise=nse, sig=sig)

    print("\n" + "=" * 84)
    print("2. 시즌 간 안정성 — 과거 시즌으로 같은 보정을 만들 수 있는가")
    print("=" * 84)
    M = {}
    for f in (2022, 2023, 2024):
        m = season == f
        M[f] = pd.DataFrame({"p": PID[m], "r": res[f]}).groupby("p")["r"].agg(
            ["mean", "size"])
    print(f"  {'시즌쌍':<14}{'공통 투수':>10}{'상관':>9}{'감쇠보정 상관':>14}")
    rel = out.get("rel_n300", out.get("rel_n100", {})).get("corr", np.nan)
    for a, b in ((2022, 2024), (2023, 2024), (2022, 2023)):
        J2 = M[a].join(M[b], lsuffix="_a", rsuffix="_b", how="inner")
        J2 = J2[(J2["size_a"] >= 300) & (J2["size_b"] >= 300)]
        if len(J2) < 10:
            continue
        c = float(np.corrcoef(J2["mean_a"], J2["mean_b"])[0, 1])
        print(f"  {a}~{b:<9}{len(J2):>10}{c:>+9.3f}"
              f"{c/max(rel, 1e-9):>+14.3f}")
        out[f"cross_{a}_{b}"] = dict(n=len(J2), corr=c)

    print("\n" + "=" * 84)
    print("3. 합법 대리 — 라벨 없이 투수별 잔차평균을 예측할 수 있는가")
    print("=" * 84)
    rows = []
    for f in (2022, 2023):
        m = season == f
        df = pd.DataFrame({c: tr[c].to_numpy(np.float64)[m] for c in FEAT})
        df["p"] = PID[m]
        g = df.groupby("p").mean()
        g["r"] = M[f]["mean"]
        g["n"] = M[f]["size"]
        rows.append(g[g["n"] >= 300])
    T = pd.concat(rows).dropna()
    df24 = pd.DataFrame({c: tr[c].to_numpy(np.float64)[m24] for c in FEAT})
    df24["p"] = pid24
    G24 = df24.groupby("p").mean()
    X = np.column_stack([T[FEAT].to_numpy(), np.ones(len(T))])
    w = np.linalg.lstsq(X, T["r"].to_numpy(), rcond=None)[0]
    pred_r = np.column_stack([G24[FEAT].to_numpy(), np.ones(len(G24))]) @ w
    S24 = M[2024]
    J3 = pd.DataFrame({"hat": pred_r}, index=G24.index).join(S24, how="inner")
    J3 = J3[J3["size"] >= 300]
    c = float(np.corrcoef(J3["hat"], J3["mean"])[0, 1])
    print(f"  학습 시즌(2022+2023) 투수 {len(T)}명으로 적합, 2024 투수 {len(J3)}명 평가")
    print(f"  corr(예측 잔차평균, 실제 잔차평균) = {c:+.3f}")
    out["proxy_corr"] = c
    tab = pd.Series(pred_r, index=G24.index)
    add = tab.reindex(pid24).fillna(0.0).to_numpy()
    for wgt in (0.25, 0.5, 1.0):
        print(f"    가중 {wgt:.2f} 로 적용 -> {r2(p24 + wgt*add, y24) - base:+.1f}")

    print("\n" + "=" * 84)
    print("4. 같은 자 위에 놓고 비교")
    print("=" * 84)
    print(f"  오라클 (2024 라벨, 교차적합, k=2000)            +17.9")
    print(f"  과거 시즌 라벨로 만든 표 (2023->2024)            +1.0")
    print(f"  과거 시즌 라벨로 만든 표 (2022+2023->2024)       +0.5")
    print(f"  라벨 없이 합법 피처로 예측 (위 3절)             위 표")
    json.dump(out, io.open(os.path.join(ROOT, "exp", "oracle_decomp.json"), "w",
                           encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
