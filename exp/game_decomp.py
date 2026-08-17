r"""`+187.8` 분해 — 경기 잠재상태 중 **행 하나로 사전 추정 가능한 몫**이 있는가.

오라클 값을 피처로 만들지 않는다. 같은 경기의 다른 행도 참조하지 않는다.
여기서 하는 것은 **진단**이다.

## 세 갈래

    1 신뢰도    경기 잔차평균이 경기 안에서 재현되는가 (절반 A vs B).
                경기 중앙값이 19투구라 절반은 ~10투구다. 잡음 비중을 먼저 본다.
    2 중첩 분해 투수 -> 투수x월 -> 순수 경기 순으로 **얹어가며** 증분을 잰다.
                각 단계의 그룹 키는 전부 **그 행의 컬럼만으로 만들어진다**.
    3 합법 대리 경기 잔차평균을 그 경기 **시작 시점의 행 값**(cur_*, prev_*, 월,
                구장, 상대팀)으로 예측할 수 있는가. 사상은 2022+2023 에서 적합하고
                2024 로 평가한다 — 라벨 없이 만들 수 있는지의 직접 검증이다.

## 규정

경기 경계 복원은 train 안에서만 쓰는 진단 도구다. 대리 후보는 전부
`행 자신의 컬럼 + 학습 구간 상수` 로만 계산되는 것이어야 승격 대상이 된다.

    .\.venv\Scripts\python.exe -u exp\game_decomp.py
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
from d_limits import cv2_gain, KS                          # noqa: E402
from traj_probe import cells, look                         # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

GFEAT = ["cur_succ", "cur_mid", "cur_ball", "cur_rev", "cur_str",
         "cur_logn_pitch", "asof_pitcher_success_rate",
         "asof_pitcher_prev1_game_success_rate",
         "asof_pitcher_prev3_game_success_rate",
         "asof_pitcher_prev5_game_success_rate",
         "asof_pitcher_prev1_game_middle_rate", "game_month"]


def r2(p, y):
    return 1e5 * np.corrcoef(p, y)[0, 1] ** 2


def games(pid, an, p1, p1m):
    """투수별 asof_n 순으로 prev 값이 바뀌는 지점을 끊는다."""
    o = np.lexsort((an, pid))
    key = np.zeros(len(o), np.int64)
    gid, pp, pv, pv2 = 0, -1, np.nan, np.nan
    for i in o:
        if pid[i] != pp or not (np.isclose(p1[i], pv, equal_nan=True)
                                and np.isclose(p1m[i], pv2, equal_nan=True)):
            gid += 1
        key[i] = gid
        pp, pv, pv2 = pid[i], p1[i], p1m[i]
    return key


def main():
    tr = build_df()
    season = tr["season"].to_numpy()
    y = tr["control_success"].to_numpy(np.float64)
    PID = tr["pitcher_id"].to_numpy(np.int64)
    g = lambda c: tr[c].to_numpy(np.float64)
    pv, res, GK = {}, {}, {}
    for f in (2022, 2023, 2024):
        m = season == f
        P = np.load(os.path.join(ROOT, "exp", f"prod_champ_{f}.npy"))
        pv[f] = P[:3].mean(0) + post_for(tr, y, season < f, m)
        res[f] = y[m] - pv[f]
        GK[f] = games(PID[m], g("asof_pitcher_n")[m],
                      g("asof_pitcher_prev1_game_success_rate")[m],
                      g("asof_pitcher_prev1_game_middle_rate")[m])
        print(f"  폴드 {f}: 경기 {len(np.unique(GK[f])):,}개", flush=True)
    m24 = season == 2024
    y24, p24, r24, k24 = y[m24], pv[2024], res[2024], GK[2024]
    base = r2(p24, y24)
    half = np.random.default_rng(0).random(len(y24)) < 0.5
    out = {"base": base}

    print("\n" + "=" * 84)
    print("1. 신뢰도 — 경기 잔차평균의 얼마가 잡음인가")
    print("=" * 84)
    d = pd.DataFrame({"k": k24, "r": r24, "h": half})
    A = d[d.h].groupby("k")["r"].agg(["mean", "size"])
    B = d[~d.h].groupby("k")["r"].agg(["mean", "size"])
    J = A.join(B, lsuffix="_a", rsuffix="_b", how="inner")
    for nmin in (5, 15, 40):
        S = J[(J["size_a"] >= nmin) & (J["size_b"] >= nmin)]
        if len(S) < 20:
            continue
        c = float(np.corrcoef(S["mean_a"], S["mean_b"])[0, 1])
        sd = float(np.std(np.concatenate([S["mean_a"], S["mean_b"]])))
        nse = float(np.sqrt(np.mean(0.25 / S[["size_a", "size_b"]].to_numpy())))
        sig = max(sd ** 2 - nse ** 2, 0) ** 0.5
        print(f"  절반당 {nmin:>2}투구 이상 경기 {len(S):>5}개  A~B 상관 {c:>+6.3f}"
              f"   관측 sd {sd:.4f}  잡음 sd {nse:.4f}  신호 sd {sig:.4f}"
              f"   신호분산비 {(sig/sd)**2:.2f}")
        out[f"rel_{nmin}"] = dict(n=len(S), corr=c, sd=sd, noise=nse, sig=sig)

    print("\n" + "=" * 84)
    print("2. 중첩 분해 — 행 컬럼만으로 만든 그룹을 얹어가며")
    print("=" * 84)
    MON = g("game_month").astype(np.int64)[m24]
    TB = tr["top_bottom"].to_numpy()[m24]
    HOME = np.where(TB == "T", tr["pitcher_team_id"].to_numpy(np.int64)[m24],
                    tr["batter_team_id"].to_numpy(np.int64)[m24])
    BT = tr["batter_team_id"].to_numpy(np.int64)[m24]
    STEPS = [("투수", PID[m24]),
             ("+ 투수 x 월", PID[m24] * 100 + MON),
             ("+ 투수 x 월 x 구장", (PID[m24] * 100 + MON) * 100 + HOME),
             ("+ (투수, 경기)", k24)]
    cur = p24.copy()
    prev_total = 0.0
    print(f"{'단계':<24}{'셀':>8}{'누적':>10}{'증분':>10}")
    for name, key in STEPS:
        gain, nc, bk = cv2_gain(key, cur if False else p24, y24, half)
        # 누적은 이전 보정 위에서 다시 잰다
        res_now = y24 - cur
        add = np.zeros(len(y24))
        best, bk2 = -1e9, None
        for k in KS:
            a2 = np.zeros(len(y24))
            for msk in (half, ~half):
                u, tab, _ = cells(key[msk], res_now[msk], k)
                a2[~msk] = look(u, tab, key[~msk])
            v = r2(cur + a2, y24)
            if v > best:
                best, bk2, add = v, k, a2
        inc = best - r2(cur, y24)
        cur = cur + add
        tot = best - base
        print(f"{name:<24}{len(np.unique(key)):>8,}{tot:>+10.1f}{inc:>+10.1f}")
        out[f"nest|{name}"] = dict(total=tot, inc=inc, k=bk2)
        prev_total = tot

    print("\n" + "=" * 84)
    print("3. 합법 대리 — 경기 시작 시점의 행 값으로 경기 잔차평균을 맞히는가")
    print("=" * 84)
    tab = {}
    for f in (2022, 2023, 2024):
        m = season == f
        df = pd.DataFrame({c: g(c)[m] for c in GFEAT})
        df["k"] = GK[f]
        df["r"] = res[f]
        agg = df.groupby("k").agg(**{c: (c, "first") for c in GFEAT},
                                  r=("r", "mean"), n=("r", "size"))
        tab[f] = agg[agg["n"] >= 10]
        print(f"  {f}: 10투구 이상 경기 {len(tab[f]):,}개")
    T = pd.concat([tab[2022], tab[2023]]).dropna()
    X = np.column_stack([T[GFEAT].to_numpy(), np.ones(len(T))])
    w = np.linalg.lstsq(X, T["r"].to_numpy(), rcond=None)[0]
    for f in (2023, 2024):
        S = tab[f].dropna()
        hat = np.column_stack([S[GFEAT].to_numpy(), np.ones(len(S))]) @ w
        c = float(np.corrcoef(hat, S["r"])[0, 1])
        print(f"  {f} 경기 {len(S):,}개   corr(예측, 실제 경기잔차평균) = {c:+.3f}")
        out[f"proxy_corr_{f}"] = c
    # 2024 행에 적용해 실제 이득
    S24 = tab[2024].dropna()
    hat24 = pd.Series(
        np.column_stack([S24[GFEAT].to_numpy(), np.ones(len(S24))]) @ w,
        index=S24.index)
    addg = hat24.reindex(k24).fillna(0.0).to_numpy()
    print(f"\n  {'가중':>7}{'2024 이득':>11}")
    for wt in (0.1, 0.25, 0.5, 1.0):
        print(f"  {wt:>7.2f}{r2(p24 + wt * addg, y24) - base:>+11.1f}")
    print("  (주의 — 이 대리조차 '경기 시작 행'을 알아야 하는데 2025 에서는 "
          "그 식별이 불가능하다. 상한 참고용이다)")

    json.dump(out, io.open(os.path.join(ROOT, "exp", "game_decomp.json"), "w",
                           encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
