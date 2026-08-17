r"""latent 대리 탐침 — row-only 정보가 경기/타석 잠재축의 **방향**을 설명하는가.

오라클을 피처로 만들지 않는다. 단순 보정(pred + w*proxy)부터 하지 않는다.
먼저 **방향 정합**을 본다.

## 방법

경기 잠재축의 실체는 교차적합으로 만든 보정 벡터 `add_game` 이다 (오라클 +187.8
을 만든 그 벡터). row-only 그룹 `R` 도 같은 방식으로 보정 벡터 `add_R` 을 만든다.

    정합도 = corr(add_R, add_game)      두 보정이 같은 행을 같은 방향으로 미는가

정합도가 낮으면 그 관계는 잠재축과 무관한 다른 것을 잡고 있다는 뜻이고,
높으면 잠재축의 그림자일 수 있다 -> 그때만 시즌 간 전이를 잰다.

## 후보 (사용자 지정 + 미검증 조합)

`pitcher current-state x {월, game_type, 타자손, 카운트, 이닝, 주자/아웃, 팀,
점수}` 중 **이미 닫힌 것은 제외**한다 — cur x {adv,onb,sh,bs}=X/H1, cur x 2S=K2,
cur x is_F=st(−6.7), cur x 이닝/li=REGIME C(−3.0), cur x 점수차(오늘 −2.1/−3.6).
남은 미검증은 **cur x 월**과 **cur x 팀**이다. 여기에 투수 정체성 계열
(투수x타자 매치업, 투수x점수차 등)을 정합도 비교용으로 함께 놓는다.

    .\.venv\Scripts\python.exe -u exp\latent_proxy.py
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
from traj_probe import cells, look, r2                     # noqa: E402
from game_decomp import games                              # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

KS = [0, 50, 150, 500, 2000, 10000]


def corr_fit(key, pred, y, half):
    """교차적합 보정 벡터와 그 이득. 축소는 이득 최대인 것으로 고른다."""
    resid = y - pred
    b = r2(pred, y)
    best, badd = -1e9, None
    for k in KS:
        add = np.zeros(len(y))
        for m in (half, ~half):
            u, tab, _ = cells(key[m], resid[m], k)
            add[~m] = look(u, tab, key[~m])
        v = r2(pred + add, y) - b
        if v > best:
            best, badd = v, add
    return best, badd


def main():
    tr = build_df()
    season = tr["season"].to_numpy()
    y = tr["control_success"].to_numpy(np.float64)
    g = lambda c: tr[c].to_numpy(np.float64)
    PID = tr["pitcher_id"].to_numpy(np.int64)
    BID = tr["batter_id"].to_numpy(np.int64)
    pv, res, GK = {}, {}, {}
    for f in (2022, 2023, 2024):
        m = season == f
        P = np.load(os.path.join(ROOT, "exp", f"prod_champ_{f}.npy"))
        pv[f] = P[:3].mean(0) + post_for(tr, y, season < f, m)
        res[f] = y[m] - pv[f]
        GK[f] = games(PID[m], g("asof_pitcher_n")[m],
                      g("asof_pitcher_prev1_game_success_rate")[m],
                      g("asof_pitcher_prev1_game_middle_rate")[m])
    m24 = season == 2024
    y24, p24 = y[m24], pv[2024]
    base = r2(p24, y24)
    half = np.random.default_rng(0).random(len(y24)) < 0.5

    gain_g, add_g = corr_fit(GK[2024], p24, y24, half)
    print(f"기준선 {base:.1f}   경기 잠재축 보정: 이득 {gain_g:+.1f}, "
          f"보정 sd {add_g.std():.4f}")

    def dec(x, q=10):
        v = x[m24]
        qq = np.nanquantile(v, np.linspace(0, 1, q + 1)[1:-1])
        return np.searchsorted(qq, np.nan_to_num(v, nan=np.nanmedian(v)))

    MON = g("game_month").astype(np.int64)[m24]
    PT = tr["pitcher_team_id"].to_numpy(np.int64)[m24]
    CS = dec(g("cur_succ"))
    CB = dec(g("cur_ball"))
    GR = {
        "cur_succ 십분위 x 월 (미검증)": CS * 100 + MON,
        "cur_ball 십분위 x 월 (미검증)": CB * 100 + MON,
        "cur_succ 십분위 x 투수팀 (미검증)": CS * 100 + PT,
        "투수 x 타자 매치업": PID[m24] * 100000 + BID[m24],
        "투수 x 월": PID[m24] * 100 + MON,
        "투수 x 점수차": PID[m24] * 10
        + np.clip(g("score_diff_pitcher_team").astype(np.int64)[m24], -3, 3) + 3,
        "투수 (참조)": PID[m24],
        "cur_succ 십분위 (참조)": CS,
    }
    print("\n" + "=" * 88)
    print("1. 방향 정합 — 그 관계의 보정이 경기 잠재축 보정과 같은 곳을 미는가")
    print("=" * 88)
    print(f"{'관계':<32}{'셀':>9}{'이득':>9}{'정합도':>9}{'설명 분산':>10}")
    out = {}
    for n, k in GR.items():
        gg, aa = corr_fit(k, p24, y24, half)
        c = float(np.corrcoef(aa, add_g)[0, 1]) if aa.std() > 0 else 0.0
        out[n] = dict(gain=gg, align=c, cells=int(len(np.unique(k))))
        print(f"  {n:<30}{len(np.unique(k)):>9,}{gg:>+9.1f}{c:>+9.3f}"
              f"{c**2:>10.3f}")
    print("\n  정합도 = corr(그 관계의 보정, 경기 잠재축 보정). "
          "1 이면 같은 축, 0 이면 무관하다.")

    print("\n" + "=" * 88)
    print("2. 미검증 두 관계의 시즌 간 전이 (2022+2023 -> 2024)")
    print("=" * 88)
    CSall = np.searchsorted(np.nanquantile(g("cur_succ")[season < 2025],
                                           np.linspace(0, 1, 11)[1:-1]),
                            np.nan_to_num(g("cur_succ"), nan=0.5))
    CBall = np.searchsorted(np.nanquantile(g("cur_ball")[season < 2025],
                                           np.linspace(0, 1, 11)[1:-1]),
                            np.nan_to_num(g("cur_ball"), nan=0.3))
    MONall = g("game_month").astype(np.int64)
    PTall = tr["pitcher_team_id"].to_numpy(np.int64)
    CAND = {"cur_succ x 월": CSall * 100 + MONall,
            "cur_ball x 월": CBall * 100 + MONall,
            "cur_succ x 투수팀": CSall * 100 + PTall,
            "투수 x 타자 매치업": PID * 100000 + BID}
    msrc = np.isin(season, (2022, 2023))
    rs = np.concatenate([res[2022], res[2023]])
    print(f"{'관계':<24}{'최적 k':>8}{'2024 이득':>11}{'k=0':>9}")
    for n, kk in CAND.items():
        best, bk, v0 = -1e9, None, None
        for k in KS + [50000]:
            u, tab, _ = cells(kk[msrc], rs, k)
            v = r2(p24 + look(u, tab, kk[m24]), y24) - base
            if k == 0:
                v0 = v
            if v > best:
                best, bk = v, k
        out[f"transfer|{n}"] = dict(gain=best, k=bk)
        print(f"  {n:<22}{bk:>8}{best:>+11.1f}{v0:>+9.1f}")

    json.dump(out, io.open(os.path.join(ROOT, "exp", "latent_proxy.json"), "w",
                           encoding="utf-8"), indent=1, ensure_ascii=False)
    print("\n판정 — 정합도가 낮으면 그 관계는 잠재축의 그림자가 아니다. "
          "전이 이득이 +3.8%(약 +36점) 미만이면 승격하지 않는다.")


if __name__ == "__main__":
    main()
