r"""데이터 생성·문제 정의의 기본 가정 감사. 학습 0회, 새 피처 0개.

새 상호작용을 만들지 않는다. **우리가 맞다고 가정해 온 것들을 원자료로 직접
확인**한다. 확인 항목은 여섯 개다.

    Q1  F/R 은 왜 생성관계가 다른가 — 야구 메커니즘인가 집계/라벨링인가
    Q2  game_type 외에 같은 종류의 "생성 체제" 구분자가 더 있는가
    Q3  target/row/asof 가 만들어지는 **순서**가 우리 가정과 일치하는가
    Q4  train/test 에 같은 생성 규칙이 적용되는가
    Q5  관측 불가능한 고해상도 상태의 합법 대리가 있는가
    Q6  "47열을 다 썼다" 와 "47열의 생성 의미를 다 이해했다" 를 분리

가장 값진 검사는 Q3 의 **연속성**이다. `asof_pitcher_n` 이 "직전까지 누적 투구
수"라면 한 투수의 행들은 `asof_n` 이 1씩 증가하는 **연속 정수**여야 한다.
빠진 정수가 있다면 그 투수의 투구 중 **train 에 행으로 없는 것**이 있다는 뜻이고,
그것은 우리가 `cur_n` 을 시즌내 순번으로 읽어 온 근거를 흔든다.

    .\.venv\Scripts\python.exe -u exp\gen_audit.py
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

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def sec(t):
    print("\n" + "=" * 86)
    print(t)
    print("=" * 86, flush=True)


def main():
    tr = build_df()
    season = tr["season"].to_numpy()
    y = tr["control_success"].to_numpy(np.float64)
    gt = tr["game_type"].to_numpy()
    PID = tr["pitcher_id"].to_numpy(np.int64)
    an = tr["asof_pitcher_n"].to_numpy(np.float64)
    out = {}

    sec("Q3-a. asof_pitcher_n 은 연속 정수인가 (투수-시즌 단위)")
    d = pd.DataFrame({"p": PID, "s": season, "n": an, "gt": gt, "y": y})
    gsz = d.groupby(["p", "s"])["n"].agg(["size", "min", "max", "nunique"])
    gsz["span"] = gsz["max"] - gsz["min"] + 1
    gsz["gap"] = gsz["span"] - gsz["size"]
    dup = int((gsz["nunique"] != gsz["size"]).sum())
    print(f"  투수-시즌 조합 {len(gsz):,}개")
    print(f"  asof_n 중복이 있는 조합 {dup:,}개 ({dup/len(gsz):.1%})")
    print(f"  빈틈(span-size)이 0 인 조합 {(gsz['gap'] == 0).mean():.1%}")
    print(f"  빈틈 중앙값 {gsz['gap'].median():.0f}   평균 {gsz['gap'].mean():.1f}"
          f"   최대 {gsz['gap'].max():.0f}")
    print(f"  빈틈 비율(빈틈/span) 중앙값 {(gsz['gap']/gsz['span']).median():.3f}")
    out["Q3_gap_zero_share"] = float((gsz["gap"] == 0).mean())

    sec("Q3-b. 빈틈은 어디서 오는가 — F/R 을 섞으면 메워지는가")
    for lab, m in (("전체(F+R)", np.ones(len(d), bool)),
                   ("R 행만", gt == "R"), ("F 행만", gt == "F")):
        dd = d[m]
        g2 = dd.groupby(["p", "s"])["n"].agg(["size", "min", "max"])
        g2["gap"] = g2["max"] - g2["min"] + 1 - g2["size"]
        print(f"  {lab:<10} 조합 {len(g2):>7,}   빈틈0 비율 {(g2['gap']==0).mean():>6.1%}"
              f"   빈틈 중앙값 {g2['gap'].median():>8.0f}")
    both = d.groupby(["p", "s"])["gt"].nunique()
    print(f"  한 시즌에 F 와 R 을 모두 던진 투수-시즌 {int((both > 1).sum()):,}"
          f" / {len(both):,} ({(both > 1).mean():.1%})")

    sec("Q3-c. 시즌 경계에서 이어지는가 (통산 누적 가정의 핵심)")
    mx = d.groupby(["p", "s"])["n"].max().rename("max_n").reset_index()
    mn = d.groupby(["p", "s"])["n"].min().rename("min_n").reset_index()
    j = mx.merge(mn, on=["p", "s"])
    j["s_next"] = j["s"] + 1
    k = j.merge(j, left_on=["p", "s_next"], right_on=["p", "s"],
                suffixes=("", "_n"))
    d_boundary = k["min_n_n"] - k["max_n"]
    print(f"  연속 시즌쌍 {len(k):,}개")
    print(f"  다음 시즌 첫 asof_n - 이번 시즌 마지막 asof_n:")
    print(f"    = 1 인 비율 {(d_boundary == 1).mean():.1%}"
          f"   > 1 인 비율 {(d_boundary > 1).mean():.1%}"
          f"   <= 0 인 비율 {(d_boundary <= 0).mean():.1%}")
    print(f"    중앙값 {d_boundary.median():.0f}   90분위 {d_boundary.quantile(.9):.0f}")
    out["Q3_boundary_continuous"] = float((d_boundary == 1).mean())

    sec("Q1. F/R — 야구 메커니즘인가 집계/라벨링인가")
    tm = np.isfinite(tr["tmc_speed_dev"].to_numpy(np.float64))
    r = []
    for s in sorted(set(season)):
        m = season == s
        for t in ("R", "F"):
            mm = m & (gt == t)
            r.append(dict(season=s, type=t, rows=int(mm.sum()),
                          y=float(y[mm].mean()),
                          tm_cover=float(tm[mm].mean()),
                          pitchers=int(len(set(PID[mm])))))
    R = pd.DataFrame(r).pivot(index="season", columns="type")
    print(f"{'시즌':<6}{'R 성공률':>10}{'F 성공률':>10}{'F-R':>9}"
          f"{'R TM커버':>10}{'F TM커버':>10}{'F 행비율':>10}")
    for s in sorted(set(season)):
        yr, yf = R[("y", "R")][s], R[("y", "F")][s]
        cr, cf = R[("tm_cover", "R")][s], R[("tm_cover", "F")][s]
        nr, nf = R[("rows", "R")][s], R[("rows", "F")][s]
        print(f"{s:<6}{yr:>10.4f}{yf:>10.4f}{yf-yr:>+9.4f}"
              f"{cr:>10.1%}{cf:>10.1%}{nf/(nr+nf):>10.1%}")
    out["Q1_by_season"] = R.to_dict()

    sec("Q1-b. 2023 이동이 커버리지 때문인가 — 커버리지로 층화")
    print(f"{'시즌':<6}{'F·TM있음':>11}{'F·TM없음':>11}{'R·TM있음':>11}{'R·TM없음':>11}")
    for s in sorted(set(season)):
        m = season == s
        v = []
        for t in ("F", "R"):
            for c in (True, False):
                mm = m & (gt == t) & (tm == c)
                v.append(y[mm].mean() if mm.sum() > 500 else np.nan)
        print(f"{s:<6}" + "".join(f"{x:>11.4f}" for x in v))

    sec("Q2. 같은 종류의 '생성 체제' 구분자가 더 있는가")
    print("  판정 = 그 그룹의 성공률이 전체 추세와 **다르게** 급변한 해가 있는가")
    print(f"{'구분자':<22}{'최대 |초과변화|':>16}{'해당 시즌':>12}{'셀수':>7}")
    cands = {"game_type": gt,
             "top_bottom": tr["top_bottom"].to_numpy(),
             "투수팀": tr["pitcher_team_id"].to_numpy(),
             "타자팀": tr["batter_team_id"].to_numpy(),
             "game_month": tr["game_month"].to_numpy(),
             "요일": tr["game_dayofweek"].to_numpy(),
             "이닝(1-3/4-6/7+)": np.digitize(tr["inning"].to_numpy(), [4, 7]),
             "투수손": tr["pitcher_hand"].to_numpy(),
             "타자손": tr["batter_hand"].to_numpy(),
             "TM커버": tm.astype(int)}
    gy = pd.Series(y).groupby(season).mean()
    for name, v in cands.items():
        t = pd.DataFrame({"s": season, "g": v, "y": y}).groupby(["s", "g"])["y"].agg(
            ["mean", "size"])
        t = t[t["size"] >= 3000]["mean"].unstack()
        excess = t.sub(gy, axis=0)                 # 전체 추세 제거
        dif = excess.diff().abs()
        if dif.size == 0 or dif.isna().all().all():
            continue
        mv = np.nanmax(dif.to_numpy())
        wh = dif.stack().idxmax()
        print(f"{name:<22}{mv:>16.4f}{str(wh[0]):>12}{t.shape[1]:>7}")
        out[f"Q2_{name}"] = dict(max_excess_shift=float(mv), season=int(wh[0]))

    sec("Q6. 47열의 생성 의미 — 파생 컬럼은 정말 파생인가")
    g = lambda c: tr[c].to_numpy(np.float64)
    print(f"  run_total == run_top + run_bot            "
          f"{np.mean(g('run_total_before') == g('run_top_before') + g('run_bot_before')):.4%}")
    print(f"  score_diff_home == run_bot - run_top      "
          f"{np.mean(g('score_diff_home') == g('run_bot_before') - g('run_top_before')):.4%}")
    print(f"  num_runners == 1b+2b+3b                   "
          f"{np.mean(g('num_runners_on') == g('runner_on_1b') + g('runner_on_2b') + g('runner_on_3b')):.4%}")
    bs = (tr["runner_on_1b"].astype(int).astype(str)
          + tr["runner_on_2b"].astype(int).astype(str)
          + tr["runner_on_3b"].astype(int).astype(str)).to_numpy()
    bs2 = pd.Series(tr["base_state"].to_numpy()).map(
        lambda s: "".join("1" if ch not in "_" else "0" for ch in s)).to_numpy()
    print(f"  base_state == 주자 3플래그                  {np.mean(bs == bs2):.4%}")
    print(f"  home_we + away_we == 100                  "
          f"{np.mean(np.abs(g('home_win_expectancy') + g('away_win_expectancy') - 100) < 1e-6):.4%}")
    sdp = np.where(tr["top_bottom"].to_numpy() == "T",
                   g("run_bot_before") - g("run_top_before"),
                   g("run_top_before") - g("run_bot_before"))
    print(f"  score_diff_pitcher_team == 수비팀 기준 차   "
          f"{np.mean(sdp == g('score_diff_pitcher_team')):.4%}")

    sec("Q6-b. li 와 승률기대는 경기상태만의 함수인가 (숨은 정보 검사)")
    st = (tr["inning"].astype(str) + "|" + tr["top_bottom"].astype(str) + "|"
          + tr["outs_before"].astype(str) + "|" + tr["base_state"].astype(str)
          + "|" + tr["score_diff_home"].astype(str)).to_numpy()
    for col in ("li", "home_win_expectancy"):
        v = g(col)
        t = pd.DataFrame({"k": st, "v": v}).groupby("k")["v"].agg(
            ["nunique", "size", "std"])
        big = t[t["size"] >= 30]
        print(f"  {col:<20} 상태셀 {len(t):,}   30행 이상 셀 {len(big):,}"
              f"   그 안에서 값이 유일한 셀 {(big['nunique'] == 1).mean():.1%}"
              f"   셀내 표준편차 중앙값 {big['std'].median():.4f}")
        out[f"Q6_{col}_unique_share"] = float((big["nunique"] == 1).mean())

    json.dump(out, io.open(os.path.join(ROOT, "exp", "gen_audit.json"), "w",
                           encoding="utf-8"), indent=1, ensure_ascii=False,
              default=str)


if __name__ == "__main__":
    main()
