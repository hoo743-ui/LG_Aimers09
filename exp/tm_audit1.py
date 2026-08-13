r"""TrackMan 조인 감사 1단계 — 기본 사실. 추정 없이 파일에서만 읽는다.

    .\.venv\Scripts\python.exe exp\tm_audit1.py
"""
import os

import pandas as pd

D = r"C:\Users\GACHON\Desktop\open\data"

TR_COLS = ["row_id", "season", "game_month", "game_dayofweek", "inning",
           "top_bottom", "game_type", "balls_before", "strikes_before",
           "outs_before", "pitcher_id", "batter_id", "pitcher_hand",
           "batter_hand", "pitcher_team_id", "batter_team_id"]
TM_COLS = ["season", "game_date", "game_month", "game_dayofweek",
           "trackman_game_id", "pitch_no", "inning", "top_bottom",
           "balls_before", "strikes_before", "outs_before", "pitch_of_pa",
           "pitcher_trackman_id", "batter_trackman_id", "pitcher_hand",
           "batter_hand", "pitcher_team", "batter_team"]


def main():
    tr = pd.read_csv(os.path.join(D, "train.csv"), usecols=TR_COLS,
                     encoding="utf-8-sig")
    tm = pd.read_csv(os.path.join(D, "trackman_history.csv"), usecols=TM_COLS,
                     encoding="utf-8-sig")
    te = pd.read_csv(os.path.join(D, "test.csv"), encoding="utf-8-sig")

    print(f"TRAIN ROWS    = {len(tr):,}")
    print(f"TRACKMAN ROWS = {len(tm):,}")
    print(f"TEST ROWS     = {len(te):,}  (로컬 파일)")

    print("\n=== 시즌 분포 ===")
    a = tr.season.value_counts().sort_index()
    b = tm.season.value_counts().sort_index()
    print(f"{'season':>8}{'train':>12}{'trackman':>12}")
    for s in sorted(set(a.index) | set(b.index)):
        print(f"{s:>8}{a.get(s, 0):>12,}{b.get(s, 0):>12,}")
    print(f"{'test':>8}{'':>12}  ->  " + str(sorted(te.season.unique())))

    print("\n=== 표현형 (조인 전에 맞춰야 하는 것들) ===")
    for nm, s in [("train.top_bottom", tr.top_bottom),
                  ("tm.top_bottom", tm.top_bottom),
                  ("train.pitcher_hand", tr.pitcher_hand),
                  ("tm.pitcher_hand", tm.pitcher_hand),
                  ("train.batter_hand", tr.batter_hand),
                  ("tm.batter_hand", tm.batter_hand)]:
        print(f"  {nm:<20} {sorted(s.dropna().unique().tolist())[:8]}")
    print(f"  {'train.pitcher_team_id':<20} "
          f"{sorted(tr.pitcher_team_id.unique().tolist())}")
    print(f"  {'tm.pitcher_team':<20} "
          f"{sorted(tm.pitcher_team.unique().tolist())[:30]}")
    print(f"  {'train.game_type':<20} {sorted(tr.game_type.unique().tolist())}")
    print(f"  tm.game_type         <없음>")

    print("\n=== ID 공간 ===")
    for nm, s in [("train.pitcher_id", tr.pitcher_id),
                  ("tm.pitcher_trackman_id", tm.pitcher_trackman_id),
                  ("train.batter_id", tr.batter_id),
                  ("tm.batter_trackman_id", tm.batter_trackman_id)]:
        print(f"  {nm:<24} 고유 {s.nunique():>6,}  "
              f"범위 {s.min():>8,} ~ {s.max():>9,}")
    ov_p = len(set(tr.pitcher_id.unique()) & set(tm.pitcher_trackman_id.unique()))
    ov_b = len(set(tr.batter_id.unique()) & set(tm.batter_trackman_id.unique()))
    print(f"  값 자체의 교집합: 투수 {ov_p}개, 타자 {ov_b}개")

    print("\n=== 시간 정보 ===")
    print(f"  train 에 있는 시간 컬럼: "
          f"{[c for c in tr.columns if 'game' in c or 'season' in c]}")
    print(f"  tm    에 있는 시간 컬럼: "
          f"{[c for c in tm.columns if 'game' in c or 'season' in c]}")
    print(f"  train 에 game_date 있는가: {'game_date' in tr.columns}")
    print(f"  train 에 game_id  있는가: "
          f"{any('game_id' in c for c in tr.columns)}")

    print("\n=== 🚩 TrackMan 팀 구성 — train 은 13팀, tm 은 26코드 ===")
    vc = tm.pitcher_team.value_counts()
    minor = [t for t in vc.index if t.startswith("MIN_")]
    other = [t for t in vc.index if not t.startswith("MIN_")
             and not t.startswith("KBO_") and t != "ACE_MEX"]
    spec = [t for t in vc.index if t.startswith("KBO_") or t == "ACE_MEX"]
    print(f"  1군으로 보이는 코드 {len(other)}개  행 {vc[other].sum():>10,}")
    print(f"  MIN_* (2군)   {len(minor)}개  행 {vc[minor].sum():>10,}")
    print(f"  KBO_*/ACE_*   {len(spec)}개  행 {vc[spec].sum():>10,}")
    print(f"  train 전체                     {len(tr):>10,}")
    m1 = ~(tm.pitcher_team.str.startswith("MIN_")
           | tm.pitcher_team.str.startswith("KBO_")
           | (tm.pitcher_team == "ACE_MEX")
           | tm.batter_team.str.startswith("MIN_")
           | tm.batter_team.str.startswith("KBO_")
           | (tm.batter_team == "ACE_MEX"))
    print(f"  양팀 다 1군인 행               {m1.sum():>10,}  "
          f"(train 대비 {100 * m1.sum() / len(tr):.1f}%)")
    print("\n  시즌별 — 1군만 남긴 tm vs train")
    c1 = tm[m1].season.value_counts().sort_index()
    for s in sorted(a.index):
        print(f"    {s}  train {a[s]:>9,}   tm(1군) {c1.get(s, 0):>9,}   "
              f"차 {c1.get(s, 0) - a[s]:>+9,}")

    # 날짜 -> (월, 요일) 로 얼마나 뭉개지는가
    d = pd.to_datetime(tm.game_date, format="mixed")
    tm["_dd"] = d
    per = tm.groupby(["season", "game_month", "game_dayofweek"])["_dd"].nunique()
    print(f"\n  (season, game_month, game_dayofweek) 조합 {len(per):,}개")
    print(f"  그 안에 들어가는 서로 다른 날짜 수: 중앙 {per.median():.0f}  "
          f"최대 {per.max()}  합 {per.sum():,}")
    print(f"  -> train 의 시간 해상도는 '월 x 요일' 이고 실제 날짜는 "
          f"평균 {per.mean():.1f} 개가 겹친다")

    ng = tm.groupby(["season", "game_month", "game_dayofweek"])[
        "trackman_game_id"].nunique()
    print(f"  같은 조합 안의 서로 다른 경기 수: 중앙 {ng.median():.0f}  "
          f"최대 {ng.max()}")


if __name__ == "__main__":
    main()
