r"""TrackMan 조인 감사 2단계 — 합성키 유일성과 ID 매핑 식별가능성.

ID 매핑이 있다고 **가정해도** 조인이 되는지 먼저 본다. 안 되면 매핑 조사는 무의미하다.

    .\.venv\Scripts\python.exe exp\tm_audit2.py
"""
import os

import numpy as np
import pandas as pd

D = r"C:\Users\GACHON\Desktop\open\data"


def uniq_report(df, keys, name):
    g = df.groupby(keys, observed=True).size()
    n = len(df)
    u = int((g == 1).sum())
    tot_u = u
    print(f"  {name:<52} 키 {len(g):>9,}  "
          f"유일행 {100 * tot_u / n:5.1f}%  최대충돌 {g.max():>5,}")
    return g


def main():
    tr = pd.read_csv(os.path.join(D, "train.csv"), encoding="utf-8-sig",
                     usecols=["season", "inning", "top_bottom", "balls_before",
                              "strikes_before", "outs_before", "pitcher_id",
                              "batter_id", "pitcher_hand", "batter_hand",
                              "pitcher_team_id", "batter_team_id", "game_type"])
    tm = pd.read_csv(os.path.join(D, "trackman_history.csv"),
                     encoding="utf-8-sig",
                     usecols=["season", "game_date", "trackman_game_id",
                              "pitch_no", "inning", "top_bottom",
                              "balls_before", "strikes_before", "outs_before",
                              "pitch_of_pa", "pitcher_trackman_id",
                              "batter_trackman_id", "pitcher_hand",
                              "batter_hand", "pitcher_team", "batter_team"])

    # 1군만
    bad = (tm.pitcher_team.str.startswith(("MIN_", "KBO_"))
           | tm.batter_team.str.startswith(("MIN_", "KBO_"))
           | tm.pitcher_team.eq("ACE_MEX") | tm.batter_team.eq("ACE_MEX"))
    tm1 = tm[~bad].copy()
    tm1["tb"] = np.where(tm1.top_bottom.eq("Top"), "T", "B")
    print(f"train {len(tr):,}   trackman(1군) {len(tm1):,}   "
          f"차 {len(tm1) - len(tr):+,}\n")

    print("=== A. 합성키가 TRACKMAN 안에서 유일한가 (ID 매핑이 완벽하다고 가정) ===")
    base = ["season", "inning", "tb", "balls_before", "strikes_before",
            "outs_before"]
    uniq_report(tm1, base, "season+inning+TB+B+S+O")
    uniq_report(tm1, base + ["pitcher_trackman_id"], "  + 투수")
    uniq_report(tm1, base + ["pitcher_trackman_id", "batter_trackman_id"],
                "  + 투수 + 타자")
    uniq_report(tm1, base + ["pitcher_trackman_id", "batter_trackman_id",
                             "pitcher_team", "batter_team"],
                "  + 투수 + 타자 + 양팀")

    print("\n=== B. 같은 합성키가 TRAIN 안에서 유일한가 ===")
    baset = ["season", "inning", "top_bottom", "balls_before",
             "strikes_before", "outs_before"]
    uniq_report(tr, baset, "season+inning+TB+B+S+O")
    uniq_report(tr, baset + ["pitcher_id"], "  + 투수")
    uniq_report(tr, baset + ["pitcher_id", "batter_id"], "  + 투수 + 타자")
    uniq_report(tr, baset + ["pitcher_id", "batter_id", "pitcher_team_id",
                             "batter_team_id"], "  + 투수 + 타자 + 양팀")

    print("\n=== C. 날짜를 안다고 가정하면? (train 에는 없는 정보) ===")
    uniq_report(tm1, ["game_date"] + base + ["pitcher_trackman_id",
                                             "batter_trackman_id"],
                "game_date + 위 전체")
    uniq_report(tm1, ["trackman_game_id"] + base + ["pitcher_trackman_id",
                                                    "batter_trackman_id"],
                "trackman_game_id + 위 전체")
    uniq_report(tm1, ["trackman_game_id", "pitch_no"], "게임ID + 투구번호")

    print("\n=== D. ID 매핑이 식별 가능한가 — 투수 프로파일 ===")
    # train: (시즌, 팀) 별 등장수 / tm: 같은 형태. 손은 표기만 다름.
    hand_map = {1: "Right", 2: "Left"}
    tp = (tr.assign(h=tr.pitcher_hand.map(hand_map))
            .groupby("pitcher_id")
            .agg(n=("season", "size"), hand=("h", "first"),
                 seasons=("season", lambda s: tuple(sorted(s.unique()))),
                 teams=("pitcher_team_id", lambda s: len(s.unique()))))
    mp = (tm1.groupby("pitcher_trackman_id")
             .agg(n=("season", "size"), hand=("pitcher_hand", "first"),
                  seasons=("season", lambda s: tuple(sorted(s.unique()))),
                  teams=("pitcher_team", lambda s: len(s.unique()))))
    print(f"  train 투수 {len(tp):,}명   tm(1군) 투수 {len(mp):,}명   "
          f"차 {len(mp) - len(tp):+,}")

    # 손 + 시즌집합 만으로 후보가 몇 명인가
    tp["key"] = list(zip(tp.hand, tp.seasons))
    mp["key"] = list(zip(mp.hand, mp.seasons))
    cnt = mp.groupby("key").size()
    cand = tp["key"].map(cnt).fillna(0).astype(int)
    print(f"\n  (손 + 시즌집합) 으로 후보를 좁히면 train 투수당 tm 후보 수:")
    print(f"    후보 0명 {int((cand == 0).sum()):>4}명   "
          f"1명 {int((cand == 1).sum()):>4}명   "
          f"2~5명 {int(((cand >= 2) & (cand <= 5)).sum()):>4}명   "
          f"6명+ {int((cand > 5).sum()):>4}명")
    print(f"    -> 유일 식별 {100 * (cand == 1).mean():.1f}%")

    print(f"\n  등장수(n) 로 더 좁힐 수 있는가 — train n 과 tm n 의 관계")
    print(f"    train 투수 총투구 합 {tp.n.sum():,}   "
          f"tm(1군) 합 {mp.n.sum():,}   비율 {mp.n.sum() / tp.n.sum():.3f}")
    print(f"    train n 중앙 {tp.n.median():.0f}  tm n 중앙 {mp.n.median():.0f}")
    print("    비율이 1.0 이 아니고 선수마다 다르면 n 은 결정적 키가 못 된다")

    print("\n=== E. game_type 축 — train 에만 있다 ===")
    print(tr.game_type.value_counts().to_string())
    print("    tm 에는 정규/포스트 구분 컬럼이 없다 -> 필터를 맞출 수 없다")


if __name__ == "__main__":
    main()
