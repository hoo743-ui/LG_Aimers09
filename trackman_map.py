r"""train 과 trackman_history 를 경기 단위로 맞춰 선수 ID 대응표를 역산한다.

4-3 은 "ID 체계가 달라 조인 불가"로 결론냈지만, 확인한 건 ID 직접 비교뿐이었다.
양쪽에 공통 키가 많고 trackman 에는 경기 식별자와 정확한 날짜까지 있다.

    train      season, game_month, game_dayofweek, 양팀 id, inning,
               top_bottom, balls/strikes/outs_before, 좌우
    trackman   trackman_game_id, game_date, pitch_no, 같은 상황 컬럼들

전략은 세 단계다.
  1. 경기 맞추기 — (시즌, 월, 요일) 안에서 '한 팀이 던진 투구 수'를 지문으로 쓴다.
     경기마다 값이 제각각이라(중앙값 155) 후보가 크게 좁혀진다.
  2. 투구 정렬 — 맞은 경기 안에서 (이닝, 초말, 아웃, 볼, 스트라이크) 순서로 맞춘다.
  3. 대응표 — 정렬된 투구의 pitcher_id 와 pitcher_trackman_id 를 다수결로 묶는다.

규칙 확인: trackman 은 2019~2024 만 있고 평가는 2025 다. 따라서 투구 단위 물리량은
평가에 못 쓴다. 대응표로 만들 수 있는 건 **투수별 과거 요약값**이고, 규칙이
"과거 투구 특성, 구종 특성, 투수 단위 요약값 등 추가 피처를 만들 수 있다"고
명시적으로 허용하는 용도다.

이 스크립트는 1단계 실현 가능성만 잰다. 매칭률이 낮으면 여기서 접는다.

사용법:
    .\.venv\Scripts\python.exe trackman_map.py --season 2024
"""
import argparse

import pandas as pd

DATA_DIR = "./data"
TARGET = "control_success"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, default=2024)
    return p.parse_args()


def train_games(season):
    """train 을 '한 경기에서 한 팀이 던진 투구' 단위로 자른다.

    game_id 가 없으므로 (시즌, 월, 요일, 양팀)을 키로 쓰고, row_id 가 시간순인
    점을 이용해 이닝이 되감기는 지점에서 경기를 가른다 (4-7 말미와 같은 방법).
    """
    df = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                     usecols=["row_id", "season", "game_month", "game_dayofweek",
                              "pitcher_team_id", "batter_team_id", "inning",
                              "top_bottom", "balls_before", "strikes_before",
                              "outs_before", "pitcher_id", TARGET])
    df = df[df["season"] == season].reset_index(drop=True)
    key = ["game_month", "game_dayofweek", "pitcher_team_id", "batter_team_id"]
    new = df.groupby(key, sort=False)["inning"].diff().fillna(0) < 0
    df["seg"] = new.groupby([df[c] for c in key], sort=False).cumsum()
    df["gkey"] = list(zip(*[df[c] for c in key + ["seg"]]))
    return df


def trackman_games(season):
    """trackman 을 같은 단위로 자른다 — (경기, 투수팀)."""
    df = pd.read_csv(f"{DATA_DIR}/trackman_history.csv", encoding="utf-8-sig",
                     usecols=["season", "game_date", "trackman_game_id",
                              "pitch_no", "inning", "top_bottom",
                              "balls_before", "strikes_before", "outs_before",
                              "pitcher_trackman_id", "pitcher_team",
                              "batter_team"])
    df = df[df["season"] == season].copy()
    d = pd.to_datetime(df["game_date"], format="mixed")
    df["game_month"] = d.dt.month
    # train 의 game_dayofweek 규약을 모르므로 두 가지를 다 만들어 뒤에서 고른다
    df["dow0"] = d.dt.dayofweek          # 월=0
    df["dow1"] = d.dt.dayofweek + 1      # 월=1
    df["gkey"] = list(zip(df["trackman_game_id"], df["pitcher_team"]))
    return df


def main():
    args = parse_args()
    tr = train_games(args.season)
    tm = trackman_games(args.season)

    a = tr.groupby("gkey").size()
    b = tm.groupby("gkey").size()
    print(f"=== {args.season} 시즌 ===")
    print(f"train    투구 {len(tr):,}  경기·팀 단위 {len(a):,}  "
          f"투구수 중앙값 {a.median():.0f}")
    print(f"trackman 투구 {len(tm):,}  경기·팀 단위 {len(b):,}  "
          f"투구수 중앙값 {b.median():.0f}")

    # 요일 규약 확인 — 월/요일 조합별 경기 수가 맞아떨어지는 쪽을 고른다
    print("\n=== 요일 규약 맞추기 ===")
    ta = tr.groupby("gkey")[["game_month", "game_dayofweek"]].first()
    cnt_a = ta.groupby(["game_month", "game_dayofweek"]).size()
    for col in ("dow0", "dow1"):
        tb = tm.groupby("gkey")[["game_month", col]].first()
        cnt_b = tb.groupby(["game_month", col]).size()
        joined = cnt_a.to_frame("train").join(
            cnt_b.rename_axis(["game_month", "game_dayofweek"]).to_frame("tm"),
            how="outer").fillna(0)
        diff = (joined["train"] - joined["tm"]).abs().sum()
        print(f"  {col}: 조합 {len(joined)}개, 경기수 절대차 합계 {diff:.0f}")

    # 투구 수 지문이 얼마나 유일한지 — 이게 낮으면 경기 매칭이 어렵다
    print("\n=== 투구 수 지문의 변별력 (train) ===")
    sig = ta.join(a.rename("n"))
    grp = sig.groupby(["game_month", "game_dayofweek", "n"]).size()
    print(f"  (월, 요일, 투구수) 조합 {len(grp):,}개")
    print(f"  그중 유일한 것 {(grp == 1).sum():,}개 ({(grp == 1).mean():.1%})")
    print(f"  최대 충돌 {grp.max()}개")


if __name__ == "__main__":
    main()
