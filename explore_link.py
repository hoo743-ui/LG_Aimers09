r"""train/test 의 pitcher_id 와 trackman_history 의 pitcher_trackman_id 가
서로 연결 가능한 ID 체계인지 확인한다.

붙일 수 있으면 Trackman 기반 투수 요약 피처(구속/회전/무브먼트/구종믹스)를
만들 수 있고, 아니면 그 경로는 포기해야 한다.

사용법:
    .\.venv\Scripts\python.exe explore_link.py
"""
import pandas as pd

DATA = "./data"


def summarize(name, s):
    print(f"  {name:26s} n={s.nunique():6d}  min={s.min():8.0f}  max={s.max():8.0f}")


print("== 1. ID 집합 비교 ==")
train_ids = pd.read_csv(f"{DATA}/train.csv", encoding="utf-8-sig",
                        usecols=["pitcher_id", "batter_id", "season"])
tm_ids = pd.read_csv(f"{DATA}/trackman_history.csv", encoding="utf-8-sig",
                     usecols=["pitcher_trackman_id", "batter_trackman_id", "season"])

summarize("train.pitcher_id", train_ids["pitcher_id"])
summarize("trackman.pitcher_id", tm_ids["pitcher_trackman_id"])
summarize("train.batter_id", train_ids["batter_id"])
summarize("trackman.batter_id", tm_ids["batter_trackman_id"])

p_main = set(train_ids["pitcher_id"].unique())
p_tm = set(tm_ids["pitcher_trackman_id"].dropna().unique())
b_main = set(train_ids["batter_id"].unique())
b_tm = set(tm_ids["batter_trackman_id"].dropna().unique())

print("\n== 2. 교집합 ==")
print(f"  투수: main {len(p_main)} / trackman {len(p_tm)} / 교집합 {len(p_main & p_tm)}"
      f"  ({len(p_main & p_tm) / len(p_main):.1%} of main)")
print(f"  타자: main {len(b_main)} / trackman {len(b_tm)} / 교집합 {len(b_main & b_tm)}"
      f"  ({len(b_main & b_tm) / len(b_main):.1%} of main)")

print("\n== 3. 시즌별 행 수 ==")
print("  train    :", train_ids["season"].value_counts().sort_index().to_dict())
print("  trackman :", tm_ids["season"].value_counts().sort_index().to_dict())

print("\n== 4. test.csv 의 pitcher_id 가 train 에 있는지 ==")
test = pd.read_csv(f"{DATA}/test.csv", encoding="utf-8-sig",
                   usecols=["pitcher_id", "batter_id"])
print(f"  test 투수 {test['pitcher_id'].nunique()}명 중 train 에 있는 수: "
      f"{test['pitcher_id'].isin(p_main).sum()} / {len(test)} 행")
print("  (test 는 형식확인용 5건뿐이라 참고만)")
