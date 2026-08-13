import os

import pandas as pd

D = r"C:\Users\GACHON\Desktop\open\data"
R = r"C:\Users\GACHON\Desktop\open"

mp = pd.read_csv(os.path.join(R, "pitcher_id_map.csv"))
tr = pd.read_csv(os.path.join(D, "train.csv"), encoding="utf-8-sig",
                 usecols=["pitcher_id", "pitcher_hand", "season"])
tm = pd.read_csv(os.path.join(D, "trackman_history.csv"), encoding="utf-8-sig",
                 usecols=["pitcher_trackman_id", "pitcher_hand",
                          "pitcher_team", "season"])
tm1 = tm[~(tm.pitcher_team.str.startswith(("MIN_", "KBO_"))
           | tm.pitcher_team.eq("ACE_MEX"))]

th = tr.groupby("pitcher_id").pitcher_hand.first()
mh = tm1.groupby("pitcher_trackman_id").pitcher_hand.first()
a, b = mp.pitcher_id.map(th), mp.pitcher_trackman_id.map(mh)
m = a.notna() & b.notna()
print(f"매핑 {int(m.sum())}쌍에서 투수 손 일치율")
for nm, mm in [("1=Right / 2=Left", {1: "Right", 2: "Left"}),
               ("1=Left  / 2=Right", {1: "Left", 2: "Right"})]:
    print(f"  {nm:<20} {100 * (a[m].map(mm) == b[m]).mean():6.2f}%")

print("\ntrain  pitcher_hand 비율")
print((tr.pitcher_hand.value_counts(normalize=True) * 100).round(1).to_string())
print("tm(1군) pitcher_hand 비율")
print((tm1.pitcher_hand.value_counts(normalize=True) * 100).round(1).to_string())
print(f"\ntrackman 시즌 범위 {int(tm.season.min())}~{int(tm.season.max())}"
      f"   test 시즌 2025")
