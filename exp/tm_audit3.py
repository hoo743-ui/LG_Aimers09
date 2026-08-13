r"""TrackMan 조인 감사 3단계 — 기존 pitcher_id_map 독립 검증 + 행 단위 상한.

README 4-4 는 경기 단위 매칭에 성공해 pitcher_id_map.csv 를 만들었다. 기록을 믿지
않고 **매핑을 만들 때 쓰지 않은 신호**로 검증한다. 그리고 이 매핑을 실제로 써서
행 단위 조인의 상한을 낸다.

    .\.venv\Scripts\python.exe exp\tm_audit3.py
"""
import os

import numpy as np
import pandas as pd

D = r"C:\Users\GACHON\Desktop\open\data"
R = r"C:\Users\GACHON\Desktop\open"


def main():
    mp = pd.read_csv(os.path.join(R, "pitcher_id_map.csv"))
    print(f"=== pitcher_id_map.csv — {len(mp):,}행 ===")
    print(f"  train 투수 792명 중 {len(mp)}명 매핑 ({100 * len(mp) / 792:.1f}%)")
    print(f"  conf: 중앙 {mp.conf.median():.3f}  최소 {mp.conf.min():.3f}  "
          f"1.0 인 건수 {(mp.conf == 1.0).sum()}")
    print(f"  중복 배정 — train_id 중복 {mp.pitcher_id.duplicated().sum()}, "
          f"tm_id 중복 {mp.pitcher_trackman_id.duplicated().sum()}")

    tr = pd.read_csv(os.path.join(D, "train.csv"), encoding="utf-8-sig",
                     usecols=["season", "inning", "top_bottom", "balls_before",
                              "strikes_before", "outs_before", "pitcher_id",
                              "batter_id", "pitcher_hand", "batter_hand",
                              "pitcher_team_id"])
    tm = pd.read_csv(os.path.join(D, "trackman_history.csv"),
                     encoding="utf-8-sig",
                     usecols=["season", "inning", "top_bottom", "balls_before",
                              "strikes_before", "outs_before",
                              "pitcher_trackman_id", "batter_trackman_id",
                              "pitcher_hand", "batter_hand", "pitcher_team"])
    bad = (tm.pitcher_team.str.startswith(("MIN_", "KBO_"))
           | tm.pitcher_team.eq("ACE_MEX"))
    tm1 = tm[~bad].copy()
    tm1["tb"] = np.where(tm1.top_bottom.eq("Top"), "T", "B")

    # ---------- 독립 검증 1: 손 일치 ----------
    hmap = {1: "Right", 2: "Left"}
    th = tr.groupby("pitcher_id").pitcher_hand.first().map(hmap)
    mh = tm1.groupby("pitcher_trackman_id").pitcher_hand.first()
    j = mp.assign(train_hand=mp.pitcher_id.map(th),
                  tm_hand=mp.pitcher_trackman_id.map(mh)).dropna()
    agree = (j.train_hand == j.tm_hand).mean()
    print(f"\n=== 독립 검증 1) 투수 손 일치 ===")
    print(f"  매핑 {len(j)}쌍 중 손 일치 {100 * agree:.2f}%  "
          f"(무작위 매칭이면 ~62% 기대: 우완 74.9%^2+좌완 25.1%^2)")

    # ---------- 독립 검증 2: 시즌 등장 집합 일치 ----------
    ts = tr.groupby("pitcher_id").season.apply(lambda s: frozenset(s.unique()))
    ms = tm1.groupby("pitcher_trackman_id").season.apply(
        lambda s: frozenset(s.unique()))
    a = mp.pitcher_id.map(ts)
    b = mp.pitcher_trackman_id.map(ms)
    ok = [(x is not None and y is not None
           and len(x & y) / max(len(x | y), 1)) for x, y in zip(a, b)]
    ok = [v for v in ok if v is not False]
    print(f"\n=== 독립 검증 2) 활동 시즌 집합 자카드 유사도 ===")
    print(f"  중앙 {np.median(ok):.3f}   1.0(완전일치) 비율 "
          f"{100 * np.mean([v == 1.0 for v in ok]):.1f}%   "
          f"0.5 미만 {100 * np.mean([v < 0.5 for v in ok]):.1f}%")

    # ---------- 독립 검증 3: 투구 수 상관 ----------
    tn = tr.groupby("pitcher_id").size()
    mn = tm1.groupby("pitcher_trackman_id").size()
    x, y = mp.pitcher_id.map(tn), mp.pitcher_trackman_id.map(mn)
    m = x.notna() & y.notna()
    print(f"\n=== 독립 검증 3) 투구 수 상관 ===")
    print(f"  피어슨 {np.corrcoef(x[m], y[m])[0, 1]:.4f}  "
          f"스피어만 {pd.Series(x[m]).corr(pd.Series(y[m]), method='spearman'):.4f}")
    print(f"  비율 tm/train 중앙 {np.median(y[m] / x[m]):.3f}")

    # ---------- 타자 매핑이 존재하는가 ----------
    print(f"\n=== 🚩 타자 매핑 ===")
    fs = [f for f in os.listdir(R) if f.endswith(".csv") and "map" in f.lower()]
    print(f"  루트의 map csv: {fs}")
    print(f"  batter_id -> batter_trackman_id 대응표: "
          f"{'있음' if any('batter' in f for f in fs) else '**없음**'}")

    # ---------- 행 단위 조인 상한 ----------
    print(f"\n=== 행 단위 조인 상한 (기존 매핑을 실제로 적용) ===")
    d = dict(zip(mp.pitcher_id, mp.pitcher_trackman_id))
    tr["ptm"] = tr.pitcher_id.map(d)
    cov = tr.ptm.notna().mean()
    print(f"  매핑되는 train 행 비율            {100 * cov:5.1f}%")

    key = ["season", "inning", "tb", "balls_before", "strikes_before",
           "outs_before", "pitcher_trackman_id"]
    keyt = ["season", "inning", "top_bottom", "balls_before",
            "strikes_before", "outs_before", "ptm"]
    gm = tm1.groupby(key, observed=True).size()
    sub = tr[tr.ptm.notna()].copy()
    gt = sub.groupby(keyt, observed=True).size()

    idx_tm = set(map(tuple, gm.reset_index()[key].values))
    kt = pd.MultiIndex.from_frame(sub[keyt])
    n_tm = pd.Series(kt.map(gm.to_dict()), index=sub.index)
    hit = n_tm.notna()
    exact = hit & (n_tm == 1)
    print(f"  그중 tm 에 키가 존재            {100 * hit.mean():5.1f}%")
    print(f"  그중 tm 에서 유일               {100 * exact.mean():5.1f}%")
    print(f"  -> 전체 train 대비 유일 대응     "
          f"{100 * exact.sum() / len(tr):5.1f}%   (목표 95%)")
    print(f"  나머지는 다대다. 최대 충돌 {int(n_tm[hit].max())}건")
    del idx_tm

    print(f"\n  타자를 넣을 수 있다면? (타자 매핑이 없어 계산만)")
    print(f"    tm 에서 투수+타자까지 넣으면 유일행 84.0% (감사 2 A)")
    print(f"    그래도 95% 미만이고, 타자 대응표 자체가 없다")


if __name__ == "__main__":
    main()
