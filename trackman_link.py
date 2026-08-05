r"""train 과 trackman_history 를 연결해 선수 ID 대응표를 만든다.

4-3 은 "ID 체계가 달라 조인 불가"로 결론냈지만 확인한 건 ID 직접 비교뿐이었다.
양쪽에 공통 상황 컬럼이 많고 trackman 에는 경기 식별자와 날짜가 있으므로,
경기를 맞추면 선수 대응은 따라 나온다.

부트스트랩으로 푼다 — 팀 코드 대응표를 모르는 채로 시작한다.

  1. (월, 요일, 투구수) 가 양쪽에서 유일한 경기부터 맞춘다 (train 의 78.8%)
  2. 그 매칭에서 팀 코드 ↔ 팀 id 대응을 다수결로 역산한다
  3. 팀 대응표로 나머지 모호한 경기를 확정한다
  4. 맞은 경기 안에서 투수를 등판 순서·투구 수로 짝지어 표를 쌓는다

투구를 하나씩 정렬하지 않는 이유는 견고성이다. train 이 경기의 모든 투구를
담고 있지 않아도 투구 수 순위와 등판 순서는 유지된다.

규칙: trackman 은 2019~2024 뿐이고 평가는 2025 다. 따라서 투구 단위 물리량은
평가에 쓸 수 없고, 만들 수 있는 건 투수별 과거 요약값이다. 규칙이 명시적으로
허용하는 용도다 ("과거 투구 특성, 구종 특성, 투수 단위 요약값").

사용법:
    .\.venv\Scripts\python.exe trackman_link.py --seasons 2024
    .\.venv\Scripts\python.exe trackman_link.py --seasons 2019,2020,2021,2022,2023,2024
"""
import argparse
from collections import Counter, defaultdict

import pandas as pd

DATA_DIR = "./data"
OUT = "pitcher_id_map.csv"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", default="2024")
    p.add_argument("--out", default=OUT)
    return p.parse_args()


def load_train(seasons):
    df = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                     usecols=["season", "game_month", "game_dayofweek",
                              "pitcher_team_id", "batter_team_id", "inning",
                              "pitcher_id"])
    df = df[df["season"].isin(seasons)].reset_index(drop=True)
    key = ["season", "game_month", "game_dayofweek",
           "pitcher_team_id", "batter_team_id"]
    # row_id 순 = 시간 순. 같은 키 안에서 이닝이 되감기면 다른 경기다.
    new = df.groupby(key, sort=False)["inning"].diff().fillna(0) < 0
    df["seg"] = new.groupby([df[c] for c in key], sort=False).cumsum()
    df["unit"] = list(zip(*[df[c] for c in key + ["seg"]]))
    return df


def load_trackman(seasons):
    df = pd.read_csv(f"{DATA_DIR}/trackman_history.csv", encoding="utf-8-sig",
                     usecols=["season", "game_date", "trackman_game_id",
                              "pitch_no", "pitcher_trackman_id",
                              "pitcher_team", "batter_team"])
    df = df[df["season"].isin(seasons)].copy()
    d = pd.to_datetime(df["game_date"], format="mixed")
    df["game_month"] = d.dt.month
    df["game_dayofweek"] = d.dt.dayofweek        # 월=0 (trackman_map.py 에서 확인)
    df = df.sort_values(["trackman_game_id", "pitch_no"], kind="stable")
    df["unit"] = list(zip(df["trackman_game_id"], df["pitcher_team"]))
    return df


def unit_table(df, team_col, other_col):
    """경기·투수팀 단위 요약 — 투구 수와 시즌/월/요일, 양팀."""
    g = df.groupby("unit", sort=False)
    return pd.DataFrame({
        "n": g.size(),
        "season": g["season"].first(),
        "month": g["game_month"].first(),
        "dow": g["game_dayofweek"].first(),
        "pteam": g[team_col].first(),
        "bteam": g[other_col].first(),
    })


def pitcher_order(df, id_col):
    """단위별 [(투수, 투구수)] — 등판 순서 유지."""
    out = defaultdict(list)
    for unit, sub in df.groupby("unit", sort=False):
        seen = {}
        for pid in sub[id_col]:
            seen[pid] = seen.get(pid, 0) + 1
        out[unit] = list(seen.items())      # dict 는 삽입 순서 = 등판 순서
    return out


def main():
    args = parse_args()
    seasons = [int(s) for s in args.seasons.split(",")]

    tr = load_train(seasons)
    tm = load_trackman(seasons)
    A = unit_table(tr, "pitcher_team_id", "batter_team_id")
    B = unit_table(tm, "pitcher_team", "batter_team")
    print(f"단위: train {len(A):,}  trackman {len(B):,}")

    # --- 1단계: (시즌, 월, 요일, 투구수) 가 양쪽에서 유일한 것 ---
    sig = ["season", "month", "dow", "n"]
    a1 = A.groupby(sig).filter(lambda g: len(g) == 1)
    b1 = B.groupby(sig).filter(lambda g: len(g) == 1)
    m1 = a1.reset_index().merge(b1.reset_index(), on=sig,
                                suffixes=("_a", "_b"))
    print(f"1단계 유일 매칭: {len(m1):,} 단위 "
          f"(train 의 {len(m1)/len(A):.1%})")

    # --- 2단계: 팀 대응표 역산 ---
    votes = defaultdict(Counter)
    for r in m1.itertuples():
        votes[r.pteam_a][r.pteam_b] += 1
        votes[r.bteam_a][r.bteam_b] += 1
    team_map, conf = {}, []
    for tid, c in votes.items():
        code, k = c.most_common(1)[0]
        team_map[tid] = code
        conf.append(k / sum(c.values()))
    print(f"팀 대응 {len(team_map)}개, 다수결 신뢰도 "
          f"최소 {min(conf):.1%} 평균 {sum(conf)/len(conf):.1%}")
    for tid in sorted(team_map):
        print(f"    team_id {tid:>3} -> {team_map[tid]}")

    # --- 3단계: 팀 대응표로 전체 매칭 ---
    A2 = A.copy()
    A2["pteam"] = A2["pteam"].map(team_map)
    A2["bteam"] = A2["bteam"].map(team_map)
    full = ["season", "month", "dow", "pteam", "bteam", "n"]
    a2 = A2.dropna(subset=["pteam", "bteam"]).groupby(full).filter(
        lambda g: len(g) == 1)
    b2 = B.groupby(full).filter(lambda g: len(g) == 1)
    m2 = a2.reset_index().merge(b2.reset_index(), on=full,
                                suffixes=("_a", "_b"))
    print(f"\n3단계 매칭: {len(m2):,} 단위 (train 의 {len(m2)/len(A):.1%})")

    # --- 4단계: 투수 대응 ---
    pa = pitcher_order(tr, "pitcher_id")
    pb = pitcher_order(tm, "pitcher_trackman_id")
    pv = defaultdict(Counter)
    used = 0
    for r in m2.itertuples():
        la, lb = pa[r.unit_a], pb[r.unit_b]
        if len(la) != len(lb):
            continue                     # 투수 수가 다르면 건너뛴다
        used += 1
        for (ia, na), (ib, nb) in zip(la, lb):
            if na == nb:                 # 투구 수까지 같을 때만 표를 준다
                pv[ia][ib] += 1
    print(f"투수 대응에 쓴 단위 {used:,} / {len(m2):,}")

    rows = []
    for pid, c in pv.items():
        tid, k = c.most_common(1)[0]
        rows.append({"pitcher_id": pid, "pitcher_trackman_id": tid,
                     "votes": k, "total": sum(c.values()),
                     "conf": k / sum(c.values())})
    mp = pd.DataFrame(rows).sort_values("total", ascending=False)
    print(f"\n대응된 투수 {len(mp)}명 / train 투수 {tr['pitcher_id'].nunique()}명")
    if len(mp):
        print(f"신뢰도 90% 이상 {(mp['conf'] >= 0.9).sum()}명, "
              f"중앙값 {mp['conf'].median():.1%}")
        # 1:1 인지 확인 — 한 trackman id 에 여러 pitcher_id 가 붙으면 문제다
        dup = mp["pitcher_trackman_id"].duplicated().sum()
        print(f"trackman id 중복 배정 {dup}건")
        mp.to_csv(args.out, index=False)
        print(f"저장: {args.out}")
        print(mp.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
