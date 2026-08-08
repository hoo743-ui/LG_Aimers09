r"""상호작용 신호가 실재하는지 싸게 반증한다 — 학습 없이 안정성만 잰다.

배경. `asof_*` 는 전부 주변부 통계다. 투수 통산 성공률, 직전 N경기 성공률,
타자 통산 성공률. **조건부 통계는 하나도 없다.** "이 투수가 2스트라이크에서",
"이 투수가 좌타 상대로" 같은 것 말이다. 게다가 확정 설정은 max_leaf_nodes=10
이라 트리 한 그루가 만들 수 있는 영역이 10개뿐이고, 800명 투수 x 12 카운트의
상호작용을 스스로 합성할 수 없다. 준다면 피처로 미리 계산해 줘야 한다.

그런데 상호작용 항 대부분은 표본 노이즈다. 학습을 돌려보기 전에 이걸 거른다.

방법. 2019~2022 에서 상호작용 편차를 재고, 2023~2024 에서 다시 재서, 둘이
상관되는지 본다.

    dev[p,c] = rate[p,c] - rate[p] - (rate[c] - rate[all])

투수 고유 수준(rate[p])과 리그 전체의 카운트 효과(rate[c])를 뺀 나머지다.
앞의 둘은 이미 asof_* 와 balls/strikes 로 모델에 들어가 있으므로, 여기서
남는 것만이 새 정보다.

상관이 0 이면 그 상호작용은 노이즈이고 그 자리에서 끝난다. 0 보다 유의하게
크면 그 크기가 곧 신호의 상한이다.

노이즈 바닥은 같은 기간을 무작위로 반 갈라 재서 잡는다 — 그쪽이 0 이 아니면
셀 표본이 부족해 측정 자체가 못 미더운 것이다.

    .\.venv\Scripts\python.exe probe_interact.py
"""
import os

import numpy as np
import pandas as pd

DATA = "./data/train.csv"
TARGET = "control_success"
EARLY = [2019, 2020, 2021, 2022]
LATE = [2023, 2024]

# 상호작용 후보. (이름, 조건 컬럼을 만드는 함수, 최소 셀 표본)
# 최소 표본은 셀 하나가 노이즈로 뒤덮이지 않을 만큼 — 성공률 0.5 기준
# 표준오차가 표본 100 에서 0.05, 400 에서 0.025 다.
MIN_N = 150


def load():
    cols = ["season", "pitcher_id", "batter_id", "balls_before", "strikes_before",
            "outs_before", "inning", "pitcher_hand", "batter_hand",
            "num_runners_on", "asof_pitcher_n", "asof_batter_success_rate",
            "pitcher_team_id", "batter_team_id", "top_bottom", "li",
            "score_diff_pitcher_team", "game_month", TARGET]
    df = pd.read_csv(DATA, encoding="utf-8-sig", usecols=cols)
    # 카운트 상태 12종 -> 하나의 코드로. 트리가 balls/strikes 를 각각 나눠서는
    # 12칸을 다 못 만든다.
    df["count_state"] = df["balls_before"] * 3 + df["strikes_before"]
    # 이닝 구간 — 선발의 2~3순회 시점을 거칠게 나눈다
    df["inning_bkt"] = np.clip(df["inning"], 1, 9) // 3
    # 투수 경력 구간 (누적 투구수) — 신인/베테랑
    df["career_bkt"] = pd.cut(df["asof_pitcher_n"].fillna(0),
                              [-1, 500, 2000, 6000, 10**9], labels=False)
    df["runners"] = (df["num_runners_on"] > 0).astype(int)

    # 구장. 데이터에 직접 없지만 복원된다 — 초(T)는 원정팀 공격이므로 홈팀이
    # 던지고 있고, 그 홈팀의 구장이다. 말(B)이면 반대. 스트라이크 판정 편향은
    # 구장마다 실재하고, 잎 10개짜리 트리가 팀 x 초말 2단 분할로 이걸 스스로
    # 만들어낼 가능성은 낮다.
    df["park"] = np.where(df["top_bottom"].eq("T"),
                          df["pitcher_team_id"], df["batter_team_id"])
    # 타자 수준 구간 — "이 투수가 좋은 타자에게 유독 약한가"
    df["bat_tier"] = pd.cut(df["asof_batter_success_rate"],
                            [-0.01, .46, .49, .52, 1.01], labels=False)
    # 상황 압박 — 흔히 말하는 클러치. 대개 노이즈지만 싸게 확인한다
    df["li_bkt"] = pd.cut(df["li"], [-0.01, .7, 1.3, 2.5, 10**9], labels=False)
    df["sdiff_bkt"] = pd.cut(df["score_diff_pitcher_team"],
                             [-10**9, -4, -1, 1, 4, 10**9], labels=False)
    return df


def resid_main(df, key):
    """key 단위의 잔차 평균 — 투수 고유 수준을 뺀 나머지.

    구장처럼 '상호작용'이 아니라 그 자체가 새 축인 후보용이다. 그냥 rate[key]
    를 보면 그 구장 홈팀 투수진의 실력이 절반 섞여 들어오므로, 행마다 그 투수의
    평균을 빼고 남는 것만 센다.
    """
    ru = df.groupby("pitcher_id")[TARGET].mean()
    resid = df[TARGET].to_numpy() - ru.reindex(df["pitcher_id"]).to_numpy()
    g = pd.DataFrame({"k": df[key].to_numpy(), "r": resid}).groupby("k")["r"]
    return g.mean() - resid.mean(), g.size()


def main_stability(df_a, df_b, key, min_n=MIN_N):
    dev_a, n_a = resid_main(df_a, key)
    dev_b, n_b = resid_main(df_b, key)
    common = dev_a.index.intersection(dev_b.index)
    keep = common[(n_a.reindex(common) >= min_n) & (n_b.reindex(common) >= min_n)]
    if len(keep) < 4:
        return np.nan, len(keep), np.nan, np.nan
    a = dev_a.reindex(keep).to_numpy()
    b = dev_b.reindex(keep).to_numpy()
    return (float(np.corrcoef(a, b)[0, 1]), len(keep),
            float(np.cov(a, b, bias=True)[0, 1] / a.var()), float(a.std()))


def interaction_dev(df, unit, cond):
    """dev[unit,cond] = rate[u,c] - rate[u] - (rate[c] - rate[all]) 와 표본 수.

    unit 고유 수준과 리그 전체의 cond 효과를 제거한 나머지. 모델이 이미
    아는 두 축을 빼고 남는 것만 본다.
    """
    g = df.groupby([unit, cond])[TARGET].agg(["mean", "size"])
    ru = df.groupby(unit)[TARGET].mean()
    rc = df.groupby(cond)[TARGET].mean()
    r_all = df[TARGET].mean()

    idx = g.index
    dev = (g["mean"].to_numpy()
           - ru.reindex(idx.get_level_values(0)).to_numpy()
           - (rc.reindex(idx.get_level_values(1)).to_numpy() - r_all))
    return pd.Series(dev, index=idx), g["size"]


def stability(df_a, df_b, unit, cond, min_n=MIN_N):
    """두 기간의 상호작용 편차가 상관되는가. (상관, 셀 수, 회귀 기울기)"""
    dev_a, n_a = interaction_dev(df_a, unit, cond)
    dev_b, n_b = interaction_dev(df_b, unit, cond)

    common = dev_a.index.intersection(dev_b.index)
    keep = common[(n_a.reindex(common) >= min_n) & (n_b.reindex(common) >= min_n)]
    if len(keep) < 30:
        return np.nan, len(keep), np.nan, np.nan

    a = dev_a.reindex(keep).to_numpy()
    b = dev_b.reindex(keep).to_numpy()
    corr = float(np.corrcoef(a, b)[0, 1])
    # 기울기 = 축소계수. 이후 기간에 편차가 얼마나 남아 있는지 (1 이면 그대로)
    slope = float(np.cov(a, b, bias=True)[0, 1] / a.var())
    return corr, len(keep), slope, float(a.std())


def main():
    if not os.path.exists(DATA):
        raise SystemExit(f"{DATA} 없음")

    print("로드 중 ...", flush=True)
    df = load()
    print(f"{len(df):,} 행\n")

    # ---- 0. pitcher_id 가 순서 정보를 갖는가 ----
    # keep-ids 가 +8.5 였는데, ID 는 익명 정수라 트리의 순서 분할이 의미를
    # 가지려면 부여 규칙에 정보가 있어야 한다. 있다면 그 +8.5 는 "정체성"이
    # 아니라 "데뷔 시기" 를 잰 것이고, 진짜 정체성 신호는 아직 미개봉이다.
    first = df.groupby("pitcher_id")["season"].min()
    npitch = df.groupby("pitcher_id").size()
    print("=== 0. pitcher_id 부여 규칙 ===")
    print(f"  투수 {len(first)}명, id 범위 {first.index.min()}~{first.index.max()}")
    print(f"  corr(id, 첫 시즌)     {np.corrcoef(first.index, first.to_numpy())[0,1]:+.3f}")
    print(f"  corr(id, 총 투구수)   {np.corrcoef(npitch.index, npitch.to_numpy())[0,1]:+.3f}")
    print("  첫 시즌별 id 중앙값:",
          {int(s): int(first.index[first == s].to_series().median())
           for s in sorted(first.unique())})

    # ---- 1. 상호작용 편차의 시간 안정성 ----
    early = df[df["season"].isin(EARLY)]
    late = df[df["season"].isin(LATE)]
    # 노이즈 바닥 — 같은 기간을 무작위로 반 갈라 잰다. 시간 요인이 없으므로
    # 여기서 나오는 상관이 측정 가능한 상한이다.
    rng = np.random.default_rng(0)
    half = rng.random(len(early)) < 0.5

    cands = [
        # 1차 탐색에서 살아남은 것들 — 재확인용으로 남겨 둔다
        ("pitcher_id", "batter_hand"),
        ("batter_id", "pitcher_hand"),
        ("pitcher_id", "strikes_before"),
        ("pitcher_id", "count_state"),
        # 1차에서 기각된 것들
        ("pitcher_id", "inning_bkt"),
        ("pitcher_id", "runners"),
        ("batter_id", "count_state"),
        # 2차 후보
        ("pitcher_id", "bat_tier"),        # 좋은 타자에게 유독 약한가
        ("pitcher_id", "batter_team_id"),  # 특정 라인업에 대한 익숙함
        ("pitcher_id", "park"),            # 구장별 적응 (홈/원정과 다르다)
        ("pitcher_id", "li_bkt"),          # 클러치
        ("pitcher_id", "sdiff_bkt"),       # 점수차 (대량 리드/추격)
        ("pitcher_id", "top_bottom"),      # 홈/원정
        ("pitcher_id", "game_month"),      # 시즌 내 시기 (컨디션 곡선)
        ("batter_id", "strikes_before"),   # 타자 접근법
        ("batter_id", "park"),
        ("batter_team_id", "pitcher_hand"),
    ]

    print(f"\n=== 1. 상호작용 편차의 안정성 (셀 최소 {MIN_N} 투구) ===")
    print(f"{'unit x cond':>28} {'셀':>6} {'상관':>7} {'기울기':>7} {'편차sd':>8}"
          f"   | {'같은기간 반분':>13}")
    print("-" * 88)
    for unit, cond in cands:
        c, n, sl, sd = stability(early, late, unit, cond)
        c0, n0, sl0, _ = stability(early[half], early[~half], unit, cond)
        f = lambda v: "  n/a " if np.isnan(v) else f"{v:+.3f}"
        print(f"{unit + ' x ' + cond:>28} {n:6d} {f(c)} {f(sl)} "
              f"{'  n/a ' if np.isnan(sd) else f'{sd:.4f}'}"
              f"   | {f(c0)} ({n0})")

    # ---- 2. 그 자체가 새 축인 후보 (상호작용이 아니라 주효과) ----
    # 투수 고유 수준을 뺀 잔차로 잰다. 구장을 그냥 rate 로 보면 그 구장 홈팀
    # 투수진의 실력이 절반 섞여 들어온다.
    print(f"\n=== 2. 주효과 후보 (투수 수준 제거 후 잔차) ===")
    print(f"{'key':>20} {'셀':>6} {'상관':>7} {'기울기':>7} {'잔차sd':>8}"
          f"   | {'같은기간 반분':>13}")
    print("-" * 80)
    for key in ["park", "batter_team_id", "pitcher_team_id", "game_month"]:
        c, n, sl, sd = main_stability(early, late, key)
        c0, n0, sl0, _ = main_stability(early[half], early[~half], key)
        f = lambda v: "  n/a " if np.isnan(v) else f"{v:+.3f}"
        print(f"{key:>20} {n:6d} {f(c)} {f(sl)} "
              f"{'  n/a ' if np.isnan(sd) else f'{sd:.4f}'}"
              f"   | {f(c0)} ({n0})")

    print("""
읽는 법.
  같은기간 반분 상관 ~ 0   -> 셀 표본이 모자라 측정 불가. 신호 유무를 말할 수 없다
  반분은 크나 시즌간이 0  -> 상호작용은 있으나 해마다 바뀐다. 예측에 못 쓴다
  둘 다 유의하게 > 0      -> 실재하고 지속된다. 기울기 x 편차sd 가 쓸 수 있는 폭
""")


if __name__ == "__main__":
    main()
