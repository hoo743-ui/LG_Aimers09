r"""Trackman 의 **상황 조건부** 정보가 새 정보인지 값싸게 검산한다.

메인 데이터는 투수의 구종 비율을 `asof_pitcher_fastball_rate` 처럼 **전체 평균
하나**로만 준다. trackman 에는 balls_before/strikes_before/batter_hand 가 있어
"이 투수가 0-2 에서 변화구를 얼마나 던지는가"를 만들 수 있다.

4-4/4-7 이 실패한 이유는 프로필이 투수당 상수여서 pitcher_id 와 중복이었던 것
이다. 상황 조건부 값은 **행의 볼카운트에 따라 변하므로** 그 비판을 피한다.
4-8 이 보인 "HGB 가 이미 카운트 상호작용을 잡는다"와도 다르다 — 트리는 train.csv
에 없는 정보를 만들어낼 수 없다.

## 진짜 물어야 할 것

"카운트별로 구종이 달라지는가"는 잘못된 질문이다. 그건 리그 전체 경향이고,
카운트는 이미 피처라 트리가 안다. 새 정보가 되려면 **카운트 조정이 투수마다
달라야** 한다. 즉 투수 A 는 0-2 에서 변화구를 확 늘리고 투수 B 는 안 늘리는,
그 **개인차**가 있어야 한다.

그래서 재는 것은 `s_pc - s_p` 의 투수 간 분산이다.
  s_pc : 투수 p 의 카운트 c 에서의 구종 비율
  s_p  : 투수 p 의 전체 구종 비율  (<- 메인 데이터가 이미 주는 것)

이 편차가 투수마다 제각각이면 새 정보다. 다만 표본이 적으면 **표집 잡음**만으로도
분산이 생기므로, 이항 잡음 기대값과 반드시 비교해야 한다. 4-7 에서 산포의 추정
오차를 신호로 착각한 실수를 반복하지 않는다.

사용법:
    .\.venv\Scripts\python.exe ctx_probe.py
"""
import numpy as np
import pandas as pd

DATA_DIR = "./data"
MAP = "pitcher_id_map.csv"
MIN_CONF = 0.9
MIN_CELL = 30      # 이 미만 표본의 (투수,칸) 은 통계를 못 낸다


def load():
    id_map = pd.read_csv(MAP)
    id_map = id_map[id_map["conf"] >= MIN_CONF]
    tm = pd.read_csv(f"{DATA_DIR}/trackman_history.csv", encoding="utf-8-sig",
                     usecols=["season", "pitcher_trackman_id", "pitch_type_group",
                              "balls_before", "strikes_before", "batter_hand",
                              "rel_speed"])
    tm["pitcher_id"] = tm["pitcher_trackman_id"].map(
        id_map.set_index("pitcher_trackman_id")["pitcher_id"])
    return tm.dropna(subset=["pitcher_id"])


def decompose(df, cell_cols, label, kind="breaking"):
    """s_pc - s_p 의 투수 간 분산을 표집 잡음과 비교한다."""
    df = df.copy()
    df["is_kind"] = (df["pitch_type_group"] == kind).astype(float)

    s_p = df.groupby("pitcher_id")["is_kind"].agg(["mean", "size"])
    s_pc = df.groupby(["pitcher_id"] + cell_cols)["is_kind"].agg(["mean", "size"])
    s_pc = s_pc[s_pc["size"] >= MIN_CELL]
    if not len(s_pc):
        print(f"  {label}: 표본 부족")
        return

    # 리그의 칸별 평균 (이건 카운트만으로 아는 것 = 새 정보 아님)
    league_c = df.groupby(cell_cols)["is_kind"].mean()

    j = s_pc.join(s_p["mean"].rename("s_p"))
    j = j.join(league_c.rename("league_c"), on=cell_cols)
    # 투수 전체 비율 대비 이 칸에서의 조정폭
    j["adj"] = j["mean"] - j["s_p"]
    # 리그가 이미 설명하는 조정폭을 뺀 나머지 = 개인차
    j["resid"] = j["adj"] - (j["league_c"] - df["is_kind"].mean())

    # 표집 잡음 기대: Var = p(1-p)/n
    noise = (j["mean"] * (1 - j["mean"]) / j["size"]).mean()
    obs = j["resid"].var()
    real = max(0.0, obs - noise)

    print(f"  {label:28s} 셀 {len(j):>6,}  관측분산 {obs:.5f}  "
          f"잡음 {noise:.5f}  실분산 {real:.5f}  "
          f"신호/잡음 {real/noise if noise else float('nan'):5.2f}")
    return real, noise


def main():
    tm = load()
    print(f"대응된 trackman 투구 {len(tm):,}건, 투수 "
          f"{tm['pitcher_id'].nunique()}명\n")

    print("=== 리그 전체: 카운트별 구종 비율 (이건 카운트만으로 아는 것) ===")
    mix = (tm.groupby(["balls_before", "strikes_before"])["pitch_type_group"]
           .value_counts(normalize=True).unstack(fill_value=0))
    print(mix.round(3).to_string())
    print("\n  → 폭이 크다면 '카운트가 구종을 정한다'는 뜻이지만, 그건 리그 경향이라")
    print("     카운트 피처만으로 트리가 이미 안다. 새 정보가 아니다.\n")

    print("=== 개인차 검산: s_pc - s_p 의 투수 간 분산 vs 표집 잡음 ===")
    print("  (신호/잡음 이 1 을 크게 넘어야 새 정보다)\n")
    for kind in ("breaking", "fastball", "offspeed"):
        decompose(tm, ["balls_before", "strikes_before"],
                  f"구종 {kind} × 볼카운트", kind)
    print()
    for kind in ("breaking", "fastball"):
        decompose(tm, ["strikes_before"], f"구종 {kind} × 스트라이크수", kind)
    print()
    for kind in ("breaking", "fastball"):
        decompose(tm, ["batter_hand"], f"구종 {kind} × 타자 좌우", kind)

    # 커버리지 — 메인 데이터의 투구 중 몇 %가 (투수,카운트) 셀을 갖는가
    print("\n=== 커버리지 ===")
    train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                        usecols=["season", "pitcher_id", "balls_before",
                                 "strikes_before"])
    for Y in (2022, 2024):
        past = tm[tm["season"] < Y]
        cells = (past.groupby(["pitcher_id", "balls_before", "strikes_before"])
                 .size())
        cells = set(cells[cells >= MIN_CELL].index)
        va = train[train["season"] == Y]
        key = list(zip(va["pitcher_id"], va["balls_before"],
                       va["strikes_before"]))
        cov = np.mean([k in cells for k in key])
        print(f"  fold {Y}: 검증 투구의 {cov:.1%} 가 (투수,카운트) 셀을 갖는다 "
              f"(셀 {MIN_CELL}구 이상)")


if __name__ == "__main__":
    main()
