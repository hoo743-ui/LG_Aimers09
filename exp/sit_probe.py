r"""상황별 **성공률** 편차가 새 정보인지 값싸게 검산한다. 학습 0회.

## 착상

4-9 가 채택한 것은 trackman 의 **구종 성향**을 상황별로 준 것이다
(`이 투수가 0-2 에서 변화구를 얼마나 던지는가`). 그런데 메인 데이터에는
**결과 자체**가 있다 — `control_success`. 그러니 같은 구조로

    dev = P(성공 | 투수, 상황) - P(성공 | 투수)

를 만들 수 있다. 뒤쪽 항은 `asof_pitcher_success_rate` 로 이미 모델에 있으므로
편차가 곧 순수 증분이다 (4-9 와 같은 논리).

## 왜 트리가 이미 알고 있지 않은가

4-8 은 "게이트가 이미 피처면 하드 MoE 는 이득이 없다"고 했다. 하지만 그건
**분할**의 이야기다. 여기서 주는 것은 `(투수 592명) x (카운트 12칸)` 짜리 셀
평균이고, `min_samples_leaf=1000` / `max_leaf_nodes=10` 인 트리는 그 깊이의
교호작용을 원리적으로 만들 수 없다. 트리가 못 만드는 것을 표로 건네는 것이다.

## 위험 — 이건 타깃에서 나온 값이다

`tmc_*` 는 구종 비율이라 타깃과 무관했지만 이건 **타깃 평균**이다. 누설을
막는 장치가 셋 필요하다.

  1. **시점 규칙** — season S 행에는 S 미만 시즌 데이터만. 4-9 와 동일
  2. **편차 형태** — 투수 전체 수준을 빼서 `asof_pitcher_success_rate` 와의
     중복을 없앤다
  3. **연도 누설 검사** — 만든 뒤 `era_screen.py` 를 반드시 통과시킨다

## 물어야 할 질문

"카운트에 따라 성공률이 달라지는가"가 아니다 (그건 리그 경향이고 카운트는 이미
피처다). **그 조정폭이 투수마다 다른가**이다. 그래서 리그 칸 평균을 뺀 잔차의
투수 간 분산을 이항 잡음과 비교한다 — `ctx_probe.py` 와 같은 잣대다.

    .\.venv\Scripts\python.exe exp\sit_probe.py
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "exp", "cache")
MIN_CELL = 30


def load():
    X = np.load(f"{CACHE}/X.npy", mmap_mode="r")
    meta = json.load(open(f"{CACHE}/cols.json"))
    ix = {c: i for i, c in enumerate(meta["cols"])}
    need = ["pitcher_id", "batter_id", "balls_before", "strikes_before",
            "outs_before", "base_state", "batter_hand", "inning",
            "num_runners_on"]
    d = {c: np.asarray(X[:, ix[c]]) for c in need}
    d["season"] = np.load(f"{CACHE}/season.npy")
    d["y"] = np.load(f"{CACHE}/y.npy").astype(np.float64)
    return pd.DataFrame(d)


def decompose(df, who, cell_cols, label):
    """s_pc - s_p 의 개인 간 분산을 표집 잡음과 비교한다 (ctx_probe 와 동일)."""
    s_p = df.groupby(who)["y"].mean().rename("s_p")
    g = df.groupby([who] + cell_cols)["y"]
    s_pc = pd.DataFrame({"mean": g.mean(), "size": g.size()})
    s_pc = s_pc[s_pc["size"] >= MIN_CELL]
    if not len(s_pc):
        print(f"  {label:34s} 표본 부족")
        return
    league_c = df.groupby(cell_cols)["y"].mean().rename("league_c")

    j = s_pc.join(s_p, on=who).join(league_c, on=cell_cols)
    j["adj"] = j["mean"] - j["s_p"]
    # 리그가 이미 설명하는 조정폭을 뺀 나머지 = 개인차
    j["resid"] = j["adj"] - (j["league_c"] - df["y"].mean())

    noise = (j["mean"] * (1 - j["mean"]) / j["size"]).mean()
    obs = j["resid"].var()
    real = max(0.0, obs - noise)
    print(f"  {label:34s} 셀 {len(j):>7,}  관측 {obs:.5f}  잡음 {noise:.5f}  "
          f"실 {real:.5f}  신호/잡음 {real / noise if noise else np.nan:6.2f}")
    return real / noise if noise else np.nan


def coverage(df, who, cell_cols, label):
    for Y in (2022, 2024):
        past = df[df["season"] < Y]
        cells = past.groupby([who] + cell_cols).size()
        keep = set(cells[cells >= MIN_CELL].index)
        va = df[df["season"] == Y]
        key = list(zip(*[va[c] for c in [who] + cell_cols]))
        cov = np.mean([k in keep for k in key])
        print(f"    fold {Y} 커버리지 {cov:6.1%}   ({label})")


def main():
    df = load()
    print(f"행 {len(df):,}  투수 {df['pitcher_id'].nunique()}명  "
          f"타자 {df['batter_id'].nunique()}명  전체 성공률 {df['y'].mean():.4f}\n")

    print("=== 리그 경향 (이건 카운트 피처만으로 트리가 이미 안다) ===")
    piv = df.pivot_table(index="balls_before", columns="strikes_before",
                         values="y", aggfunc="mean")
    print(piv.round(4).to_string())

    print("\n=== 개인차 검산: 상황 조정폭이 사람마다 다른가 ===")
    print("  (신호/잡음 이 1 을 크게 넘어야 새 정보. 4-9 채택 피처는 2.9~19.0 이었다)\n")
    res = {}
    res["p×count"] = decompose(df, "pitcher_id",
                               ["balls_before", "strikes_before"],
                               "투수 x 볼카운트(12칸)")
    res["p×strikes"] = decompose(df, "pitcher_id", ["strikes_before"],
                                 "투수 x 스트라이크수(3칸)")
    res["p×balls"] = decompose(df, "pitcher_id", ["balls_before"],
                               "투수 x 볼수(4칸)")
    res["p×hand"] = decompose(df, "pitcher_id", ["batter_hand"],
                              "투수 x 타자좌우(2칸)")
    res["p×base"] = decompose(df, "pitcher_id", ["base_state"],
                              "투수 x 주자상황(8칸)")
    res["p×outs"] = decompose(df, "pitcher_id", ["outs_before"],
                              "투수 x 아웃카운트(3칸)")
    print()
    res["b×count"] = decompose(df, "batter_id",
                               ["balls_before", "strikes_before"],
                               "타자 x 볼카운트(12칸)")
    res["b×hand"] = decompose(df, "batter_id", ["pitcher_hand"]
                              if "pitcher_hand" in df else ["batter_hand"],
                              "타자 x 투수좌우(2칸)")

    print("\n=== 커버리지 (시점 규칙 적용: 그 시즌 미만 데이터로 만든 셀) ===")
    coverage(df, "pitcher_id", ["balls_before", "strikes_before"], "투수x카운트")
    coverage(df, "pitcher_id", ["batter_hand"], "투수x타자좌우")
    coverage(df, "batter_id", ["balls_before", "strikes_before"], "타자x카운트")

    print("\n판단: 신호/잡음이 1 을 크게 넘고 커버리지가 60% 이상인 조합만")
    print("      실제 학습으로 넘긴다. 4-7 은 전제를 통과하고도 실패했으니")
    print("      이 검산은 가설을 죽일 수는 있어도 살리지는 못한다 (일하는 방식 ①).")


if __name__ == "__main__":
    main()
