r"""4-9 가 통한 패턴을 trackman 의 **나머지 물리량**으로 확장할 수 있는지 검산.

## 왜 이 방향인가

README 4-7 은 "Trackman 은 세 갈래 모두 닫혔다"고 적었지만, 그 뒤 4-9 가
**네 번째 갈래**로 점수를 냈다 — 상황 조건부 편차. 그런데 4-9 가 실제로 쓴
trackman 컬럼은 `pitch_type_group` 과 `rel_speed` **둘뿐**이다. 파일에는
물리량이 7개 더 있다.

    spin_rate  induced_vert_break  horz_break  extension  rel_height
    rel_side   zone_speed

같은 패턴(그 투수 전체 대비 이 상황에서의 편차)을 이것들에 적용하면 4-9 와
같은 성질을 갖는다 — 행마다 변하고, train.csv 에 없고, 투수당 상수가 아니다.

## 4-7 과 무엇이 다른가

4-7 은 **시즌 간 변화량**이었고 실패 원인은 "한 시즌 300구로 추정한 산포의
표준오차가 신호를 덮었다"였다. 여기서는 시즌을 가로지르지 않고 전 기간을 묶어
(투수, 상황) 셀을 만든다 — 셀당 표본이 수백~수천이라 4-9 와 같은 안정성을 갖는다.

## 검산 방법

연속량이므로 이항 잡음이 아니라 **셀 평균의 표준오차**와 비교한다.

    관측분산 = Var_투수( 셀평균 - 그 투수 전체평균 - 리그 상황효과 )
    잡음     = mean( 셀내 분산 / 셀 표본수 )

신호/잡음이 1 을 크게 넘어야 새 정보다. 4-9 채택 피처는 2.9~19.0 이었다.

    .\.venv\Scripts\python.exe exp\tmx_probe.py
"""
import numpy as np
import pandas as pd

DATA = "./data"
MAP = "pitcher_id_map.csv"
MIN_CONF = 0.9
MIN_COUNT_CELL = 30
MIN_HAND_CELL = 50

QUANT = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break",
         "extension", "rel_height", "rel_side", "zone_speed"]


def load():
    id_map = pd.read_csv(MAP)
    id_map = id_map[id_map["conf"] >= MIN_CONF]
    tm = pd.read_csv(f"{DATA}/trackman_history.csv", encoding="utf-8-sig",
                     usecols=["season", "pitcher_trackman_id", "balls_before",
                              "strikes_before", "batter_hand"] + QUANT)
    tm["pitcher_id"] = tm["pitcher_trackman_id"].map(
        id_map.set_index("pitcher_trackman_id")["pitcher_id"])
    return tm.dropna(subset=["pitcher_id"])


def decompose(df, cell_cols, q, label, min_cell):
    d = df[["pitcher_id", q] + cell_cols].dropna()
    if not len(d):
        print(f"  {label:38s} 값 없음")
        return np.nan

    s_p = d.groupby("pitcher_id")[q].mean().rename("s_p")
    g = d.groupby(["pitcher_id"] + cell_cols)[q]
    cell = pd.DataFrame({"mean": g.mean(), "size": g.size(), "var": g.var()})
    cell = cell[cell["size"] >= min_cell]
    if len(cell) < 50:
        print(f"  {label:38s} 셀 부족 ({len(cell)})")
        return np.nan

    league_c = d.groupby(cell_cols)[q].mean().rename("league_c")
    j = cell.join(s_p, on="pitcher_id").join(league_c, on=cell_cols)
    # 그 투수 전체 대비 조정폭에서, 리그가 이미 설명하는 몫을 뺀 나머지
    j["resid"] = (j["mean"] - j["s_p"]) - (j["league_c"] - d[q].mean())

    noise = (j["var"] / j["size"]).mean()          # 셀 평균의 표준오차 제곱
    obs = j["resid"].var()
    real = max(0.0, obs - noise)
    ratio = real / noise if noise else np.nan
    # 투수 간 전체 산포로 정규화해 "이 편차가 실무적으로 큰가"도 같이 본다
    scale = s_p.std()
    print(f"  {label:38s} 셀 {len(j):>6,}  실분산 {real:10.5f}  "
          f"신호/잡음 {ratio:7.2f}  편차SD/투수간SD {np.sqrt(real)/scale:6.3f}")
    return ratio


def main():
    tm = load()
    print(f"대응된 trackman {len(tm):,}건, 투수 {tm['pitcher_id'].nunique()}명")
    print(f"결측률: " + "  ".join(f"{q}={tm[q].isna().mean():.1%}" for q in QUANT))

    print(f"\n=== 투수 x 볼카운트(12칸), 셀 {MIN_COUNT_CELL}구 이상 ===")
    print("  (4-9 채택 피처 = 구종비율/구속. 신호잡음 2.9~19.0 이 기준선)\n")
    rc = {q: decompose(tm, ["balls_before", "strikes_before"], q,
                       f"{q} x 볼카운트", MIN_COUNT_CELL) for q in QUANT}

    print(f"\n=== 투수 x 타자좌우(2칸), 셀 {MIN_HAND_CELL}구 이상 ===\n")
    rh = {q: decompose(tm, ["batter_hand"], q, f"{q} x 타자좌우", q and
                       MIN_HAND_CELL) for q in QUANT}

    print("\n=== 요약: 신호/잡음 ===")
    print(f"{'물리량':22s} {'x카운트':>10s} {'x좌우':>10s}")
    for q in QUANT:
        print(f"{q:22s} {rc[q]:10.2f} {rh[q]:10.2f}")
    print("\n  1 을 크게 넘는 것만 학습으로 넘긴다. 다만 rel_speed 는 이미")
    print("  채택돼 있으므로(tmc_speed_dev/tmh_speed_dev) 그 값이 이 검산의")
    print("  '통과 기준선' 역할을 한다 — 그보다 낮으면 추가 가치가 의심스럽다.")


if __name__ == "__main__":
    main()
