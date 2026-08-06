r"""7-1 의 두 가설을 돈 들이기 전에 값싸게 검증한다.

7-1 은 Trackman 대응표를 '정적 프로필'이 아닌 방식으로 쓰자고 한다. 두 갈래인데
둘 다 모델을 돌리기 전에 확인할 수 있는 전제를 깔고 있다.

  가설 A — pitcher_id 를 빼고 프로필로 대체하면 **미지 투수**에게도 물리 정보가
  붙는다. 전제: 미지 투수가 trackman 에는 있다. 그런데 train 에 없는 투수가
  trackman 에는 있을 이유가 딱히 없다. 같은 리그 같은 기간이다. 전제가 깨지면
  가설 A 는 '아는 투수를 더 나쁜 좌표계로 다시 말하기'가 되어 4-4 와 같은 실패다.

  가설 B — 시즌별 프로필의 **변화량**은 정적 상수가 아니므로 pitcher_id 와
  중복되지 않는다. 전제: 변화량을 계산할 표본이 충분하다. 두 시즌 연속 등판한
  투수가 적거나 시즌당 투구 수가 적으면 변화량은 대부분 결측이거나 잡음이다.

여기서는 모델을 학습하지 않는다. 커버리지와 표본 수만 센다.

사용법:
    .\.venv\Scripts\python.exe dyn_probe.py
"""
import numpy as np
import pandas as pd

DATA_DIR = "./data"
MAP = "pitcher_id_map.csv"
MIN_CONF = 0.9
FOLDS = [2021, 2022, 2024]


def main():
    id_map = pd.read_csv(MAP)
    id_map = id_map[id_map["conf"] >= MIN_CONF]
    tm_to_train = id_map.set_index("pitcher_trackman_id")["pitcher_id"]
    print(f"대응표 {len(id_map)}명 (신뢰도 {MIN_CONF} 이상)")

    tm = pd.read_csv(f"{DATA_DIR}/trackman_history.csv", encoding="utf-8-sig",
                     usecols=["season", "pitcher_trackman_id", "rel_speed"])
    train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                        usecols=["season", "pitcher_id"])

    # trackman 투수를 train 의 pitcher_id 공간으로 옮긴다. 대응 안 된 투수는 버린다.
    tm["pitcher_id"] = tm["pitcher_trackman_id"].map(tm_to_train)
    tm_mapped = tm.dropna(subset=["pitcher_id"])
    tm_mapped["pitcher_id"] = tm_mapped["pitcher_id"].astype(np.int64)

    # --- 가설 A: 미지 투수에게 프로필이 붙는가 ---
    print("\n=== 가설 A: 미지 투수 커버리지 ===")
    print("미지 = 그 폴드의 학습 시즌(<Y)에 등장하지 않은 pitcher_id")
    print(f"\n{'fold':>6} {'미지투구비율':>12} {'미지투수수':>10} "
          f"{'프로필보유':>10} {'미지투구중 프로필':>16}")
    for Y in FOLDS:
        seen = set(train.loc[train["season"] < Y, "pitcher_id"].unique())
        va = train[train["season"] == Y]
        unknown_rows = ~va["pitcher_id"].isin(seen)
        unknown_ids = set(va.loc[unknown_rows, "pitcher_id"].unique())

        # 프로필은 Y 이전 시즌 trackman 에서만 만든다 (시점 규칙)
        prof_ids = set(tm_mapped.loc[tm_mapped["season"] < Y, "pitcher_id"].unique())
        covered_ids = unknown_ids & prof_ids
        covered_rows = va.loc[unknown_rows, "pitcher_id"].isin(prof_ids).mean()

        print(f"{Y:>6} {unknown_rows.mean():>11.1%} {len(unknown_ids):>10} "
              f"{len(covered_ids):>10} "
              f"{(covered_rows if len(unknown_ids) else float('nan')):>15.1%}")

    print("\n  → 이 마지막 칸이 0% 에 가까우면 가설 A 의 전제가 깨진 것이다.")
    print("     train 에 없는 투수는 trackman 에도 없다는 뜻이므로,")
    print("     프로필은 '이미 아는 투수'만 덮는다 = 4-4 의 실패 구조 그대로다.")

    # --- 가설 B: 변화량을 잴 표본이 있는가 ---
    print("\n=== 가설 B: 시즌별 변화량 표본 ===")
    per = (tm_mapped.groupby(["pitcher_id", "season"])
           .size().rename("n").reset_index())
    print(f"\n{'fold':>6} {'검증투구':>10} {'직전2시즌 연속':>14} {'양시즌 300구+':>14}")
    for Y in FOLDS:
        va = train[train["season"] == Y]
        a = per[per["season"] == Y - 1].set_index("pitcher_id")["n"]
        b = per[per["season"] == Y - 2].set_index("pitcher_id")["n"]
        both = a.index.intersection(b.index)
        solid = [i for i in both if a[i] >= 300 and b[i] >= 300]

        cov_both = va["pitcher_id"].isin(both).mean()
        cov_solid = va["pitcher_id"].isin(solid).mean()
        print(f"{Y:>6} {len(va):>10,} {cov_both:>13.1%} {cov_solid:>13.1%}")

    print("\n  → 변화량 피처는 이 비율만큼만 값이 있고 나머지는 결측이다.")
    print("     HGB 는 결측을 분기로 처리하니 치명적이진 않지만, 커버리지가 낮으면")
    print("     이득의 상한도 그만큼 낮다.")

    # 변화량이 실제로 움직이는 값인지 — 안 움직이면 정적 상수와 다를 게 없다
    print("\n=== 가설 B 보조: 변화량이 실제로 변하는가 (rel_speed 평균) ===")
    tm_full = pd.read_csv(f"{DATA_DIR}/trackman_history.csv",
                          encoding="utf-8-sig",
                          usecols=["season", "pitcher_trackman_id", "rel_speed",
                                   "pitch_type_group"])
    fb = tm_full[tm_full["pitch_type_group"] == "fastball"].copy()
    fb["pitcher_id"] = fb["pitcher_trackman_id"].map(tm_to_train)
    fb = fb.dropna(subset=["pitcher_id"])
    sp = fb.groupby(["pitcher_id", "season"])["rel_speed"].agg(["mean", "size"])
    sp = sp[sp["size"] >= 300]["mean"].unstack()

    d = sp.diff(axis=1)
    flat = d.to_numpy().ravel()
    flat = flat[~np.isnan(flat)]
    print(f"  연도간 패스트볼 구속 변화: 표본 {len(flat)}건")
    print(f"    평균 {flat.mean():+.3f} km/h   표준편차 {flat.std():.3f}")
    print(f"    |변화| 1km/h 초과 비율 {np.mean(np.abs(flat) > 1):.1%}")
    print("\n  → 표준편차가 투수간 구속 차이(보통 5~8km/h)에 비해 아주 작으면")
    print("     변화량은 사실상 잡음이고, 트리가 쓸 신호가 없다.")


if __name__ == "__main__":
    main()
