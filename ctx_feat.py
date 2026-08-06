r"""Trackman 의 **상황 조건부 구종 성향**을 피처로 붙여 이득을 잰다.

ctx_probe.py 가 전제를 확인했다 — 카운트별/좌우별 구종 조정은 투수마다 다르고
(신호/잡음 2.8~19.0), 커버리지도 77% 다.

## 왜 이건 4-4/4-7 과 다른가

4-4(정적 프로필)와 4-7(변화량)이 실패한 자리를 정확히 피한다.

  - 4-4 비판은 "투수당 상수라 pitcher_id 와 중복". 여기 값은 **행의 볼카운트와
    타자 좌우에 따라 변한다.** 같은 투수도 0-0 과 3-0 에서 다른 값을 받는다.
  - 4-8 이 보인 "HGB 가 카운트 상호작용을 이미 잡는다"와도 다르다. 트리는
    train.csv 에 **없는** 정보를 만들어낼 수 없고, 카운트별 구종 성향은 메인
    데이터에 아예 없다 (`asof_pitcher_*_rate` 는 투수별 전체 평균 하나뿐이다).

## 편차로 준다

조건부 비율을 그대로 주면 그 투수의 전체 비율과 강하게 겹친다. 그리고 트리는
`A - B` 를 축 수직 분할로 아주 비효율적으로 근사한다(4-5). 그래서 **새 정보만
분리해서** 넣는다.

    dev = (이 카운트에서의 변화구 비율) - (그 투수 전체 변화구 비율)

전체 비율은 이미 `asof_pitcher_breaking_rate` 로 모델에 있으므로, dev 가 곧
순수한 증분 정보다.

## 시점 규칙

season S 의 행에는 **S 미만 시즌의 trackman** 만 쓴다. 평가(2025)는 2019~2024
전량을 쓰고, 학습 행도 같은 규칙이라 의미가 일치한다.

행 독립도 지킨다 — 값은 (투수, 상황) 단위이고 평가셋의 다른 행을 보지 않는다.

사용법:
    .\.venv\Scripts\python.exe ctx_feat.py --variants base,hand,count,both --seeds 42
"""
import argparse
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

DATA_DIR = "./data"
ID = "row_id"
TARGET = "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state"]
LR, LEAVES, MIN_LEAF, L2, N_ITER = 0.02, 10, 1000, 1.0, 1400
VAL_SEASON = 2024

# 메인 데이터의 손 인코딩. 비율 대조로 확정했다 (투수 74:26 이 결정적).
HAND = {2: "Right", 1: "Left"}

MIN_COUNT_CELL = 30
MIN_HAND_CELL = 50

KINDS = ["breaking", "fastball", "offspeed"]
COUNT_FEATS = [f"tmc_{k}_dev" for k in KINDS] + ["tmc_speed_dev", "tmc_n"]
HAND_FEATS = [f"tmh_{k}_dev" for k in KINDS] + ["tmh_speed_dev", "tmh_n"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variants", default="base,hand,count,both")
    p.add_argument("--seeds", default="42")
    p.add_argument("--folds", default="2024",
                   help="쉼표로 구분. 4-6 기준 최소 3폴드 x 2시드")
    p.add_argument("--map", default="pitcher_id_map.csv")
    p.add_argument("--min-conf", type=float, default=0.9)
    return p.parse_args()


def summarize(df, keys, prefix, min_cell):
    """keys 로 묶어 구종 비율과 구속 평균을 낸다."""
    g = df.groupby(keys)
    out = pd.DataFrame({f"{prefix}_n": g.size()})
    for k in KINDS:
        out[f"{prefix}_{k}"] = g["is_" + k].mean()
    out[f"{prefix}_speed"] = g["rel_speed"].mean()
    return out[out[f"{prefix}_n"] >= min_cell]


def build_tables(tm, seasons):
    """시즌별로 (그 시즌 미만 trackman) 에서 조건부 편차표를 만든다."""
    for k in KINDS:
        tm["is_" + k] = (tm["pitch_type_group"] == k).astype(float)

    count_parts, hand_parts = [], []
    for S in seasons:
        past = tm[tm["season"] < S]
        if not len(past):
            continue
        # 투수 전체 (기준선) — 이건 메인 데이터가 이미 아는 것
        base = summarize(past, ["pitcher_id"], "tmp", 0)

        cnt = summarize(past, ["pitcher_id", "balls_before", "strikes_before"],
                        "tmc", MIN_COUNT_CELL)
        cnt = cnt.join(base, on="pitcher_id")
        for k in KINDS:
            cnt[f"tmc_{k}_dev"] = cnt[f"tmc_{k}"] - cnt[f"tmp_{k}"]
        cnt["tmc_speed_dev"] = cnt["tmc_speed"] - cnt["tmp_speed"]
        cnt = cnt[COUNT_FEATS].reset_index()
        cnt["season"] = S
        count_parts.append(cnt)

        hnd = summarize(past, ["pitcher_id", "batter_hand"], "tmh",
                        MIN_HAND_CELL)
        hnd = hnd.join(base, on="pitcher_id")
        for k in KINDS:
            hnd[f"tmh_{k}_dev"] = hnd[f"tmh_{k}"] - hnd[f"tmp_{k}"]
        hnd["tmh_speed_dev"] = hnd["tmh_speed"] - hnd["tmp_speed"]
        hnd = hnd[HAND_FEATS].reset_index()
        hnd["season"] = S
        hand_parts.append(hnd)

    return (pd.concat(count_parts).set_index(
                ["pitcher_id", "season", "balls_before", "strikes_before"]),
            pd.concat(hand_parts).set_index(
                ["pitcher_id", "season", "batter_hand"]))


def make_model(cols, seed):
    num = [c for c in cols if c not in CAT_COLS]
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                               unknown_value=-1), CAT_COLS),
        ("num", "passthrough", num),
    ])
    clf = HistGradientBoostingClassifier(
        max_iter=N_ITER, learning_rate=LR, max_leaf_nodes=LEAVES,
        min_samples_leaf=MIN_LEAF, l2_regularization=L2,
        early_stopping=False, random_state=seed)
    return Pipeline([("pre", pre), ("clf", clf)])


def best_curve(clf, X_t, y, base):
    best = (0, 0.0)
    for i, proba in enumerate(clf.staged_predict_proba(X_t), start=1):
        s = max(0.0, 100000 * (1 - ((proba[:, 1] - y) ** 2).mean() / base))
        if s > best[1]:
            best = (i, s)
    return best


def main():
    args = parse_args()
    variants = args.variants.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    folds = [int(f) for f in args.folds.split(",")]

    id_map = pd.read_csv(args.map)
    id_map = id_map[id_map["conf"] >= args.min_conf]
    tm = pd.read_csv(f"{DATA_DIR}/trackman_history.csv", encoding="utf-8-sig",
                     usecols=["season", "pitcher_trackman_id", "pitch_type_group",
                              "balls_before", "strikes_before", "batter_hand",
                              "rel_speed"])
    tm["pitcher_id"] = tm["pitcher_trackman_id"].map(
        id_map.set_index("pitcher_trackman_id")["pitcher_id"])
    tm = tm.dropna(subset=["pitcher_id"])
    tm["pitcher_id"] = tm["pitcher_id"].astype(np.int64)
    print(f"대응된 trackman {len(tm):,}건")

    test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig",
                            nrows=0).columns
    cols = [c for c in test_cols if c != ID]
    train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                        usecols=cols + [TARGET])

    seasons = sorted(train["season"].unique())
    cnt_tab, hnd_tab = build_tables(tm, seasons)
    print(f"카운트 조건부 셀 {len(cnt_tab):,} | 좌우 조건부 셀 {len(hnd_tab):,}")

    train["_hand"] = train["batter_hand"].map(HAND)
    df = train.join(cnt_tab, on=["pitcher_id", "season", "balls_before",
                                 "strikes_before"])
    df = df.join(hnd_tab, on=["pitcher_id", "season", "_hand"])

    print("\n시즌별 커버리지 (카운트 / 좌우):")
    for S in seasons:
        m = df["season"] == S
        print(f"  {S}  {df.loc[m,'tmc_n'].notna().mean():6.1%} / "
              f"{df.loc[m,'tmh_n'].notna().mean():6.1%}")

    sets = {"base": cols, "hand": cols + HAND_FEATS,
            "count": cols + COUNT_FEATS,
            "both": cols + COUNT_FEATS + HAND_FEATS}
    res = {}
    for Y in folds:
        tr, va = df[df["season"] < Y], df[df["season"] == Y]
        y = va[TARGET].to_numpy()
        base = y.mean() * (1 - y.mean())
        cov_tr = tr["tmc_n"].notna().mean()
        print(f"\n--- fold {Y} | 학습 {len(tr):,} 검증 {len(va):,} "
              f"| 학습 커버리지 {cov_tr:.1%} ---")
        for seed in seeds:
            for v in variants:
                use = sets[v]
                t = time.time()
                m = make_model(use, seed)
                m.fit(tr[use], tr[TARGET])
                X_t = m.named_steps["pre"].transform(va[use])
                it, sc = best_curve(m.named_steps["clf"], X_t, y, base)
                res[(Y, seed, v)] = sc
                print(f"  seed{seed} {v:6s} {sc:9.2f} (iter {it:4d})  "
                      f"[{time.time()-t:.0f}s]")

    print("\n=== 요약 (폴드 안에서 base 대비) ===")
    for v in variants:
        if v == "base":
            continue
        d = []
        cells = []
        for Y in folds:
            for s in seeds:
                if (Y, s, v) in res and (Y, s, "base") in res:
                    x = res[(Y, s, v)] - res[(Y, s, "base")]
                    d.append(x)
                    cells.append(f"{Y}s{s} {x:+.1f}")
        if d:
            se = np.std(d, ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
            print(f"  {v:6s} 평균 {np.mean(d):+7.2f}  표준오차 {se:5.2f}  "
                  f"(n={len(d)})")
            print(f"         {' | '.join(cells)}")
    print("\n  4-6: 시드 노이즈 ±15, 폴드 간 ±45. 평균이 표준오차의 2배를 못 넘으면")
    print("  채택 근거가 못 된다. 폴드별 부호가 갈리는지도 반드시 볼 것.")


if __name__ == "__main__":
    main()
