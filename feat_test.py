r"""피처 제거 실험 — 어떤 컬럼이 오히려 해로운가.

가장 의심스러운 것은 pitcher_id / batter_id 다. 익명 정수 ID 가 그대로 들어가면
트리가 "ID 21813 이상" 같은 무의미한 분기로 특정 선수를 외울 수 있다. 평가 대상인
2025 시즌에는 학습 때 없던 신인 ID 가 등장하므로, 외운 내용이 그대로 오답이 된다.
투수의 실력 정보는 이미 asof_* 컬럼이 담고 있어 ID 없이도 손실이 적을 수 있다.

여기서는 컬럼을 빼기만 한다. 새 컬럼을 만들면 추론 쪽에도 같은 코드를 넣어야 하고
평가 서버에서 pickle 이 그 코드를 못 찾는 사고가 나는데, 제거는 그 위험이 없다.

검증은 sweep2 와 같은 2겹 시즌 방식이다.

사용법:
    .\.venv\Scripts\python.exe feat_test.py --leaves 15 --min-leaf 1000 --lr 0.02
"""
import argparse
import os
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
# 연도별 난이도 차이가 커서(2022 는 2250점, 2023 은 0점) 한두 해로 고르면
# 그 해에만 맞는 답을 뽑게 된다. 네 해를 모두 본다.
FOLDS = [2021, 2022, 2023, 2024]

# (이름, 제거할 컬럼)
VARIANTS = [
    ("전체 유지",            []),
    ("투수ID 제거",          ["pitcher_id"]),
    ("타자ID 제거",          ["batter_id"]),
    ("양쪽 ID 제거",         ["pitcher_id", "batter_id"]),
    ("ID+팀 제거",           ["pitcher_id", "batter_id",
                              "pitcher_team_id", "batter_team_id"]),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--leaves", type=int, default=15)
    p.add_argument("--min-leaf", type=int, default=1000)
    p.add_argument("--l2", type=float, default=1.0)
    p.add_argument("--max-iter", type=int, default=1800)
    return p.parse_args()


def make_model(args, features, max_iter):
    cat = [c for c in CAT_COLS if c in features]
    num = [c for c in features if c not in cat]
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                               unknown_value=-1), cat),
        ("num", "passthrough", num),
    ])
    clf = HistGradientBoostingClassifier(
        max_iter=max_iter, learning_rate=args.lr, max_leaf_nodes=args.leaves,
        min_samples_leaf=args.min_leaf, l2_regularization=args.l2,
        early_stopping=False, random_state=42,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def best_on_fold(args, features, train, val_season):
    """반복수는 실제 제출에 쓸 값으로 고정한다.

    폴드마다 최고 반복수를 골라 쓰면 그 폴드의 정답을 훔쳐보는 셈이라 변형 간
    비교가 왜곡된다. 고정하면 비교가 공정해지고 속도도 훨씬 빠르다.
    """
    tr = train[train["season"] < val_season]
    va = train[train["season"] == val_season]
    y_va = va[TARGET].to_numpy()
    r = y_va.mean()
    base = r * (1 - r)

    model = make_model(args, features, args.max_iter)
    model.fit(tr[features], tr[TARGET])
    p = model.predict_proba(va[features])[:, 1]
    s = max(0, 100000 * (1 - ((p - y_va) ** 2).mean() / base))
    return args.max_iter, s


def main():
    args = parse_args()
    test_cols = pd.read_csv(os.path.join(DATA_DIR, "test.csv"),
                            encoding="utf-8-sig", nrows=0).columns
    all_features = [c for c in test_cols if c != ID]
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"),
                        encoding="utf-8-sig", usecols=all_features + [TARGET])

    print(f"설정: lr={args.lr} leaves={args.leaves} "
          f"min_leaf={args.min_leaf} l2={args.l2}")
    print(f"폴드 {FOLDS} | 전체 피처 {len(all_features)}개\n")

    header = (f"{'변형':22s}{'피처':>5s}"
              + "".join(f"{f'val{s}':>18s}" for s in FOLDS) + f"{'평균':>10s}")
    print(header)
    print("-" * len(header))

    results = []
    for name, drop in VARIANTS:
        features = [c for c in all_features if c not in drop]
        scores, iters = [], []
        t = time.time()
        for season in FOLDS:
            it, sc = best_on_fold(args, features, train, season)
            scores.append(sc)
            iters.append(it)
        mean = float(np.mean(scores))
        cells = "".join(f"{sc:11.2f}({it:4d})" for sc, it in zip(scores, iters))
        print(f"{name:22s}{len(features):5d}{cells}{mean:10.2f}  [{time.time()-t:.0f}s]")
        results.append((name, drop, scores, iters, mean))

    baseline = results[0][4]
    print("-" * len(header))
    results.sort(key=lambda x: -x[4])
    print("\n=== 순위 (전체 유지 대비) ===")
    for rank, (name, drop, scores, iters, mean) in enumerate(results, 1):
        spread = max(scores) - min(scores)
        print(f"  {rank}. {name:22s} 평균 {mean:7.2f} "
              f"({mean - baseline:+7.2f}) | 폴드차 {spread:6.2f} | 반복 {iters}")

    win = results[0]
    print(f"\n★ 최고: {win[0]}  평균 {win[4]:.2f}")
    if win[1]:
        print(f"  제거 컬럼: {win[1]}")
    print(f"  전체 유지 대비 {win[4] - baseline:+.2f}")


if __name__ == "__main__":
    main()
