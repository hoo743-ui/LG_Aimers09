r"""시드 앙상블 vs 다양성 앙상블 비교.

시드만 바꾼 모델들은 서로 너무 닮아서 평균의 이득이 작다. 잎 개수를 달리하면
잡아내는 상호작용의 깊이가 달라져 오차의 성격이 갈리고, 평균이 서로의 실수를
상쇄해 준다. 실제로 그런지 같은 폴드에서 잰다.

폴드마다 모델 풀을 한 번만 학습해 두고, 그 예측을 조합만 바꿔 가며 평가한다.
같은 모델을 조합마다 다시 학습하지 않으므로 훨씬 빠르다.

2023 은 어떤 구성으로도 0점이라(국면 이동) 비교에 정보가 없어 제외한다.

사용법:
    .\.venv\Scripts\python.exe ens_test.py
"""
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
FOLDS = [2021, 2022, 2024]
LR, L2 = 0.02, 1.0

# 이름 -> (leaves, min_leaf, n_iter). 반복수는 조합별 곡선에서 찾은 최적값.
SHAPES = {
    "L7":  (7,  1000, 1400),
    "L10": (10, 1000, 1000),
    "L15": (15, 1000,  860),
    "L20": (20, 1000,  700),
    "L31": (31,  200,  530),
}

# 풀에 담을 (모양, 시드) 목록
POOL = ([("L10", s) for s in (42, 43, 44, 45, 46)]
        + [(shape, 42) for shape in ("L7", "L15", "L20", "L31")])

# 비교할 조합
COMBOS = [
    ("단일 L10",        [("L10", 42)]),
    ("시드 2개",        [("L10", s) for s in (42, 43)]),
    ("시드 3개",        [("L10", s) for s in (42, 43, 44)]),
    ("시드 5개",        [("L10", s) for s in (42, 43, 44, 45, 46)]),
    ("다양성 3",        [("L7", 42), ("L10", 42), ("L15", 42)]),
    ("다양성 4",        [("L7", 42), ("L10", 42), ("L15", 42), ("L31", 42)]),
    ("다양성 5",        [("L7", 42), ("L10", 42), ("L15", 42),
                         ("L20", 42), ("L31", 42)]),
    ("혼합 7",          [("L10", 42), ("L10", 43), ("L10", 44),
                         ("L7", 42), ("L15", 42), ("L20", 42), ("L31", 42)]),
]


def make_model(features, shape, seed):
    leaves, min_leaf, n_iter = SHAPES[shape]
    num = [c for c in features if c not in CAT_COLS]
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                               unknown_value=-1), CAT_COLS),
        ("num", "passthrough", num),
    ])
    clf = HistGradientBoostingClassifier(
        max_iter=n_iter, learning_rate=LR, max_leaf_nodes=leaves,
        min_samples_leaf=min_leaf, l2_regularization=L2,
        early_stopping=False, random_state=seed,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def main():
    test_cols = pd.read_csv(os.path.join(DATA_DIR, "test.csv"),
                            encoding="utf-8-sig", nrows=0).columns
    features = [c for c in test_cols if c != ID]
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"),
                        encoding="utf-8-sig", usecols=features + [TARGET])
    print(f"폴드 {FOLDS} | 풀 {len(POOL)}개 모델 | 조합 {len(COMBOS)}개\n")

    per_fold = {}
    for Y in FOLDS:
        tr = train[train["season"] < Y]
        va = train[train["season"] == Y]
        y = va[TARGET].to_numpy()
        base = y.mean() * (1 - y.mean())
        print(f"--- fold {Y} (학습 {len(tr)}) ---")

        preds = {}
        for shape, seed in POOL:
            t = time.time()
            m = make_model(features, shape, seed)
            m.fit(tr[features], tr[TARGET])
            preds[(shape, seed)] = m.predict_proba(va[features])[:, 1]
            print(f"    {shape}-s{seed} [{time.time()-t:.0f}s]")
        per_fold[Y] = (y, base, preds)

    print(f"\n{'조합':16s}{'n':>4}"
          + "".join(f"{f'val{Y}':>12s}" for Y in FOLDS) + f"{'평균':>11s}")
    print("-" * (20 + 12 * len(FOLDS) + 11))

    results = []
    for name, members in COMBOS:
        scores = []
        for Y in FOLDS:
            y, base, preds = per_fold[Y]
            p = np.mean([preds[m] for m in members], axis=0)
            scores.append(max(0, 100000 * (1 - ((p - y) ** 2).mean() / base)))
        mean = float(np.mean(scores))
        print(f"{name:16s}{len(members):>4}"
              + "".join(f"{s:12.2f}" for s in scores) + f"{mean:11.2f}")
        results.append((name, members, scores, mean))

    print("-" * (20 + 12 * len(FOLDS) + 11))
    single = results[0][3]
    results.sort(key=lambda x: -x[3])
    print("\n=== 순위 (단일 L10 대비) ===")
    for rank, (name, members, scores, mean) in enumerate(results, 1):
        print(f"  {rank}. {name:16s} 평균 {mean:9.2f}  ({mean - single:+8.2f})"
              f"  모델 {len(members)}개")

    win = results[0]
    print(f"\n★ 최고: {win[0]}  평균 {win[3]:.2f}  (단일 대비 {win[3] - single:+.2f})")
    print(f"  구성: {win[1]}")


if __name__ == "__main__":
    main()
