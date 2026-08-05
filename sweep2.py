r"""2차 탐색 — 잎을 더 줄이는 방향, 시즌 2겹 검증.

1차에서 잎이 적을수록(L63<L31<L15) 점수가 올랐으므로 그 방향을 더 밀어본다.

검증은 두 겹으로 한다. 조합 하나를 검증셋 하나로만 고르면 그 시즌에 우연히
맞는 조합을 뽑게 되기 때문이다. 두 겹 모두 "과거로 학습해 다음 시즌을 맞춘다"는
평가 상황과 같은 구조다.

    fold A : 2019~2022 학습 → 2023 검증
    fold B : 2019~2023 학습 → 2024 검증

평균이 높고 두 겹의 차이가 작은 조합이 안전하다.

사용법:
    .\.venv\Scripts\python.exe sweep2.py
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
FOLDS = [2023, 2024]

# (이름, lr, leaves, min_leaf, l2, max_iter)
CONFIGS = [
    ("L15 min1k  lr.02",   0.02, 15, 1000, 1.0, 1800),   # 1차 우승자, 상한 확장
    ("L10 min1k  lr.02",   0.02, 10, 1000, 1.0, 2500),
    ("L7  min1k  lr.02",   0.02,  7, 1000, 1.0, 3500),
    ("L4  min1k  lr.02",   0.02,  4, 1000, 1.0, 5000),
    ("L15 min500 lr.02",   0.02, 15,  500, 1.0, 1800),
    ("L15 min4k  lr.02",   0.02, 15, 4000, 1.0, 1800),
    ("L15 min1k  lr.01",   0.01, 15, 1000, 1.0, 3500),
    ("L15 min1k  l2=10",   0.02, 15, 1000, 10.0, 1800),
    ("L10 min4k  lr.02",   0.02, 10, 4000, 1.0, 2500),
]


def make_model(lr, leaves, min_leaf, l2, max_iter, features):
    num_cols = [c for c in features if c not in CAT_COLS]
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                               unknown_value=-1), CAT_COLS),
        ("num", "passthrough", num_cols),
    ])
    clf = HistGradientBoostingClassifier(
        max_iter=max_iter, learning_rate=lr, max_leaf_nodes=leaves,
        min_samples_leaf=min_leaf, l2_regularization=l2,
        early_stopping=False, random_state=42,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def best_on_fold(cfg, features, train, val_season):
    """해당 폴드에서 반복수별 최고 점수와 그 반복수를 반환."""
    _, lr, leaves, min_leaf, l2, max_iter = cfg
    tr = train[train["season"] < val_season]
    va = train[train["season"] == val_season]
    y_va = va[TARGET].to_numpy()
    r = y_va.mean()
    base = r * (1 - r)

    model = make_model(lr, leaves, min_leaf, l2, max_iter, features)
    model.fit(tr[features], tr[TARGET])

    X_va_t = model.named_steps["pre"].transform(va[features])
    clf = model.named_steps["clf"]
    best_iter, best_score = 0, -1.0
    for i, proba in enumerate(clf.staged_predict_proba(X_va_t), start=1):
        p = proba[:, 1]
        s = max(0, 100000 * (1 - ((p - y_va) ** 2).mean() / base))
        if s > best_score:
            best_iter, best_score = i, s
    return best_iter, best_score, best_iter >= max_iter * 0.95


def main():
    test_cols = pd.read_csv(os.path.join(DATA_DIR, "test.csv"),
                            encoding="utf-8-sig", nrows=0).columns
    features = [c for c in test_cols if c != ID]
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"),
                        encoding="utf-8-sig", usecols=features + [TARGET])
    print(f"데이터 {train.shape} | 폴드 {FOLDS}\n")

    header = f"{'조합':20s}" + "".join(f"{f'val{s}':>18s}" for s in FOLDS) + f"{'평균':>10s}"
    print(header)
    print("-" * len(header))

    results = []
    for cfg in CONFIGS:
        row, scores, iters, ceilings = cfg[0], [], [], []
        t = time.time()
        for season in FOLDS:
            it, sc, ceil = best_on_fold(cfg, features, train, season)
            scores.append(sc)
            iters.append(it)
            ceilings.append(ceil)
        mean = float(np.mean(scores))
        cells = "".join(f"{sc:11.2f}({it:4d})" for sc, it in zip(scores, iters))
        flag = " ⚠상한" if any(ceilings) else ""
        print(f"{row:20s}{cells}{mean:10.2f}  [{time.time()-t:.0f}s]{flag}")
        results.append((row, cfg, scores, iters, mean, any(ceilings)))

    print("-" * len(header))
    results.sort(key=lambda x: -x[4])
    print("\n=== 평균 순위 ===")
    for rank, (name, cfg, scores, iters, mean, ceil) in enumerate(results, 1):
        spread = max(scores) - min(scores)
        note = "  ⚠ max_iter 상한 도달" if ceil else ""
        print(f"  {rank}. {name:20s} 평균 {mean:7.2f} "
              f"| 폴드차 {spread:6.2f} | 반복 {iters}{note}")

    win = results[0]
    print(f"\n★ 최고: {win[0]}")
    print(f"  설정: lr={win[1][1]} leaves={win[1][2]} min_leaf={win[1][3]} "
          f"l2={win[1][4]}")
    print(f"  평균 {win[4]:.2f} | 폴드별 {[f'{s:.2f}' for s in win[2]]} "
          f"| 최적반복 {win[3]}")
    print(f"\n  final_train.py 실행 예시:")
    print(f"  .\\.venv\\Scripts\\python.exe final_train.py --lr {win[1][1]} "
          f"--leaves {win[1][2]} --min-leaf {win[1][3]} --l2 {win[1][4]} "
          f"--n-iter {win[3][-1]} --seeds 5")


if __name__ == "__main__":
    main()
