r"""여러 하이퍼파라미터 조합의 검증 점수 봉우리를 비교한다.

각 조합마다 조기종료 없이 끝까지 학습한 뒤, staged_predict_proba 로
2024 시즌 점수 곡선을 그려 최고점과 그 반복수를 기록한다.
데이터는 한 번만 읽어 모든 조합이 재사용한다.

사용법:
    .\.venv\Scripts\python.exe sweep.py
"""
import os
import time

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

DATA_DIR = "./data"
ID = "row_id"
TARGET = "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state"]
VAL_SEASON = 2024
RF_BASELINE = 415.57

# (이름, lr, leaves, min_leaf, l2, max_iter)
CONFIGS = [
    ("현재최선  lr.05 L31",  0.05,  31,   200, 1.0,  400),
    ("느린학습  lr.02 L31",  0.02,  31,   200, 1.0, 1000),
    ("넓은잎    lr.02 L63",  0.02,  63,   500, 1.0,  800),
    ("강한규제  lr.02 L15",  0.02,  15,  1000, 1.0, 1200),
    ("매우보수  lr.03 L31",  0.03,  31,  2000, 5.0,  800),
]


def load_data():
    test_cols = pd.read_csv(os.path.join(DATA_DIR, "test.csv"),
                            encoding="utf-8-sig", nrows=0).columns
    features = [c for c in test_cols if c != ID]
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"),
                        encoding="utf-8-sig", usecols=features + [TARGET])
    is_val = train["season"] == VAL_SEASON
    return (features,
            train.loc[~is_val, features], train.loc[~is_val, TARGET],
            train.loc[is_val, features], train.loc[is_val, TARGET].to_numpy())


def run_config(name, lr, leaves, min_leaf, l2, max_iter,
               features, X_tr, y_tr, X_va, y_va, base):
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
    model = Pipeline([("pre", pre), ("clf", clf)])

    t = time.time()
    model.fit(X_tr, y_tr)
    fit_sec = time.time() - t

    X_va_t = model.named_steps["pre"].transform(X_va)
    best = (0, 0.0, 0.0, 0.0)   # (iter, score, pmin, pmax)
    for i, proba in enumerate(clf.staged_predict_proba(X_va_t), start=1):
        p = proba[:, 1]
        score = max(0, 100000 * (1 - ((p - y_va) ** 2).mean() / base))
        if score > best[1]:
            best = (i, score, p.min(), p.max())

    hit_ceiling = best[0] >= max_iter * 0.95
    print(f"  {name:22s} 최적 {best[0]:4d}회 → {best[1]:7.2f} "
          f"({best[1] - RF_BASELINE:+7.2f})  범위 {best[2]:.3f}~{best[3]:.3f} "
          f"  [{fit_sec:.0f}s]" + ("  ⚠상한도달" if hit_ceiling else ""))
    return name, best[0], best[1], hit_ceiling


def main():
    features, X_tr, y_tr, X_va, y_va = load_data()
    r = y_va.mean()
    base = r * (1 - r)
    print(f"학습 {len(X_tr)} | 검증 {len(X_va)} ({VAL_SEASON} 시즌)")
    print(f"기준선 r(1-r)={base:.6f} | RF 베이스라인 {RF_BASELINE}\n")
    print(f"{'조합':22s} {'결과'}")
    print("-" * 88)

    results = []
    for cfg in CONFIGS:
        results.append(run_config(*cfg, features, X_tr, y_tr, X_va, y_va, base))

    print("-" * 88)
    results.sort(key=lambda x: -x[2])
    print("\n=== 순위 ===")
    for rank, (name, n_iter, score, ceiling) in enumerate(results, start=1):
        mark = "  ⚠ max_iter 상한에 걸림 - 더 늘려야 함" if ceiling else ""
        print(f"  {rank}. {name:22s} {score:7.2f}  (반복 {n_iter}){mark}")

    win = results[0]
    print(f"\n★ 최고: {win[0]}  Score {win[2]:.2f}  반복 {win[1]}회")


if __name__ == "__main__":
    main()
