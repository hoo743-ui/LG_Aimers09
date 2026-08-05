r"""학습 구간과 최근성 가중 실험.

타깃 성공률이 시즌마다 계속 떨어지고 있고(2019 .565 → 2024 .486), 누적형
asof 피처는 그 추세를 0.025 만큼 뒤늦게 따라간다. 그래서 옛 시즌은 "누적
성공률 ≈ 타깃" 이라는, 지금은 틀린 관계를 가르친다.

따라서 최근 몇 시즌만 쓰거나 최근에 큰 가중을 주는 편이 나을 수 있다.
반대로 데이터가 줄어드는 손해도 있으니 실제로 재본다.

폴드는 두 개다. 둘 다 "과거로 학습해 다음 시즌을 맞힌다" 구조다.
    fold 2024 : 최종 모델(2019~2024 학습 → 2025 예측)과 가장 닮은 상황
    fold 2023 : 국면이 크게 꺾인 해. 무너지지 않는지 보는 스트레스 시험

사용법:
    .\.venv\Scripts\python.exe regime_test.py
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
FOLDS = [2024, 2023]

LR, LEAVES, MIN_LEAF, L2 = 0.02, 10, 1000, 1.0
MAX_ITER = 2500

# (이름, 사용할 최근 시즌 수, 시즌당 가중 배율)
# 배율 1.0 = 균등. 2.0 = 한 시즌 최신일수록 가중 2배.
SCHEMES = [
    ("전체 균등",        99, 1.0),
    ("전체 가중x1.5",    99, 1.5),
    ("전체 가중x2",      99, 2.0),
    ("전체 가중x3",      99, 3.0),
    ("최근3시즌",         3, 1.0),
    ("최근3시즌 가중x2",  3, 2.0),
    ("최근2시즌",         2, 1.0),
    ("최근1시즌",         1, 1.0),
]


def make_model(features, max_iter):
    num = [c for c in features if c not in CAT_COLS]
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                               unknown_value=-1), CAT_COLS),
        ("num", "passthrough", num),
    ])
    clf = HistGradientBoostingClassifier(
        max_iter=max_iter, learning_rate=LR, max_leaf_nodes=LEAVES,
        min_samples_leaf=MIN_LEAF, l2_regularization=L2,
        early_stopping=False, random_state=42,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def run(features, train, val_season, n_seasons, decay):
    tr = train[(train["season"] < val_season)
               & (train["season"] >= val_season - n_seasons)]
    va = train[train["season"] == val_season]
    y_va = va[TARGET].to_numpy()
    r = y_va.mean()
    base = r * (1 - r)

    # 최신 시즌 가중 1.0, 한 해 멀어질수록 1/decay 배
    age = val_season - 1 - tr["season"].to_numpy()
    w = decay ** (-age.astype(float))

    model = make_model(features, MAX_ITER)
    model.fit(tr[features], tr[TARGET], clf__sample_weight=w)

    X_va_t = model.named_steps["pre"].transform(va[features])
    clf = model.named_steps["clf"]
    best_iter, best_score, best_mean = 0, -1.0, 0.0
    for i, proba in enumerate(clf.staged_predict_proba(X_va_t), start=1):
        p = proba[:, 1]
        s = max(0, 100000 * (1 - ((p - y_va) ** 2).mean() / base))
        if s > best_score:
            best_iter, best_score, best_mean = i, s, float(p.mean())
    return best_iter, best_score, best_mean, r, len(tr)


def main():
    test_cols = pd.read_csv(os.path.join(DATA_DIR, "test.csv"),
                            encoding="utf-8-sig", nrows=0).columns
    features = [c for c in test_cols if c != ID]
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"),
                        encoding="utf-8-sig", usecols=features + [TARGET])

    print(f"모델 고정: lr={LR} leaves={LEAVES} min_leaf={MIN_LEAF} l2={L2}")
    print("예측평균/실제 = 중심이 얼마나 어긋났는가. 차이가 클수록 Brier 손해\n")

    for season in FOLDS:
        print(f"===== fold {season} =====")
        print(f"{'구성':20s}{'학습행':>10s}{'점수':>9s}{'반복':>7s}"
              f"{'예측평균':>10s}{'실제':>8s}{'편차':>8s}")
        print("-" * 72)
        rows = []
        for name, n_seasons, decay in SCHEMES:
            t = time.time()
            it, sc, pm, r, n = run(features, train, season, n_seasons, decay)
            print(f"{name:20s}{n:10d}{sc:9.2f}{it:7d}{pm:10.4f}{r:8.4f}"
                  f"{pm - r:+8.4f}  [{time.time()-t:.0f}s]")
            rows.append((name, sc, it, pm - r))
        rows.sort(key=lambda x: -x[1])
        print(f"  ★ fold {season} 최고: {rows[0][0]}  {rows[0][1]:.2f} "
              f"(반복 {rows[0][2]}, 편차 {rows[0][3]:+.4f})\n")


if __name__ == "__main__":
    main()
