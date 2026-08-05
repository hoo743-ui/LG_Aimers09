r"""HistGradientBoosting 학습/검증 스크립트.

노트북과 달리 매번 처음부터 실행되므로 "셀을 안 돌려서 옛 모델이 남아있는"
문제가 생기지 않는다.

기본 동작 : 2019~2023 학습 → 2024 검증 → Brier Skill Score 출력
--save    : 검증 후 전체 데이터로 재학습하여 ./model/rf.pkl 저장

사용법:
    .\.venv\Scripts\python.exe train_hgb.py
    .\.venv\Scripts\python.exe train_hgb.py --save
    .\.venv\Scripts\python.exe train_hgb.py --lr 0.03 --leaves 63
"""
import argparse
import os
import time

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

DATA_DIR = "./data"
MODEL_PATH = "./model/rf.pkl"   # script.py 가 찾는 이름. 내용은 HGB 여도 무관.

ID = "row_id"
TARGET = "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state"]
VAL_SEASON = 2024


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--leaves", type=int, default=31)
    p.add_argument("--min-leaf", type=int, default=200)
    p.add_argument("--l2", type=float, default=1.0)
    p.add_argument("--max-iter", type=int, default=1000)
    p.add_argument("--save", action="store_true",
                   help="검증 후 전체 데이터로 재학습하여 모델 저장")
    return p.parse_args()


def build_model(args, features):
    """전처리 + HGB 파이프라인.

    수치형은 passthrough 다. asof_* 의 결측은 '아직 기록이 없다'는 정보이므로
    중앙값으로 덮지 않고 HGB 가 결측 자체를 분기 조건으로 학습하게 둔다.
    """
    num_cols = [c for c in features if c not in CAT_COLS]
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                               unknown_value=-1), CAT_COLS),
        ("num", "passthrough", num_cols),
    ])
    clf = HistGradientBoostingClassifier(
        max_iter=args.max_iter,
        learning_rate=args.lr,
        max_leaf_nodes=args.leaves,
        min_samples_leaf=args.min_leaf,
        l2_regularization=args.l2,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=50,
        random_state=42,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def brier_score(y, pred):
    """대회 지표. 상수 예측(r) 대비 Brier 개선폭을 10만 배 한 값."""
    r = y.mean()
    brier = ((pred - y) ** 2).mean()
    base = r * (1 - r)
    return max(0, 100000 * (1 - brier / base)), brier, base


def main():
    args = parse_args()

    # 피처 목록은 test.csv 가 정한다. train 에만 있는 컬럼을 쓰면 추론이 깨진다.
    test_cols = pd.read_csv(os.path.join(DATA_DIR, "test.csv"),
                            encoding="utf-8-sig", nrows=0).columns
    features = [c for c in test_cols if c != ID]

    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"),
                        encoding="utf-8-sig", usecols=features + [TARGET])
    print(f"train: {train.shape} | 피처 {len(features)}개")

    model = build_model(args, features)
    print(f"모델: {type(model.named_steps['clf']).__name__} "
          f"(lr={args.lr}, leaves={args.leaves}, "
          f"min_leaf={args.min_leaf}, l2={args.l2})")

    # ---- 검증: 2024 시즌 홀드아웃 ----
    is_val = train["season"] == VAL_SEASON
    X_tr, y_tr = train.loc[~is_val, features], train.loc[~is_val, TARGET]
    X_va, y_va = train.loc[is_val, features], train.loc[is_val, TARGET]
    print(f"학습 {len(X_tr)} | 검증 {len(X_va)}")

    t = time.time()
    model.fit(X_tr, y_tr)
    fit_sec = time.time() - t
    n_iter = model.named_steps["clf"].n_iter_
    print(f"학습 완료 :: {fit_sec:.1f}s | 조기종료 반복수 {n_iter}")

    pred = model.predict_proba(X_va)[:, 1]
    score, brier, base = brier_score(y_va, pred)
    print(f"\nBrier: {brier:.6f} | 기준선 r(1-r): {base:.6f}")
    print(f"Validation Score: {score:.2f}")
    print(f"예측 범위: {pred.min():.4f} ~ {pred.max():.4f} (평균 {pred.mean():.4f})")

    if not args.save:
        print("\n(저장하려면 --save 를 붙여 다시 실행)")
        return

    # ---- 전체 데이터로 재학습 후 저장 ----
    # 조기종료로 찾은 반복수를 고정한다. 데이터가 늘었으니 10% 여유를 준다.
    model.named_steps["clf"].set_params(
        max_iter=int(n_iter * 1.1), early_stopping=False)
    t = time.time()
    model.fit(train[features], train[TARGET])
    print(f"\n전체 재학습 완료 :: {time.time() - t:.1f}s "
          f"(max_iter={int(n_iter * 1.1)})")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH, compress=3)
    print(f"저장 완료: {MODEL_PATH} "
          f"({os.path.getsize(MODEL_PATH)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
