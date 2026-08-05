r"""반복 횟수별 검증 점수 곡선을 뽑아 과적합 시작 지점을 찾는다.

HGB 는 staged_predict_proba 로 매 반복 시점의 예측을 꺼낼 수 있으므로,
학습 한 번으로 "몇 번째 반복이 최적인가" 를 전부 확인할 수 있다.
조기종료(무작위 10% 홀드아웃)는 같은 시즌 안에서 나뉘어 시즌 이동을
감지하지 못하므로 여기서는 끄고, 2024 시즌으로 직접 채점한다.

사용법:
    .\.venv\Scripts\python.exe diag_curve.py
    .\.venv\Scripts\python.exe diag_curve.py --lr 0.02 --leaves 15 --min-leaf 2000
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
VAL_SEASON = 2024


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--leaves", type=int, default=31)
    p.add_argument("--min-leaf", type=int, default=200)
    p.add_argument("--l2", type=float, default=1.0)
    p.add_argument("--max-iter", type=int, default=300)
    p.add_argument("--every", type=int, default=10, help="곡선 출력 간격")
    return p.parse_args()


def main():
    args = parse_args()

    test_cols = pd.read_csv(os.path.join(DATA_DIR, "test.csv"),
                            encoding="utf-8-sig", nrows=0).columns
    features = [c for c in test_cols if c != ID]
    num_cols = [c for c in features if c not in CAT_COLS]

    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"),
                        encoding="utf-8-sig", usecols=features + [TARGET])

    is_val = train["season"] == VAL_SEASON
    X_tr, y_tr = train.loc[~is_val, features], train.loc[~is_val, TARGET]
    X_va = train.loc[is_val, features]
    y_va = train.loc[is_val, TARGET].to_numpy()
    print(f"학습 {len(X_tr)} | 검증 {len(X_va)} ({VAL_SEASON} 시즌)")

    r = y_va.mean()
    base = r * (1 - r)
    print(f"검증 기준선 r={r:.4f}, r(1-r)={base:.6f}")
    print(f"설정: lr={args.lr} leaves={args.leaves} "
          f"min_leaf={args.min_leaf} l2={args.l2} max_iter={args.max_iter}\n")

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
        early_stopping=False,          # 곡선을 끝까지 보기 위해 끔
        random_state=42,
    )
    model = Pipeline([("pre", pre), ("clf", clf)])

    t = time.time()
    model.fit(X_tr, y_tr)
    print(f"학습 완료 :: {time.time() - t:.1f}s\n")

    # 전처리된 검증 행렬을 한 번만 만들어 두고 단계별 예측을 순회한다.
    X_va_t = model.named_steps["pre"].transform(X_va)

    print(f"{'iter':>5} {'Brier':>10} {'Score':>9} {'예측범위':>17}")
    print("-" * 45)
    scores = []
    t = time.time()
    for i, proba in enumerate(clf.staged_predict_proba(X_va_t), start=1):
        p = proba[:, 1]
        brier = ((p - y_va) ** 2).mean()
        score = max(0, 100000 * (1 - brier / base))
        scores.append((i, brier, score, p.min(), p.max()))
        if i % args.every == 0 or i == 1:
            print(f"{i:5d} {brier:10.6f} {score:9.2f} "
                  f"{p.min():7.4f} ~ {p.max():7.4f}")

    best = max(scores, key=lambda s: s[2])
    print("-" * 45)
    print(f"곡선 계산 :: {time.time() - t:.1f}s")
    print(f"\n★ 최적 반복수 {best[0]} → Score {best[2]:.2f} "
          f"(Brier {best[1]:.6f}, 예측범위 {best[3]:.4f}~{best[4]:.4f})")
    print(f"  마지막 반복 {scores[-1][0]} → Score {scores[-1][2]:.2f}")
    print(f"  RF 베이스라인 415.57 대비: {best[2] - 415.57:+.2f}")

    # 과적합 시작 지점 = 최고점 이후 점수가 계속 떨어지기 시작하는 곳
    if best[0] < len(scores):
        drop = scores[-1][2] - best[2]
        print(f"  최고점 이후 {len(scores) - best[0]}회 더 학습 시 {drop:+.2f}점")

    np.save("curve_scores.npy", np.array([(s[0], s[2]) for s in scores]))
    print("\n곡선 저장: curve_scores.npy")


if __name__ == "__main__":
    main()
