r"""최종 모델 학습 — 시드 앙상블 + 과신 교정(shrinkage).

배경: 이 문제는 신호가 희박해서(BSS ~0.006) 모델이 조금만 확신해도 Brier 가
손해를 본다. 두 가지로 대응한다.

1. 시드 앙상블 — 같은 설정을 여러 시드로 학습해 평균낸다. 개별 모델의 분산이
   깎이면서 Brier 가 거의 항상 내려간다.
2. 과신 교정 — 예측을 기준선 쪽으로 당긴다.  p' = c + a*(p - c)
   최적 a 는 (p-c) 로 (y-c) 를 설명하는 회귀 기울기라 닫힌 해로 구해진다.
   a < 1 이면 모델이 과신하고 있다는 뜻이다.

저장 형식은 커스텀 클래스를 쓰지 않는다. 평가 서버에서 joblib.load 가 클래스를
찾지 못해 실패하는 사고를 피하려고, sklearn 객체와 순수 dict 만 담는다.

사용법:
    .\.venv\Scripts\python.exe final_train.py                 # 검증만
    .\.venv\Scripts\python.exe final_train.py --seeds 5
    .\.venv\Scripts\python.exe final_train.py --seeds 5 --save # 전체학습+저장
"""
import argparse
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

DATA_DIR = "./data"
MODEL_PATH = "./model/rf.pkl"
ID = "row_id"
TARGET = "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state"]
VAL_SEASON = 2024
RF_BASELINE = 415.57


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--leaves", type=int, default=10)
    p.add_argument("--min-leaf", type=int, default=1000)
    p.add_argument("--l2", type=float, default=1.0)
    p.add_argument("--n-iter", type=int, default=1000,
                   help="곡선에서 찾은 최적 반복수 (조기종료 없이 고정)")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--keep-ids", action="store_true",
                   help="pitcher_id/batter_id 를 남긴다. 기본은 제거 — 점수는 "
                        "같은데(+1.27, 노이즈 범위) 2025 신인 ID 노출을 피할 수 "
                        "있다. 팀 ID 는 빼면 -20.45 라 항상 남긴다.")
    p.add_argument("--calibrate", action="store_true",
                   help="축소 보정을 실제로 적용한다. 기본은 측정만 하고 쓰지 않음 "
                        "— 연도별 편차가 불안정해 전년도 보정을 옮기면 손해였다 "
                        "(offset_test.py 참고).")
    p.add_argument("--save", action="store_true")
    return p.parse_args()


def make_pipeline(args, features, seed, n_iter):
    num_cols = [c for c in features if c not in CAT_COLS]
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                               unknown_value=-1), CAT_COLS),
        ("num", "passthrough", num_cols),
    ])
    clf = HistGradientBoostingClassifier(
        max_iter=n_iter,
        learning_rate=args.lr,
        max_leaf_nodes=args.leaves,
        min_samples_leaf=args.min_leaf,
        l2_regularization=args.l2,
        early_stopping=False,
        random_state=seed,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def score_of(y, p, base):
    return max(0, 100000 * (1 - ((p - y) ** 2).mean() / base))


def fit_shrinkage(y, p, center):
    """p' = center + a*(p-center) 의 Brier 최소화 해. 회귀 기울기와 같다."""
    d = p - center
    var = (d ** 2).mean()
    if var <= 0:
        return 1.0
    return float((d * (y - center)).mean() / var)


def main():
    args = parse_args()

    test_cols = pd.read_csv(os.path.join(DATA_DIR, "test.csv"),
                            encoding="utf-8-sig", nrows=0).columns
    all_features = [c for c in test_cols if c != ID]
    drop = [] if args.keep_ids else ["pitcher_id", "batter_id"]
    features = [c for c in all_features if c not in drop]
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"),
                        encoding="utf-8-sig", usecols=all_features + [TARGET])
    if drop:
        print(f"제거 컬럼: {drop}")

    is_val = train["season"] == VAL_SEASON
    X_tr, y_tr = train.loc[~is_val, features], train.loc[~is_val, TARGET]
    X_va = train.loc[is_val, features]
    y_va = train.loc[is_val, TARGET].to_numpy()

    r = y_va.mean()
    base = r * (1 - r)
    center = float(y_tr.mean())      # 학습 데이터 성공률. 검증 정답은 쓰지 않는다.

    print(f"학습 {len(X_tr)} | 검증 {len(X_va)} ({VAL_SEASON} 시즌)")
    print(f"설정: lr={args.lr} leaves={args.leaves} min_leaf={args.min_leaf} "
          f"l2={args.l2} n_iter={args.n_iter} seeds={args.seeds}")
    print(f"중심값 c={center:.4f} | 검증 기준선 r(1-r)={base:.6f}\n")

    # ---- 시드별 학습 & 누적 평균 ----
    print(f"{'시드':>4} {'단일점수':>9} {'누적앙상블':>11}")
    print("-" * 30)
    acc = np.zeros(len(X_va))
    for k, seed in enumerate(range(42, 42 + args.seeds), start=1):
        m = make_pipeline(args, features, seed, args.n_iter)
        t = time.time()
        m.fit(X_tr, y_tr)
        p = m.predict_proba(X_va)[:, 1]
        acc += p
        ens = acc / k
        print(f"{seed:>4} {score_of(y_va, p, base):9.2f} "
              f"{score_of(y_va, ens, base):11.2f}   [{time.time()-t:.0f}s]")

    p_ens = acc / args.seeds
    s_raw = score_of(y_va, p_ens, base)

    # ---- 과신 교정 ----
    a = fit_shrinkage(y_va, p_ens, center)
    p_cal = np.clip(center + a * (p_ens - center), 0.0, 1.0)
    s_cal = score_of(y_va, p_cal, base)

    print("\n=== 결과 ===")
    print(f"  앙상블 원본     {s_raw:8.2f}   범위 {p_ens.min():.4f}~{p_ens.max():.4f}")
    print(f"  축소계수 a      {a:8.4f}   {'(과신 → 교정 여지 있음)' if a < 0.97 else '(1 에 가까움 = 이미 잘 맞음)'}")
    print(f"  교정 시        {s_cal:8.2f}   ({s_cal - s_raw:+.2f})")
    print(f"  RF 베이스라인 대비 {s_raw - RF_BASELINE:+.2f}")

    # 이 a 는 2024 에서 잰 값이라 2025 에 그대로 옮기면 위험하다. offset_test 에서
    # 전년도 보정을 옮겨 쓴 결과가 2승이 아니라 1승 1패였고, 선형보정은 -472 였다.
    # 그래서 기본값은 보정 없이(a=1) 저장한다.
    alpha_out = a if args.calibrate else 1.0
    if not args.calibrate:
        print("  → 보정 미적용(a=1)으로 저장. 적용하려면 --calibrate")

    if not args.save:
        print("\n(저장하려면 --save)")
        return

    # ---- 전체 데이터로 재학습 ----
    # 학습량이 늘었으니 반복수에 10% 여유를 준다.
    n_iter_full = int(args.n_iter * 1.1)
    print(f"\n전체 데이터 재학습 (n_iter={n_iter_full}) ...")
    models = []
    for seed in range(42, 42 + args.seeds):
        m = make_pipeline(args, features, seed, n_iter_full)
        t = time.time()
        m.fit(train[features], train[TARGET])
        models.append(m)
        print(f"  seed {seed} 완료 [{time.time()-t:.0f}s]")

    bundle = {
        "models": models,
        "alpha": alpha_out,
        "center": float(train[TARGET].mean()),
        # 학습에 쓴 컬럼과 순서. 추론 때 test.csv 에서 이대로 골라내야 한다.
        # 열이 하나라도 다르면 ColumnTransformer 가 이름 불일치로 실패한다.
        "features": list(features),
        "note": "predict: mean(predict_proba[:,1]) -> center + alpha*(p-center) -> clip(0,1)",
    }
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(bundle, MODEL_PATH, compress=3)
    print(f"\n저장 완료: {MODEL_PATH} "
          f"({os.path.getsize(MODEL_PATH)/1e6:.1f} MB, 모델 {len(models)}개)")


if __name__ == "__main__":
    main()
