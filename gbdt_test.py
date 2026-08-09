r"""LightGBM / XGBoost — CatBoost 의 성공이 계열 특유인지 확인한다.

가설. CatBoost 가 이긴 이유는 **대칭 트리(oblivious tree)** 로 보인다 (4-14).
잎 128개(depth 7)인데 예측 표준편차가 HGB(잎 10)보다 좁다. 4-1 이 규정한
"표현력이 아니라 과신하지 않기" 문제에 잎을 조이는 것과 **다른 경로로** 답한
것이다. 그 관점에서 세 라이브러리는 제약 강도가 다르다.

    sklearn HGB   leaf-wise 히스토그램        현재 주력
    LightGBM      leaf-wise 히스토그램        HGB 와 거의 같은 기전
    XGBoost       level-wise (기본)           대칭 트리에 가장 가까운 대안
    CatBoost      oblivious (완전 대칭)       최강 제약

**LightGBM 은 HGB 와 상관이 높아 혼합 이득이 낮을 것으로 예상한다.** XGBoost 의
level-wise 성장은 제약 방향이 CatBoost 쪽이라 가능성이 있다. 예상이 빗나가면
"대칭 트리 때문"이라는 해석이 틀린 것이므로, 어느 쪽이든 정보가 된다.

CatBoost 에서 배운 것 — **기본 설정으로 재면 안 된다.** depth 6→7, 반복수
1100→600 으로 +17 이 나왔다. 그래서 여기서도 반복수 곡선을 같이 낸다.
두 라이브러리 모두 학습된 모델에서 앞 n 그루만 쓰는 예측을 지원한다.

판단은 **현재 최적 혼합 위에 얹었을 때의 기여**로 한다.

    .\.venv\Scripts\python.exe gbdt_test.py --lib lgb --depths 7,10
    .\.venv\Scripts\python.exe gbdt_test.py --lib xgb --depths 6,8
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder

DATA_DIR = "./data"
CACHE = "./.blendcache"
ID = "row_id"
TARGET = "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state"]
PREV1 = "asof_pitcher_prev1_game_success_rate"
# 현재 최적 혼합 (4-15) — hgb 0.15 / cb 0.80 / lr 0.05, lam 0.03
W_HGB, W_CB, W_LR, LAM = 0.15, 0.80, 0.05, 0.03
CB_TAG = "cb_d7_l210_it600_noid_seed42"
NEW_W = [0.05, 0.10, 0.20, 0.30, 0.40]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lib", choices=["lgb", "xgb"], required=True)
    p.add_argument("--fold", type=int, default=2024)
    p.add_argument("--depths", default="7")
    p.add_argument("--iters", type=int, default=1200)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--step", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def add_derived(df):
    df = df.copy()
    df["same_hand"] = (df["pitcher_hand"] == df["batter_hand"]).astype(int)
    return df


def main():
    args = parse_args()
    Y = args.fold
    depths = [int(d) for d in args.depths.split(",")]

    test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig",
                            nrows=0).columns
    base_cols = [c for c in test_cols if c != ID]
    train = add_derived(pd.read_csv(f"{DATA_DIR}/train.csv",
                                    encoding="utf-8-sig",
                                    usecols=base_cols + [TARGET]))
    features = base_cols + ["same_hand"]
    # 범주형 3개만 정수로. HGB 경로와 같은 전처리다.
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                               unknown_value=-1), CAT_COLS),
        ("num", "passthrough", [c for c in features if c not in CAT_COLS]),
    ])
    tr = train[train["season"] < Y]
    va = train[train["season"] == Y].reset_index(drop=True)
    Xtr = pre.fit_transform(tr[features])
    Xva = pre.transform(va[features])
    ytr = tr[TARGET].to_numpy()
    y = va[TARGET].to_numpy(dtype=float)
    denom = y.mean() * (1 - y.mean())
    c = float(ytr.mean())
    anc = va[PREV1].fillna(c).to_numpy(dtype=float) - c

    p_h = np.load(os.path.join(CACHE, f"{Y}_hgb_seed42.npy"))
    p_l = np.load(os.path.join(CACHE, f"{Y}_lr_seed42.npy"))
    p_c = np.load(os.path.join(CACHE, f"{Y}_{CB_TAG}.npy"))
    cur = W_HGB * p_h + W_CB * p_c + W_LR * p_l

    def sc(p):
        return max(0.0, 100000 * (1 - ((np.clip(p, 0, 1) - y) ** 2).mean() / denom))

    ref = sc(np.clip(cur + LAM * anc, 0, 1))
    print(f"fold {Y} | 라이브러리 {args.lib} | 학습 {len(tr):,}")
    print(f"기준 = 현재 최적 혼합 (hgb {W_HGB} / cb {W_CB} / lr {W_LR}, "
          f"lam {LAM}) = {ref:.2f}\n")

    ntrees = list(range(args.step, args.iters + 1, args.step))
    for depth in depths:
        tag = f"{args.lib}_d{depth}_it{args.iters}_s{args.seed}"
        path = os.path.join(CACHE, f"{Y}_{tag}.npz")
        t = time.time()
        if os.path.exists(path):
            z = np.load(path)
            preds = {int(k): z[k] for k in z.files}
            took = "캐시"
        else:
            if args.lib == "lgb":
                import lightgbm as lgb
                m = lgb.LGBMClassifier(
                    n_estimators=args.iters, learning_rate=args.lr,
                    num_leaves=2 ** depth, max_depth=depth,
                    min_child_samples=1000, reg_lambda=1.0,
                    subsample=1.0, colsample_bytree=1.0,
                    random_state=args.seed, verbose=-1, n_jobs=-1)
                m.fit(Xtr, ytr)
                preds = {n: m.predict_proba(Xva, num_iteration=n)[:, 1]
                         for n in ntrees}
            else:
                import xgboost as xgb
                m = xgb.XGBClassifier(
                    n_estimators=args.iters, learning_rate=args.lr,
                    max_depth=depth, min_child_weight=200, reg_lambda=1.0,
                    tree_method="hist", random_state=args.seed,
                    n_jobs=-1, eval_metric="logloss")
                m.fit(Xtr, ytr)
                preds = {n: m.predict_proba(Xva, iteration_range=(0, n))[:, 1]
                         for n in ntrees}
            np.savez_compressed(path, **{str(k): v for k, v in preds.items()})
            took = f"{time.time()-t:.0f}s"

        solo = [sc(preds[n]) for n in ntrees]
        j = int(np.argmax(solo))
        best_p = preds[ntrees[j]]
        mixes = [sc(np.clip((1 - w) * cur + w * best_p + LAM * anc, 0, 1))
                 for w in NEW_W]
        print(f"--- {args.lib} depth {depth} [{took}] ---")
        print("     " + "  ".join(f"{n:6d}" for n in ntrees))
        print("단독 " + "  ".join(f"{v:6.1f}" for v in solo))
        print(f"  최적 {ntrees[j]}그루 단독 {solo[j]:.2f}  "
              f"(cb 단독 {sc(p_c):.2f} / hgb {sc(p_h):.2f})")
        print("  혼합 기여 " + "  ".join(
            f"w{w:.2f} {v:7.2f}({v-ref:+6.2f})" for w, v in zip(NEW_W, mixes)))
        print(flush=True)

    print("""읽는 법.
  단독이 CatBoost 보다 낮아도 혼합에서 이길 수 있다 — 로지스틱이 단독 164.59 인데
  혼합 +8.24 였다 (4-8). **혼합 기여**만 본다.
  LightGBM 이 HGB 와 비슷하게 나오면 "대칭 트리가 원인"이라는 4-14 의 해석이
  강화되고, LightGBM 도 크게 이기면 그 해석이 틀린 것이다.""")


if __name__ == "__main__":
    main()
