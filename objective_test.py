r"""목적함수 정렬 — Brier 로 직접 학습하면 나아지는가.

지금 HGB 는 **log-loss** 로 학습하는데 평가는 **Brier** 다. 둘 다 proper scoring
rule 이지만 오차 가중이 다르다 — log-loss 는 확신에 찬 오답을 훨씬 크게 벌하고,
Brier 는 제곱오차라 그렇지 않다. 신호가 희박해 예측이 0.5 근처에 몰려 있는
이 문제에서 둘의 차이가 어디로 나타나는지는 재봐야 안다.

`HistGradientBoostingRegressor(loss="squared_error")` 로 0/1 타깃을 회귀하면
**평가지표를 직접 최소화**한다. 나무 구조도 파라미터도 그대로 두고 손실만 바꾼다.

이 프로젝트에서 한 번도 안 해봤고 비용이 사실상 0 이다.

주의. 회귀라 예측이 [0,1] 을 벗어날 수 있어 clip 한다. 그리고 분류기가 아니므로
`predict_proba` 가 없다 — 제출 경로에 넣으려면 script.py 를 손봐야 하지만,
그건 이 실험이 통과한 뒤 문제다.

평가는 제출 파이프라인 그대로다 (혼합 -> 중심 보정). 캐시된 lr 예측을 쓴다.

    .\.venv\Scripts\python.exe objective_test.py --folds 2024
    .\.venv\Scripts\python.exe objective_test.py --folds 2021,2022,2024 --seeds 2
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

DATA_DIR = "./data"
CACHE = "./.blendcache"
ID = "row_id"
TARGET = "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state"]
PREV1 = "asof_pitcher_prev1_game_success_rate"
LR, LEAVES, MIN_LEAF, L2, N_ITER = 0.02, 10, 1000, 1.0, 1100
W_LR, LAM = 0.10, 0.03          # 4-8 / 4-9 확정


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--folds", default="2024")
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--n-iter", type=int, default=N_ITER)
    p.add_argument("--raw", action="store_true",
                   help="혼합·중심 보정 없이 단독만, **클리핑 없는 BSS** 로. "
                        "lr 캐시가 없는 폴드(2020/2023)까지 넣어 학습량 추세의 "
                        "모양을 본다 — 4-11 에서 2023 이 추세를 깬 전례가 있다")
    return p.parse_args()


def add_derived(df):
    df = df.copy()
    df["same_hand"] = (df["pitcher_hand"] == df["batter_hand"]).astype(int)
    return df


def make(features, kind, n_iter, seed):
    """kind 만 바꾸고 나무 구조·파라미터는 동일하게 둔다."""
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                               unknown_value=-1), CAT_COLS),
        ("num", "passthrough", [c for c in features if c not in CAT_COLS]),
    ])
    common = dict(max_iter=n_iter, learning_rate=LR, max_leaf_nodes=LEAVES,
                  min_samples_leaf=MIN_LEAF, l2_regularization=L2,
                  early_stopping=False, random_state=seed)
    est = (HistGradientBoostingClassifier(**common) if kind == "logloss"
           else HistGradientBoostingRegressor(loss="squared_error", **common))
    return Pipeline([("pre", pre), ("clf", est)])


def predict(m, X, kind):
    if kind == "logloss":
        return m.predict_proba(X)[:, 1]
    return np.clip(m.predict(X), 0.0, 1.0)


def score_of(y, p, denom):
    return max(0.0, 100000 * (1 - ((np.clip(p, 0, 1) - y) ** 2).mean() / denom))


def main():
    args = parse_args()
    folds = [int(f) for f in args.folds.split(",")]
    test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig",
                            nrows=0).columns
    base_cols = [c for c in test_cols if c != ID]
    train = add_derived(pd.read_csv(f"{DATA_DIR}/train.csv",
                                    encoding="utf-8-sig",
                                    usecols=base_cols + [TARGET]))
    features = base_cols + ["same_hand"]
    print(f"{len(train):,} 행 | 피처 {len(features)}개 | "
          f"n_iter={args.n_iter} seeds={args.seeds}")
    print(f"평가는 제출 파이프라인 그대로 (lr 혼합 {W_LR}, 중심 보정 {LAM})\n")

    print(f"{'폴드':>6}{'목적함수':>10}{'단독':>10}{'+혼합':>10}{'+중심':>10}"
          f"{'예측sd':>9}{'중심편차':>10}")
    print("-" * 65)
    summary = {}
    for Y in folds:
        tr = train[train["season"] < Y]
        va = train[train["season"] == Y].reset_index(drop=True)
        y = va[TARGET].to_numpy(dtype=float)
        denom = y.mean() * (1 - y.mean())
        c = float(tr[TARGET].mean())
        anc = va[PREV1].fillna(c).to_numpy(dtype=float) - c
        p_lr = None
        if not args.raw:
            lp = os.path.join(CACHE, f"{Y}_lr_seed42.npy")
            if not os.path.exists(lp):
                raise SystemExit(f"{lp} 없음 — blend_test.py 를 먼저 돌리거나 "
                                 f"--raw 로 단독 비교만 할 것")
            p_lr = np.load(lp)

        for kind in ("logloss", "brier"):
            t = time.time()
            acc = np.zeros(len(va))
            for s in range(42, 42 + args.seeds):
                m = make(features, kind, args.n_iter, s)
                m.fit(tr[features], tr[TARGET])
                acc += predict(m, va[features], kind)
            p = acc / args.seeds
            if args.raw:
                # 클리핑 없는 BSS — 2023 은 음수가 나오는데 그 자체가 정보다 (4-11)
                s_raw = 100000 * (1 - ((p - y) ** 2).mean() / denom)
                s_mix = s_full = s_raw
            else:
                s_raw = score_of(y, p, denom)
                mix = (1 - W_LR) * p + W_LR * p_lr
                s_mix = score_of(y, mix, denom)
                s_full = score_of(y, np.clip(mix + LAM * anc, 0, 1), denom)
            summary[(Y, kind)] = s_full
            print(f"{Y:>6}{kind:>10}{s_raw:10.2f}{s_mix:10.2f}{s_full:10.2f}"
                  f"{p.std():9.4f}{p.mean()-y.mean():+10.4f}  [{time.time()-t:.0f}s]",
                  flush=True)

    if len(folds) >= 1:
        d = [summary[(Y, "brier")] - summary[(Y, "logloss")] for Y in folds]
        print(f"\nbrier − logloss (최종 기준): "
              + "  ".join(f"{Y} {x:+.2f}" for Y, x in zip(folds, d))
              + f"   평균 {np.mean(d):+.2f}")
        if len(folds) >= 2:
            ok = all(x > 0 for x in d) or all(x < 0 for x in d)
            print(f"부호 {'일치' if ok else '★엇갈림'}")
    print("""
읽는 법. 채택 기준은 로컬 +15~20 이다 (4-8 의 RF 실패 이후 올렸다).
그보다 작으면 제출 기회를 쓰지 않는다 — 로컬 +4.54 가 LB −9.45 였다.""")


if __name__ == "__main__":
    main()
