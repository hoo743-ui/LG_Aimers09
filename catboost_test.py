r"""CatBoost — ordered target statistics 가 ID 한계를 넘는가.

왜 다른가. `pitcher_id` 는 데뷔 순 정수라 트리의 순서형 분할로는 사실상
**데뷔 코호트**만 뽑힌다 (4-7). CatBoost 는 고차원 범주형을 **타깃 통계**로
인코딩하되 행 순서를 지켜 누수를 막는다(ordered boosting). 그 기전이 트리
**안에서** 작동한다는 점이 핵심이다.

embed_test.py 에서 기각된 것은 **선형** 모델의 one-hot 이었다 (4-12). 계수
하나로 선수 효과를 표현하는 방식이라 상황과 상호작용하지 못했다. CatBoost 의
타깃 통계는 분할 조건으로 쓰이므로 다른 피처와 결합될 수 있다.

`cat_features` 로 선언해야 그 기전이 켜진다. 숫자로 넣으면 HGB 와 다를 게 없다.

**RF 실패에서 배운 검증을 처음부터 넣는다** (4-8). RF 는 2024 폴드가 양수여서
채택했는데 실전에서 -9.45 였고, 원인은 단독 성적이 학습 시즌에 따라 단조
감소한다는 것이었다. 그래서 여기서는 **학습량 추세를 같이 낸다** — 감소
방향이면 2024 가 양수여도 채택하지 않는다.

판단은 단독이 아니라 **혼합 기여**로 한다. 로지스틱이 단독 164.59 인데 혼합에서
+8.24 를 냈다 (4-8).

    .\.venv\Scripts\python.exe catboost_test.py --folds 2024
    .\.venv\Scripts\python.exe catboost_test.py --folds 2020,2021,2022,2023,2024 --raw
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

DATA_DIR = "./data"
CACHE = "./.blendcache"
ID = "row_id"
TARGET = "control_success"
PREV1 = "asof_pitcher_prev1_game_success_rate"
W_LR, LAM = 0.10, 0.03            # 4-8 / 4-9 확정
BLEND_W = [0.05, 0.10, 0.15, 0.20, 0.30]

# 범주형으로 선언할 컬럼. 고차원 ID 를 여기 넣는 것이 이 실험의 전부다.
CAT_FEATURES = ["top_bottom", "game_type", "base_state",
                "pitcher_hand", "batter_hand",
                "pitcher_id", "batter_id",
                "pitcher_team_id", "batter_team_id"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--folds", default="2024")
    p.add_argument("--iters", type=int, default=1100)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--l2", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--raw", action="store_true",
                   help="혼합 없이 단독만, 클리핑 없는 BSS. 학습량 추세용")
    p.add_argument("--no-id-cats", action="store_true",
                   help="고차원 ID 를 범주형에서 빼고 숫자로 넘긴다. CatBoost 의 "
                        "타깃 통계 인코딩이 해를 끼치는지 분리하는 대조군 — "
                        "4-12 에서 학습된 선수별 파라미터가 asof_* 를 못 넘는 것을 "
                        "이미 확인했고, 평가 시즌은 투구의 20% 가 신인이다")
    return p.parse_args()


def add_derived(df):
    df = df.copy()
    df["same_hand"] = (df["pitcher_hand"] == df["batter_hand"]).astype(int)
    return df


def score_of(y, p, denom, clip=True):
    v = 100000 * (1 - ((np.clip(p, 0, 1) - y) ** 2).mean() / denom)
    return max(0.0, v) if clip else v


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
    cats = [c for c in CAT_FEATURES if c in features]
    if args.no_id_cats:
        cats = [c for c in cats
                if c not in ("pitcher_id", "batter_id",
                             "pitcher_team_id", "batter_team_id")]
    # CatBoost 는 범주형을 문자열/정수로 받는다. 결측 없는 정수형이라 그대로 쓴다.
    for c in cats:
        train[c] = train[c].astype(str)
    print(f"{len(train):,} 행 | 피처 {len(features)}개 | "
          f"범주형 {len(cats)}개 {cats}")
    print(f"iters={args.iters} depth={args.depth} lr={args.lr} l2={args.l2}\n")

    for Y in folds:
        tr = train[train["season"] < Y]
        va = train[train["season"] == Y].reset_index(drop=True)
        y = va[TARGET].to_numpy(dtype=float)
        denom = y.mean() * (1 - y.mean())
        c = float(tr[TARGET].mean())
        anc = va[PREV1].fillna(c).to_numpy(dtype=float) - c

        # 폴드당 7분이라 재학습하면 가중치 격자를 못 훑는다. 예측을 캐시한다.
        tag = (f"cb_d{args.depth}_l2{int(args.l2)}_it{args.iters}"
               f"{'_noid' if args.no_id_cats else ''}_seed{args.seed}")
        path = os.path.join(CACHE, f"{Y}_{tag}.npy")
        t = time.time()
        if os.path.exists(path):
            p_cb = np.load(path)
            took = "캐시"
        else:
            m = CatBoostClassifier(
                iterations=args.iters, depth=args.depth, learning_rate=args.lr,
                l2_leaf_reg=args.l2, loss_function="Logloss",
                random_seed=args.seed, verbose=0, allow_writing_files=False)
            m.fit(Pool(tr[features], tr[TARGET], cat_features=cats))
            p_cb = m.predict_proba(Pool(va[features], cat_features=cats))[:, 1]
            np.save(path, p_cb)
            took = f"{time.time()-t:.0f}s"
        solo = score_of(y, p_cb, denom, clip=not args.raw)
        print(f"--- fold {Y} (학습 {len(tr):,}, {Y-2019}시즌) ---")
        print(f"  catboost 단독 {solo:9.2f}  예측sd {p_cb.std():.4f}  "
              f"중심편차 {p_cb.mean()-y.mean():+.4f}  [{time.time()-t:.0f}s]",
              flush=True)

        if args.raw:
            continue

        hp = os.path.join(CACHE, f"{Y}_hgb_seed42.npy")
        lp = os.path.join(CACHE, f"{Y}_lr_seed42.npy")
        if not (os.path.exists(hp) and os.path.exists(lp)):
            print("  (혼합 캐시 없음 — blend_test.py 를 먼저 돌릴 것)")
            continue
        p_h, p_l = np.load(hp), np.load(lp)
        print(f"  hgb 단독 {score_of(y, p_h, denom):9.2f}")

        # 현재 확정 구성: hgb 0.90 / lr 0.10 + 중심 보정
        cur = np.clip((1 - W_LR) * p_h + W_LR * p_l + LAM * anc, 0, 1)
        ref = score_of(y, cur, denom)
        print(f"  현재 구성 {ref:9.2f}  (hgb .90 / lr .10 / 중심 {LAM})")

        # catboost 를 얹는다. hgb 비중에서 덜어낸다.
        row = []
        for w in BLEND_W:
            mix = (1 - W_LR - w) * p_h + W_LR * p_l + w * p_cb
            row.append(score_of(y, np.clip(mix + LAM * anc, 0, 1), denom))
        print("  + catboost " + "  ".join(
            f"w{w:.2f} {v:8.2f}({v-ref:+6.2f})" for w, v in zip(BLEND_W, row)))
        print()

    print("""읽는 법.
  판단은 단독이 아니라 **혼합 기여**로 한다 — 로지스틱은 단독 164.59 인데 혼합에서
  +8.24 를 냈다 (4-8). 그리고 채택 전에 **학습량 추세**를 반드시 볼 것
  (--raw 로 5폴드). RF 는 2024 폴드가 양수라 채택했다가 실전에서 -9.45 였고,
  원인이 단조 감소 추세였다 (4-8). 채택 기준은 로컬 +15~20.""")


if __name__ == "__main__":
    main()
