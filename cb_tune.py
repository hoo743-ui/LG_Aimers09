r"""CatBoost 하이퍼파라미터 튜닝 — 한 번도 안 한 작업.

4-14 에서 CatBoost 를 채택했지만 설정은 `depth=6 / l2=10 / lr=0.02 / iters=1100`
으로 HGB 쪽에서 대충 옮긴 값이다. HGB 는 4-1 에서 잎·반복수·규제를 전부 훑어
최적점을 찾았는데 CatBoost 는 그 작업을 안 했다.

**정정 하나.** `depth=4` 를 시험해 나쁘다고 기록했지만, 그건 ID 를 범주형으로
선언한 **망가진 설정**에서 잰 것이다 (539.89 vs 535.45, 둘 다 나쁜 쪽).
제대로 된 설정에서 depth 는 미탐색이다.

반복수는 공짜로 얻는다. CatBoost 의 `predict_proba(ntree_end=n)` 이 앞 n 그루만
쓴 예측을 준다 — 4-1 에서 HGB 에 `staged_predict_proba` 를 쓴 것과 같다.
한 번 학습하고 곡선 전체를 뽑으므로 격자에서 반복수 축이 빠진다.

판단은 **혼합 기여**로 한다. 현재 확정 구성이 기준이다 —
`hgb 0.30 / catboost 0.60 / lr 0.10`, `lam 0.03` (LB 875.66).

    .\.venv\Scripts\python.exe cb_tune.py --depths 4,5,6,7
    .\.venv\Scripts\python.exe cb_tune.py --depths 6 --l2 3,10,30
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
W_HGB, W_CB, W_LR, LAM = 0.30, 0.60, 0.10, 0.03      # 4-14 확정
STR_COLS = ["top_bottom", "game_type", "base_state",
            "pitcher_hand", "batter_hand"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, default=2024)
    p.add_argument("--depths", default="6")
    p.add_argument("--l2", default="10")
    p.add_argument("--iters", type=int, default=1600)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--step", type=int, default=200)
    p.add_argument("--rsm", type=float, default=1.0,
                   help="분할당 피처 샘플링 비율. CatBoost 기본 1.0(전부)이라 "
                        "미탐색 축이다. 낮추면 트리 간 다양성이 늘고 규제 효과도 난다")
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
    l2s = [float(v) for v in args.l2.split(",")]

    test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig",
                            nrows=0).columns
    base_cols = [c for c in test_cols if c != ID]
    train = add_derived(pd.read_csv(f"{DATA_DIR}/train.csv",
                                    encoding="utf-8-sig",
                                    usecols=base_cols + [TARGET]))
    features = base_cols + ["same_hand"]
    for c in STR_COLS:
        train[c] = train[c].astype(str)

    tr = train[train["season"] < Y]
    va = train[train["season"] == Y].reset_index(drop=True)
    y = va[TARGET].to_numpy(dtype=float)
    denom = y.mean() * (1 - y.mean())
    c = float(tr[TARGET].mean())
    anc = va[PREV1].fillna(c).to_numpy(dtype=float) - c
    p_h = np.load(os.path.join(CACHE, f"{Y}_hgb_seed42.npy"))
    p_l = np.load(os.path.join(CACHE, f"{Y}_lr_seed42.npy"))

    def sc(p):
        return max(0.0, 100000 * (1 - ((np.clip(p, 0, 1) - y) ** 2).mean() / denom))

    def mixed(p_cb):
        return sc(np.clip(W_HGB * p_h + W_CB * p_cb + W_LR * p_l + LAM * anc,
                          0, 1))

    ref_cb = np.load(os.path.join(
        CACHE, f"{Y}_cb_d6_l210_it1100_noid_seed42.npy"))
    ref = mixed(ref_cb)
    print(f"fold {Y} | 학습 {len(tr):,} | 검증 {len(va):,}")
    print(f"기준 = 현재 확정 구성 (d6/l2=10/it1100) 혼합 {ref:.2f}  "
          f"[단독 {sc(ref_cb):.2f}]  — LB 875.66\n")

    va_pool = Pool(va[features], cat_features=STR_COLS)
    ntrees = list(range(args.step, args.iters + 1, args.step))
    best = []
    for depth in depths:
        for l2 in l2s:
            tag = (f"cbt_d{depth}_l2{int(l2)}_it{args.iters}_s{args.seed}"
                   + ("" if args.rsm >= 1.0 else f"_rsm{int(args.rsm*100)}"))
            path = os.path.join(CACHE, f"{Y}_{tag}.npz")
            t = time.time()
            if os.path.exists(path):
                z = np.load(path)
                preds = {int(k): z[k] for k in z.files}
                took = "캐시"
            else:
                m = CatBoostClassifier(
                    iterations=args.iters, depth=depth, learning_rate=args.lr,
                    l2_leaf_reg=l2, loss_function="Logloss",
                    random_seed=args.seed, verbose=0, allow_writing_files=False,
                    rsm=args.rsm)
                m.fit(Pool(tr[features], tr[TARGET], cat_features=STR_COLS))
                preds = {n: m.predict_proba(va_pool, ntree_end=n)[:, 1]
                         for n in ntrees}
                np.savez_compressed(path, **{str(k): v for k, v in preds.items()})
                took = f"{time.time()-t:.0f}s"

            solo = [sc(preds[n]) for n in ntrees]
            mix = [mixed(preds[n]) for n in ntrees]
            j = int(np.argmax(mix))
            best.append((depth, l2, ntrees[j], solo[j], mix[j], mix[j] - ref))
            print(f"--- depth {depth} / l2 {l2:g} [{took}] ---")
            print("    " + "  ".join(f"{n}" for n in ntrees))
            print("단독" + "".join(f"{v:8.1f}" for v in solo))
            print("혼합" + "".join(f"{v:8.1f}" for v in mix))
            print(f"  → 최적 {ntrees[j]}그루  단독 {solo[j]:.2f}  "
                  f"혼합 {mix[j]:.2f}  ({mix[j]-ref:+.2f})\n", flush=True)

    best.sort(key=lambda r: -r[4])
    print("=== 요약 (혼합 기준) ===")
    print(f"{'depth':>6}{'l2':>6}{'그루':>7}{'단독':>10}{'혼합':>10}{'vs 기준':>9}")
    for d, l2, n, s, m, dd in best:
        print(f"{d:6d}{l2:6g}{n:7d}{s:10.2f}{m:10.2f}{dd:+9.2f}")
    print("""
곡선 최적점은 검증셋에 맞춘 낙관 편향이 있다 (4-6). 여기서 고른 설정은
반드시 3폴드로 다시 확인할 것. 그리고 제출 기회를 쓰려면 로컬 +25~30 이
필요하다 — 전달률이 그 아래에서 노이즈에 묻힌다 (1장).""")


if __name__ == "__main__":
    main()
