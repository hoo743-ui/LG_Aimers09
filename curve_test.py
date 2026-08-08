r"""반복수 곡선을 **현재 파이프라인 기준으로** 다시 그린다.

왜. 2024 최적 축소계수가 a=1.128 이다 — 예측을 13% 벌려야 이득이라는 뜻이고,
부스팅에서 **과소확신은 학습 부족의 전형적 신호**다. 그리고 error_map.py 는
남은 격차가 전부 해상도라고 말한다. 두 사실이 같은 방향을 가리킨다.

그런데 `n_iter=1100` / `leaves=10` / `min_leaf=1000` 은 4-1 에서 정한 값이고,
**그때는 중심 보정(4-9)도 계열 혼합(4-8)도 없었다.** 당시 점수는 중심 오차가
지배했으므로, 복잡한 모델이 중심을 어긋나게 만들면 그대로 점수 손실이었다.
즉 "복잡도를 낮춰라"는 결론이 실제로는 "중심 오차를 줄여라"였을 수 있다.
지금은 중심을 따로 잡으므로 최적점이 이동했을 가능성이 있다.

무엇을 재는가. 한 번 학습하고 `staged_predict_proba` 로 반복수별 예측을 뽑아,
**제출 파이프라인 그대로**(혼합 -> 중심 보정) 채점한다. 세 곡선을 같이 낸다.

    raw     HGB 단독
    +mix    계열 혼합까지 (rf/lr 캐시 사용)
    +full   중심 보정까지 = 실제 제출 구성

raw 의 최적점과 full 의 최적점이 다르면, 지금 반복수는 **옛 목적함수에 맞춰진
값**이라는 뜻이다.

주의. 곡선 최적점은 검증셋에 맞춘 낙관 편향이 있다 (4-6). 절대값이 아니라
**최적점의 위치**를 보는 도구다. 위치가 바뀌면 3폴드 x 2시드로 확인할 것.

    .\.venv\Scripts\python.exe curve_test.py --max-iter 2400 --leaves 10
    .\.venv\Scripts\python.exe curve_test.py --max-iter 1600 --leaves 20
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
CACHE = "./.blendcache"
ID = "row_id"
TARGET = "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state"]
PREV1 = "asof_pitcher_prev1_game_success_rate"
VAL = 2024
W_RF, W_LR, LAM = 0.20, 0.05, 0.03      # 4-8 / 4-9 확정


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max-iter", type=int, default=2400)
    p.add_argument("--leaves", type=int, default=10)
    p.add_argument("--min-leaf", type=int, default=1000)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--l2", type=float, default=1.0)
    p.add_argument("--every", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def add_derived(df):
    df = df.copy()
    df["same_hand"] = (df["pitcher_hand"] == df["batter_hand"]).astype(int)
    return df


def main():
    args = parse_args()
    test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig",
                            nrows=0).columns
    base_cols = [c for c in test_cols if c != ID]
    train = add_derived(pd.read_csv(f"{DATA_DIR}/train.csv",
                                    encoding="utf-8-sig",
                                    usecols=base_cols + [TARGET]))
    features = base_cols + ["same_hand"]
    tr = train[train["season"] < VAL]
    va = train[train["season"] == VAL].reset_index(drop=True)
    y = va[TARGET].to_numpy(dtype=float)
    denom = y.mean() * (1 - y.mean())
    c = float(tr[TARGET].mean())
    anc = va[PREV1].fillna(c).to_numpy(dtype=float) - c

    # 혼합용 예측은 반복수와 무관하므로 캐시를 그대로 쓴다
    def cached(name):
        p = os.path.join(CACHE, f"{VAL}_{name}_seed42.npy")
        if not os.path.exists(p):
            raise SystemExit(f"{p} 없음 — blend_test.py 를 먼저 돌릴 것")
        return np.load(p)
    p_rf, p_lr = cached("rf"), cached("lr")

    print(f"설정: lr={args.lr} leaves={args.leaves} min_leaf={args.min_leaf} "
          f"l2={args.l2} max_iter={args.max_iter}")
    print(f"학습 {len(tr):,} | 검증 {len(va):,} ({VAL})")
    print(f"혼합 rf {W_RF} / lr {W_LR} | 중심 lam {LAM}\n", flush=True)

    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                               unknown_value=-1), CAT_COLS),
        ("num", "passthrough", [c2 for c2 in features if c2 not in CAT_COLS]),
    ])
    clf = HistGradientBoostingClassifier(
        max_iter=args.max_iter, learning_rate=args.lr,
        max_leaf_nodes=args.leaves, min_samples_leaf=args.min_leaf,
        l2_regularization=args.l2, early_stopping=False,
        random_state=args.seed)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    t = time.time()
    pipe.fit(tr[features], tr[TARGET])
    print(f"학습 완료 [{time.time()-t:.0f}s] — 곡선 채점 중 ...", flush=True)

    Xva = pipe.named_steps["pre"].transform(va[features])
    w_h = 1.0 - W_RF - W_LR

    def sc(p):
        return max(0.0, 100000 * (1 - ((np.clip(p, 0, 1) - y) ** 2).mean() / denom))

    rows = []
    t = time.time()
    for i, proba in enumerate(clf.staged_predict_proba(Xva), start=1):
        if i % args.every and i != args.max_iter:
            continue
        p = proba[:, 1]
        mix = w_h * p + W_RF * p_rf + W_LR * p_lr
        full = mix + LAM * anc
        rows.append((i, sc(p), sc(mix), sc(full), float(p.std()),
                     float(p.mean() - y.mean())))
    print(f"채점 완료 [{time.time()-t:.0f}s]\n")

    arr = np.array([(r[0], r[1], r[2], r[3]) for r in rows])
    best_raw = int(arr[np.argmax(arr[:, 1]), 0])
    best_mix = int(arr[np.argmax(arr[:, 2]), 0])
    best_full = int(arr[np.argmax(arr[:, 3]), 0])

    print(f"{'iter':>7}{'raw':>10}{'+mix':>10}{'+full':>10}"
          f"{'예측sd':>9}{'중심편차':>10}")
    print("-" * 56)
    for i, s0, s1, s2, sd, dev in rows:
        mark = ""
        if i == best_raw:
            mark += " ←raw최적"
        if i == best_full:
            mark += " ←full최적"
        print(f"{i:7d}{s0:10.2f}{s1:10.2f}{s2:10.2f}{sd:9.4f}{dev:+10.4f}{mark}")

    print(f"\n최적 반복수:  raw {best_raw}   +mix {best_mix}   +full {best_full}")
    print(f"현재 확정값 1100 (전체 재학습 시 1210)")
    if best_full > best_raw * 1.15:
        print("\n★ full 최적점이 raw 보다 뚜렷이 뒤에 있다. 중심을 따로 잡으면서")
        print("  더 오래 학습해도 되게 됐다는 뜻이다 — 4-1 의 반복수는 옛 목적함수")
        print("  기준값이므로 3폴드 x 2시드로 다시 확인할 것.")
    elif best_full < best_raw * 0.85:
        print("\n★ full 최적점이 raw 보다 앞이다. 혼합/보정이 이미 역할을 해서")
        print("  HGB 는 덜 학습해도 된다는 뜻이다.")
    else:
        print("\n두 최적점이 비슷하다 — 반복수는 목적함수 변화에 둔감하다.")
    print("""
곡선 최적점은 검증셋에 맞춘 낙관 편향이 있다 (4-6). 절대값이 아니라 **위치**를
보는 도구다. 위치가 바뀌었으면 3폴드 x 2시드로 확인할 것.""")


if __name__ == "__main__":
    main()
