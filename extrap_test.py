r"""폴드 평균 대신 **학습 시즌 수로 추세를 그려 2025 를 외삽한다.**

왜 평균이 틀린 집계인가. 폴드마다 학습 시즌 수가 다르고 실제 평가는 그 어느
폴드보다 많다.

    fold 2020 -> 1시즌   2021 -> 2   2022 -> 3   2023 -> 4   2024 -> 5
    실제 2025 -> 6시즌  (외삽 구간)

용량을 늘리는 변경은 데이터가 많을수록 유리해진다. 이 프로젝트 기록에 그 패턴이
남아 있다 — ID 복원이 2021 -9.6 / 2022 +13.5 / 2024 +21.6 (4-5), cat_only3 가
-16.6 / -14.2 / +11.2 로 둘 다 단조 증가다. **평균을 내면 이 추세가 뭉개진다.**

2023 과 2020 을 되살린다. 2023 을 뺀 이유는 "어떤 구성으로도 0점"이었는데
(4-6) 그건 `max(0, ...)` 클리핑 때문이지 폴드 탓이 아니다. **클리핑하지 않은
BSS 로 재면 구성 간 비교가 완벽히 가능하다.** 1/6 의 검증 신호를 버려온 셈이다.

지표. `bss = 100000 * (1 - Brier / (r(1-r)))`, **클리핑 없음**. 음수도 그대로
쓴다. 2023 은 음수가 나오는데 그 자체가 정보다.

방법 검증. 추세 외삽이 믿을 만한지 먼저 확인한다 — 앞의 폴드들로 적합해
마지막 폴드를 예측하고 실제와 대조한다. 그게 안 맞으면 6시즌 외삽도 못 믿는다.

    .\.venv\Scripts\python.exe extrap_test.py --folds 2020,2021,2022
    .\.venv\Scripts\python.exe extrap_test.py            # 전체
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
CACHE = "./.extrapcache"
ID = "row_id"
TARGET = "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state"]
LR, MIN_LEAF, L2 = 0.02, 1000, 1.0
FOLDS = [2020, 2021, 2022, 2023, 2024]

# (이름, leaves, n_iter). n_iter 은 2024 곡선에서 각 잎 수의 최적점을 썼다
# (curve_test.py). 잎이 많을수록 한 번의 갱신이 커서 최적 반복수가 줄어든다.
CONFIGS = [("L10", 10, 1200), ("L15", 15, 1100), ("L20", 20, 1000)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--folds", default=",".join(map(str, FOLDS)))
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--only", default=None)
    return p.parse_args()


def add_derived(df):
    df = df.copy()
    df["same_hand"] = (df["pitcher_hand"] == df["batter_hand"]).astype(int)
    return df


def bss(y, p):
    """클리핑하지 않은 BSS x 100000. 음수도 그대로 반환한다."""
    r = y.mean()
    return 100000.0 * (1 - ((p - y) ** 2).mean() / (r * (1 - r)))


def main():
    args = parse_args()
    folds = [int(f) for f in args.folds.split(",")]
    os.makedirs(CACHE, exist_ok=True)

    test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig",
                            nrows=0).columns
    base_cols = [c for c in test_cols if c != ID]
    train = add_derived(pd.read_csv(f"{DATA_DIR}/train.csv",
                                    encoding="utf-8-sig",
                                    usecols=base_cols + [TARGET]))
    features = base_cols + ["same_hand"]
    cfgs = CONFIGS
    if args.only:
        want = {s.strip() for s in args.only.split(",")}
        cfgs = [c for c in CONFIGS if c[0] in want]

    print(f"{len(train):,} 행 | 피처 {len(features)}개 | 시드 {args.seeds}")
    print("지표: 클리핑 없는 BSS x 100000\n")

    res = {}
    for name, leaves, n_iter in cfgs:
        for Y in folds:
            tr = train[train["season"] < Y]
            va = train[train["season"] == Y]
            y = va[TARGET].to_numpy(dtype=float)
            acc, made = np.zeros(len(va)), 0
            t = time.time()
            for s in range(42, 42 + args.seeds):
                path = os.path.join(CACHE, f"{Y}_{name}_seed{s}.npy")
                if os.path.exists(path):
                    acc += np.load(path)
                    continue
                pre = ColumnTransformer([
                    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                                           unknown_value=-1), CAT_COLS),
                    ("num", "passthrough",
                     [c for c in features if c not in CAT_COLS]),
                ])
                m = Pipeline([("pre", pre), ("clf",
                    HistGradientBoostingClassifier(
                        max_iter=n_iter, learning_rate=LR,
                        max_leaf_nodes=leaves, min_samples_leaf=MIN_LEAF,
                        l2_regularization=L2, early_stopping=False,
                        random_state=s))])
                m.fit(tr[features], tr[TARGET])
                p = m.predict_proba(va[features])[:, 1]
                np.save(path, p)
                acc += p
                made += 1
            v = bss(y, acc / args.seeds)
            res[(name, Y)] = v
            tag = "캐시" if made == 0 else f"{time.time()-t:.0f}s"
            print(f"  {name} fold {Y} (학습 {Y-2019}시즌, {len(tr):,} 행) "
                  f"bss {v:9.2f}  [{tag}]", flush=True)

    # ---- 추세 ----
    print(f"\n{'구성':>6}" + "".join(f"{f'{Y}({Y-2019}시즌)':>15}" for Y in folds))
    print("-" * (6 + 15 * len(folds)))
    for name, _, _ in cfgs:
        print(f"{name:>6}" + "".join(f"{res.get((name, Y), float('nan')):15.2f}"
                                     for Y in folds))

    if len(cfgs) < 2:
        return
    ref = cfgs[0][0]
    print(f"\n=== {ref} 대비 효과와 6시즌(2025) 외삽 ===")
    x = np.array([Y - 2019 for Y in folds], dtype=float)
    for name, _, _ in cfgs[1:]:
        d = np.array([res[(name, Y)] - res[(ref, Y)] for Y in folds])
        print(f"\n  {name} − {ref}")
        print("    " + "  ".join(f"{int(xi)}시즌 {di:+8.2f}"
                                 for xi, di in zip(x, d)))
        print(f"    단순 평균 {d.mean():+8.2f}")
        if len(x) >= 3:
            a, b = np.polyfit(x, d, 1)
            pred6 = a * 6 + b
            fit = a * x + b
            rms = float(np.sqrt(((d - fit) ** 2).mean()))
            print(f"    선형 추세 {a:+.2f}/시즌, 잔차 RMS {rms:.2f}"
                  f"  →  **6시즌 예측 {pred6:+.2f}**")
            # 방법 검증 — 앞 폴드로 적합해 마지막 폴드를 예측
            if len(x) >= 4:
                a2, b2 = np.polyfit(x[:-1], d[:-1], 1)
                hat = a2 * x[-1] + b2
                print(f"    [방법 검증] 앞 {len(x)-1}개로 {int(x[-1])}시즌 예측 "
                      f"{hat:+.2f}  실제 {d[-1]:+.2f}  오차 {hat-d[-1]:+.2f}")
    print("""
읽는 법.
  단순 평균은 학습량이 다른 폴드를 뭉갠 값이라 **용량 관련 변경에서는 편향된다**.
  선형 추세의 기울기가 뚜렷하면 6시즌 예측을 쓰고, 잔차 RMS 가 크거나 방법 검증
  오차가 크면 추세를 믿지 말고 평균으로 돌아갈 것.""")


if __name__ == "__main__":
    main()
