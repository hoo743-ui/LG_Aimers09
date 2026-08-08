r"""모델 계열 혼합 — survey.py 가 찾은 길을 정식 정밀도로 판다.

배경. survey.py 화면 결과(축소 정밀도, 1시드):

    구조 축 6종      전부 음수 — 확정 설정이 국소 최적
    rf     단독 +1.47      혼합 w=0.3 +40.67   폴드 부호 전 가중치 일치
    logreg 단독 -245.22    혼합 w=0.2 +38.25
    et     단독 -81.43     혼합 w=0.1 +15.54

단독으로 지는 모델이 섞으면 이긴다. 이득의 정체는 중심 보정이 아니라 **오차
상쇄**로 확인했다 — 혼합 전후로 중심 편차가 거의 안 변한다. 그래서 2025 국면이
어느 쪽으로 움직여도 비교적 안전하다.

여기서는 정식 정밀도(lr=0.02, n_iter=1100)로 다시 재고, 3자 혼합까지 본다.
로지스틱은 선형 제약으로, RF 는 분산 축소로 실패하므로 실패 방식이 서로 달라
합쳤을 때 각각보다 나을 수 있다.

설계. 폴드당 모델을 **한 번만** 학습하고 예측을 캐시한다. 가중치 격자는 그
위에서 공짜로 훑는다 — 격자가 커져도 학습 비용은 그대로다.

주의. 격자에서 최고점을 골라 쓰면 그 3폴드를 훔쳐보는 셈이다. **부호가 일치
하는 것 중 보수적인 가중치를 고를 것.** 4-6 이 경고한 그대로다.

    .\.venv\Scripts\python.exe blend_test.py
    .\.venv\Scripts\python.exe blend_test.py --seeds 2
"""
import argparse
import itertools
import os
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

DATA_DIR = "./data"
ID = "row_id"
TARGET = "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state"]
FOLDS = [2021, 2022, 2024]
# 확정 설정 (README 5장)
LR, LEAVES, MIN_LEAF, L2, N_ITER = 0.02, 10, 1000, 1.0, 1100


def add_derived(df):
    """final_train.py / script.py 와 같은 정의 (4-7)."""
    df = df.copy()
    df["same_hand"] = (df["pitcher_hand"] == df["batter_hand"]).astype(int)
    return df


def score_of(y, p, base):
    return max(0.0, 100000 * (1 - ((p - y) ** 2).mean() / base))


def make_hgb(features, seed):
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                               unknown_value=-1), CAT_COLS),
        ("num", "passthrough", [c for c in features if c not in CAT_COLS]),
    ])
    return Pipeline([("pre", pre), ("clf", HistGradientBoostingClassifier(
        max_iter=N_ITER, learning_rate=LR, max_leaf_nodes=LEAVES,
        min_samples_leaf=MIN_LEAF, l2_regularization=L2,
        early_stopping=False, random_state=seed))])


def make_rf(features, seed):
    """규제를 세게 건 RF. 1회차 베이스라인(depth 10)이 415.57 이었던 건 RF 가
    나빠서가 아니라 규제가 없어 과신했기 때문이다 — min_samples_leaf 를 크게
    잡으면 튜닝된 HGB 와 대등하다 (survey.py 에서 +1.47)."""
    num = [c for c in features if c not in CAT_COLS]
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                               unknown_value=-1), CAT_COLS),
        ("num", SimpleImputer(strategy="median", add_indicator=True), num),
    ])
    return Pipeline([("pre", pre), ("clf", RandomForestClassifier(
        n_estimators=200, min_samples_leaf=2000, max_features=0.4,
        n_jobs=-1, random_state=seed))])


def make_lr(features, seed=0):
    """결측은 중앙값으로 메우고 결측 표시를 따로 준다 — HGB 는 결측을 분기로
    직접 배우지만 선형 모형은 그걸 못 한다."""
    num = [c for c in features if c not in CAT_COLS]
    return Pipeline([
        ("pre", ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=0.001),
             CAT_COLS),
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median", add_indicator=True)),
                ("sc", StandardScaler()),
            ]), num),
        ])),
        ("clf", LogisticRegression(C=0.1, max_iter=200, solver="lbfgs")),
    ])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=1,
                   help="HGB/RF 시드 수. 폴드당 그만큼 더 학습한다")
    p.add_argument("--folds", default=",".join(map(str, FOLDS)))
    p.add_argument("--top", type=int, default=18)
    p.add_argument("--cache", default="./.blendcache",
                   help="폴드별 예측을 npy 로 저장해 재실행을 싸게 만든다. "
                        "학습이 12분인데 출력 단계에서 잘리면 전부 날아가므로 "
                        "도구 쪽에서 막는다. 지우면 다시 학습한다")
    return p.parse_args()


def main():
    args = parse_args()
    folds = [int(f) for f in args.folds.split(",")]

    test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig",
                            nrows=0).columns
    cols = [c for c in test_cols if c != ID]
    train = add_derived(pd.read_csv(f"{DATA_DIR}/train.csv",
                                    encoding="utf-8-sig",
                                    usecols=cols + [TARGET]))
    features = cols + ["same_hand"]
    print(f"{len(train):,} 행 | 피처 {len(features)}개 | "
          f"lr={LR} n_iter={N_ITER} seeds={args.seeds}")
    print(f"폴드 {folds}\n")

    # ---- 폴드당 한 번만 학습하고 예측을 캐시 ----
    cache = {}          # fold -> dict(name -> pred), y, base
    for Y in folds:
        tr = train[train["season"] < Y]
        va = train[train["season"] == Y]
        y = va[TARGET].to_numpy()
        base = y.mean() * (1 - y.mean())
        os.makedirs(args.cache, exist_ok=True)
        preds = {}
        for name, maker, n_seed in (("hgb", make_hgb, args.seeds),
                                    ("rf", make_rf, args.seeds),
                                    ("lr", make_lr, 1)):
            # 캐시는 **시드별로** 둔다. 시드를 하나 늘릴 때 이미 학습한 것을
            # 다시 돌리지 않기 위해서다.
            t = time.time()
            acc, n_new = np.zeros(len(va)), 0
            for s in range(42, 42 + n_seed):
                path = os.path.join(args.cache, f"{Y}_{name}_seed{s}.npy")
                if os.path.exists(path):
                    acc += np.load(path)
                    continue
                m = maker(features, s)
                m.fit(tr[features], tr[TARGET])
                p = m.predict_proba(va[features])[:, 1]
                np.save(path, p)
                acc += p
                n_new += 1
            preds[name] = acc / n_seed
            tag = "캐시" if n_new == 0 else f"{time.time()-t:.0f}s, 신규 {n_new}"
            print(f"  fold {Y} {name:3s} {score_of(y, preds[name], base):9.2f}"
                  f"   [{tag}]", flush=True)
        cache[Y] = (preds, y, base)

    ref = float(np.mean([score_of(cache[Y][1], cache[Y][0]["hgb"], cache[Y][2])
                         for Y in folds]))
    print(f"\nhgb 단독 3폴드 평균 {ref:.2f} — 이게 기준이다\n")

    # ---- 가중치 격자 ----
    # hgb 비중을 0.5 아래로는 내리지 않는다. 검증된 주력을 소수파로 만들면
    # 폴드 3개로 고른 가중치의 위험이 그대로 커진다.
    grid = []
    steps = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    for w_rf, w_lr in itertools.product(steps, steps):
        w_h = 1.0 - w_rf - w_lr
        if w_h < 0.5 - 1e-9:
            continue
        per = []
        for Y in folds:
            p, y, base = cache[Y][0], cache[Y][1], cache[Y][2]
            mix = w_h * p["hgb"] + w_rf * p["rf"] + w_lr * p["lr"]
            per.append(score_of(y, mix, base))
        base_per = [score_of(cache[Y][1], cache[Y][0]["hgb"], cache[Y][2])
                    for Y in folds]
        d = [a - b for a, b in zip(per, base_per)]
        grid.append((w_h, w_rf, w_lr, float(np.mean(per)),
                     float(np.mean(d)), d, all(x > 0 for x in d)))

    grid.sort(key=lambda g: -g[4])
    print(f"{'hgb':>6}{'rf':>6}{'lr':>6}{'평균':>10}{'vs hgb':>9}"
          f"   {'폴드별':>26}  부호")
    print("-" * 74)
    for w_h, w_rf, w_lr, m, d, per, ok in grid[:args.top]:
        print(f"{w_h:6.2f}{w_rf:6.2f}{w_lr:6.2f}{m:10.2f}{d:+9.2f}   "
              + " ".join(f"{x:+8.2f}" for x in per)
              + ("  일치" if ok else "  ★엇갈림"))

    best_ok = next((g for g in grid if g[6]), None)
    if best_ok:
        w_h, w_rf, w_lr, m, d, per, _ = best_ok
        print(f"\n부호 일치 중 최고(3폴드 평균): hgb {w_h:.2f} / rf {w_rf:.2f} / "
              f"lr {w_lr:.2f}  →  {d:+.2f}")

    # ---- 마지막 폴드 기준으로 다시 본다 ----
    # 3폴드 평균은 이 경우 구조적으로 과대평가한다. 폴드마다 학습 시즌 수가
    # 다르고(2021 은 2시즌, 2024 는 5시즌) RF 의 이점이 시즌이 쌓일수록
    # 단조 감소하기 때문이다. 실제 평가는 6시즌이라 어느 폴드보다도 많다.
    last = folds[-1]
    print(f"\n=== {last} 폴드 기준 재정렬 (학습 시즌이 가장 많아 실제 조건에 "
          f"제일 가깝다) ===")
    ok_only = [g for g in grid if g[6]]
    ok_only.sort(key=lambda g: -g[5][-1])
    print(f"{'hgb':>6}{'rf':>6}{'lr':>6}{'3폴드평균':>10}{f'{last}':>9}"
          f"   {'폴드별':>26}")
    print("-" * 66)
    for w_h, w_rf, w_lr, m, d, per, _ in ok_only[:12]:
        print(f"{w_h:6.2f}{w_rf:6.2f}{w_lr:6.2f}{d:+10.2f}{per[-1]:+9.2f}   "
              + " ".join(f"{x:+8.2f}" for x in per))
    if not ok_only:
        print("  부호가 일치하는 조합이 없다")

    # ---- 2자 혼합만 따로 ----
    # 배선 비용이 계열마다 크게 다르다. 로지스틱은 계수 몇백 개라 pkl 위험이
    # 사실상 없지만, RF 200그루는 직렬화 형식이 HGB 와 달라 6-4(낮은 numpy 에서
    # pkl 로드 실패)를 처음부터 다시 검증해야 한다. 이득이 비슷하면 로지스틱만
    # 쓰는 쪽이 훨씬 싸다.
    for who, idx in (("hgb+lr (rf 제외)", 2), ("hgb+rf (lr 제외)", 1)):
        rows = [g for g in grid if g[3 - idx] == 0.0]
        rows.sort(key=lambda g: -g[4])
        print(f"\n--- {who} ---")
        print(f"{'hgb':>6}{'rf':>6}{'lr':>6}{'3폴드평균':>10}{f'{last}':>9}"
              f"   {'폴드별':>26}  부호")
        for w_h, w_rf, w_lr, m, d, per, ok in rows[:6]:
            print(f"{w_h:6.2f}{w_rf:6.2f}{w_lr:6.2f}{d:+10.2f}{per[-1]:+9.2f}   "
                  + " ".join(f"{x:+8.2f}" for x in per)
                  + ("  일치" if ok else "  ★엇갈림"))
    print("""
격자에서 최고점을 그대로 쓰면 이 3폴드를 훔쳐보는 셈이다. 부호가 일치하는 것
중에서 보수적인(주력 비중이 큰) 가중치를 고를 것 — 4-6 이 경고한 지점이다.""")


if __name__ == "__main__":
    main()
