r"""연도별 중심 편차가 안정적인지, 작년 보정을 올해에 써도 되는지 확인한다.

모델은 예측 평균이 실제보다 높게 나오는 경향이 있다(타깃 비율이 해마다 떨어지는데
누적형 asof 피처가 이를 뒤늦게 따라가기 때문). 2025 의 실제 비율은 알 수 없고
test.csv 로 계산하는 것은 규정 위반이므로, 쓸 수 있는 방법은 "작년에 관측된 보정을
올해에 그대로 적용" 하는 것뿐이다.

그게 통하려면 편차가 해마다 비슷해야 한다. 여러 해를 재서 확인한다.

각 연도 Y 마다:
    Y 이전 시즌으로 학습 → Y 예측
    - 원본 점수
    - 그 해 정답으로 맞춘 보정 (오라클, 실제로는 불가능한 상한선)
    - 전년도에서 구한 보정을 그대로 적용 (실제로 쓸 수 있는 방법)

사용법:
    .\.venv\Scripts\python.exe offset_test.py
"""
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

LR, LEAVES, MIN_LEAF, L2 = 0.02, 10, 1000, 1.0
N_ITER = 1000            # 폴드마다 최적반복을 고르면 오라클이 되므로 고정한다
YEARS = [2021, 2022, 2023, 2024]


def make_model(features):
    num = [c for c in features if c not in CAT_COLS]
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                               unknown_value=-1), CAT_COLS),
        ("num", "passthrough", num),
    ])
    clf = HistGradientBoostingClassifier(
        max_iter=N_ITER, learning_rate=LR, max_leaf_nodes=LEAVES,
        min_samples_leaf=MIN_LEAF, l2_regularization=L2,
        early_stopping=False, random_state=42,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def score(y, p, base):
    return max(0, 100000 * (1 - ((p - y) ** 2).mean() / base))


def fit_linear(y, p):
    """Brier 를 최소화하는 p' = a + b*p. 최소제곱해와 같다."""
    b, a = np.polyfit(p, y, 1)
    return float(a), float(b)


def main():
    test_cols = pd.read_csv(os.path.join(DATA_DIR, "test.csv"),
                            encoding="utf-8-sig", nrows=0).columns
    features = [c for c in test_cols if c != ID]
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"),
                        encoding="utf-8-sig", usecols=features + [TARGET])

    print(f"고정 설정: lr={LR} leaves={LEAVES} min_leaf={MIN_LEAF} "
          f"n_iter={N_ITER}\n")

    rows = []
    for Y in YEARS:
        tr = train[train["season"] < Y]
        va = train[train["season"] == Y]
        y = va[TARGET].to_numpy()
        r = y.mean()
        base = r * (1 - r)

        t = time.time()
        m = make_model(features)
        m.fit(tr[features], tr[TARGET])
        p = m.predict_proba(va[features])[:, 1]

        offset = float(p.mean() - r)
        a, b = fit_linear(y, p)
        rows.append({
            "Y": Y, "p_mean": float(p.mean()), "r": float(r), "offset": offset,
            "base": base, "raw": score(y, p, base),
            "a": a, "b": b, "y": y, "p": p,
            "sec": time.time() - t,
        })
        print(f"  {Y} 완료 [{rows[-1]['sec']:.0f}s]")

    print(f"\n{'연도':>6}{'예측평균':>10}{'실제':>9}{'편차':>9}"
          f"{'원본점수':>10}{'오라클보정':>11}{'절편a':>9}{'기울기b':>9}")
    print("-" * 74)
    for d in rows:
        p_cal = np.clip(d["a"] + d["b"] * d["p"], 0, 1)
        d["oracle"] = score(d["y"], p_cal, d["base"])
        print(f"{d['Y']:>6}{d['p_mean']:10.4f}{d['r']:9.4f}{d['offset']:+9.4f}"
              f"{d['raw']:10.2f}{d['oracle']:11.2f}{d['a']:9.4f}{d['b']:9.4f}")

    print("\n=== 전년도 보정을 그대로 적용했을 때 ===")
    print(f"{'연도':>6}{'원본':>10}{'단순편차빼기':>14}{'선형보정':>11}"
          f"{'오라클':>10}")
    print("-" * 53)
    for prev, cur in zip(rows, rows[1:]):
        # (1) 전년도 편차만큼 상수로 빼기
        p1 = np.clip(cur["p"] - prev["offset"], 0, 1)
        s1 = score(cur["y"], p1, cur["base"])
        # (2) 전년도에서 맞춘 선형식을 그대로 적용
        p2 = np.clip(prev["a"] + prev["b"] * cur["p"], 0, 1)
        s2 = score(cur["y"], p2, cur["base"])
        print(f"{cur['Y']:>6}{cur['raw']:10.2f}{s1:14.2f}{s2:11.2f}"
              f"{cur['oracle']:10.2f}")

    offs = [d["offset"] for d in rows]
    print(f"\n편차 추이: {[f'{o:+.4f}' for o in offs]}")
    print(f"  평균 {np.mean(offs):+.4f} | 표준편차 {np.std(offs):.4f} "
          f"| 최소 {min(offs):+.4f} 최대 {max(offs):+.4f}")
    print("\n판단 기준: '단순편차빼기'와 '선형보정'이 '원본'보다 꾸준히 높으면"
          " 2025 에도 적용할 만하다.")


if __name__ == "__main__":
    main()
