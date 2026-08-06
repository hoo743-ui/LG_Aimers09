r"""중심 편차를 **합법적으로** 줄일 수 있는지 잰다.

국면 보정(4-3)은 평가셋 전체의 prev1 평균을 봐서 규칙 위반이었다. 그게 LB 에서
+57.65 였으니, 중심은 이 문제에서 가장 큰 단일 레버다. 합법 대체를 다시 본다.

## 합법의 경계

금지된 것은 **평가셋의 다른 행을 보는 것**이다. 그러므로:

  ❌ 평가셋 prev1 평균으로 중심 이동          (4-3, 위반)
  ❌ 우리 예측값을 평가셋에서 평균내어 이동    (구조적으로 같은 위반)
  ✅ **학습 데이터만으로 정한 상수**를 더하기  (행 독립, 평가셋 미참조)

세 번째만 남는다. 즉 `p' = p + Δ` 이고 Δ 는 학습 시점에 확정돼 pkl 에 박히는 수다.

## 무엇을 재는가

모델은 학습 시절 관계에 닻을 내려 최근 국면보다 높게 예측한다. 그 편향이
**해마다 비슷하다면** 상수로 뺄 수 있다. 흔들린다면 못 뺀다.

정직하게 재려고 **워크포워드**로 한다 — fold Y 의 보정값은 Y 이전 폴드에서만
추정한다. 전체 폴드의 평균 편향을 알고 그걸 되먹이면 미래를 본 것이다.

비교 대상 셋:
  raw        보정 없음 (현재 제출본)
  wf_bias    이전 폴드들의 평균 편향만큼 뺀다
  wf_trend   시즌 성공률을 이전 시즌들로 선형 외삽해 그 값에 중심을 맞춘다

4-5 는 wf_trend 를 "폴드별 오차 0.002~0.032, 2021 은 410점 손실"로 기각했다.
여기서는 wf_bias 를 따로 본다 — 시즌 성공률이 아니라 **모델 편향**을 외삽하는
것이라 다른 양이다.

사용법:
    .\.venv\Scripts\python.exe center_test.py
"""
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
LR, LEAVES, MIN_LEAF, L2, N_ITER = 0.02, 10, 1000, 1.0, 1100
FOLDS = [2021, 2022, 2023, 2024]


def make_model(cols, seed=42):
    num = [c for c in cols if c not in CAT_COLS]
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                               unknown_value=-1), CAT_COLS),
        ("num", "passthrough", num),
    ])
    clf = HistGradientBoostingClassifier(
        max_iter=N_ITER, learning_rate=LR, max_leaf_nodes=LEAVES,
        min_samples_leaf=MIN_LEAF, l2_regularization=L2,
        early_stopping=False, random_state=seed)
    return Pipeline([("pre", pre), ("clf", clf)])


def score(y, p, base):
    return max(0.0, 100000 * (1 - ((np.clip(p, 0, 1) - y) ** 2).mean() / base))


def main():
    test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig",
                            nrows=0).columns
    cols = [c for c in test_cols if c != ID]
    train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                        usecols=cols + [TARGET])

    rates = train.groupby("season")[TARGET].mean()
    print("시즌별 실제 성공률")
    print(rates.round(4).to_string(), "\n")

    rec = []
    for Y in FOLDS:
        tr, va = train[train["season"] < Y], train[train["season"] == Y]
        y = va[TARGET].to_numpy()
        r = y.mean()
        base = r * (1 - r)
        t = time.time()
        m = make_model(cols)
        m.fit(tr[cols], tr[TARGET])
        p = m.predict_proba(va[cols])[:, 1]
        bias = p.mean() - r
        rec.append({"Y": Y, "r": r, "pred": p.mean(), "bias": bias,
                    "base": base, "p": p, "y": y})
        print(f"  fold {Y}  실제 {r:.4f}  예측평균 {p.mean():.4f}  "
              f"편향 {bias:+.4f}  raw {score(y, p, base):8.2f}  "
              f"[{time.time()-t:.0f}s]")

    print("\n=== 편향의 안정성 ===")
    b = np.array([d["bias"] for d in rec])
    print(f"  편향 {[f'{x:+.4f}' for x in b]}")
    print(f"  평균 {b.mean():+.4f}  표준편차 {b.std(ddof=1):.4f}  "
          f"부호 일치 {'예' if (b > 0).all() or (b < 0).all() else '아니오'}")

    print("\n=== 워크포워드 평가 (보정값은 이전 폴드에서만 추정) ===")
    print(f"{'fold':>6} {'raw':>9} {'wf_bias':>9} {'차이':>8} "
          f"{'wf_trend':>9} {'차이':>8}")
    tot = {"wf_bias": [], "wf_trend": []}
    for i, d in enumerate(rec):
        if i == 0:
            print(f"{d['Y']:>6} {score(d['y'], d['p'], d['base']):>9.2f} "
                  f"{'-':>9} {'-':>8} {'-':>9} {'-':>8}   (이전 폴드 없음)")
            continue
        s_raw = score(d["y"], d["p"], d["base"])

        # wf_bias — 이전 폴드들의 평균 편향만큼 뺀다
        delta = np.mean([rec[j]["bias"] for j in range(i)])
        s_bias = score(d["y"], d["p"] - delta, d["base"])

        # wf_trend — 이전 시즌 성공률을 선형 외삽해 그 값에 중심을 맞춘다
        past = rates[rates.index < d["Y"]]
        if len(past) >= 2:
            k, c = np.polyfit(past.index.values, past.values, 1)
            r_hat = k * d["Y"] + c
            s_trend = score(d["y"], d["p"] + (r_hat - d["p"].mean()), d["base"])
        else:
            s_trend = float("nan")

        tot["wf_bias"].append(s_bias - s_raw)
        tot["wf_trend"].append(s_trend - s_raw)
        print(f"{d['Y']:>6} {s_raw:>9.2f} {s_bias:>9.2f} {s_bias-s_raw:>+8.2f} "
              f"{s_trend:>9.2f} {s_trend-s_raw:>+8.2f}")

    print()
    for k, v in tot.items():
        v = [x for x in v if not np.isnan(x)]
        wins = sum(1 for x in v if x > 0)
        print(f"  {k:9s} 평균 {np.mean(v):+8.2f}   {wins}승 {len(v)-wins}패")

    print("\n  2023 은 어떤 구성으로도 0점이라 점수 비교엔 정보가 없지만,")
    print("  편향 자체는 유효한 관측이라 편향 통계에는 포함했다.")


if __name__ == "__main__":
    main()
