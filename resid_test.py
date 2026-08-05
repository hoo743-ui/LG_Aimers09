r"""국면 이동을 규칙 안에서 잡는다 — 행 단위 기준선 위의 잔차 학습.

배경: 4-6 의 국면 보정(평가셋 전체의 prev1 평균으로 예측 중심을 이동)은
대회 규칙 위반이다. data_description.md 5) 가 "평가 데이터 전체를 보고 만든
사후 보정값"과 "test.csv 내부 빈도값 또는 분포 통계"를 명시적으로 금지한다.

목표 자체는 유효하다 — 타깃 성공률이 해마다 떨어지는데 모델이 학습 시즌의
수준에 고정되면 중심이 통째로 어긋난다 (4-2). 다른 행을 보지 않고 같은 일을
하는 방법은 **각 행이 자기 자신의 asof 기록을 기준선으로 삼는 것**이다.

    b_i = 그 행의 prev 계열 (공식 asof 피처, 규칙상 허용)
    학습: y_i - b_i 를 맞춘다
    추론: p_i = clip(b_i + g(x_i), 0, 1)

b 의 단면 평균이 그 시즌의 성공률을 따라가므로(4-6 에서 표준편차 0.0031 로
측정), 예측 중심도 자동으로 따라간다. 전체 평균을 재지 않으므로 행 독립이다.

비교 대상:
  base        분류기, 보정 없음 — 합법 기준선
  offset      분류기 + 행 단위 오프셋 (b_i - 학습셋 평균 b)
  resid_prev1 회귀기, 기준선 prev1
  resid_prev5 회귀기, 기준선 prev5 (더 안정적)
  reg_plain   회귀기, 기준선 없음 — 제곱오차 손실 자체의 효과를 분리
  [참고] shift 전역 보정 — 위반이지만 상한 확인용

제곱오차 회귀는 Brier 를 직접 최소화한다. 분류기의 로그손실과 목적이 다르므로
기준선 효과와 손실함수 효과를 나눠 봐야 한다 (reg_plain 이 그 역할).

사용법:
    .\.venv\Scripts\python.exe resid_test.py --folds 2021,2022,2024
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
ID = "row_id"
TARGET = "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state"]
PREV = {k: f"asof_pitcher_prev{k}_game_success_rate" for k in (1, 3, 5)}
CUMUL = "asof_pitcher_success_rate"
LR, LEAVES, MIN_LEAF, L2, N_ITER = 0.02, 10, 1000, 1.0, 1400


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--folds", default="2021,2022,2024")
    p.add_argument("--only", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def baseline(df, kind, fallback):
    """행 단위 기준선. 다른 행을 보지 않는다.

    prev 계열이 없는 행(경력 초기)은 누적형으로, 그것도 없으면 학습셋에서 구한
    상수로 메운다. fallback 은 학습 데이터에서만 계산해 넘긴다.
    """
    b = df[PREV[kind]]
    b = b.fillna(df[PREV[5]]).fillna(df[PREV[3]]).fillna(df[PREV[1]])
    return b.fillna(df[CUMUL]).fillna(fallback).to_numpy()


def make_pre(cols):
    num = [c for c in cols if c not in CAT_COLS]
    return ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                               unknown_value=-1), CAT_COLS),
        ("num", "passthrough", num),
    ])


def make_model(cols, seed, regressor):
    cls = HistGradientBoostingRegressor if regressor \
        else HistGradientBoostingClassifier
    est = cls(max_iter=N_ITER, learning_rate=LR, max_leaf_nodes=LEAVES,
              min_samples_leaf=MIN_LEAF, l2_regularization=L2,
              early_stopping=False, random_state=seed)
    return Pipeline([("pre", make_pre(cols)), ("est", est)])


def best_curve(est, X_t, y, base, offset):
    """반복 시점별 최고 점수. offset 은 행 단위 기준선(없으면 0)."""
    # 분류기에도 staged_predict 가 있지만 그건 하드 레이블(0/1)이다. 확률이
    # 필요하므로 staged_predict_proba 를 먼저 본다.
    staged = (est.staged_predict_proba(X_t)
              if hasattr(est, "staged_predict_proba")
              else est.staged_predict(X_t))
    best = (0, 0.0)
    for i, out in enumerate(staged, start=1):
        p = out if out.ndim == 1 else out[:, 1]
        p = np.clip(p + offset, 0.0, 1.0)
        s = max(0.0, 100000 * (1 - ((p - y) ** 2).mean() / base))
        if s > best[1]:
            best = (i, s)
    return best


def main():
    args = parse_args()
    folds = [int(f) for f in args.folds.split(",")]

    test_cols = pd.read_csv(os.path.join(DATA_DIR, "test.csv"),
                            encoding="utf-8-sig", nrows=0).columns
    cols = [c for c in test_cols if c != ID]
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"),
                        encoding="utf-8-sig", usecols=cols + [TARGET])

    variants = ["base", "offset", "resid_prev1", "resid_prev5", "reg_plain"]
    if args.only:
        variants = args.only.split(",")

    results = {v: {} for v in variants}
    illegal = {}
    for Y in folds:
        tr = train[train["season"] < Y]
        va = train[train["season"] == Y]
        y = va[TARGET].to_numpy()
        base = y.mean() * (1 - y.mean())
        fb = float(tr[TARGET].mean())          # 학습셋 상수 (합법)
        print(f"--- fold {Y} (학습 {len(tr)}) ---")

        for v in variants:
            t = time.time()
            if v == "base":
                m = make_model(cols, args.seed, regressor=False)
                m.fit(tr[cols], tr[TARGET])
                X_t = m.named_steps["pre"].transform(va[cols])
                it, sc = best_curve(m.named_steps["est"], X_t, y, base, 0.0)
                # 참고용 — 전역 보정을 걸었을 때의 상한 (규칙 위반, 제출 불가)
                r_hat = float(va[PREV[1]].dropna().mean())
                proba = m.predict_proba(va[cols])[:, 1]
                q = np.clip(proba + (r_hat - proba.mean()), 0, 1)
                illegal[Y] = max(0.0, 100000 * (1 - ((q - y) ** 2).mean() / base))
            elif v == "offset":
                m = make_model(cols, args.seed, regressor=False)
                m.fit(tr[cols], tr[TARGET])
                b_tr = baseline(tr, 5, fb)
                off = baseline(va, 5, fb) - b_tr.mean()
                X_t = m.named_steps["pre"].transform(va[cols])
                it, sc = best_curve(m.named_steps["est"], X_t, y, base, off)
            elif v == "reg_plain":
                m = make_model(cols, args.seed, regressor=True)
                m.fit(tr[cols], tr[TARGET])
                X_t = m.named_steps["pre"].transform(va[cols])
                it, sc = best_curve(m.named_steps["est"], X_t, y, base, 0.0)
            else:
                kind = 1 if v.endswith("prev1") else 5
                b_tr = baseline(tr, kind, fb)
                b_va = baseline(va, kind, fb)
                m = make_model(cols, args.seed, regressor=True)
                m.fit(tr[cols], tr[TARGET].to_numpy() - b_tr)
                X_t = m.named_steps["pre"].transform(va[cols])
                it, sc = best_curve(m.named_steps["est"], X_t, y, base, b_va)
            results[v][Y] = sc
            print(f"  {v:12s} {sc:9.2f} (iter {it:4d})  [{time.time()-t:.0f}s]")
        print()

    print("=== 요약 (행 독립, 규칙 준수) ===")
    print(f"{'변형':14s}" + "".join(f"{f'fold{Y}':>11s}" for Y in folds)
          + f"{'평균':>10s}{'base대비':>10s}")
    b0 = np.mean([results["base"][Y] for Y in folds]) if "base" in results else 0
    for v in variants:
        vals = [results[v][Y] for Y in folds]
        m = float(np.mean(vals))
        print(f"{v:14s}" + "".join(f"{x:11.2f}" for x in vals)
              + f"{m:10.2f}{m - b0:+10.2f}")
    if illegal:
        vals = [illegal[Y] for Y in folds]
        print(f"\n[참고·제출불가] 전역보정"
              + "".join(f"{x:11.2f}" for x in vals)
              + f"{np.mean(vals):10.2f}{np.mean(vals) - b0:+10.2f}")


if __name__ == "__main__":
    main()
