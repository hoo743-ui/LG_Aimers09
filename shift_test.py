r"""국면 이동 대응 — 평가셋 피처로 시즌 성공률을 추정해 예측 중심을 옮긴다.

배경: 이 문제에서 점수를 통째로 날리는 건 순위가 아니라 중심이다. 누적형
`asof_pitcher_success_rate` 는 경력 전체 평균이라 해마다 떨어지는 국면을 늦게
따라가고, 그 격차가 벌어진 2023 에서 예측 중심이 +0.0394 어긋나 621점을 잃어
점수가 0이 됐다 (README 4-2).

offset_test.py 는 "전년도에서 잰 편차를 다음 해에 이식"하는 방법을 검증했고
실패했다. 편차가 해마다 부호까지 바뀌기 때문이다. 여기서는 옮겨오지 않는다.
평가 시즌의 성공률을 그 시즌 피처에서 직접 추정한다.

    r_hat = mean(asof_pitcher_prev1_game_success_rate)   # 정답 불필요
    p'    = clip(p + (r_hat - mean(p)), 0, 1)

`prev1_game` 은 직전 경기의 성공률이라 현재 국면을 실시간으로 반영한다. 기울기는
건드리지 않고 절편만 옮긴다 — 선형보정이 2024 를 -472 로 무너뜨린 경로를 피한다.

판정 기준은 "2023 이 0에서 살아나는가, 나머지 폴드가 손해를 보지 않는가" 다.

사용법:
    .\.venv\Scripts\python.exe shift_test.py
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
PREV1 = "asof_pitcher_prev1_game_success_rate"
CUMUL = "asof_pitcher_success_rate"
FOLDS = [2021, 2022, 2023, 2024]

# 확정 설정 (README 5). 앙상블 없이 단일 모델로 잰다 — 여기서 보는 건 분산이
# 아니라 중심이고, 중심은 시드를 늘려도 움직이지 않는다.
LR, LEAVES, MIN_LEAF, L2, N_ITER = 0.02, 10, 1000, 1.0, 1000


def make_model(features, seed=42):
    num = [c for c in features if c not in CAT_COLS]
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                               unknown_value=-1), CAT_COLS),
        ("num", "passthrough", num),
    ])
    clf = HistGradientBoostingClassifier(
        max_iter=N_ITER, learning_rate=LR, max_leaf_nodes=LEAVES,
        min_samples_leaf=MIN_LEAF, l2_regularization=L2,
        early_stopping=False, random_state=seed,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def score_of(y, p, base):
    return max(0, 100000 * (1 - ((p - y) ** 2).mean() / base))


def shift_to(p, target):
    return np.clip(p + (target - p.mean()), 0.0, 1.0)


def main():
    test_cols = pd.read_csv(os.path.join(DATA_DIR, "test.csv"),
                            encoding="utf-8-sig", nrows=0).columns
    features = [c for c in test_cols if c != ID]
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"),
                        encoding="utf-8-sig", usecols=features + [TARGET])

    # ---- 추정기 자체의 정확도 (모델과 무관, 데이터만 본다) ----
    print("=== 추정기 점검: 시즌 성공률 vs 피처 평균 ===")
    print(f"{'시즌':>6}{'실제':>9}{'prev1':>9}{'오차':>9}"
          f"{'누적asof':>10}{'오차':>9}")
    est = {}
    for s, g in train.groupby("season"):
        actual = g[TARGET].mean()
        prev1 = g[PREV1].mean()
        cumul = g[CUMUL].mean()
        est[s] = (actual, prev1)
        print(f"{s:>6}{actual:9.4f}{prev1:9.4f}{actual - prev1:+9.4f}"
              f"{cumul:10.4f}{actual - cumul:+9.4f}")
    gaps = np.array([a - p for a, p in est.values()])
    print(f"\nprev1 오차: 평균 {gaps.mean():+.4f}  표준편차 {gaps.std():.4f}"
          f"  → 이 크기의 손실 {gaps.std()**2 / 0.25 * 100000:.1f}점\n")

    rows = []
    for Y in FOLDS:
        tr = train[train["season"] < Y]
        va = train[train["season"] == Y]
        y = va[TARGET].to_numpy()
        r = y.mean()
        base = r * (1 - r)

        t = time.time()
        m = make_model(features)
        m.fit(tr[features], tr[TARGET])
        p = m.predict_proba(va[features])[:, 1]
        print(f"--- fold {Y} (학습 {len(tr)}) [{time.time()-t:.0f}s] ---")

        # 추정기는 평가 시즌의 피처만 본다. 편향 보정 b 는 과거 시즌에서만 잰다.
        r_hat = float(va[PREV1].mean())
        past = [est[s] for s in est if s < Y]
        b = float(np.mean([a - q for a, q in past])) if past else 0.0

        variants = [
            ("원본", p),
            ("이동(prev1)", shift_to(p, r_hat)),
            ("이동(prev1+편향보정)", shift_to(p, r_hat + b)),
            ("이동(정답 r) — 상한", shift_to(p, r)),
        ]
        for name, q in variants:
            print(f"  {name:24s} 중심 {q.mean():.4f} (실제 {r:.4f}, "
                  f"편차 {q.mean() - r:+.4f})  {score_of(y, q, base):9.2f}")
        rows.append((Y, [score_of(y, q, base) for _, q in variants]))
        print()

    names = ["원본", "이동(prev1)", "이동(prev1+b)", "이동(정답)"]
    print("=== 요약 ===")
    print(f"{'폴드':>6}" + "".join(f"{n:>16s}" for n in names))
    for Y, scores in rows:
        print(f"{Y:>6}" + "".join(f"{s:16.2f}" for s in scores))
    means = np.mean([s for _, s in rows], axis=0)
    print(f"{'평균':>6}" + "".join(f"{v:16.2f}" for v in means))
    print(f"\n이동(prev1) − 원본 = {means[1] - means[0]:+.2f}")
    print("판정: 2023 이 0에서 살아나고 나머지 폴드가 손해를 안 봐야 채택.")


if __name__ == "__main__":
    main()
