r"""연도 클러스터 MoE 의 급소 — **게이트가 한 행만 보고 국면을 알 수 있는가**.

연도별 DGP 가 다르다고 보고 클러스터별 전문가를 두는 구조는, 추론 시점에 "이
샘플이 어느 클러스터인가"를 정해야 성립한다. 규칙상 평가셋 전체 분포는 못 쓰므로
**행 하나만 보고** 판정해야 한다.

그런데 국면은 본질적으로 **모집단의 성질**(그 시즌의 분포)이지 행의 성질이 아니다.
행 하나에 국면 정보가 얼마나 실려 있는지가 이 구조의 상한을 정한다.

여기서는 그걸 직접 잰다 — 행 피처로 **시즌을 맞히는** 분류기를 학습한다.

  - 다중분류 정확도 vs 우연 수준(1/6)
  - 이웃 두 시즌 이진 분류 AUC (2023 vs 2024 처럼 실제로 갈라야 할 쌍)

정확도가 우연에 가까우면 게이트는 동전던지기이고, 그 위에 쌓는 전문가 혼합은
가중치가 무작위인 앙상블에 지나지 않는다.

주의 — `season` 컬럼 자체는 당연히 빼고 잰다. 그건 정답을 주는 것이다.
평가셋(2025)에서는 season 이 학습에 없던 값이라 트리가 외삽도 못 한다.

사용법:
    .\.venv\Scripts\python.exe gate_probe.py
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

DATA_DIR = "./data"
TARGET = "control_success"
SAMPLE = 300_000
SEED = 42

# season 은 정답 누설이라 뺀다. 팀/선수 ID 도 뺀다 — 로스터가 해마다 바뀌므로
# ID 만으로 연도를 맞히는 건 국면 추론이 아니라 명부 외우기다.
DROP = ["season", "pitcher_id", "batter_id", "pitcher_team_id",
        "batter_team_id", TARGET]


def main():
    test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig",
                            nrows=0).columns
    cols = [c for c in test_cols if c != "row_id"]
    df = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                     usecols=cols + [TARGET])
    feats = [c for c in df.columns if c not in DROP]
    df = df.sample(SAMPLE, random_state=SEED)
    # 문자 범주형 3개는 코드로 바꾼다 (top_bottom='T'/'B' 등)
    for c in ("top_bottom", "game_type", "base_state"):
        if c in df.columns and df[c].dtype == object:
            df[c] = df[c].astype("category").cat.codes
    print(f"표본 {len(df):,}행, 피처 {len(feats)}개 (season/ID 제외)\n")

    # --- 1. 시즌 다중분류 ---
    X_tr, X_te, y_tr, y_te = train_test_split(
        df[feats], df["season"], test_size=0.3, random_state=SEED,
        stratify=df["season"])
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                       max_leaf_nodes=31, random_state=SEED)
    m.fit(X_tr, y_tr)
    acc = (m.predict(X_te) == y_te).mean()
    chance = y_te.value_counts(normalize=True).max()
    print("=== 행 하나로 시즌 맞히기 (6-클래스) ===")
    print(f"  정확도 {acc:.3f} | 최빈 클래스 기준선 {chance:.3f} | "
          f"균등 우연 {1/6:.3f}")
    print(f"  기준선 대비 {acc - chance:+.3f}")

    # --- 2. 이웃 시즌 이진 분류 ---
    print("\n=== 이웃 두 시즌 가르기 (AUC) ===")
    print("  AUC 0.5 = 구분 불가, 1.0 = 완벽")
    for a, b in [(2019, 2020), (2020, 2021), (2021, 2022),
                 (2022, 2023), (2023, 2024)]:
        sub = df[df["season"].isin([a, b])]
        Xa, Xb, ya, yb = train_test_split(
            sub[feats], (sub["season"] == b).astype(int),
            test_size=0.3, random_state=SEED, stratify=sub["season"])
        mm = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                            max_leaf_nodes=31,
                                            random_state=SEED)
        mm.fit(Xa, ya)
        auc = roc_auc_score(yb, mm.predict_proba(Xb)[:, 1])
        print(f"  {a} vs {b}   AUC {auc:.4f}")

    print("\n  → AUC 가 0.5 에 가까우면 행 하나로는 국면을 못 가른다는 뜻이고,")
    print("     그 위의 게이트는 무작위 가중치가 된다.")
    print("     0.5 에서 멀면 행에 국면 정보가 있다는 뜻이지만, 그건 대개")
    print("     asof_* 누적값이 해마다 커지는 것(=시간 추세)을 읽은 것이라")
    print("     2025 에서는 학습 범위 밖이라 외삽이 안 된다는 점을 유의할 것.")


if __name__ == "__main__":
    main()
