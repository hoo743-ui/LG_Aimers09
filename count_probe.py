r"""7-2 볼카운트 구간별 분할 모델링 — 값싼 사전 검산.

가설: 피처와 타깃의 관계가 볼카운트 구간마다 다르다면 하나의 모델로 못 담는다.
특히 지금 확정 설정은 **잎 10개**짜리 아주 얕은 모델이라 상호작용을 표현할 여력이
적다. 구간을 갈라 각자에게 잎 10개를 주면 나아질 수 있다.

반대 논리도 있다. 나누면 모델당 학습량이 1/N 로 준다. 4-5 에서 "최근 시즌만 학습"이
−41~−95 였던 것처럼 이 데이터는 학습량에 민감하다. 그러니 **관계가 실제로 다른지**
부터 확인해야 한다.

HGB 를 12번 돌리기 전에 로지스틱 회귀로 같은 질문을 던진다. 로지스틱은 수십 초면
끝나고, "전역 계수 하나 vs 구간별 계수"의 차이가 곧 상호작용의 크기다.

  global  : 전체에 로지스틱 하나
  percell : (balls, strikes) 12칸에 각각 로지스틱

percell 이 global 을 못 이기면 상호작용이 없다는 뜻이고, HGB 분할도 기대할 게
없다. 이기면 그 크기가 HGB 로 넘어갈지 판단할 근거가 된다.

로지스틱의 절대 점수는 HGB 보다 낮다. 여기서 보는 건 **두 로지스틱의 차이**다.

사용법:
    .\.venv\Scripts\python.exe count_probe.py
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = "./data"
TARGET = "control_success"
VAL_SEASON = 2024

FEATS = [
    "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_batter_success_rate", "asof_batter_middle_rate",
    "asof_pitcher_n", "asof_batter_n",
    "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
    "li", "inning", "outs_before", "num_runners_on",
]
CELL = ["balls_before", "strikes_before"]


def score(y, p, base):
    return max(0.0, 100000 * (1 - ((p - y) ** 2).mean() / base))


def fit_predict(tr_X, tr_y, va_X):
    """표본이 너무 적으면 상수를 돌려준다 (그 칸의 학습 성공률)."""
    if len(tr_X) < 500 or tr_y.nunique() < 2:
        return np.full(len(va_X), tr_y.mean() if len(tr_y) else 0.5)
    m = make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=1000, C=1.0))
    m.fit(tr_X, tr_y)
    return m.predict_proba(va_X)[:, 1]


def main():
    cols = FEATS + CELL + ["season", TARGET]
    df = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                     usecols=cols)
    # 로지스틱은 결측을 못 받는다. 중앙값으로 채운다 — HGB 와 달리 여기서는
    # 결측 처리 방식이 비교의 관심사가 아니고, 두 변형에 똑같이 적용된다.
    med = df.loc[df["season"] < VAL_SEASON, FEATS].median()
    df[FEATS] = df[FEATS].fillna(med)

    tr = df[df["season"] < VAL_SEASON]
    va = df[df["season"] == VAL_SEASON]
    y_va = va[TARGET].to_numpy()
    r = y_va.mean()
    base = r * (1 - r)
    print(f"학습 {len(tr):,} | 검증 {len(va):,} (2024)  실제 성공률 {r:.4f}\n")

    # --- 구간별 실태 ---
    print("=== 볼카운트별 성공률과 표본 ===")
    print(f"{'B-S':>5} {'학습표본':>10} {'검증표본':>10} "
          f"{'학습성공률':>10} {'검증성공률':>10}")
    g_tr = tr.groupby(CELL)[TARGET].agg(["size", "mean"])
    g_va = va.groupby(CELL)[TARGET].agg(["size", "mean"])
    for key in sorted(set(g_tr.index) | set(g_va.index)):
        a = g_tr.loc[key] if key in g_tr.index else None
        b = g_va.loc[key] if key in g_va.index else None
        print(f"{key[0]}-{key[1]:>3} "
              f"{int(a['size']) if a is not None else 0:>10,} "
              f"{int(b['size']) if b is not None else 0:>10,} "
              f"{a['mean'] if a is not None else float('nan'):>10.4f} "
              f"{b['mean'] if b is not None else float('nan'):>10.4f}")

    overall = tr[TARGET].mean()
    spread = g_tr["mean"].max() - g_tr["mean"].min()
    print(f"\n  전체 {overall:.4f} | 칸 간 성공률 폭 {spread:.4f}")
    print("  → 이 폭은 **주효과**다. 트리는 balls/strikes 로 쪼개 이미 잡을 수 있다.")
    print("    분할 모델이 이득을 보려면 주효과가 아니라 **기울기 차이**가 있어야 한다.")

    # --- global vs percell ---
    # 공정한 비교의 핵심: global 에 볼카운트를 **원핫으로** 준다. 안 그러면
    # percell 의 이득 대부분이 칸별 절편(=주효과)에서 나오는데, 그건 HGB 가
    # balls_before/strikes_before 로 이미 잡는 것이라 분할의 이득이 아니다.
    # 원핫을 주고 나서 남는 차이만이 **상호작용**이다.
    oh_tr = pd.get_dummies(tr[CELL].astype(str).agg("-".join, axis=1),
                           prefix="c")
    oh_va = pd.get_dummies(va[CELL].astype(str).agg("-".join, axis=1),
                           prefix="c").reindex(columns=oh_tr.columns,
                                               fill_value=0)
    Xg_tr = pd.concat([tr[FEATS].reset_index(drop=True),
                       oh_tr.reset_index(drop=True)], axis=1)
    Xg_va = pd.concat([va[FEATS].reset_index(drop=True),
                       oh_va.reset_index(drop=True)], axis=1)

    print("\n=== 로지스틱 비교 (2024 검증) ===")
    p_naive = fit_predict(tr[FEATS], tr[TARGET], va[FEATS])
    s_naive = score(y_va, p_naive, base)
    print(f"  global (카운트 없음)  {s_naive:9.2f}   ← 비교 대상 아님")
    p_global = fit_predict(Xg_tr, tr[TARGET], Xg_va)
    s_global = score(y_va, p_global, base)
    print(f"  global + 카운트 원핫  {s_global:9.2f}   ← 공정한 기준선 "
          f"(주효과 {s_global - s_naive:+.2f})")

    p_cell = np.empty(len(va))
    rows = []
    for key, idx in va.groupby(CELL).indices.items():
        m = (tr["balls_before"] == key[0]) & (tr["strikes_before"] == key[1])
        sub = tr[m]
        pred = fit_predict(sub[FEATS], sub[TARGET], va.iloc[idx][FEATS])
        p_cell[idx] = pred
        yy = y_va[idx]
        rows.append((key, len(idx), score(yy, pred, base),
                     score(yy, p_global[idx], base)))
    s_cell = score(y_va, p_cell, base)
    print(f"  percell  {s_cell:9.2f}   차이 {s_cell - s_global:+.2f}")

    print(f"\n{'B-S':>5} {'검증표본':>10} {'percell':>10} {'global':>10} {'차이':>9}")
    for key, n, sc, sg in sorted(rows):
        print(f"{key[0]}-{key[1]:>3} {n:>10,} {sc:>10.2f} {sg:>10.2f} "
              f"{sc - sg:>+9.2f}")

    print("\n  → percell 이 global 을 의미 있게 못 이기면 상호작용이 약하다는 뜻이고,")
    print("     HGB 분할도 학습량만 잃는다. 이기면 그 크기를 HGB 로 확인할 값이 있다.")

    # --- 기울기가 실제로 다른가 (해석용) ---
    print("\n=== 주요 피처의 구간별 표준화 계수 ===")
    key_feats = ["asof_pitcher_success_rate", "asof_pitcher_middle_rate",
                 "asof_pitcher_ball_rate", "asof_batter_success_rate"]
    coefs = {}
    for key in sorted(g_tr.index):
        m = (tr["balls_before"] == key[0]) & (tr["strikes_before"] == key[1])
        sub = tr[m]
        if len(sub) < 500:
            continue
        mm = make_pipeline(StandardScaler(),
                           LogisticRegression(max_iter=1000))
        mm.fit(sub[FEATS], sub[TARGET])
        c = mm[-1].coef_[0]
        coefs[key] = {f: c[FEATS.index(f)] for f in key_feats}
    cd = pd.DataFrame(coefs).T
    cd.index = [f"{a}-{b}" for a, b in cd.index]
    print(cd.round(4).to_string())
    print("\n  칸마다 부호가 바뀌거나 크기가 몇 배씩 다르면 상호작용이 실재한다.")
    print("  전부 비슷하면 관계는 하나이고 주효과만 다른 것이다.")


if __name__ == "__main__":
    main()
