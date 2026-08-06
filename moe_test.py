r"""7-2 상황별 분할 모델(MoE) 을 HGB 로 직접 잰다.

count_probe.py 는 로지스틱으로 상호작용이 실재함을 보였다(+24.03). 하지만
로지스틱은 칸 안에서 상호작용을 표현할 수단이 아예 없어서 분할이 유일한 통로다.
HGB 는 트리로 그걸 공짜로 하므로, **그 +24 중 얼마가 이미 잡혀 있는지**는
HGB 로 다시 재야만 알 수 있다.

## 측정 함정 — 칸마다 반복수를 고르면 안 된다

곡선 최적점은 검증셋에 맞춘 낙관 편향이 있다(4-6). 전역 모델은 자유 파라미터가
반복수 하나인데, 12칸 MoE 가 칸마다 최적점을 고르면 **자유 파라미터가 12개**가
된다. 그대로 비교하면 MoE 가 편향만으로 이긴다.

그래서 두 값을 따로 낸다.

  shared : 모든 칸이 **같은 반복수**를 쓰고, 그 공유 반복수만 곡선에서 고른다.
           자유 파라미터 1개 — 전역 모델과 정확히 같은 조건이다. **이게 판단 기준.**
  percell: 칸마다 제 최적점을 고른다. 자유 파라미터 N개라 낙관 편향이 있다.
           **상한선**으로만 읽는다.

둘의 차이가 크면 그 차이는 실력이 아니라 편향이다.

## 게이트

  strikes : strikes_before 0/1/2 — 3칸. 계수표에서 기울기가 가장 체계적으로
            갈린 축이다(투스트라이크에서 투수 기량 의존이 크다).
  count   : (balls, strikes) 12칸. 가장 세밀하지만 칸당 학습량이 1/12 로 준다.

사용법:
    .\.venv\Scripts\python.exe moe_test.py --gates strikes --seeds 42
    .\.venv\Scripts\python.exe moe_test.py --gates strikes,count --seeds 42,43
"""
import argparse
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
# 칸 모델은 학습량이 적어 더 오래 돌 수 있어야 한다. 곡선에서 고르므로 넉넉히.
N_ITER = 1600
VAL_SEASON = 2024


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gates", default="strikes")
    p.add_argument("--seeds", default="42")
    p.add_argument("--n-iter", type=int, default=N_ITER)
    return p.parse_args()


def make_model(cols, seed, n_iter):
    num = [c for c in cols if c not in CAT_COLS]
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                               unknown_value=-1), CAT_COLS),
        ("num", "passthrough", num),
    ])
    clf = HistGradientBoostingClassifier(
        max_iter=n_iter, learning_rate=LR, max_leaf_nodes=LEAVES,
        min_samples_leaf=MIN_LEAF, l2_regularization=L2,
        early_stopping=False, random_state=seed)
    return Pipeline([("pre", pre), ("clf", clf)])


def sse_curve(model, X_va, y_va, n_iter):
    """반복 시점별 잔차제곱합. 점수가 아니라 SSE 를 돌려주는 게 요점이다 —
    칸별 SSE 는 그냥 더할 수 있어서 공유 반복수 점수를 만들 수 있다."""
    pre, clf = model.named_steps["pre"], model.named_steps["clf"]
    X_t = pre.transform(X_va)
    out = np.empty(n_iter)
    out.fill(np.nan)
    for i, proba in enumerate(clf.staged_predict_proba(X_t)):
        if i >= n_iter:
            break
        out[i] = ((proba[:, 1] - y_va) ** 2).sum()
    # 조기 종료된 경우 마지막 값으로 채운다 (곡선 비교를 위해 길이를 맞춘다)
    last = np.where(~np.isnan(out))[0]
    if len(last):
        out[last[-1] + 1:] = out[last[-1]]
    return out


def gate_labels(df, gate):
    if gate == "strikes":
        return df["strikes_before"].astype(str)
    if gate == "count":
        return (df["balls_before"].astype(str) + "-"
                + df["strikes_before"].astype(str))
    raise SystemExit(f"모르는 게이트: {gate}")


def main():
    args = parse_args()
    gates = args.gates.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    n_iter = args.n_iter

    test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig",
                            nrows=0).columns
    cols = [c for c in test_cols if c != ID]
    df = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                     usecols=cols + [TARGET])

    tr = df[df["season"] < VAL_SEASON]
    va = df[df["season"] == VAL_SEASON]
    y_va = va[TARGET].to_numpy().astype(float)
    n_va = len(va)
    r = y_va.mean()
    base = r * (1 - r)
    denom = n_va * base
    print(f"학습 {len(tr):,} | 검증 {n_va:,} (2024)  기준선 r(1-r)={base:.6f}\n")

    def to_score(sse):
        return np.maximum(0.0, 100000 * (1 - sse / denom))

    for seed in seeds:
        print(f"########## seed {seed} ##########")
        t = time.time()
        m = make_model(cols, seed, n_iter)
        m.fit(tr[cols], tr[TARGET])
        sse_g = sse_curve(m, va[cols], y_va, n_iter)
        sc_g = to_score(sse_g)
        i_g = int(np.argmax(sc_g))
        print(f"  전역 baseline   {sc_g[i_g]:9.2f}  (iter {i_g+1})  "
              f"[{time.time()-t:.0f}s]\n")

        for gate in gates:
            g_tr = gate_labels(tr, gate)
            g_va = gate_labels(va, gate)
            keys = sorted(set(g_tr) & set(g_va))
            total = np.zeros(n_iter)
            per_best = 0.0
            print(f"  --- 게이트 {gate} ({len(keys)}칸) ---")
            for k in keys:
                mtr, mva = g_tr == k, g_va == k
                sub_tr, sub_va = tr[mtr], va[mva]
                yy = y_va[mva.to_numpy()]
                t = time.time()
                mm = make_model(cols, seed, n_iter)
                mm.fit(sub_tr[cols], sub_tr[TARGET])
                sse = sse_curve(mm, sub_va[cols], yy, n_iter)
                total += sse
                per_best += sse.min()
                # 이 칸만 떼서 본 최적점 (진단용)
                bi = int(np.argmin(sse))
                print(f"    {k:>5}  학습 {len(sub_tr):>8,} 검증 {len(sub_va):>7,}"
                      f"  칸최적 iter {bi+1:>4}  [{time.time()-t:.0f}s]")

            sc_shared = to_score(total)
            i_s = int(np.argmax(sc_shared))
            s_shared = sc_shared[i_s]
            s_percell = float(to_score(per_best))
            print(f"\n    shared  {s_shared:9.2f}  (공유 iter {i_s+1})   "
                  f"vs 전역 {s_shared - sc_g[i_g]:+8.2f}   ← 판단 기준")
            print(f"    percell {s_percell:9.2f}  (칸별 최적)          "
                  f"vs 전역 {s_percell - sc_g[i_g]:+8.2f}   ← 낙관 상한")
            print(f"    편향 폭 {s_percell - s_shared:+.2f}"
                  f"  (칸마다 반복수를 고른 것만으로 벌어진 차이)\n")


if __name__ == "__main__":
    main()
