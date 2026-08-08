r"""학습된 선수별 파라미터가 asof_* 를 넘어서는가 — 임베딩의 1차원 반증.

배경. `pitcher_id` 는 데뷔 순으로 부여된 정수다 (4-7). 트리는 순서형 분할만
하므로 이 컬럼에서 뽑아내는 것은 사실상 **데뷔 코호트**이고, 선수 정체성 자체는
접근하지 못한다. 4-5 의 `keep-ids +8.5` 가 개인차가 아니라 코호트였던 이유다.

임베딩(엔티티 임베딩 + MLP)의 주장은 정확히 이 한계를 없앤다는 것이다 —
792명 각각에 학습된 벡터를 준다. 그런데 **그 주장의 1차원 버전은 torch 없이
지금 바로 검증할 수 있다.** 로지스틱에 ID 를 one-hot 으로 넣으면 선수마다
학습된 계수가 하나씩 생기고, L2 정규화까지 걸린다.

    1차원 학습 파라미터가 아무것도 못 하면 다차원 임베딩도 기대하기 어렵다.

그리고 혼합 성분이 이미 로지스틱이라(4-8) 붙이는 비용도 없다.

무엇과 비교하는가. `asof_pitcher_success_rate` 는 그 선수의 **과거 평균**이다.
one-hot 계수는 **다른 모든 피처를 통제한 뒤 남는 선수 고유 효과**라 성격이
다르다. 후자가 전자를 넘어서는지가 질문이다.

단독 점수는 중요하지 않다. 로지스틱은 단독으로 크게 지지만(2024 에서 164.59)
혼합에서 +8.24 를 냈다 (4-8). **혼합 기여로 판단한다.**

    .\.venv\Scripts\python.exe embed_test.py
    .\.venv\Scripts\python.exe embed_test.py --folds 2021,2022,2024
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_DIR = "./data"
CACHE = "./.blendcache"
ID = "row_id"
TARGET = "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state"]
PREV1 = "asof_pitcher_prev1_game_success_rate"
LAM = 0.03                       # 4-9 확정
BLEND_W = [0.05, 0.10, 0.15, 0.20, 0.30]
# one-hot 으로 넣을 고차원 ID. 트리가 순서로만 쓰던 것들이다.
ID_COLS = ["pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--folds", default="2024")
    p.add_argument("--C", type=float, default=0.1)
    p.add_argument("--max-iter", type=int, default=200)
    return p.parse_args()


def add_derived(df):
    df = df.copy()
    df["same_hand"] = (df["pitcher_hand"] == df["batter_hand"]).astype(int)
    return df


def make_lr(features, with_ids, C, max_iter):
    """4-8 에서 검증된 로지스틱 구성 그대로. with_ids 면 ID 블록만 얹는다.

    수치형은 **반드시 StandardScaler** 다. 중심을 옮기지 않으면 season(2019~2024)
    같은 컬럼이 최댓값으로 나뉘어 전부 0.9999 근처가 되고, 분산이 사라져 lbfgs 가
    제대로 못 푼다 — 실제로 단독 점수가 164.59 에서 0.00 으로 무너졌다.

    ID one-hot 은 min_frequency 로 희귀 선수를 한 칸에 몰아 과적합을 막는다.
    """
    cats = list(CAT_COLS)
    ids = ID_COLS if with_ids else []
    num = [c for c in features if c not in cats + ids]
    blocks = [
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=0.001), cats),
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median", add_indicator=True)),
            ("sc", StandardScaler()),
        ]), num),
    ]
    if ids:
        blocks.insert(1, ("ids", OneHotEncoder(handle_unknown="ignore",
                                               min_frequency=20), ids))
    return Pipeline([
        ("pre", ColumnTransformer(blocks)),
        ("clf", LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs")),
    ])


def score_of(y, p, denom):
    return max(0.0, 100000 * (1 - ((np.clip(p, 0, 1) - y) ** 2).mean() / denom))


def main():
    args = parse_args()
    folds = [int(f) for f in args.folds.split(",")]
    test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig",
                            nrows=0).columns
    base_cols = [c for c in test_cols if c != ID]
    train = add_derived(pd.read_csv(f"{DATA_DIR}/train.csv",
                                    encoding="utf-8-sig",
                                    usecols=base_cols + [TARGET]))
    features = base_cols + ["same_hand"]
    print(f"{len(train):,} 행 | 피처 {len(features)}개 | C={args.C}\n")

    for Y in folds:
        tr = train[train["season"] < Y]
        va = train[train["season"] == Y].reset_index(drop=True)
        y = va[TARGET].to_numpy(dtype=float)
        denom = y.mean() * (1 - y.mean())
        c = float(tr[TARGET].mean())
        anc = va[PREV1].fillna(c).to_numpy(dtype=float) - c

        hp = os.path.join(CACHE, f"{Y}_hgb_seed42.npy")
        if not os.path.exists(hp):
            raise SystemExit(f"{hp} 없음 — blend_test.py 를 먼저 돌릴 것")
        p_h = np.load(hp)

        # 현재 확정 구성 (hgb 0.90 / lr 0.10 + 중심 보정) 을 기준으로 삼는다
        print(f"--- fold {Y} (학습 {len(tr):,}) ---")
        base_lr = None
        for label, with_ids in (("lr (현재)", False), ("lr+ID onehot", True)):
            t = time.time()
            m = make_lr(features, with_ids, args.C, args.max_iter)
            m.fit(tr[features], tr[TARGET])
            p_l = m.predict_proba(va[features])[:, 1]
            n_feat = m.named_steps["pre"].transform(va[features][:5]).shape[1]
            solo = score_of(y, p_l, denom)
            if not with_ids:
                base_lr = p_l
            row = []
            for w in BLEND_W:
                mix = (1 - w) * p_h + w * p_l
                row.append(score_of(y, np.clip(mix + LAM * anc, 0, 1), denom))
            print(f"  {label:14s} 입력 {n_feat:5d}차원  단독 {solo:8.2f}  "
                  f"[{time.time()-t:.0f}s]")
            print(f"{'':16s} 혼합 " + "  ".join(
                f"w{w:.2f} {v:8.2f}" for w, v in zip(BLEND_W, row)))

        # 기준: 혼합 없이 hgb + 중심 보정만
        ref = score_of(y, np.clip(p_h + LAM * anc, 0, 1), denom)
        print(f"{'':16s} (혼합 없음 = {ref:.2f})\n")

    print("""읽는 법.
  단독 점수는 판단 기준이 아니다. 로지스틱은 단독으로 크게 지지만 혼합에서
  이득을 낸다 (4-8). **lr+ID 의 혼합 열이 lr 보다 높은지**만 본다.
  높지 않으면 학습된 선수별 파라미터가 asof_* 를 넘지 못한다는 뜻이고,
  다차원 임베딩(torch)도 기대하기 어렵다.""")


if __name__ == "__main__":
    main()
