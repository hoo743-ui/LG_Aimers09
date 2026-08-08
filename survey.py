r"""넓고 얕게 — 안 건드린 방법들을 한 번에 훑어 방향부터 찾는다.

왜 이렇게 하는가. 6회차 실측(로컬 661.03 -> LB 822.57)으로 1등과의 격차가
로컬 환산 **160점**이고, 국면 보정을 우리에게 허용해줘도 **120점**이 남는다는
게 확인됐다 (README 1장). 이 크기는 하나를 깊게 파서 닿는 거리가 아니다.
후보를 전부 싸게 훑어 방향을 고른 뒤에 무겁게 가야 한다.

무엇을 후보로 삼는가. 실패 목록(4-5)의 교훈이 기준이다 — **피처를 더하는 축은
죽었다.** 기각된 12개가 전부 `asof_*` 의 재표현이었다. 유일하게 통한 것은
`same_hand` 였고, 그건 새 정보가 아니라 **알고리즘이 구조적으로 못 보던 자리**를
메운 것이었다 (4-7). 그래서 후보를 전부 구조·모델계열 축으로 잡는다.

정밀도. 화면용으로 lr=0.04 n_iter=550 을 쓴다 (확정 설정 lr=0.02 n_iter=1100 의
절반 시간). 절대값은 확정 설정과 다르므로 **같은 정밀도의 base 와의 차이만**
읽는다. 승자는 interact_feat.py 로 정식 프로토콜(3폴드 x 2시드, 전체 반복수)
에서 다시 잰다.

덤으로 축소계수도 같이 낸다. base 예측에서 사후 계산되므로 추가 학습이 없고,
README 7장 3번(“a 가 1.01 -> 1.1254 로 뒤집혔다”)에 바로 답한다. 각 폴드의
최적 a 와, **다른 폴드들의 a 를 옮겨 썼을 때**의 점수를 함께 낸다 — 4-5 가
기각한 이유가 정확히 이 이식 가능성이었다.

    .\.venv\Scripts\python.exe survey.py --list
    .\.venv\Scripts\python.exe survey.py --only tree
    .\.venv\Scripts\python.exe survey.py --only family
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

DATA_DIR = "./data"
ID = "row_id"
TARGET = "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state"]
FOLDS = [2021, 2022, 2024]      # 2023 은 어떤 구성으로도 0점 (README 4-6)

# 화면용 축소 정밀도. 확정 설정(lr=0.02, n_iter=1100)의 절반 시간.
LR, N_ITER = 0.04, 550
LEAVES, MIN_LEAF, L2 = 10, 1000, 1.0

# 저차원 범주형 — HGB 의 native categorical 은 최대 255종까지다. 팀 ID(13종)
# 까지는 되지만 pitcher_id(792명)는 안 된다.
LOWCARD = CAT_COLS + ["pitcher_hand", "batter_hand", "same_hand",
                      "pitcher_team_id", "batter_team_id",
                      "balls_before", "strikes_before", "outs_before"]

# 국면이 바뀌어도 방향이 뒤집힐 이유가 없는 것들. 단조 제약은 표현력을 깎는
# 대신 시즌 이동에 강해진다 — 이 대회의 핵심 실패 모드가 국면 이동이다 (4-2).
MONO_UP = ["asof_pitcher_success_rate", "asof_pitcher_strike_rate",
           "asof_pitcher_prev1_game_success_rate",
           "asof_pitcher_prev3_game_success_rate",
           "asof_pitcher_prev5_game_success_rate"]
MONO_DOWN = ["asof_pitcher_ball_rate", "asof_pitcher_middle_rate",
             "asof_pitcher_reverse_rate"]
# asof_batter_success_rate 는 뺐다. "그 타자가 상대한 투구의 제구 성공률"이라
# 부호가 자명하지 않다 — 좋은 타자일수록 투수가 조심해서 빗나갈 수도, 반대로
# 존 안으로 넣을 수밖에 없을 수도 있다. 틀린 방향을 강제하면 그냥 손해다.


def add_derived(df):
    """final_train.py / script.py 와 같은 정의 (4-7)."""
    df = df.copy()
    df["same_hand"] = (df["pitcher_hand"] == df["batter_hand"]).astype(int)
    return df


def score_of(y, p, base):
    return max(0.0, 100000 * (1 - ((p - y) ** 2).mean() / base))


def fit_alpha(y, p, center):
    """p' = c + a*(p-c) 의 Brier 최소해. a>1 이면 예측을 벌려야 이득이다."""
    d = p - center
    v = (d ** 2).mean()
    return 1.0 if v <= 0 else float((d * (y - center)).mean() / v)


# ---------------------------------------------------------------- 모델 구성

def hgb(features, **kw):
    """확정 설정의 HGB. kw 로 한 축씩만 바꿔 무엇이 효과인지 분리한다."""
    cat = kw.pop("cat_cols", None)
    mono = kw.pop("mono", False)
    params = dict(max_iter=N_ITER, learning_rate=LR, max_leaf_nodes=LEAVES,
                  min_samples_leaf=MIN_LEAF, l2_regularization=L2,
                  early_stopping=False, random_state=42)
    params.update(kw)

    if cat is None:
        # 지금 확정 경로: 범주형 3개만 정수로 바꾸고 나머지는 통과
        pre = ColumnTransformer([
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                                   unknown_value=-1), CAT_COLS),
            ("num", "passthrough", [c for c in features if c not in CAT_COLS]),
        ])
        if mono:
            names = [c for c in features if c not in CAT_COLS]
            cst = [0] * len(CAT_COLS) + [
                1 if n in MONO_UP else -1 if n in MONO_DOWN else 0 for n in names]
            params["monotonic_cst"] = cst
        return Pipeline([("pre", pre), ("clf",
                         HistGradientBoostingClassifier(**params))])

    # native categorical — 부분집합 분할을 쓴다. 순서 코드로는 비인접 묶음을
    # 한 번에 못 가르는데(hand_mix 4수준이 same_hand 이진보다 나빴던 이유, 4-7),
    # 이걸 켜면 알고리즘이 스스로 처리한다.
    use = [c for c in cat if c in features]
    rest = [c for c in features if c not in use]
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                               unknown_value=-1), use),
        ("num", "passthrough", rest),
    ])
    params["categorical_features"] = list(range(len(use)))
    return Pipeline([("pre", pre), ("clf",
                     HistGradientBoostingClassifier(**params))])


def linear(features):
    """L2 로지스틱. 트리와 실패 방식이 달라 섞을 때 의미가 있다.

    결측은 중앙값으로 메우고 결측 표시를 따로 준다 — HGB 는 결측을 분기로
    직접 배우지만 선형 모형은 그걸 못 한다.
    """
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
        # 화면용이라 반복수를 조인다. 표준화된 입력이라 lbfgs 는 보통 그 전에
        # 수렴하고, 덜 수렴해도 순위 비교에는 지장이 없다.
        ("clf", LogisticRegression(C=0.1, max_iter=120, solver="lbfgs")),
    ])


def forest(features, extra=False):
    num = [c for c in features if c not in CAT_COLS]
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                               unknown_value=-1), CAT_COLS),
        ("num", SimpleImputer(strategy="median", add_indicator=True), num),
    ])
    cls = ExtraTreesClassifier if extra else RandomForestClassifier
    # 화면용이라 나무 수를 줄였다. 1회차 베이스라인(depth 10)이 415.57 이었는데
    # 그건 규제가 거의 없는 판이었다 — 여기서는 잎을 크게 잡아 과신을 막는다.
    return Pipeline([("pre", pre), ("clf", cls(
        n_estimators=100, min_samples_leaf=2000, max_features=0.4,
        n_jobs=-1, random_state=42))])


# ---------------------------------------------------------------- 후보 목록
# (그룹, 이름, 만드는 함수, 한 줄 설명)
CONFIGS = [
    ("tree", "base", lambda f: hgb(f),
     "확정 설정 + same_hand (축소 정밀도)"),
    ("tree", "cat_native", lambda f: hgb(f, cat_cols=LOWCARD),
     "저차원 범주형에 부분집합 분할 — 4-7 사각지대의 일반해"),
    ("tree", "cat_only3", lambda f: hgb(f, cat_cols=CAT_COLS),
     "범주형 3개에만. cat_native 의 효과가 어디서 오는지 분리"),
    ("tree", "mono", lambda f: hgb(f, mono=True),
     "asof 성공률 계열에 단조 제약 — 국면 이동 견딤 (4-2)"),
    ("tree", "deep", lambda f: hgb(f, max_leaf_nodes=31, min_samples_leaf=200),
     "잎 31. 4-5 의 기각은 same_hand 이전 값이다"),
    ("tree", "wide_l2", lambda f: hgb(f, l2_regularization=10.0),
     "L2 10배 — 규제를 세게"),
    ("family", "base", lambda f: hgb(f),
     "비교 기준 (tree 그룹과 동일)"),
    ("family", "logreg", linear,
     "L2 로지스틱. 트리와 실패 방식이 다르다"),
    ("family", "rf", lambda f: forest(f),
     "RandomForest — 1회차 베이스라인의 제대로 된 판"),
    ("family", "et", lambda f: forest(f, extra=True),
     "ExtraTrees — 분할점을 무작위로, 분산이 더 낮다"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--only", default=None, help="그룹 또는 이름 (쉼표 구분)")
    p.add_argument("--folds", default=",".join(map(str, FOLDS)))
    p.add_argument("--list", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.list:
        for g, n, _, d in CONFIGS:
            print(f"  [{g:6s}] {n:12s} {d}")
        return

    folds = [int(f) for f in args.folds.split(",")]
    picked = CONFIGS
    if args.only:
        want = {s.strip() for s in args.only.split(",")}
        picked = [c for c in CONFIGS if c[0] in want or c[1] in want]
        if not picked:
            raise SystemExit(f"고른 것이 없다: {sorted(want)}")
        # base 는 항상 기준으로 넣는다
        if not any(n == "base" for _, n, _, _ in picked):
            picked = [c for c in CONFIGS if c[1] == "base"][:1] + picked

    test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig",
                            nrows=0).columns
    cols = [c for c in test_cols if c != ID]
    train = add_derived(pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                                    usecols=cols + [TARGET]))
    features = cols + ["same_hand"]

    print(f"{len(train):,} 행 | 피처 {len(features)}개")
    print(f"화면 정밀도: lr={LR} n_iter={N_ITER} (확정은 0.02/1100) — "
          f"base 와의 차이만 읽을 것")
    print(f"폴드 {folds}\n")

    header = (f"{'구성':13s}" + "".join(f"{f'val{s}':>10s}" for s in folds)
              + f"{'평균':>10s}{'vs base':>10s}   {'a(폴드별)':>22s}{'a이식':>9s}")
    print(header)
    print("-" * (len(header) + 6))

    # 혼합 가중치. 폴드 정답으로 최적값을 고르면 그 폴드를 훔쳐보는 셈이라
    # 고정값으로만 본다. 계열이 다른 모델은 단독 점수보다 이쪽이 관건이다.
    BLEND_W = [0.1, 0.2, 0.3]

    ref, ref_preds = None, None
    for group, name, make, desc in picked:
        scores, alphas, preds = [], [], []
        t = time.time()
        try:
            for Y in folds:
                tr = train[train["season"] < Y]
                va = train[train["season"] == Y]
                y = va[TARGET].to_numpy()
                base = y.mean() * (1 - y.mean())
                center = float(tr[TARGET].mean())
                m = make(features)
                m.fit(tr[features], tr[TARGET])
                p = m.predict_proba(va[features])[:, 1]
                scores.append(score_of(y, p, base))
                alphas.append(fit_alpha(y, p, center))
                preds.append((y, p, base, center))
        except Exception as e:                       # 계열마다 터지는 지점이 다르다
            print(f"{name:13s} 실패: {type(e).__name__}: {str(e)[:70]}")
            continue

        # a 이식 — 각 폴드에 '나머지 폴드들의 평균 a' 를 적용한다. 자기 폴드의
        # 정답으로 맞춘 a 를 쓰면 당연히 이득이라 의미가 없다 (4-5 가 기각한 지점).
        trans = []
        for i, (y, p, base, center) in enumerate(preds):
            a_out = float(np.mean([a for j, a in enumerate(alphas) if j != i]))
            trans.append(score_of(y, np.clip(center + a_out * (p - center), 0, 1),
                                  base))

        mean = float(np.mean(scores))
        if ref is None:
            ref, ref_preds = scores, [p for _, p, _, _ in preds]
        d = mean - float(np.mean(ref))
        dt = float(np.mean(trans)) - mean
        print(f"{name:13s}" + "".join(f"{s:10.2f}" for s in scores)
              + f"{mean:10.2f}{d:+10.2f}   "
              + " ".join(f"{a:6.3f}" for a in alphas) + f"{dt:+9.2f}"
              f"   [{time.time()-t:.0f}s]", flush=True)

        # 혼합 — base 와 섞었을 때 이득이 있는가. 단독으로 지더라도 실패 방식이
        # 다르면 여기서 살아난다. 그게 계열을 바꿔 보는 이유다.
        if name != "base" and ref_preds is not None:
            out = []
            for w in BLEND_W:
                bl = [score_of(y, (1 - w) * pb + w * p, base)
                      for (y, p, base, _), pb in zip(preds, ref_preds)]
                out.append(f"w={w:.1f} {float(np.mean(bl)) - float(np.mean(ref)):+7.2f}"
                           + ("" if all(x > 0 for x in
                                        (np.array(bl) - np.array(ref)))
                              else "*"))
            print(f"{'':13s}  혼합(vs base): " + "   ".join(out)
                  + "     * = 폴드 부호 엇갈림", flush=True)

            # 중심 편차가 줄어드는가 — 혼합 이득의 정체를 가른다.
            # 트리는 season 을 외삽하지 못해 평가 시즌을 '작년처럼' 본다. 선형
            # 모형은 하락 추세를 외삽하므로, 섞으면 중심이 내려가 4-2 의 편차가
            # 줄어들 수 있다. 그렇다면 이건 국면 보정(4-3, 규칙 위반)이 하던
            # 일을 평가셋을 안 보고 대신하는 셈이다.
            w = 0.2
            b_dev = [float(pb.mean() - y.mean())
                     for (y, _, _, _), pb in zip(preds, ref_preds)]
            m_dev = [float(((1 - w) * pb + w * p).mean() - y.mean())
                     for (y, p, _, _), pb in zip(preds, ref_preds)]
            s_dev = [float(p.mean() - y.mean()) for (y, p, _, _) in preds]
            print(f"{'':13s}  중심편차 base " + " ".join(f"{x:+7.4f}" for x in b_dev)
                  + f" | 단독 " + " ".join(f"{x:+7.4f}" for x in s_dev)
                  + f" | 혼합w.2 " + " ".join(f"{x:+7.4f}" for x in m_dev),
                  flush=True)

    print("""
읽는 법.
  vs base  폴드 부호가 갈리면 그 변동폭만큼 할인한다 (4-6). 여기는 1시드라
           폴드당 노이즈가 ±15 다. 20점 미만 차이는 구분되지 않는다
  a        1 보다 크면 예측을 벌려야 이득 (과소확신), 작으면 좁혀야 이득
  a이식    자기 폴드가 아닌 나머지 폴드의 a 를 적용했을 때의 이득. 이게
           양수여야 실전에 쓸 수 있다. 4-5 는 여기서 음수라 기각했었다
""")


if __name__ == "__main__":
    main()
