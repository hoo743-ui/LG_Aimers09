r"""최종 모델 학습 — 시드 앙상블 + 과신 교정(shrinkage).

배경: 이 문제는 신호가 희박해서(BSS ~0.006) 모델이 조금만 확신해도 Brier 가
손해를 본다. 두 가지로 대응한다.

1. 시드 앙상블 — 같은 설정을 여러 시드로 학습해 평균낸다. 개별 모델의 분산이
   깎이면서 Brier 가 거의 항상 내려간다.
2. 과신 교정 — 예측을 기준선 쪽으로 당긴다.  p' = c + a*(p - c)
   최적 a 는 (p-c) 로 (y-c) 를 설명하는 회귀 기울기라 닫힌 해로 구해진다.
   a < 1 이면 모델이 과신하고 있다는 뜻이다.

저장 형식은 커스텀 클래스를 쓰지 않는다. 평가 서버에서 joblib.load 가 클래스를
찾지 못해 실패하는 사고를 피하려고, sklearn 객체와 순수 dict 만 담는다.

앙상블 구성은 두 가지로 지정한다. 기본은 한 모양(leaves 고정)을 시드만 바꿔
쌓는 시드 앙상블이고, --spec 을 주면 잎 개수가 다른 모델을 섞는 다양성 앙상블이
된다. 어느 쪽이 나은지는 ens_test.py 가 폴드에서 재 준다.

사용법:
    .\.venv\Scripts\python.exe final_train.py                 # 검증만
    .\.venv\Scripts\python.exe final_train.py --seeds 5
    .\.venv\Scripts\python.exe final_train.py --seeds 5 --save # 전체학습+저장
    .\.venv\Scripts\python.exe final_train.py --spec "L7,L10,L15,L20,L31" --save
    .\.venv\Scripts\python.exe final_train.py --spec "L10x3,L7,L15,L20,L31" --save
"""
import argparse
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

DATA_DIR = "./data"
MODEL_PATH = "./model/rf.pkl"
ID = "row_id"
TARGET = "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state"]
VAL_SEASON = 2024
RF_BASELINE = 415.57

# 국면 이동 보정에 쓰는 피처. 직전 경기의 성공률이라 평가 시즌의 국면을 실시간
# 으로 반영한다 — 6시즌 내내 실제 성공률을 표준편차 0.0031 로 맞혔다.
# 실제 이동은 추론 시점(script.py)에 평가셋 전체 평균으로 수행한다.
PREV1 = "asof_pitcher_prev1_game_success_rate"

# 경력 통산 비율 — 국면이 바뀌어도 늦게 따라가는 것들. --detrend 를 주면 시즌별
# 리그 평균을 빼서 "리그 대비 상대 위치"로 바꾼다. 구종 비율은 성향이지 성과가
# 아니라 제외했다. sweep_shift.py 와 같은 목록을 쓴다.
CUMUL = [
    "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_batter_success_rate", "asof_batter_middle_rate",
]

# 이름 -> (leaves, min_leaf, n_iter). ens_test.py 와 같은 표를 쓴다. 반복수는
# 모양마다 다르다 — 잎이 적을수록 한 번의 갱신이 작아 더 많이 돌아야 한다.
SHAPES = {
    "L7":  (7,  1000, 1400),
    "L10": (10, 1000, 1000),
    "L15": (15, 1000,  860),
    "L20": (20, 1000,  700),
    "L31": (31,  200,  530),
}


def add_derived(df):
    """파생 컬럼. script.py 의 같은 이름 함수와 **반드시 일치**해야 한다.

    same_hand — 투수와 타자의 좌우가 같은가.

    동일 손 매치업은 투수에게 유리한데, 그 효과는 pitcher_hand 단독으로도
    batter_hand 단독으로도 주변부 평균이 상쇄돼 0 이다.

              좌타   우타
        좌완    +      -
        우완    -      +

    탐욕적 부스팅은 루트에서 주변부 이득으로 분할을 고르므로 이런 순수
    상호작용을 영원히 선택하지 않는다. 두 컬럼 다 4년 내내 모델 안에 있었는데
    트리 1100 그루가 못 찾았다 — 구조적 사각지대다 (README 4-7).

    3폴드 x 2시드 +24.66. 4-5 기각 목록의 어떤 항목보다 크고, 유일한 합법
    채택이었던 pitcher_id 복원(+8.5)의 세 배다.

    양쪽 다 int {1,2} 라 산술로 결정적이다. pd.factorize 는 등장 순서로 코드를
    매겨 train 과 test 에서 값이 갈리므로 쓰지 말 것.
    """
    df = df.copy()
    df["same_hand"] = (df["pitcher_hand"] == df["batter_hand"]).astype(int)
    return df


DERIVED = ["same_hand"]

# 계열 혼합 (4-8) 과 중심 보정 (4-9) 의 기본값. 둘 다 3폴드에서 부호가 일치하고
# 서로 거의 독립이다 (겹침 -0.1 ~ -3.4). 2024 폴드 기준 각각 +9.34 / +7.81,
# 합쳐서 +16.03.
BLEND_LR_W = 0.05
BLEND_RF_W = 0.20
CENTER_LAM = 0.03


def make_forest(features, seed=42):
    """혼합용 RandomForest (4-8).

    규제를 세게 걸면 튜닝된 HGB 와 대등하다. 1회차 베이스라인이 415.57 이었던
    것은 RF 가 나빠서가 아니라 max_depth=10 으로 규제가 없어 과신했기 때문이다.
    폴드별 강약이 HGB 와 정반대라(2021 +128 / 2024 -92) 섞을 때 값이 있다.
    """
    num = [c for c in features if c not in CAT_COLS]
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                               unknown_value=-1), CAT_COLS),
        ("num", SimpleImputer(strategy="median", add_indicator=True), num),
    ])
    return Pipeline([("pre", pre), ("clf", RandomForestClassifier(
        n_estimators=200, min_samples_leaf=2000, max_features=0.4,
        n_jobs=-1, random_state=seed))])


def make_linear(features):
    """혼합용 로지스틱. 단독으로는 HGB 에 크게 지지만(2024 에서 164.59) 실패
    방식이 달라 섞으면 이득이 난다 — 오차 상쇄다 (4-8).

    결측은 중앙값으로 메우고 결측 표시를 따로 준다. HGB 는 결측을 분기로 직접
    배우지만 선형 모형은 그걸 못 한다.
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
        ("clf", LogisticRegression(C=0.1, max_iter=200, solver="lbfgs")),
    ])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--leaves", type=int, default=10)
    p.add_argument("--min-leaf", type=int, default=1000)
    p.add_argument("--l2", type=float, default=1.0)
    p.add_argument("--n-iter", type=int, default=1000,
                   help="곡선에서 찾은 최적 반복수 (조기종료 없이 고정)")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--spec", default=None,
                   help="다양성 앙상블 구성. 쉼표로 모양 이름을 나열하고 "
                        "'L10x3' 처럼 개수를 붙이면 그 모양을 시드만 바꿔 그만큼 "
                        f"쌓는다. 쓸 수 있는 이름: {', '.join(SHAPES)}. "
                        "주면 --leaves/--min-leaf/--n-iter/--seeds 는 무시된다.")
    p.add_argument("--keep-ids", action="store_true",
                   help="pitcher_id/batter_id 를 남긴다. 기본은 제거 — 점수는 "
                        "같은데(+1.27, 노이즈 범위) 2025 신인 ID 노출을 피할 수 "
                        "있다. 팀 ID 는 빼면 -20.45 라 항상 남긴다.")
    p.add_argument("--calibrate", action="store_true",
                   help="축소 보정을 실제로 적용한다. 기본은 측정만 하고 쓰지 않음 "
                        "— 연도별 편차가 불안정해 전년도 보정을 옮기면 손해였다 "
                        "(offset_test.py 참고).")
    p.add_argument("--detrend", action="store_true",
                   help="누적형 비율 피처에서 시즌별 리그 평균을 뺀다. 절대 수준 "
                        "대신 리그 대비 상대 위치만 남아 국면 이동에 둔감해진다. "
                        "추론에서도 같은 변환이 적용된다 (script.py).")
    p.add_argument("--shift", action="store_true",
                   help="🚫 규칙 위반. 켜지 말 것. 평가셋 전체의 prev1 평균으로 "
                        "예측 중심을 옮기는데, data_description.md 5) 가 "
                        "'평가 데이터 전체를 보고 만든 사후 보정값'을 금지한다. "
                        "제출 불가이며 실험 재현용으로만 남겨둔다 (README 4-6).")
    p.add_argument("--no-derived", action="store_true",
                   help="파생 컬럼(same_hand)을 빼고 학습한다. 기존 결과와 대조용")
    p.add_argument("--blend-lr", type=float, default=BLEND_LR_W,
                   help="로지스틱 혼합 비중 (4-8). 0 이면 끈다")
    p.add_argument("--blend-rf", type=float, default=BLEND_RF_W,
                   help="RandomForest 혼합 비중 (4-8). 0 이면 끈다")
    p.add_argument("--center-lam", type=float, default=CENTER_LAM,
                   help="중심 보정 계수 (4-9). p' = p + lam*(그 행의 prev1 - c). "
                        "lam 과 c 는 학습 시점 상수이고 anchor 는 그 행 자신의 "
                        "공식 입력이므로 행 독립이다 — 4-3 위반(평가셋 전체의 "
                        "prev1 평균으로 중심 이동)과 근본적으로 다르다. 0 이면 끈다")
    p.add_argument("--skip-val", action="store_true",
                   help="홀드아웃 검증을 건너뛰고 전체 재학습·저장만 한다. 같은 "
                        "설정의 검증 점수를 이미 알고 있을 때 시간을 절반으로 "
                        "줄인다. 저장되는 alpha/center 는 검증과 무관하게 정해지므로 "
                        "결과물은 --skip-val 없이 돌린 것과 동일하다. "
                        "--calibrate 와는 같이 쓸 수 없다 (a 를 잴 데가 없다)")
    p.add_argument("--save", action="store_true")
    return p.parse_args()


def build_members(args):
    """앙상블 구성원 목록 -> [(라벨, leaves, min_leaf, n_iter, seed), ...]

    --spec 이 없으면 예전 그대로 한 모양을 시드만 바꿔 쌓는다.
    """
    if not args.spec:
        return [(f"s{seed}", args.leaves, args.min_leaf, args.n_iter, seed)
                for seed in range(42, 42 + args.seeds)]

    members = []
    for token in args.spec.split(","):
        token = token.strip()
        name, _, count = token.partition("x")
        if name not in SHAPES:
            raise SystemExit(f"모르는 모양: {name!r} (가능: {', '.join(SHAPES)})")
        leaves, min_leaf, n_iter = SHAPES[name]
        for k in range(int(count) if count else 1):
            # 같은 모양을 여러 개 쌓을 때만 시드를 벌린다. 모양이 다르면
            # 시드가 같아도 예측이 충분히 갈린다.
            members.append((f"{name}-s{42 + k}", leaves, min_leaf, n_iter, 42 + k))
    if not members:
        raise SystemExit("--spec 이 비어 있다")
    return members


def make_pipeline(args, features, leaves, min_leaf, n_iter, seed):
    num_cols = [c for c in features if c not in CAT_COLS]
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                               unknown_value=-1), CAT_COLS),
        ("num", "passthrough", num_cols),
    ])
    clf = HistGradientBoostingClassifier(
        max_iter=n_iter,
        learning_rate=args.lr,
        max_leaf_nodes=leaves,
        min_samples_leaf=min_leaf,
        l2_regularization=args.l2,
        early_stopping=False,
        random_state=seed,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def strip_rng(pipe):
    """pkl 에서 numpy 난수 객체를 걷어낸다.

    HGB 는 학습 중 피처 서브샘플링에 쓰려고 `_feature_subsample_rng`
    (np.random.Generator, PCG64) 를 들고 있는데, 이게 그대로 직렬화된다.
    numpy 는 이 객체의 피클 형식을 버전마다 바꿔 왔기 때문에, 평가 서버의
    numpy 가 학습 환경보다 낮으면 로드가 통째로 실패한다.

        ValueError: <class 'numpy.random._pcg64.PCG64'> is not a known
                    BitGenerator module.

    추론 경로는 `_predictors` 만 쓰므로 이 속성은 필요 없다. requirements.txt
    의 numpy 핀과 별개로, 애초에 안 담는 쪽이 안전하다.
    """
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "_feature_subsample_rng"):
        del clf._feature_subsample_rng
    return pipe


def score_of(y, p, base):
    return max(0, 100000 * (1 - ((p - y) ** 2).mean() / base))


def fit_shrinkage(y, p, center):
    """p' = center + a*(p-center) 의 Brier 최소화 해. 회귀 기울기와 같다."""
    d = p - center
    var = (d ** 2).mean()
    if var <= 0:
        return 1.0
    return float((d * (y - center)).mean() / var)


def main():
    args = parse_args()
    members = build_members(args)

    test_cols = pd.read_csv(os.path.join(DATA_DIR, "test.csv"),
                            encoding="utf-8-sig", nrows=0).columns
    all_features = [c for c in test_cols if c != ID]
    drop = [] if args.keep_ids else ["pitcher_id", "batter_id"]
    features = [c for c in all_features if c not in drop]
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"),
                        encoding="utf-8-sig", usecols=all_features + [TARGET])
    if drop:
        print(f"제거 컬럼: {drop}")

    if not args.no_derived:
        train = add_derived(train)
        features = features + DERIVED
        print(f"파생 컬럼: {DERIVED}")

    if args.detrend:
        # 시즌별 리그 평균 차감. 평가셋은 한 시즌이므로 추론에서 평가셋 전체
        # 평균을 빼는 것과 같은 연산이다.
        for c in CUMUL:
            train[c] = train[c] - train.groupby("season")[c].transform("mean")
        print(f"상대화: 누적형 {len(CUMUL)}개에서 시즌별 리그 평균 차감")

    if args.skip_val and args.calibrate:
        raise SystemExit("--skip-val 과 --calibrate 는 같이 못 쓴다 (a 를 잴 데가 없다)")

    if args.skip_val:
        # 저장되는 alpha/center 는 검증과 무관하게 정해지므로, 검증을 건너뛰어도
        # 결과물은 동일하다. 같은 설정의 검증 점수를 이미 아는 경우에만 쓸 것.
        print(f"학습 {len(train)} | 검증 건너뜀 (--skip-val)")
        print(f"설정: lr={args.lr} leaves={args.leaves} min_leaf={args.min_leaf} "
              f"l2={args.l2} n_iter={args.n_iter} seeds={args.seeds}")
        if not args.save:
            raise SystemExit("--skip-val 은 --save 와 같이 써야 의미가 있다")
        save_bundle(args, members, train, features, 1.0)
        return

    is_val = train["season"] == VAL_SEASON
    X_tr, y_tr = train.loc[~is_val, features], train.loc[~is_val, TARGET]
    X_va = train.loc[is_val, features]
    y_va = train.loc[is_val, TARGET].to_numpy()

    r = y_va.mean()
    base = r * (1 - r)
    center = float(y_tr.mean())      # 학습 데이터 성공률. 검증 정답은 쓰지 않는다.

    print(f"학습 {len(X_tr)} | 검증 {len(X_va)} ({VAL_SEASON} 시즌)")
    if args.spec:
        print(f"설정: lr={args.lr} l2={args.l2} spec={args.spec} "
              f"({len(members)}개 모델)")
    else:
        print(f"설정: lr={args.lr} leaves={args.leaves} min_leaf={args.min_leaf} "
              f"l2={args.l2} n_iter={args.n_iter} seeds={args.seeds}")
    print(f"중심값 c={center:.4f} | 검증 기준선 r(1-r)={base:.6f}\n")

    # ---- 구성원별 학습 & 누적 평균 ----
    # 보정 후 열이 실제 판단 기준이다. 앙상블은 예측의 퍼짐을 줄이는데, 보정 전
    # 점수에서는 그게 중심 오차를 덮어주는 이득으로 보여 과대평가된다 (4-5).
    r_hat = float(X_va[PREV1].dropna().mean())

    def shifted(p):
        return score_of(y_va, np.clip(p + (r_hat - p.mean()), 0.0, 1.0), base)

    print(f"{'구성원':>9} {'단일':>9} {'단일+보정':>11} "
          f"{'누적':>9} {'누적+보정':>11}")
    print("-" * 54)
    acc = np.zeros(len(X_va))
    for k, (label, leaves, min_leaf, n_iter, seed) in enumerate(members, start=1):
        m = make_pipeline(args, features, leaves, min_leaf, n_iter, seed)
        t = time.time()
        m.fit(X_tr, y_tr)
        p = m.predict_proba(X_va)[:, 1]
        acc += p
        ens = acc / k
        print(f"{label:>9} {score_of(y_va, p, base):9.2f} {shifted(p):11.2f} "
              f"{score_of(y_va, ens, base):9.2f} {shifted(ens):11.2f}"
              f"   [{time.time()-t:.0f}s]")

    p_ens = acc / len(members)
    s_raw = score_of(y_va, p_ens, base)

    # ---- 계열 혼합 (4-8) 과 중심 보정 (4-9) ----
    # 저장 경로와 **같은 순서**로 적용해 홀드아웃 수치를 낸다. 여기서 재지 않으면
    # 실제 추론이 무엇을 내는지 모른 채 제출하게 된다.
    w_h = 1.0 - args.blend_lr - args.blend_rf
    p_mix = w_h * p_ens
    for label, w, maker in (("로지스틱", args.blend_lr, make_linear),
                            ("RF", args.blend_rf, make_forest)):
        if w <= 0:
            continue
        t = time.time()
        mdl = maker(features)
        mdl.fit(X_tr, y_tr)
        p_o = mdl.predict_proba(X_va)[:, 1]
        p_mix = p_mix + w * p_o
        print(f"\n  혼합용 {label} 단독 {score_of(y_va, p_o, base):8.2f} "
              f"(비중 {w}) [{time.time()-t:.0f}s]")
    s_mix = score_of(y_va, p_mix, base)

    p_fin = p_mix
    if args.center_lam > 0:
        a = X_va[PREV1].fillna(center).to_numpy(dtype=float)
        p_fin = np.clip(p_mix + args.center_lam * (a - center), 0.0, 1.0)
    s_fin = score_of(y_va, p_fin, base)

    # ---- 과신 교정 ----
    a = fit_shrinkage(y_va, p_ens, center)
    p_cal = np.clip(center + a * (p_ens - center), 0.0, 1.0)
    s_cal = score_of(y_va, p_cal, base)

    # ---- 국면 이동 보정 ----
    # 검증 정답은 쓰지 않는다. 검증 시즌의 prev1 평균으로 그 해 성공률을 추정하고
    # 예측 중심을 거기에 맞춘다. 추론 때 script.py 가 하는 것과 같은 계산이다.
    r_hat = float(X_va[PREV1].dropna().mean())
    p_shift = np.clip(p_ens + (r_hat - p_ens.mean()), 0.0, 1.0)
    s_shift = score_of(y_va, p_shift, base)

    print("\n=== 결과 ===")
    print(f"  앙상블 원본     {s_raw:8.2f}   범위 {p_ens.min():.4f}~{p_ens.max():.4f}")
    print(f"  중심 편차      {p_ens.mean() - r:+8.4f}   (예측 {p_ens.mean():.4f} vs 실제 {r:.4f})")
    if args.blend_lr > 0 or args.blend_rf > 0:
        print(f"  + 계열 혼합    {s_mix:8.2f}   ({s_mix - s_raw:+.2f}) "
              f"hgb {w_h:.2f} / rf {args.blend_rf} / lr {args.blend_lr}")
    if args.center_lam > 0:
        print(f"  + 중심 보정    {s_fin:8.2f}   ({s_fin - s_mix:+.2f}) "
              f"lam={args.center_lam}  편차 {p_fin.mean() - r:+.4f}")
    print(f"  ** 최종        {s_fin:8.2f}   (원본 대비 {s_fin - s_raw:+.2f})")
    print(f"  국면 보정 후    {s_shift:8.2f}   ({s_shift - s_raw:+.2f}) "
          f"추정 {r_hat:.4f} {'[적용]' if args.shift else '[미적용 — --no-shift]'}")
    if a < 0.97:
        verdict = "(과신 → 예측을 좁혀야 이득)"
    elif a > 1.03:
        verdict = "(과소확신 → 예측을 벌려야 이득)"
    else:
        verdict = "(1 에 가까움 = 이미 잘 맞음)"
    print(f"  축소계수 a      {a:8.4f}   {verdict}")
    print(f"  교정 시        {s_cal:8.2f}   ({s_cal - s_raw:+.2f})")
    print(f"  RF 베이스라인 대비 {s_raw - RF_BASELINE:+.2f}")

    # 이 a 는 2024 에서 잰 값이라 2025 에 그대로 옮기면 위험하다. offset_test 에서
    # 전년도 보정을 옮겨 쓴 결과가 2승이 아니라 1승 1패였고, 선형보정은 -472 였다.
    # 그래서 기본값은 보정 없이(a=1) 저장한다.
    alpha_out = a if args.calibrate else 1.0
    if not args.calibrate:
        print("  → 보정 미적용(a=1)으로 저장. 적용하려면 --calibrate")

    if not args.save:
        print("\n(저장하려면 --save)")
        return

    save_bundle(args, members, train, features, alpha_out)


def save_bundle(args, members, train, features, alpha_out):
    """전체 데이터로 재학습해 pkl 로 저장한다.

    --skip-val 경로와 공유한다. 저장물의 내용은 검증 여부와 무관하다 —
    alpha 는 --calibrate 없이는 1.0 이고 center 는 전체 학습 데이터의 평균이다.
    """
    # 학습량이 늘었으니 반복수에 10% 여유를 준다.
    print(f"\n전체 데이터 재학습 (반복수 +10%) ...")
    models = []
    for label, leaves, min_leaf, n_iter, seed in members:
        n_iter_full = int(n_iter * 1.1)
        m = make_pipeline(args, features, leaves, min_leaf, n_iter_full, seed)
        t = time.time()
        m.fit(train[features], train[TARGET])
        models.append(strip_rng(m))
        print(f"  {label} (n_iter={n_iter_full}) 완료 [{time.time()-t:.0f}s]")

    # ---- 혼합용 모델들 ----
    # script.py 는 dict 하나(구형)와 리스트(신형)를 모두 읽는다.
    blend = []
    for label, w, maker in (("로지스틱", args.blend_lr, make_linear),
                            ("RF", args.blend_rf, make_forest)):
        if w <= 0:
            continue
        t = time.time()
        mdl = maker(features)
        mdl.fit(train[features], train[TARGET])
        blend.append({"model": mdl, "weight": float(w), "kind": label})
        print(f"  혼합 {label} (비중 {w}) 완료 [{time.time()-t:.0f}s]")
    blend = blend or None

    center_val = float(train[TARGET].mean())
    shift_cfg = None
    if args.center_lam > 0:
        # anchor 는 그 행 자신의 prev1. c 는 학습 데이터 성공률(상수).
        shift_cfg = {"feature": PREV1, "lam": float(args.center_lam),
                     "c": center_val}
        print(f"  중심 보정: {PREV1} lam={args.center_lam} c={center_val:.4f}")

    bundle = {
        "models": models,
        "alpha": alpha_out,
        "center": center_val,
        # 계열 혼합 (4-8). 순수 sklearn 객체라 6-3 을 지킨다
        "blend": blend,
        # 중심 보정 (4-9). 그 행 자신의 anchor 만 쓰므로 행 독립이다
        "center_shift": shift_cfg,
        # 학습에 쓴 컬럼과 순서. 추론 때 test.csv 에서 이대로 골라내야 한다.
        # 열이 하나라도 다르면 ColumnTransformer 가 이름 불일치로 실패한다.
        "features": list(features),
        # 어떤 구성으로 만든 pkl 인지 남긴다. script.py 는 읽지 않는다.
        "spec": [label for label, *_ in members],
        # 국면 이동 보정. script.py 가 평가셋에서 이 피처의 평균을 내어
        # 예측 중심을 옮긴다. 없으면 보정을 하지 않는다.
        "shift": {"feature": PREV1, "bias": 0.0} if args.shift else None,
        # 피처 정의의 일부다. 학습에서 뺐으면 추론에서도 반드시 뺀다.
        "detrend": {"columns": list(CUMUL)} if args.detrend else None,
        "note":"predict: mean(predict_proba[:,1]) -> center + alpha*(p-center) -> clip(0,1)",
    }
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(bundle, MODEL_PATH, compress=3)
    print(f"\n저장 완료: {MODEL_PATH} "
          f"({os.path.getsize(MODEL_PATH)/1e6:.1f} MB, 모델 {len(models)}개)")


if __name__ == "__main__":
    main()
