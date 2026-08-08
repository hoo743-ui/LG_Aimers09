r"""피처 48개 중 무엇이 실제로 일하는가 — 용량이 조인 모델에서는 이게 해상도다.

왜 지금인가. error_map.py 로 재보니 캘리브레이션으로 되찾을 수 있는 총량은
62.9 점뿐이고 이미 16 을 가져갔다. 남은 격차(증분 전달률 0.66 기준 로컬 +251)는
**해상도**에서 나와야 한다.

그런데 이 모델은 `max_leaf_nodes=10` 이 최적이라고 확정돼 있다 (4-1). 의도적으로
용량을 조인 모델이다. 그런 모델에 48개를 밀어넣으면 탐욕적 분할기가 매 노드에서
48개를 평가하고 노이즈 피처가 우연히 이기는 일이 반복된다. **잎 10개짜리 예산에서
분할 하나를 노이즈에 쓰는 것이 곧 해상도 손실이다.**

feat_test.py 는 ID 제거만 봤다. 체계적 기여도 측정은 한 번도 없었다.

두 가지를 잰다.

  1) 분할 사용량 — 저장된 pkl 의 트리를 뜯어 피처별 분할 횟수와 이득 합.
     학습이 필요 없다. 다만 "많이 쓰인다"가 "유용하다"는 아니다
  2) 순열 중요도 — 2024 폴드에서 그 컬럼만 섞고 점수가 얼마나 떨어지는지.
     이쪽이 실제 기여다. 음수면 **있어서 해로운** 피처다

순열 중요도는 홀드아웃에서 재야 의미가 있다. 저장된 pkl 은 2024 를 포함해
학습했으므로 여기서는 2024 미만으로 따로 학습한다.

    .\.venv\Scripts\python.exe feat_audit.py --splits-only   # 즉시 (학습 없음)
    .\.venv\Scripts\python.exe feat_audit.py                 # 순열까지 (~6분)
"""
import argparse
import os
import time
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

DATA_DIR = "./data"
MODEL_PATH = "./model/rf.pkl"
ID = "row_id"
TARGET = "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state"]
LR, LEAVES, MIN_LEAF, L2, N_ITER = 0.02, 10, 1000, 1.0, 1100
VAL = 2024


# 분할 이득 0.1% 미만. 상당수가 다른 컬럼과 중복이다 —
# runner_on_* <-> base_state, score_diff_home <-> score_diff_pitcher_team,
# run_bot/top_before <-> run_total_before.
NEAR_DEAD = ["score_diff_home", "outs_before", "run_bot_before",
             "away_win_expectancy", "runner_on_2b", "top_bottom",
             "runner_on_3b", "num_runners_on", "runner_on_1b",
             "asof_pitcher_pitchmix_n"]

# 절제 실험. 잎 10개짜리 예산에서 무엇을 빼면 판별이 좋아지는가.
ABLATIONS = [
    ("base", []),
    ("dead10 제거", NEAR_DEAD),
    ("season 제거", ["season"]),
    ("dead10+season", NEAR_DEAD + ["season"]),
    ("run/주자 계열 정리", ["run_bot_before", "run_top_before",
                            "score_diff_home", "runner_on_1b",
                            "runner_on_2b", "runner_on_3b",
                            "num_runners_on", "away_win_expectancy",
                            "home_win_expectancy"]),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--splits-only", action="store_true")
    p.add_argument("--ablate", action="store_true",
                   help="절제 실험 (3폴드). 순열 중요도 대신 실제 점수로 잰다")
    p.add_argument("--only", default=None)
    p.add_argument("--folds", default="2021,2022,2024")
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def run_ablation(args):
    """피처를 빼고 3폴드에서 점수를 잰다.

    순열 중요도는 상관된 피처끼리 서로를 가려 개별 값이 과소평가된다. 묶어서
    빼는 쪽이 '예산이 풀리면 좋아지는가'라는 질문에 직접 답한다.
    """
    folds = [int(f) for f in args.folds.split(",")]
    test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig",
                            nrows=0).columns
    base_cols = [c for c in test_cols if c != ID]
    train = add_derived(pd.read_csv(f"{DATA_DIR}/train.csv",
                                    encoding="utf-8-sig",
                                    usecols=base_cols + [TARGET]))
    all_feats = base_cols + ["same_hand"]

    picked = ABLATIONS
    if args.only:
        want = {s.strip() for s in args.only.split(",")} | {"base"}
        picked = [v for v in ABLATIONS if v[0] in want]

    print(f"{len(train):,} 행 | 피처 {len(all_feats)}개 | "
          f"폴드 {folds} | 시드 {args.seeds}\n")
    header = (f"{'변형':18s}{'피처':>5s}"
              + "".join(f"{f'val{s}':>11s}" for s in folds) + f"{'평균':>10s}")
    print(header)
    print("-" * (len(header) + 8))

    ref = None
    for name, drop in picked:
        feats = [c for c in all_feats if c not in drop]
        scores = []
        t = time.time()
        for Y in folds:
            tr = train[train["season"] < Y]
            va = train[train["season"] == Y]
            y = va[TARGET].to_numpy()
            denom = y.mean() * (1 - y.mean())
            acc = np.zeros(len(va))
            for s in range(args.seed, args.seed + args.seeds):
                cat = [c for c in CAT_COLS if c in feats]
                pre = ColumnTransformer([
                    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                                           unknown_value=-1), cat),
                    ("num", "passthrough", [c for c in feats if c not in cat]),
                ])
                m = Pipeline([("pre", pre), ("clf",
                    HistGradientBoostingClassifier(
                        max_iter=N_ITER, learning_rate=LR, max_leaf_nodes=LEAVES,
                        min_samples_leaf=MIN_LEAF, l2_regularization=L2,
                        early_stopping=False, random_state=s))])
                m.fit(tr[feats], tr[TARGET])
                acc += m.predict_proba(va[feats])[:, 1]
            scores.append(score_of(y, acc / args.seeds, denom))
        mean = float(np.mean(scores))
        if ref is None:
            ref = scores
        d = [a - b for a, b in zip(scores, ref)]
        tag = "" if name == "base" else (
            "  일치" if all(x > 0 for x in d) or all(x < 0 for x in d)
            else "  ★엇갈림")
        print(f"{name:18s}{len(feats):5d}"
              + "".join(f"{s:11.2f}" for s in scores)
              + f"{mean:10.2f}"
              + ("" if name == "base" else
                 f"  ({mean - float(np.mean(ref)):+7.2f})" + tag)
              + f"  [{time.time()-t:.0f}s]", flush=True)


def add_derived(df):
    df = df.copy()
    df["same_hand"] = (df["pitcher_hand"] == df["batter_hand"]).astype(int)
    return df


def feature_order(features):
    """ColumnTransformer 출력 순서 = 범주형 먼저, 그 다음 나머지."""
    cat = [c for c in CAT_COLS if c in features]
    return cat + [c for c in features if c not in cat]


def split_stats(bundle):
    """트리 노드를 뜯어 피처별 분할 횟수와 이득 합을 센다.

    HGB 의 predictor 는 구조화 배열이고 `is_leaf`, `feature_idx`, `gain` 을
    들고 있다. 학습 없이 pkl 만으로 읽힌다.
    """
    names = feature_order(bundle["features"])
    cnt = defaultdict(int)
    gain = defaultdict(float)
    n_tree = 0
    for pipe in bundle["models"]:
        clf = pipe.named_steps["clf"]
        for stage in clf._predictors:
            for pred in stage:
                nodes = pred.nodes
                internal = nodes[nodes["is_leaf"] == 0]
                n_tree += 1
                for fi, g in zip(internal["feature_idx"], internal["gain"]):
                    cnt[names[fi]] += 1
                    gain[names[fi]] += float(g)
    return names, cnt, gain, n_tree


def score_of(y, p, base):
    return max(0.0, 100000 * (1 - ((p - y) ** 2).mean() / base))


def main():
    args = parse_args()

    if args.ablate:
        run_ablation(args)
        return

    # ---- 1) 분할 사용량 (학습 없음) ----
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"{MODEL_PATH} 없음")
    bundle = joblib.load(MODEL_PATH)
    names, cnt, gain, n_tree = split_stats(bundle)
    tot_split = sum(cnt.values())
    tot_gain = sum(gain.values())
    print(f"저장된 모델: 앙상블 {len(bundle['models'])}개, 트리 {n_tree:,}그루, "
          f"내부 분할 {tot_split:,}개")
    print(f"피처 {len(names)}개\n")

    rows = [(n, cnt.get(n, 0), gain.get(n, 0.0)) for n in names]
    rows.sort(key=lambda r: -r[2])
    print("=== 분할 사용량 (이득 합 기준) ===")
    print(f"{'피처':>40}{'분할수':>9}{'비중':>8}{'이득비중':>10}")
    print("-" * 68)
    dead = []
    for n, c, g in rows:
        share_g = g / tot_gain if tot_gain else 0
        print(f"{n:>40}{c:9d}{c/tot_split:8.2%}{share_g:10.3%}")
        if share_g < 0.001:
            dead.append(n)
    print(f"\n이득 비중 0.1% 미만 = {len(dead)}개")
    print("  " + ", ".join(dead) if dead else "  없음")

    if args.splits_only:
        print("\n(순열 중요도까지 보려면 --splits-only 를 빼고 실행)")
        return

    # ---- 2) 순열 중요도 (2024 홀드아웃) ----
    test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig",
                            nrows=0).columns
    base_cols = [c for c in test_cols if c != ID]
    train = add_derived(pd.read_csv(f"{DATA_DIR}/train.csv",
                                    encoding="utf-8-sig",
                                    usecols=base_cols + [TARGET]))
    features = base_cols + ["same_hand"]
    tr = train[train["season"] < VAL]
    va = train[train["season"] == VAL].reset_index(drop=True)
    y = va[TARGET].to_numpy()
    denom = y.mean() * (1 - y.mean())

    print(f"\n=== 순열 중요도 ({VAL} 홀드아웃, 학습 {len(tr):,} 행) ===")
    t = time.time()
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                               unknown_value=-1), CAT_COLS),
        ("num", "passthrough", [c for c in features if c not in CAT_COLS]),
    ])
    m = Pipeline([("pre", pre), ("clf", HistGradientBoostingClassifier(
        max_iter=N_ITER, learning_rate=LR, max_leaf_nodes=LEAVES,
        min_samples_leaf=MIN_LEAF, l2_regularization=L2,
        early_stopping=False, random_state=args.seed))])
    m.fit(tr[features], tr[TARGET])
    Xva = va[features]
    s0 = score_of(y, m.predict_proba(Xva)[:, 1], denom)
    print(f"기준 {s0:.2f}  [{time.time()-t:.0f}s]\n", flush=True)

    rng = np.random.default_rng(0)
    imp = []
    t = time.time()
    for i, c in enumerate(features, 1):
        drops = []
        for _ in range(args.repeats):
            X2 = Xva.copy()
            X2[c] = rng.permutation(X2[c].to_numpy())
            drops.append(s0 - score_of(y, m.predict_proba(X2)[:, 1], denom))
        imp.append((c, float(np.mean(drops))))
        if i % 12 == 0:
            print(f"  {i}/{len(features)} [{time.time()-t:.0f}s]", flush=True)

    imp.sort(key=lambda r: -r[1])
    print(f"\n{'피처':>40}{'섞었을 때 손실':>14}{'분할 비중':>11}")
    print("-" * 68)
    for c, d in imp:
        print(f"{c:>40}{d:14.2f}{cnt.get(c, 0)/tot_split:11.2%}")

    harmful = [c for c, d in imp if d < -1.0]
    useless = [c for c, d in imp if -1.0 <= d <= 1.0]
    print(f"\n섞어도 1점 미만 변화 (= 사실상 무기여) {len(useless)}개")
    print("  " + ", ".join(useless) if useless else "  없음")
    print(f"\n섞으면 오히려 좋아짐 (= 해로운 후보) {len(harmful)}개")
    print("  " + ", ".join(harmful) if harmful else "  없음")
    print("""
주의. 순열 중요도는 상관된 피처끼리 서로를 가려준다 — asof_* 계열처럼 겹치는
컬럼이 많으면 개별 값이 과소평가된다. 여기서 낮게 나온 것을 바로 지우지 말고,
**묶어서 제거하는 실험**(interact_feat.py 의 변형 방식)으로 확인할 것.""")


if __name__ == "__main__":
    main()
