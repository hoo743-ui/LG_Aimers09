r"""선수별 상호작용 편차를 피처로 주고 이득을 잰다.

가설. `asof_*` 는 전부 주변부 통계다 — 투수 통산 성공률, 직전 N경기 성공률.
조건부 통계는 하나도 없다. 리그 평균 좌우 효과는 pitcher_hand/batter_hand 로
이미 모델에 있지만 "이 투수가 유독 좌타에 약하다"는 개인차는 어디에도 없다.
그리고 max_leaf_nodes=10 짜리 트리는 800명 x 좌우 의 상호작용을 스스로
합성하지 못한다. 준다면 피처로 미리 계산해 줘야 한다.

probe_interact.py 로 실재 여부는 이미 확인했다 (2019~22 vs 2023~24 편차 상관
pitcher x batter_hand +0.53, batter x pitcher_hand +0.42). 여기서는 그게 실제
점수로 환산되는지를 잰다.

무엇을 주는가. 원시 조건부 비율이 아니라 **상호작용 편차**다.

    dev[p,h] = rate[p,h] - rate[p] - (rate[h] - rate[all])

rate[p] 는 asof_pitcher_success_rate 가, rate[h] 는 batter_hand 가 이미 모델에
들어 있다. 원시 비율을 주면 그 둘을 다시 말하는 셈이고, 그게 4-5 기각 목록
절반의 사인이다. 잔차만 넘긴다.

표본이 작은 셀은 노이즈이므로 n/(n+k) 로 0 쪽으로 당긴다. k 는 잡음분산/참분산
= 0.25/var(t) 로 잡았다 — probe 의 기울기 x 편차분산에서 나온 값이다.

시점 규칙. 시즌 s 행의 인코딩은 **시즌 < s** 이력으로만 만든다. asof_* 와 같은
규약이고, 평가 조건(2025 <- 2019~2024)과 같다. 자기 타깃이 새지 않는다.
2019 행은 이력이 없어 결측이며, HGB 는 결측을 분기로 직접 학습한다.

행 독립. 인코딩은 train.csv 로만 만들고 평가행에는 그 행 자신의 pitcher_id /
batter_hand 로 조인한다. 평가셋의 다른 행을 일절 보지 않는다.

사용법:
    .\.venv\Scripts\python.exe interact_feat.py                 # 1시드 빠른 판정
    .\.venv\Scripts\python.exe interact_feat.py --seeds 2       # 확정용
"""
import argparse
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
LR, LEAVES, MIN_LEAF, L2 = 0.02, 10, 1000, 1.0
# 2023 은 어떤 구성으로도 0점이라 비교에 정보가 없다 (README 4-6). 제외한다.
FOLDS = [2021, 2022, 2024]

# (접두사, unit, cond, k)
#   k = 0.25 / var(참신호). probe_interact.py 의 기울기 x 편차분산에서 나온다.
#   참신호가 작을수록 k 가 커지고 더 세게 0 으로 당겨진다.
#   group 이 있으면 cond 효과를 리그 전체가 아니라 그 group 안에서 뺀다.
#   px_hand 는 리그 전체 rate[batter_hand] 를 빼므로 좌완/우완의 리그 평균
#   플래툰 효과가 개인 편차에 섞여 남는다. px_hand_i 는 pitcher_hand 안에서
#   빼서 순수 개인차만 남긴다. 둘을 비교하면 이득이 리그 매치업에서 온 것인지
#   개인차에서 온 것인지 갈린다 — 전자면 선수별 인코딩 없이 컬럼 하나로 끝난다.
SPECS = [
    ("px_hand",    "pitcher_id", "batter_hand",    650, None),
    ("bx_hand",    "batter_id",  "pitcher_hand",  1225, None),
    ("px_strikes", "pitcher_id", "strikes_before", 3050, None),
    ("px_count",   "pitcher_id", "count_state",   1590, None),
    ("px_hand_i",  "pitcher_id", "batter_hand",    650, "pitcher_hand"),
    ("bx_hand_i",  "batter_id",  "pitcher_hand",  1225, "batter_hand"),
]

# 이름 -> 실제 컬럼. 한꺼번에 다 넣으면 유효한 것이 나머지에 희석된다 —
# trackman 26개를 한 번에 넣어 실패한 것과 같은 구조를 피한다 (4-4).
#   hand_mix : pitcher_hand x batter_hand 를 한 컬럼으로. 선수별 인코딩이 아니라
#              리그 매치업 그 자체다. 커버리지 100%, 누수 위험 0
#   park     : 구장. top_bottom 으로 복원한다 (초 = 홈팀이 투구)
#
# 교차 컬럼 — 선수별 인코딩이 아니라 기존 피처 두 개를 곱한 것뿐이다. 커버리지
# 100%, 누수 위험 0, 파생 1줄. 노리는 것은 **주변부 효과가 0 인 쌍**이다.
# hand_mix 가 그랬다 — 동일 손 매치업이 투수에게 유리한데 좌/좌든 우/우든
# 방향이 같아서, pitcher_hand 만 봐도 batter_hand 만 봐도 아무 이득이 없다.
# 탐욕적 트리는 루트에서 주변부 이득으로 분할을 고르므로 이런 쌍을 영원히
# 선택하지 않는다. 1100 그루를 돌려도 못 찾는다.
BLOCKS = {
    "handmix":  ["hand_mix"],
    "samehand": ["same_hand"],
    "park":     ["park"],
    "cnt":      ["count_state"],       # balls x strikes 명시 교차
    "hand1b":   ["hand_1b"],           # 좌완의 1루 견제/슬라이드스텝
    "handbase": ["hand_base"],         # 위를 주자 상황 전체로 확장
    "mixcnt":   ["mix_count"],         # 좌우 매치업 x 카운트
    "platoon":  ["px_hand_dev", "px_hand_n", "bx_hand_dev", "bx_hand_n"],
    "indiv":    ["px_hand_i_dev", "px_hand_i_n", "bx_hand_i_dev", "bx_hand_i_n"],
    "count":    ["px_strikes_dev", "px_strikes_n", "px_count_dev", "px_count_n"],
}

VARIANTS = [
    ("base",            []),
    ("handmix",         ["handmix"]),                 # 리그 매치업만 (4수준)
    ("samehand",        ["samehand"]),                # 같은 것, 이진 1컬럼
    ("platoon",         ["platoon"]),                 # 리그 + 개인 (1차 확인분)
    ("indiv",           ["indiv"]),                   # 순수 개인차만
    ("handmix+indiv",   ["handmix", "indiv"]),        # 분해 후 재조립
    ("samehand+indiv",  ["samehand", "indiv"]),       # 위를 제출 형태로
    ("park",            ["park"]),
    ("platoon+park",    ["platoon", "park"]),
    ("platoon+count",   ["platoon", "count"]),
    # 교차 컬럼 단독 — 어느 쌍이 실제로 주변부 0 인지 하나씩 가려낸다
    ("cnt",             ["cnt"]),
    ("hand1b",          ["hand1b"]),
    ("handbase",        ["handbase"]),
    ("mixcnt",          ["mixcnt"]),
    ("교차 전부",         ["handmix", "cnt", "hand1b", "handbase", "mixcnt"]),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-iter", type=int, default=1100,
                   help="확정 설정과 같은 값으로 고정한다. 폴드마다 최적점을 "
                        "고르면 그 폴드의 정답을 훔쳐보는 셈이라 변형 간 비교가 "
                        "왜곡된다 (feat_test.py 와 같은 규약)")
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--folds", default=",".join(map(str, FOLDS)))
    p.add_argument("--only", default=None,
                   help="쉼표로 변형 이름을 골라 그것만 돌린다. base 는 대비 기준이라 "
                        "항상 먼저 들어간다")
    p.add_argument("--k-scale", type=float, default=1.0,
                   help="모든 축소 상수 k 에 곱한다. 크면 더 세게 0 으로 당긴다 — "
                        "이론값(0.25/참분산)이 최적인지 확인용")
    return p.parse_args()


def build_dev(train, prefix, unit, cond, k, group=None):
    """시즌 s 의 행에 '시즌 < s' 이력으로 만든 상호작용 편차와 셀 표본수를 붙인다.

    인코딩이 시즌별로 한 번씩만 계산되므로 폴드가 바뀌어도 재계산이 필요 없다 —
    시즌 s 의 값은 언제나 s 이전만 본다.

    group 을 주면 cond 효과를 리그 전체가 아니라 그 group 안에서 뺀다.
    """
    n_rows = len(train)
    dev_out = np.full(n_rows, np.nan)
    n_out = np.full(n_rows, np.nan)
    seasons = sorted(train["season"].unique())

    for s in seasons[1:]:            # 첫 시즌은 이력이 없어 결측으로 둔다
        hist = train[train["season"] < s]
        g = hist.groupby([unit, cond], observed=True)[TARGET].agg(["mean", "size"])
        ru = hist.groupby(unit, observed=True)[TARGET].mean()

        idx = g.index
        u_vals = idx.get_level_values(0)
        c_vals = idx.get_level_values(1)

        if group is None:
            rc = hist.groupby(cond, observed=True)[TARGET].mean()
            cond_eff = rc.reindex(c_vals).to_numpy() - hist[TARGET].mean()
        else:
            # unit -> group 대응 (투수의 좌우는 고정이라 first 로 충분하다)
            u2g = hist.groupby(unit, observed=True)[group].first()
            rgc = hist.groupby([group, cond], observed=True)[TARGET].mean()
            rg = hist.groupby(group, observed=True)[TARGET].mean()
            gv = u2g.reindex(u_vals).to_numpy()
            cond_eff = (rgc.reindex(pd.MultiIndex.from_arrays([gv, c_vals])).to_numpy()
                        - rg.reindex(gv).to_numpy())

        dev = g["mean"].to_numpy() - ru.reindex(u_vals).to_numpy() - cond_eff
        n = g["size"].to_numpy()
        # 표본 축소 — 셀이 작을수록 0(= 상호작용 없음) 쪽으로 당긴다
        table = pd.Series(dev * (n / (n + k)), index=idx)
        n_tab = pd.Series(n, index=idx)

        mask = (train["season"] == s).to_numpy()
        key = pd.MultiIndex.from_arrays(
            [train.loc[mask, unit].to_numpy(), train.loc[mask, cond].to_numpy()])
        dev_out[mask] = table.reindex(key).to_numpy()
        n_out[mask] = n_tab.reindex(key).to_numpy()

    return {f"{prefix}_dev": dev_out, f"{prefix}_n": n_out}


def make_model(features, n_iter, seed):
    cat = [c for c in CAT_COLS if c in features]
    num = [c for c in features if c not in cat]
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                               unknown_value=-1), cat),
        ("num", "passthrough", num),
    ])
    clf = HistGradientBoostingClassifier(
        max_iter=n_iter, learning_rate=LR, max_leaf_nodes=LEAVES,
        min_samples_leaf=MIN_LEAF, l2_regularization=L2,
        early_stopping=False, random_state=seed,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def score_of(y, p, base):
    return max(0.0, 100000 * (1 - ((p - y) ** 2).mean() / base))


def main():
    args = parse_args()
    folds = [int(f) for f in args.folds.split(",")]

    test_cols = pd.read_csv(os.path.join(DATA_DIR, "test.csv"),
                            encoding="utf-8-sig", nrows=0).columns
    base_cols = [c for c in test_cols if c != ID]
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"),
                        encoding="utf-8-sig", usecols=base_cols + [TARGET])
    # 카운트 12종을 하나의 코드로. balls/strikes 를 따로 나눠서는 잎 10개짜리
    # 트리가 12칸을 만들지 못한다.
    train["count_state"] = train["balls_before"] * 3 + train["strikes_before"]

    # 리그 좌우 매치업. 선수별 인코딩이 아니므로 커버리지 100% 다.
    # 양쪽 다 int64 {1,2} 라 산술로 결정적이다 — factorize 는 등장 순서로 코드를
    # 매겨 train 과 test 에서 값이 갈리므로 제출 경로에 쓰면 안 된다.
    train["hand_mix"] = train["pitcher_hand"] * 2 + train["batter_hand"]
    # 메커니즘을 그대로 쓴 형태. 동일 손 매치업이 투수에게 유리하다는 것이
    # 전부이므로 이진 1컬럼이면 충분하고, 분할도 한 번이면 된다.
    # hand_mix 4수준은 동일 손 {3,6} 이 순서상 인접하지 않아 두 번 갈라야 한다.
    train["same_hand"] = (train["pitcher_hand"] == train["batter_hand"]).astype(int)
    # 구장. 초(T)는 원정팀 공격이므로 홈팀이 던지고 있고 그 홈팀의 구장이다.
    tb = train["top_bottom"].astype(str).str.upper().str[0]
    if not tb.isin(["T", "B"]).all():
        raise SystemExit(f"top_bottom 값이 T/B 가 아니다: {tb.unique()[:5]}")
    train["park"] = np.where(tb.eq("T"), train["pitcher_team_id"],
                             train["batter_team_id"])
    ph = pd.factorize(train["pitcher_hand"])[0]
    train["hand_1b"] = ph * 2 + train["runner_on_1b"].to_numpy()
    train["hand_base"] = pd.factorize(
        train["pitcher_hand"].astype(str) + "|" + train["base_state"].astype(str))[0]
    train["mix_count"] = train["hand_mix"] * 12 + train["count_state"]

    print(f"{len(train):,} 행 | 기본 피처 {len(base_cols)}개")
    print("  교차 컬럼: " + " | ".join(
        f"{c} {train[c].nunique()}종" for c in
        ("hand_mix", "same_hand", "park", "count_state",
         "hand_1b", "hand_base", "mix_count")))
    print("인코딩 생성 중 ...", flush=True)
    added = {}
    for prefix, unit, cond, k0, group in SPECS:
        k = k0 * args.k_scale
        t = time.time()
        cols = build_dev(train, prefix, unit, cond, k, group)
        added.update(cols)
        d = cols[f"{prefix}_dev"]
        ok = ~np.isnan(d)
        print(f"  {prefix:11s} k={k:7.0f}  커버리지 {ok.mean():6.1%}  "
              f"편차 sd {np.nanstd(d):.4f}  [{time.time()-t:.0f}s]")
    for c, v in added.items():
        train[c] = v

    print(f"\n설정: lr={LR} leaves={LEAVES} min_leaf={MIN_LEAF} "
          f"n_iter={args.n_iter} seeds={args.seeds}")
    print(f"폴드 {folds} (2023 제외 — 어떤 구성으로도 0점)\n")

    header = (f"{'변형':16s}{'피처':>5s}"
              + "".join(f"{f'val{s}':>11s}" for s in folds) + f"{'평균':>10s}")
    print(header)
    print("-" * (len(header) + 8))

    variants = VARIANTS
    if args.only:
        want = {n.strip() for n in args.only.split(",")} | {"base"}
        variants = [v for v in VARIANTS if v[0] in want]
        unknown = want - {v[0] for v in VARIANTS}
        if unknown:
            raise SystemExit(f"모르는 변형: {sorted(unknown)}")

    results = []
    for name, blocks in variants:
        extra = [c for b in blocks for c in BLOCKS[b]]
        features = base_cols + extra
        scores = []
        t = time.time()
        for Y in folds:
            tr = train[train["season"] < Y]
            va = train[train["season"] == Y]
            y = va[TARGET].to_numpy()
            base = y.mean() * (1 - y.mean())
            acc = np.zeros(len(va))
            for seed in range(42, 42 + args.seeds):
                m = make_model(features, args.n_iter, seed)
                m.fit(tr[features], tr[TARGET])
                acc += m.predict_proba(va[features])[:, 1]
            scores.append(score_of(y, acc / args.seeds, base))
        mean = float(np.mean(scores))
        print(f"{name:16s}{len(features):5d}"
              + "".join(f"{s:11.2f}" for s in scores)
              + f"{mean:10.2f}  [{time.time()-t:.0f}s]", flush=True)
        results.append((name, scores, mean))

    base_scores, base_mean = results[0][1], results[0][2]
    print("\n=== base 대비 ===")
    for name, scores, mean in results[1:]:
        d = [s - b for s, b in zip(scores, base_scores)]
        sign = "일치" if all(x > 0 for x in d) or all(x < 0 for x in d) else "★엇갈림"
        print(f"  {name:16s} {mean - base_mean:+8.2f}   폴드별 "
              + " ".join(f"{x:+8.2f}" for x in d) + f"   부호 {sign}")
    print("\n폴드별 부호가 엇갈리면 그 변동폭만큼 할인해서 볼 것 (README 4-6).")


if __name__ == "__main__":
    main()
