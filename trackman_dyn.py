r"""7-1 가설 B — Trackman 프로필의 **변화량**을 피처로 준다.

4-4 에서 정적 프로필이 실패한 이유는 명확했다. 프로필은 투수당 값 하나인 상수라
`pitcher_id` 가 이미 아는 것을 다른 좌표계로 다시 말하는 것에 가까웠다.

변화량은 그 비판을 피한다. '작년 대비 구속이 2km/h 떨어졌다'는 투수의 정체성이
아니라 그 시점의 상태이고, 같은 투수라도 해마다 다른 값을 갖는다. ID 로는 표현
할 수 없다.

dyn_probe.py 로 전제는 확인했다 — 연도간 패스트볼 구속 변화는 표준편차
1.88 km/h 이고 54.8% 가 1km/h 넘게 움직인다. 죽은 값이 아니다.

## 시점 규칙

season S 의 행에는 **그 투수가 마지막으로 던진 두 시즌**(둘 다 S 미만)의 차이를
준다. 연속일 필요는 없다 — 부상으로 한 해 쉰 경우 그 공백 자체가 의미다.
평가(2025)에서는 2024 와 2023 이 쓰이고, 학습 행에도 똑같은 규칙이 적용되므로
학습과 추론의 피처 의미가 일치한다. 미래 정보는 어느 쪽에도 새지 않는다.

행 독립도 지킨다 — 값은 투수·시즌 단위이고 평가셋의 다른 행을 보지 않는다.

## 폴드 선택

**fold 2021 은 쓸 수 없다.** 변화량은 두 개의 이전 시즌이 필요해 season 2021
부터만 값이 생기는데, fold 2021 의 학습 데이터는 2019~2020 뿐이라 변화량이
전부 결측이다. 학습에서 한 번도 본 적 없는 피처는 검증에서 쓸 수 없다.
fold 2022 는 학습 중 2021 한 해만 값이 있어 약하고, fold 2024 가 2021~2023
세 시즌을 갖춰 가장 정보가 많다. 그래서 2024 를 주 폴드로 본다.

4-6 의 '최소 3폴드' 기준을 여기서는 구조적으로 충족할 수 없다. 그만큼 결론의
확신도를 낮춰 읽어야 한다.

사용법:
    .\.venv\Scripts\python.exe trackman_dyn.py --folds 2024
    .\.venv\Scripts\python.exe trackman_dyn.py --folds 2022,2024 --seeds 42,43
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
LR, LEAVES, MIN_LEAF, L2, N_ITER = 0.02, 10, 1000, 1.0, 1400

# 한 시즌에 이만큼은 던져야 그 시즌 프로필을 신뢰한다. 표본이 적으면 변화량이
# 실제 변화가 아니라 추정 오차가 된다.
MIN_PITCHES = 300

# 변화량을 잴 양. 4-5 의 delta 실패(9개를 한꺼번에 넣어 상관 높은 컬럼만 늘었다)
# 를 반복하지 않도록 제구와 직결되는 것만 고른다.
#   구속 평균  — 힘의 변화. 떨어지면 부상·노쇠, 오르면 역할 변화
#   릴리스 산포 3종 — 반복성의 변화. 제구의 물리적 실체 그 자체다
DELTA_MEAN = ["rel_speed"]
DELTA_STD = ["rel_height", "rel_side", "extension"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--folds", default="2024")
    p.add_argument("--seeds", default="42")
    p.add_argument("--map", default="pitcher_id_map.csv")
    p.add_argument("--min-conf", type=float, default=0.9)
    p.add_argument("--variants", default="base,delta")
    return p.parse_args()


def season_profile(tm, id_map):
    """(pitcher_id, season) -> 그 시즌의 패스트볼 프로필."""
    tm = tm[tm["pitch_type_group"] == "fastball"].copy()
    tm["pitcher_id"] = tm["pitcher_trackman_id"].map(
        id_map.set_index("pitcher_trackman_id")["pitcher_id"])
    tm = tm.dropna(subset=["pitcher_id"])
    tm["pitcher_id"] = tm["pitcher_id"].astype(np.int64)

    g = tm.groupby(["pitcher_id", "season"])
    prof = pd.DataFrame({"n": g.size()})
    for c in DELTA_MEAN:
        prof[c] = g[c].mean()
    for c in DELTA_STD:
        prof[f"{c}_std"] = g[c].std()
    return prof[prof["n"] >= MIN_PITCHES].drop(columns="n")


def build_delta_table(prof, seasons):
    """행이 속한 시즌 S 기준으로, 그 투수가 마지막으로 던진 두 시즌의 차이.

    반환 인덱스는 (pitcher_id, season) 이고 season 은 **적용 대상 시즌 S** 다.
    프로필이 만들어진 시즌이 아니라는 점에 주의 — 조인 키를 맞추기 위한 것이다.
    """
    prof = prof.sort_index()
    cols = list(prof.columns)
    out = []
    for S in seasons:
        past = prof[prof.index.get_level_values("season") < S]
        if past.empty:
            continue
        # 투수별로 가장 최근 두 시즌을 고른다. 연속이 아니어도 된다.
        past = past.reset_index().sort_values(["pitcher_id", "season"])
        last = past.groupby("pitcher_id").tail(1).set_index("pitcher_id")
        prev = (past.groupby("pitcher_id").tail(2)
                .groupby("pitcher_id").head(1).set_index("pitcher_id"))
        # tail(2).head(1) 은 시즌이 하나뿐인 투수에게 last 와 같은 행을 준다.
        # 그 경우 차이가 0 이 되어 '변화 없음'으로 오해되므로 걸러낸다.
        both = last.index[last["season"] != prev["season"].reindex(last.index)]
        if not len(both):
            continue

        d = pd.DataFrame(index=both)
        for c in cols:
            d[f"dtm_{c}"] = last.loc[both, c] - prev.loc[both, c]
        # 공백 자체가 정보다 — 한 해 쉬었으면 2, 연속이면 1
        d["dtm_gap"] = last.loc[both, "season"] - prev.loc[both, "season"]
        d["season"] = S
        out.append(d.reset_index().rename(columns={"index": "pitcher_id"}))

    if not out:
        return None
    return pd.concat(out).set_index(["pitcher_id", "season"])


def make_model(cols, seed):
    num = [c for c in cols if c not in CAT_COLS]
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                               unknown_value=-1), CAT_COLS),
        ("num", "passthrough", num),
    ])
    clf = HistGradientBoostingClassifier(
        max_iter=N_ITER, learning_rate=LR, max_leaf_nodes=LEAVES,
        min_samples_leaf=MIN_LEAF, l2_regularization=L2,
        early_stopping=False, random_state=seed)
    return Pipeline([("pre", pre), ("clf", clf)])


def best_curve(clf, X_t, y, base):
    best = (0, 0.0)
    for i, proba in enumerate(clf.staged_predict_proba(X_t), start=1):
        p = proba[:, 1]
        s = max(0.0, 100000 * (1 - ((p - y) ** 2).mean() / base))
        if s > best[1]:
            best = (i, s)
    return best


def main():
    args = parse_args()
    folds = [int(f) for f in args.folds.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    variants = args.variants.split(",")

    id_map = pd.read_csv(args.map)
    id_map = id_map[id_map["conf"] >= args.min_conf]
    print(f"대응표 {len(id_map)}명 (신뢰도 {args.min_conf} 이상)")

    tm = pd.read_csv(f"{DATA_DIR}/trackman_history.csv", encoding="utf-8-sig",
                     usecols=["season", "pitcher_trackman_id", "pitch_type_group",
                              "rel_speed", "rel_height", "rel_side", "extension"])
    prof = season_profile(tm, id_map)
    print(f"시즌 프로필 {len(prof)}건 "
          f"(투수 {prof.index.get_level_values('pitcher_id').nunique()}명, "
          f"시즌당 {MIN_PITCHES}구 이상)")

    # 변화량이 투수간 차이에 비해 얼마나 큰지 — 작으면 신호가 없다는 뜻이다
    print("\n=== 변화량 대 투수간 차이 ===")
    for c in DELTA_MEAN + [f"{c}_std" for c in DELTA_STD]:
        between = prof[c].groupby(level="pitcher_id").mean().std()
        within = prof[c].groupby(level="pitcher_id").diff().dropna()
        if len(within):
            print(f"  {c:18s} 투수간 표준편차 {between:7.3f} | "
                  f"연도간 변화 표준편차 {within.std():7.3f} "
                  f"(비 {within.std()/between:5.2f})")

    test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig",
                            nrows=0).columns
    cols = [c for c in test_cols if c != ID]
    train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                        usecols=cols + [TARGET])

    seasons = sorted(train["season"].unique())
    dtab = build_delta_table(prof, seasons)
    if dtab is None:
        raise SystemExit("변화량 표가 비었다")
    dcols = list(dtab.columns)
    print(f"\n변화량 피처 {len(dcols)}개: {dcols}")

    joined = train.join(dtab, on=["pitcher_id", "season"])
    cov = joined[dcols[0]].notna()
    print("\n시즌별 변화량 커버리지:")
    for S in seasons:
        m = train["season"] == S
        print(f"  {S}  {cov[m].mean():6.1%}  ({int(cov[m].sum()):,} / {int(m.sum()):,})")

    res = {}
    for Y in folds:
        tr_mask = joined["season"] < Y
        va_mask = joined["season"] == Y
        tr, va = joined[tr_mask], joined[va_mask]
        y = va[TARGET].to_numpy()
        base = y.mean() * (1 - y.mean())

        tr_cov = cov[tr_mask].mean()
        print(f"\n--- fold {Y} | 학습 커버리지 {tr_cov:.1%} "
              f"검증 커버리지 {cov[va_mask].mean():.1%} ---")
        if tr_cov < 0.05:
            print("  학습에 변화량이 거의 없다. 이 폴드는 정보가 없으므로 건너뛴다.")
            continue

        for seed in seeds:
            for label in variants:
                use = cols + dcols if label == "delta" else cols
                t = time.time()
                m = make_model(use, seed)
                m.fit(tr[use], tr[TARGET])
                X_t = m.named_steps["pre"].transform(va[use])
                it, sc = best_curve(m.named_steps["clf"], X_t, y, base)
                res.setdefault((Y, seed), {})[label] = sc
                print(f"  seed{seed} {label:6s} {sc:9.2f} (iter {it:4d})"
                      f"  [{time.time()-t:.0f}s]")

    print("\n=== 요약 ===")
    diffs = []
    for (Y, seed), r in sorted(res.items()):
        if "base" in r and "delta" in r:
            d = r["delta"] - r["base"]
            diffs.append(d)
            print(f"  fold{Y} seed{seed}  base {r['base']:9.2f}  "
                  f"delta {r['delta']:9.2f}  차이 {d:+8.2f}")
    if diffs:
        print(f"\n  평균 차이 {np.mean(diffs):+.2f}  "
              f"(표본 {len(diffs)}, 표준편차 {np.std(diffs):.2f})")
        print("  4-6 기준 시드 노이즈는 ±15, 폴드 간 차이는 ±45 다. "
              "이보다 작으면 채택 근거가 못 된다.")


if __name__ == "__main__":
    main()
