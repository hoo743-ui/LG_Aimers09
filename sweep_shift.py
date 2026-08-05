r"""국면 보정을 켠 상태에서 하이퍼파라미터와 피처 구성을 다시 고른다.

지금 확정 설정(L10, min_leaf=1000)은 **보정이 없던 시절에** 고른 것이다. 강한
규제는 예측을 중심 쪽으로 오므려 중심 오차를 줄이는 부수효과가 있었고, 그게
점수로 보상받았다. 이제 중심은 4-6 의 보정이 따로 잡으므로 규제가 하던 일의
일부가 중복이 됐다 — 최적점이 더 날카로운 쪽으로 옮겨갔을 가능성이 크다.

`staged_predict_proba` 를 쓰므로 설정 하나당 학습 한 번으로 전체 반복수 곡선이
나온다. 각 반복 시점마다 보정을 적용해 채점하고, 보정 후 기준으로 최적 반복수를
고른다 (보정 전 기준으로 고른 반복수와 다를 수 있다는 게 이 실험의 요지다).

피처 구성도 같은 틀에서 비교한다.
  all      : 지금 그대로
  no_cumul : 누적형 비율 피처 제거 — 국면을 못 따라가는 것들을 아예 뺀다
  detrend  : 누적형에서 그 시즌의 리그 평균을 뺀다. 절대 수준 대신 "리그 대비
             상대 위치"만 남기므로 국면이 이동해도 의미가 유지된다. 평가 시점에는
             평가셋 자체의 평균을 쓴다 (정답 불필요, 4-6 과 같은 원리)

사용법:
    .\.venv\Scripts\python.exe sweep_shift.py --folds 2024
    .\.venv\Scripts\python.exe sweep_shift.py --folds 2021,2022,2024 --only L31_200
    .\.venv\Scripts\python.exe sweep_shift.py --feat-modes all,no_cumul,detrend --only L31_200
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
PREV1 = "asof_pitcher_prev1_game_success_rate"
DROP_IDS = ["pitcher_id", "batter_id"]

# 경력 통산 비율 — 국면이 바뀌어도 늦게 따라가는 것들. 투구 구종 비율
# (fastball/breaking/offspeed)은 성향이지 성과가 아니라 제외했다.
CUMUL = [
    "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_batter_success_rate", "asof_batter_middle_rate",
]

# 이름 -> (leaves, min_leaf, lr, max_iter). max_iter 는 곡선이 정점을 지나도록
# 넉넉히 준다 — 최적점은 곡선에서 고르지 미리 정하지 않는다.
#
# 1차 격자에서 두 손잡이가 반대로 움직인다는 걸 확인했다 (fold 2024, 보정 후):
#   L10_1000 669.85 | L20_1000 693.12 | L31_200 667.30
# 잎을 늘리면 이득, min_leaf 를 내리면 손해다. 1차 격자는 잎이 많은 설정을
# 전부 작은 min_leaf 와 묶어놔서 두 효과가 상쇄돼 있었다. 이제 분리한다.
#
CONFIGS = {
    # --- 1차에서 이미 측정 ---
    "L10_1000":  (10, 1000, 0.02, 1600),   # 669.85 — 기존 확정 설정
    "L20_1000":  (20, 1000, 0.02, 1100),   # 693.12
    "L31_200":   (31,  200, 0.02,  900),   # 667.30
    # --- 3차: 봉우리(잎 20) 주변 미세 조정 ---
    "L15_1000":  (15, 1000, 0.02, 1300),
    "L25_1000":  (25, 1000, 0.02,  950),
    # --- 2차: 잎을 늘리되 min_leaf 는 크게 유지 ---
    "L31_1000":  (31, 1000, 0.02,  800),
    "L63_1000":  (63, 1000, 0.02,  600),
    "L127_1000": (127, 1000, 0.02, 400),
    # --- min_leaf 축을 200~2000 사이에서 확인 ---
    "L20_500":   (20,  500, 0.02, 1100),
    "L20_2000":  (20, 2000, 0.02, 1200),
    "L31_2000":  (31, 2000, 0.02,  900),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--folds", default="2024",
                   help="쉼표로 구분. 2023 은 어떤 구성으로도 0점이라 뺀다")
    p.add_argument("--only", default=None, help="쉼표로 구분한 설정 이름")
    p.add_argument("--feat-modes", default="all",
                   help="all,no_cumul,detrend 중 쉼표로 구분")
    p.add_argument("--l2", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42,
                   help="시드를 바꿔 재면 봉우리가 진짜인지 노이즈인지 갈린다")
    return p.parse_args()


def add_delta(df):
    """트리가 스스로 못 만드는 차이값을 명시적으로 준다.

    부스팅 트리는 축에 수직인 분할만 쌓으므로 `A - B` 같은 양을 아주 비효율적
    으로 근사한다. 두 값을 따로 주는 것과 그 차이를 주는 것은 전혀 다르다.

    4-2 에서 발견한 구조가 정확히 이 형태다 — 누적형과 최근형의 격차. 시즌
    단위로는 국면 이동이었고, 개별 투수 단위로는 '최근 폼이 통산 대비 어떤가'다.
    지금까지 모델에게 두 값은 줬지만 그 차이는 준 적이 없다.
    """
    out = {}
    for k in (1, 3, 5):
        out[f"d_prev{k}_cumul"] = (df[f"asof_pitcher_prev{k}_game_success_rate"]
                                   - df["asof_pitcher_success_rate"])
        out[f"d_prev{k}_cumul_mid"] = (df[f"asof_pitcher_prev{k}_game_middle_rate"]
                                       - df["asof_pitcher_middle_rate"])
    out["d_prev1_prev5"] = (df["asof_pitcher_prev1_game_success_rate"]
                            - df["asof_pitcher_prev5_game_success_rate"])
    # 투수와 타자의 상대 우위. 맞대결의 방향을 한 축으로 준다.
    out["d_pitcher_batter"] = (df["asof_pitcher_success_rate"]
                               - df["asof_batter_success_rate"])
    out["d_pitcher_batter_mid"] = (df["asof_pitcher_middle_rate"]
                                   - df["asof_batter_middle_rate"])
    return out


def add_shrunk(df):
    """표본 수로 축소한 비율. 경험적 베이즈.

    asof_pitcher_n 이 5인 투수의 성공률 0.70 과 500인 투수의 0.70 은 전혀 다른
    값인데 지금은 같은 컬럼에 섞여 있다. 트리는 n 으로 분할해 이를 흉내낼 수
    있지만, 그러려면 분할을 낭비해야 한다.

        shrunk = (n*rate + k*prior) / (n + k)

    prior 는 그 시즌의 리그 평균이다. detrend 와 달리 수준을 없애지 않는다 —
    표본이 많은 투수의 값은 그대로 두고 적은 쪽만 리그 쪽으로 당긴다.
    """
    out = {}
    pairs = [("asof_pitcher_success_rate", "asof_pitcher_n", "p_succ"),
             ("asof_pitcher_middle_rate", "asof_pitcher_n", "p_mid"),
             ("asof_batter_success_rate", "asof_batter_n", "b_succ")]
    for col, ncol, tag in pairs:
        prior = df.groupby("season")[col].transform("mean")
        n = df[ncol].fillna(0)
        for k in (50, 200):
            out[f"sh{k}_{tag}"] = (n * df[col].fillna(prior) + k * prior) / (n + k)
    return out


def add_fatigue(df):
    """경기 내 시간 구조. 지금 피처에 통째로 빠져 있는 정보다.

    game_id 가 없지만 (시즌, 월, 요일, 양팀) 이 경기를 거의 식별하고(조합당
    중앙값 155투구 = 한 경기 한 팀 분량), row_id 가 시간순이라 순서도 살아 있다.
    같은 키 안에서 이닝이 되감기는 지점을 경기 경계로 삼아 나머지를 가른다.

    누적은 전부 '앞선 행까지'만 세므로 미래 정보가 새지 않는다.

    inning 만으로는 표현할 수 없는 것을 준다 — 7회에 올라온 구원투수는 이닝이
    높아도 투구 수는 적다.
    """
    key = ["season", "game_month", "game_dayofweek",
           "pitcher_team_id", "batter_team_id"]
    grp = df.groupby(key, sort=False)["inning"]
    # 같은 키 안에서 이닝이 줄면 새 경기다.
    new_game = grp.diff().fillna(0) < 0
    seg = new_game.groupby([df[c] for c in key], sort=False).cumsum()
    game = pd.util.hash_pandas_object(
        pd.concat([df[key].astype("int64"), seg.astype("int64")], axis=1),
        index=False)

    out = {}
    # 이 경기에서 이 투수가 앞서 던진 투구 수
    out["pitch_no_in_game"] = df.groupby([game, df["pitcher_id"]],
                                         sort=False).cumcount()
    # 이 경기에서 투수팀이 앞서 던진 투구 수 (교체 시점 파악)
    out["team_pitch_no_in_game"] = df.groupby(game, sort=False).cumcount()

    # 타석 단위 — 투수/타자 조합이 바뀌면 새 타석
    changed = (game.ne(game.shift()) | df["pitcher_id"].ne(df["pitcher_id"].shift())
               | df["batter_id"].ne(df["batter_id"].shift()))
    pa = changed.cumsum()
    heads = df.index[changed]
    pa_at_head = pa.loc[heads]

    # 타석 시작 행에서만 세고, 그 값을 타석 전체에 되돌린다
    faced = df.loc[heads].groupby([game.loc[heads], df.loc[heads, "pitcher_id"]],
                                  sort=False).cumcount()
    tto = df.loc[heads].groupby(
        [game.loc[heads], df.loc[heads, "pitcher_id"],
         df.loc[heads, "batter_id"]], sort=False).cumcount()
    out["batters_faced_in_game"] = pa.map(pd.Series(faced.values,
                                                    index=pa_at_head.values))
    out["times_through_order"] = pa.map(pd.Series(tto.values,
                                                  index=pa_at_head.values))
    return out


def build_features(train, mode):
    """피처 구성을 만든다. 반환은 (프레임, 컬럼목록).

    detrend 는 시즌마다 그 시즌의 리그 평균을 뺀다. 학습에서는 시즌별로,
    평가에서는 평가셋 전체 평균으로 뺀다 — 평가셋이 한 시즌이므로 같은 연산이다.
    """
    tokens = {t for t in mode.split("+") if t != "all"}
    unknown = tokens - {"ids", "no_cumul", "detrend", "delta", "shrunk",
                        "fatigue"}
    if unknown:
        raise SystemExit(f"모르는 피처 토큰: {sorted(unknown)}")

    # 기본은 ID 제거. 'ids' 를 붙이면 남긴다 — 4-4 ④ 는 ID 제거를 노이즈(±2)로
    # 기록했지만 그건 보정 없이 잰 값이라, 보정을 켜면 결론이 달라질 수 있다.
    drop = [] if "ids" in tokens else DROP_IDS
    cols = [c for c in train.columns if c not in drop + [TARGET]]
    frame = train

    if "detrend" in tokens:
        frame = frame.copy()
        for c in CUMUL:
            frame[c] = frame[c] - frame.groupby("season")[c].transform("mean")

    # 파생 피처는 한 번에 붙인다. 컬럼마다 대입하면 1.4M 행짜리 프레임이
    # 조각나서 느려진다.
    extra = {}
    for token, fn in (("delta", add_delta), ("shrunk", add_shrunk),
                      ("fatigue", add_fatigue)):
        if token in tokens:
            extra.update(fn(frame))
    if extra:
        frame = pd.concat([frame, pd.DataFrame(extra, index=frame.index)],
                          axis=1)
        cols = cols + list(extra)

    if "no_cumul" in tokens:
        cols = [c for c in cols if c not in CUMUL]

    return frame, cols


def make_model(cols, leaves, min_leaf, lr, max_iter, l2, seed=42):
    num = [c for c in cols if c not in CAT_COLS]
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                               unknown_value=-1), CAT_COLS),
        ("num", "passthrough", num),
    ])
    clf = HistGradientBoostingClassifier(
        max_iter=max_iter, learning_rate=lr, max_leaf_nodes=leaves,
        min_samples_leaf=min_leaf, l2_regularization=l2,
        early_stopping=False, random_state=seed,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def curve(clf, X_va_t, y, base, r_hat):
    """반복 시점별 (보정 전, 보정 후) 점수를 모두 잰다."""
    best_raw = (0, 0.0)
    best_shift = (0, 0.0)
    for i, proba in enumerate(clf.staged_predict_proba(X_va_t), start=1):
        p = proba[:, 1]
        s_raw = max(0.0, 100000 * (1 - ((p - y) ** 2).mean() / base))
        q = np.clip(p + (r_hat - p.mean()), 0.0, 1.0)
        s_shift = max(0.0, 100000 * (1 - ((q - y) ** 2).mean() / base))
        if s_raw > best_raw[1]:
            best_raw = (i, s_raw)
        if s_shift > best_shift[1]:
            best_shift = (i, s_shift)
    return best_raw, best_shift


def main():
    args = parse_args()
    folds = [int(f) for f in args.folds.split(",")]
    names = args.only.split(",") if args.only else list(CONFIGS)
    modes = args.feat_modes.split(",")

    test_cols = pd.read_csv(os.path.join(DATA_DIR, "test.csv"),
                            encoding="utf-8-sig", nrows=0).columns
    all_feats = [c for c in test_cols if c != ID]
    raw = pd.read_csv(os.path.join(DATA_DIR, "train.csv"),
                      encoding="utf-8-sig", usecols=all_feats + [TARGET])
    print(f"설정 {len(names)}개 | 피처모드 {modes} | 폴드 {folds}\n")

    rows = []
    for mode in modes:
        frame, cols = build_features(raw, mode)
        print(f"### 피처모드 {mode} ({len(cols)}개)")
        for name in names:
            leaves, min_leaf, lr, max_iter = CONFIGS[name]
            per_fold = []
            for Y in folds:
                tr = frame[frame["season"] < Y]
                va = frame[frame["season"] == Y]
                y = va[TARGET].to_numpy()
                base = y.mean() * (1 - y.mean())
                # 보정 추정치는 원본 prev1 에서 얻는다. detrend 는 누적형만
                # 건드리므로 prev1 은 어느 모드에서나 원본 그대로다.
                r_hat = float(va[PREV1].dropna().mean())

                t = time.time()
                m = make_model(cols, leaves, min_leaf, lr, max_iter, args.l2, args.seed)
                m.fit(tr[cols], tr[TARGET])
                X_t = m.named_steps["pre"].transform(va[cols])
                b_raw, b_shift = curve(m.named_steps["clf"], X_t, y, base, r_hat)
                per_fold.append((Y, b_raw, b_shift))
                print(f"  {name:14s} fold{Y}  "
                      f"보정전 {b_raw[1]:8.2f} (iter {b_raw[0]:4d})  "
                      f"보정후 {b_shift[1]:8.2f} (iter {b_shift[0]:4d})  "
                      f"[{time.time()-t:.0f}s]")
            mean_shift = float(np.mean([f[2][1] for f in per_fold]))
            rows.append((mode, name, per_fold, mean_shift))
        print()

    print("=== 순위 (보정 후 평균) ===")
    for mode, name, per_fold, mean_shift in sorted(rows, key=lambda x: -x[3]):
        iters = ",".join(str(f[2][0]) for f in per_fold)
        print(f"  {mean_shift:9.2f}  {mode:9s} {name:14s} 최적반복 [{iters}]")


if __name__ == "__main__":
    main()
