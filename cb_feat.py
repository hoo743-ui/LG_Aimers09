r"""CatBoost 기준 피처·인코딩 재검토 (로드맵 ②).

왜 다시 재는가. 이 프로젝트의 **모든 피처 판정이 HGB 기준**이다. 그런데
"구조 축 6종 전부 음수"(4-10)가 HGB 한정이었다는 게 4-14 에서 드러났다 —
계열을 바꾸니 +37 이 나왔다. 피처 판정도 같은 위험이 있다.

후보.

  A experience   season - 데뷔시즌.  **추론 시 퇴화하는 피처를 판별 가능하게**
                 바꾼다. season 은 평가셋 전량 2025 라 상수이고 판별 기여가 0
                 인데(4-10), 이득의 19.4% 를 쓴다. 경력 연차는 투수마다 달라
                 판별에 기여한다. same_hand 와 같은 구조적 패턴이다.
                 pitcher_id 가 데뷔 순 프록시라(상관 +0.704) 부분 중복이지만,
                 불펜 5년차와 선발 2년차는 투구수가 달라 asof_pitcher_n 과도
                 구분된다. 2025 신인은 연차 0 이 되는데 그게 의미상 정확하다.

  B prev1-cumul  최근 대비 통산. 4-5 는 이 계열 **9개를 한꺼번에** 넣어 -3.75
                 였다. 단일 컬럼으로는 미검증이다.
  C prev1-prev5  최근 추세. 미검증.
  D onehot       one_hot_max_size 를 올려 팀 ID(13종)·base_state(8종)를
                 타깃 통계 대신 **원핫**으로 처리한다. ID 축은 선언/미선언에서
                 163점이 갈렸는데(4-14) 그 사이가 비어 있다.
  E crosses      HGB 기준으로 기각했던 교차 컬럼 재측정 (4-7).
  F no_samehand  CatBoost 가 same_hand 를 스스로 찾는가. 대칭 트리는 한 깊이에서
                 같은 분할을 공유하므로 XOR 을 오히려 더 못 찾을 수도 있다.

시점 규칙. 데뷔 시즌은 **fold Y 미만 시즌**에서만 구한다. 평가(2025)가
2019~2024 를 보는 것과 같은 조건이다.

    .\.venv\Scripts\python.exe cb_feat.py --only A,B,C
    .\.venv\Scripts\python.exe cb_feat.py            # 전부
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

DATA_DIR = "./data"
CACHE = "./.blendcache"
ID = "row_id"
TARGET = "control_success"
PREV1 = "asof_pitcher_prev1_game_success_rate"
PREV5 = "asof_pitcher_prev5_game_success_rate"
CUM = "asof_pitcher_success_rate"
LAM = 0.03
STR_COLS = ["top_bottom", "game_type", "base_state",
            "pitcher_hand", "batter_hand"]
TEAM_COLS = ["pitcher_team_id", "batter_team_id"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, default=2024)
    p.add_argument("--only", default=None)
    p.add_argument("--iters", type=int, default=600)
    p.add_argument("--depth", type=int, default=7)
    p.add_argument("--l2", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--unclipped", action="store_true",
                   help="max(0,.) 클리핑 없이 잰다. 2023 은 붕괴 해라 둘 다 0 이 "
                        "돼 비교가 안 된다 — 클리핑이 정보를 지우는 것이지 폴드가 "
                        "쓸모없는 게 아니다 (4-11)")
    return p.parse_args()


def build(df, Y):
    """파생 컬럼을 모두 만든다. 시점 규칙은 experience 에만 적용된다."""
    df = df.copy()
    df["same_hand"] = (df["pitcher_hand"] == df["batter_hand"]).astype(int)
    # 데뷔 시즌 — fold Y 미만에서만 구한다. 없으면 Y(연차 0)
    hist = df[df["season"] < Y]
    debut = hist.groupby("pitcher_id")["season"].min()
    df["experience"] = (df["season"]
                        - df["pitcher_id"].map(debut).fillna(Y)).astype(float)
    df["d_prev1_cum"] = df[PREV1] - df[CUM]
    df["d_prev1_prev5"] = df[PREV1] - df[PREV5]
    df["count_state"] = df["balls_before"] * 3 + df["strikes_before"]
    ph = pd.factorize(df["pitcher_hand"])[0]
    df["hand_1b"] = ph * 2 + df["runner_on_1b"].to_numpy()
    return df


def main():
    args = parse_args()
    Y = args.fold
    test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig",
                            nrows=0).columns
    base_cols = [c for c in test_cols if c != ID]
    raw = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                      usecols=base_cols + [TARGET])
    df = build(raw, Y)
    BASE = base_cols + ["same_hand"]

    # (이름, 피처 목록, 추가 cat 컬럼, one_hot_max_size)
    VARIANTS = [
        ("base",        BASE, [], None),
        ("A_exp",       BASE + ["experience"], [], None),
        ("B_d1cum",     BASE + ["d_prev1_cum"], [], None),
        ("C_d1p5",      BASE + ["d_prev1_prev5"], [], None),
        ("D_onehot16",  BASE, TEAM_COLS, 16),
        ("D_onehot32",  BASE, TEAM_COLS, 32),
        # D 는 두 가지를 동시에 바꾼다 — base_state 를 CTR->원핫으로, 팀 ID 를
        # 숫자->원핫으로. 어느 쪽이 이득인지 분리한다.
        ("D_bs_only",   BASE, [], 16),          # base_state 만 원핫, 팀은 숫자
        ("D_team_ctr",  BASE, TEAM_COLS, 2),    # 팀을 CTR 로 (원핫 아님)
        # 카디널리티가 갈림길이었다 — 팀(13종)은 CTR 이 +30.90, 선수(792종)는
        # -163 이었다. 다만 선수 쪽은 depth 6 / l2 10 에서 잰 값이다.
        # l2 를 30배 올린 지금은 CTR 이 충분히 규제될 수도 있다.
        ("G_playerctr", BASE, TEAM_COLS + ["pitcher_id", "batter_id"], 2),
        ("H_morectr",   BASE, TEAM_COLS + ["balls_before", "strikes_before",
                                           "outs_before", "inning"], 2),
        ("E_crosses",   BASE + ["count_state", "hand_1b"], [], None),
        ("F_nosame",    [c for c in BASE if c != "same_hand"], [], None),
    ]
    if args.only:
        want = {s.strip() for s in args.only.split(",")}
        VARIANTS = [v for v in VARIANTS
                    if v[0] == "base" or v[0].split("_")[0] in want
                    or v[0] in want]

    tr_m = df["season"] < Y
    va_m = df["season"] == Y
    y = df.loc[va_m, TARGET].to_numpy(dtype=float)
    denom = y.mean() * (1 - y.mean())
    c = float(df.loc[tr_m, TARGET].mean())
    anc = df.loc[va_m, PREV1].fillna(c).to_numpy(dtype=float) - c

    def sc(p):
        v = 100000 * (1 - ((np.clip(p, 0, 1) - y) ** 2).mean() / denom)
        return v if args.unclipped else max(0.0, v)

    print(f"fold {Y} | depth {args.depth} / {args.iters}그루 / l2 {args.l2:g}")
    print(f"평가: 단독 + 중심보정(lam {LAM})\n")
    print(f"{'변형':>13}{'피처':>5}{'단독':>10}{'+중심':>10}{'vs base':>10}")
    print("-" * 50)

    ref = None
    for name, feats, extra_cat, ohms in VARIANTS:
        cats = [x for x in STR_COLS if x in feats] + list(extra_cat)
        d2 = df.copy()
        for col in cats:
            d2[col] = d2[col].astype(str)
        tag = (f"cbf_{name}_d{args.depth}_it{args.iters}"
               f"_l2{int(args.l2)}_s{args.seed}")
        path = os.path.join(CACHE, f"{Y}_{tag}.npy")
        t = time.time()
        if os.path.exists(path):
            p = np.load(path)
            took = "캐시"
        else:
            kw = {} if ohms is None else {"one_hot_max_size": ohms}
            m = CatBoostClassifier(
                iterations=args.iters, depth=args.depth, learning_rate=0.02,
                l2_leaf_reg=args.l2, loss_function="Logloss",
                random_seed=args.seed, verbose=0, allow_writing_files=False,
                **kw)
            m.fit(Pool(d2.loc[tr_m, feats], d2.loc[tr_m, TARGET],
                       cat_features=cats))
            p = m.predict_proba(Pool(d2.loc[va_m, feats],
                                     cat_features=cats))[:, 1]
            np.save(path, p)
            took = f"{time.time()-t:.0f}s"
        solo = sc(p)
        full = sc(np.clip(p + LAM * anc, 0, 1))
        if ref is None:
            ref = full
        print(f"{name:>13}{len(feats):5d}{solo:10.2f}{full:10.2f}"
              f"{full-ref:+10.2f}   [{took}]", flush=True)

    print("""
읽는 법. 판정 기준은 **+5 이상**이다 (노이즈 하한, 로드맵 ①). 단일 폴드·1시드
이므로 살아남은 것은 3폴드로 재확인해야 한다 (4-6).""")


if __name__ == "__main__":
    main()
