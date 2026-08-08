r"""모든 피처 쌍의 '순수 상호작용'을 점수로 환산해 훑는다.

왜 필요한가. hand_mix (pitcher_hand x batter_hand) 컬럼 하나가 +23.10 이었다.
그 두 피처는 4년 내내 모델 안에 있었는데도 트리가 상호작용을 못 찾았다. 이유는
플래툰 효과가 **주변부 효과가 0 인 순수 상호작용**이기 때문이다.

        좌타   우타
  좌완    +      -
  우완    -      +

pitcher_hand 만 봐도, batter_hand 만 봐도 평균 성공률은 그대로다. 탐욕적 트리는
루트에서 주변부 이득으로 분할을 고르므로 이런 쌍은 영원히 선택되지 않는다.
트리 수를 아무리 늘려도 못 찾는다 — 구조적 사각지대다.

그렇다면 나머지 쌍에도 같은 사각지대가 있을 수 있다. 전부 훑는다.

무엇을 재는가. 상관계수가 아니라 **점수**다. 앞서 상관으로 두 번 속았다 —
pitcher x batter_hand 의 +0.53 은 리그 성분이 섞인 값이었고, park 의 +0.819 는
팀 교란이었다(실제로 넣으니 -23.09). 그래서 여기서는 "이 상호작용을 넣으면
Brier 가 실제로 얼마나 줄어드는가"를 시점 분리로 직접 잰다.

    이전 시즌에서:  dev[a,b] = rate[a,b] - mu - (rate[a]-mu) - (rate[b]-mu)
                    표본 축소 n/(n+k) 후
    평가 시즌에서:  pred0 = 그 시즌 자체의 가법 모형 (주변부까지는 공짜로 준다)
                    pred1 = pred0 + dev
                    이득 = mean(2*dev*(y-pred0) - dev^2)  ->  x 100000/r(1-r)

pred0 을 평가 시즌 자체 데이터로 맞추는 것은 기준선에 유리하게 준 것이다.
주변부는 트리가 이미 잘 잡으므로, 순수 상호작용분만 신용한다.

한계. 이건 그 두 피처만 쓰는 가법 모형 대비 이득이라 상한이다. 다른 피처가
같은 정보를 이미 담고 있으면 실제 이득은 줄어든다. **후보를 좁히는 데만 쓰고
채택은 반드시 interact_feat.py 로 확인할 것** — park 이 그래서 걸렸다.

    .\.venv\Scripts\python.exe probe_cross.py
    .\.venv\Scripts\python.exe probe_cross.py --top 40
"""
import argparse
import itertools
import os
import time

import numpy as np
import pandas as pd

DATA = "./data/train.csv"
TARGET = "control_success"
FOLDS = [2021, 2022, 2024]      # 2023 은 어떤 구성으로도 0점이라 제외 (README 4-6)
K = 200                          # 표본 축소. 저차원 쌍은 셀이 커서 영향이 작다


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--top", type=int, default=25)
    p.add_argument("--folds", default=",".join(map(str, FOLDS)))
    p.add_argument("--k", type=int, default=K)
    return p.parse_args()


def qbucket(s, q):
    """분위수 구간. 연속형을 저차원 코드로 낮춘다. 결측은 별도 코드."""
    try:
        b = pd.qcut(s, q, labels=False, duplicates="drop")
    except ValueError:
        b = pd.Series(np.zeros(len(s)), index=s.index)
    b = b.fillna(-1).astype(np.int32)
    return b - b.min()


def build_codes(df):
    """피처 -> 0..K-1 정수 코드. 쌍마다 bincount 로 셀 집계를 하기 위한 준비."""
    out = {}

    def put(name, values):
        v = pd.Series(values).astype("category").cat.codes.to_numpy().astype(np.int32)
        if v.max() >= 1:
            out[name] = v

    # 저차원 원본 — 그대로 코드화
    for c in ["game_month", "game_dayofweek", "top_bottom", "game_type",
              "balls_before", "strikes_before", "outs_before",
              "runner_on_1b", "runner_on_2b", "runner_on_3b", "num_runners_on",
              "base_state", "pitcher_hand", "batter_hand",
              "pitcher_team_id", "batter_team_id"]:
        put(c, df[c])

    put("inning", np.clip(df["inning"], 1, 10))

    # 연속형 — 분위수로 낮춘다. 구간 수는 셀이 너무 잘게 쪼개지지 않을 만큼
    for c, q in [("run_total_before", 5), ("score_diff_pitcher_team", 5),
                 ("li", 5), ("home_win_expectancy", 5),
                 ("asof_pitcher_n", 5), ("asof_pitcher_success_rate", 5),
                 ("asof_pitcher_reverse_rate", 5), ("asof_pitcher_middle_rate", 5),
                 ("asof_pitcher_ball_rate", 5), ("asof_pitcher_strike_rate", 5),
                 ("asof_pitcher_prev1_game_success_rate", 5),
                 ("asof_pitcher_prev3_game_success_rate", 5),
                 ("asof_pitcher_prev5_game_success_rate", 5),
                 ("asof_pitcher_prev1_game_middle_rate", 5),
                 ("asof_batter_n", 5), ("asof_batter_success_rate", 5),
                 ("asof_batter_middle_rate", 5),
                 ("asof_pitcher_fastball_rate", 5),
                 ("asof_pitcher_breaking_rate", 5),
                 ("asof_pitcher_offspeed_rate", 5)]:
        if c in df.columns:
            put(c, qbucket(df[c], q))
    return out


def cell_stats(ia, ib, nb, y, n_rows_cells):
    """셀별 합과 개수. bincount 한 번으로 끝난다."""
    idx = ia * nb + ib
    cnt = np.bincount(idx, minlength=n_rows_cells).astype(np.float64)
    tot = np.bincount(idx, weights=y, minlength=n_rows_cells)
    return cnt, tot


def pair_gain(ca, cb, na, nb, y, hist_mask, val_mask, k):
    """이전 시즌 상호작용을 평가 시즌에 적용했을 때의 점수 이득."""
    ncell = na * nb

    # ---- 이전 시즌에서 순수 상호작용 추정 ----
    ia, ib, yh = ca[hist_mask], cb[hist_mask], y[hist_mask]
    cnt, tot = cell_stats(ia, ib, nb, yh, ncell)
    ca_cnt = np.bincount(ia, minlength=na).astype(np.float64)
    ca_tot = np.bincount(ia, weights=yh, minlength=na)
    cb_cnt = np.bincount(ib, minlength=nb).astype(np.float64)
    cb_tot = np.bincount(ib, weights=yh, minlength=nb)

    mu = yh.mean()
    a_eff = np.divide(ca_tot, ca_cnt, out=np.full(na, mu), where=ca_cnt > 0) - mu
    b_eff = np.divide(cb_tot, cb_cnt, out=np.full(nb, mu), where=cb_cnt > 0) - mu
    cell = np.divide(tot, cnt, out=np.zeros(ncell), where=cnt > 0)
    dev = cell - mu - np.repeat(a_eff, nb) - np.tile(b_eff, na)
    dev = np.where(cnt > 0, dev * (cnt / (cnt + k)), 0.0)

    # ---- 평가 시즌에서 이득 측정 ----
    # 기준선(주변부)은 평가 시즌 자체로 맞춘다 — 트리가 이미 잘 잡는 부분이므로
    # 공짜로 주고, 순수 상호작용분만 신용한다.
    ja, jb, yv = ca[val_mask], cb[val_mask], y[val_mask]
    va_cnt = np.bincount(ja, minlength=na).astype(np.float64)
    va_tot = np.bincount(ja, weights=yv, minlength=na)
    vb_cnt = np.bincount(jb, minlength=nb).astype(np.float64)
    vb_tot = np.bincount(jb, weights=yv, minlength=nb)
    muv = yv.mean()
    va = np.divide(va_tot, va_cnt, out=np.full(na, muv), where=va_cnt > 0) - muv
    vb = np.divide(vb_tot, vb_cnt, out=np.full(nb, muv), where=vb_cnt > 0) - muv

    pred0 = muv + va[ja] + vb[jb]
    d = dev[ja * nb + jb]
    red = np.mean(2 * d * (yv - pred0) - d ** 2)
    return 100000.0 * red / (muv * (1 - muv))


def main():
    args = parse_args()
    folds = [int(f) for f in args.folds.split(",")]
    if not os.path.exists(DATA):
        raise SystemExit(f"{DATA} 없음")

    print("로드 중 ...", flush=True)
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    y = df[TARGET].to_numpy(dtype=np.float64)
    season = df["season"].to_numpy()

    t = time.time()
    codes = build_codes(df)
    names = sorted(codes)
    card = {n: int(codes[n].max()) + 1 for n in names}
    print(f"피처 {len(names)}개 코드화 [{time.time()-t:.0f}s] | "
          f"쌍 {len(names)*(len(names)-1)//2}개\n")

    masks = [(Y, season < Y, season == Y) for Y in folds]

    t = time.time()
    rows = []
    for a, b in itertools.combinations(names, 2):
        ca, cb, na, nb = codes[a], codes[b], card[a], card[b]
        if na * nb > 400:        # 셀이 너무 잘게 쪼개지면 추정이 무너진다
            continue
        g = [pair_gain(ca, cb, na, nb, y, hm, vm, args.k) for _, hm, vm in masks]
        rows.append((a, b, na * nb, float(np.mean(g)), g))
    print(f"평가 완료 [{time.time()-t:.0f}s]\n")

    rows.sort(key=lambda r: -r[3])
    print(f"=== 순수 상호작용 이득 상위 {args.top} (시점 분리, 3폴드 평균) ===")
    print(f"{'A':>38} x {'B':<38}{'셀':>5}{'평균':>9}   폴드별")
    print("-" * 118)
    for a, b, nc, m, g in rows[:args.top]:
        sign = "" if all(x > 0 for x in g) or all(x < 0 for x in g) else "  ★엇갈림"
        print(f"{a:>38} x {b:<38}{nc:5d}{m:9.2f}   "
              + " ".join(f"{x:7.2f}" for x in g) + sign)

    neg = [r for r in rows if r[3] < 0]
    print(f"\n양수 {len(rows)-len(neg)} / 전체 {len(rows)}쌍")
    print("""
주의. 이 값은 그 두 피처만 쓰는 가법 모형 대비 이득이라 상한이다. 다른 피처가
같은 정보를 담고 있으면 실제 이득은 줄어든다. 후보를 좁히는 데만 쓰고 채택은
interact_feat.py 로 확인할 것 — park 이 상관 +0.819 였는데 실제로는 -23.09 였다.""")


if __name__ == "__main__":
    main()
