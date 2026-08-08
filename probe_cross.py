r"""피처 쌍 중 **현재 모델이 못 잡는 것**을 점수로 환산해 훑는다.

왜 이 도구가 있는가. hand_mix (pitcher_hand x batter_hand) 컬럼 하나가 +24.66
이었다 (4-7). 그 두 피처는 4년 내내 모델 안에 있었는데도 트리가 상호작용을
못 찾았다. 플래툰 효과가 **주변부 효과가 0 인 순수 상호작용**이기 때문이다.

        좌타   우타
  좌완    +      -
  우완    -      +

pitcher_hand 만 봐도, batter_hand 만 봐도 평균 성공률은 그대로다. 탐욕적 트리는
루트에서 주변부 이득으로 분할을 고르므로 이런 쌍은 영원히 선택되지 않는다 —
파라미터가 아니라 알고리즘의 구조적 사각지대다. 나머지 쌍에도 같은 자리가
있을 수 있으니 전부 훑는다.

기준선을 무엇으로 두느냐가 이 도구의 전부다.

  --baseline additive   그 두 피처만 쓰는 5분위 가법 모형 (예전 방식)
  --baseline model      **실제 base 모델의 예측** (기본값)

가법 기준선은 실제 48피처 HGB 보다 훨씬 약하다. 그래서 주변부 효과가 강한 쌍
(`asof x asof`)이 상위를 독식했는데, 정작 그건 HGB 가 이미 잡고 있는 것들이었다
(`asof_prev3 x asof_prev5` 가 +314 로 1위였다). 알고 싶은 것은 "이 쌍에 신호가
있는가"가 아니라 **"현재 모델이 그걸 놓치고 있는가"** 다.

model 기준선은 그 질문에 직접 답한다. 모델 잔차 `y - p` 에 셀 구조가 남아
있는지만 본다. 남아 있지 않으면 이미 잡고 있는 것이고, 컬럼으로 줘도 소용없다.

  이전 시즌 잔차에서:  dev[a,b] = 셀 평균 잔차 - a 주변부 - b 주변부 + 전체
                       표본 축소 n/(n+k) 후
  평가 시즌에서:       이득 = mean(2*dev*(y-p) - dev^2)  ->  x 100000/r(1-r)

시점을 분리한다 — 이전 시즌에서 재고 평가 시즌에서 확인한다. 같은 시즌에서
재고 그 시즌에서 확인하면 표본 노이즈가 그대로 이득으로 잡힌다.

**자체 검증.** 캐시된 예측은 `same_hand` 가 들어간 모델의 것이다. 그러므로
pitcher_hand x batter_hand 는 **0 근처**로 나와야 한다 — 이미 컬럼으로 줬으니까.
가법 기준선에서는 이게 +63.56 으로 상위였다. 그 대조가 도구가 제대로 도는지를
말해 준다. 실행하면 맨 아래 자체 검증 줄에 찍힌다.

예측 캐시는 blend_test.py 가 만든다 (`.blendcache/`). 없으면 먼저 돌릴 것.

한계. `main` 열은 그 2차원 투영에서 모델이 놓친 것 전부이고, `inter` 열은 그중
순수 상호작용분이다. 사각지대 이론이 겨냥하는 것은 `inter` 쪽이다. 어느 쪽이든
**후보를 좁히는 데만 쓰고 채택은 interact_feat.py 로 확인할 것** — 옛 도구에서
park 이 안정성 상관 +0.819 로 최상위였는데 실제로 넣으니 -23.09 였다 (4-6).

    .\.venv\Scripts\python.exe probe_cross.py
    .\.venv\Scripts\python.exe probe_cross.py --baseline additive   # 예전 방식
"""
import argparse
import itertools
import os
import time

import numpy as np
import pandas as pd

DATA = "./data/train.csv"
TARGET = "control_success"
CACHE = "./.blendcache"
HIST = [2021, 2022]      # 여기서 셀 구조를 재고
EVAL = 2024              # 여기서 확인한다 (학습 시즌이 가장 많아 실제 조건에 가깝다)
K = 200                  # 표본 축소. 저차원 쌍은 셀이 커서 영향이 작다


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--top", type=int, default=25)
    p.add_argument("--baseline", choices=["model", "additive"], default="model")
    p.add_argument("--k", type=int, default=K)
    p.add_argument("--max-cells", type=int, default=400)
    p.add_argument("--model", default="hgb", help="캐시에서 쓸 모델 이름")
    p.add_argument("--order", type=int, choices=[2, 3], default=2,
                   help="3 이면 **순수 3차 상호작용**을 훑는다. 주변부 3개와 "
                        "2차 3개를 모두 제거하고 남는 성분만 본다 — 2차가 전부 "
                        "0 으로 보이는 자리는 탐욕적 트리가 경로 자체를 못 찾기 "
                        "때문에, 사각지대 논리가 차수를 올리면 더 강해진다")
    p.add_argument("--eval-fold", type=int, default=None,
                   help="평가 폴드를 바꾼다. 이력은 그 이전 시즌 전부. 같은 "
                        "조합이 다른 폴드에서도 올라오는지 보는 안정성 검사용 "
                        "— 단일 폴드 추정은 이 프로젝트에서 두 번 배신했다")
    p.add_argument("--max-card", type=int, default=13,
                   help="3차에서 쓸 피처의 최대 수준 수. 낮출수록 셀이 커져 "
                        "추정이 안정되지만 후보가 줄어든다")
    return p.parse_args()


def group_mean(idx, n_cell, resid, k):
    """셀별 잔차 평균 (표본 축소). 없는 셀은 0."""
    cnt = np.bincount(idx, minlength=n_cell).astype(np.float64)
    tot = np.bincount(idx, weights=resid, minlength=n_cell)
    m = np.divide(tot, cnt, out=np.zeros(n_cell), where=cnt > 0)
    return np.where(cnt > 0, m * (cnt / (cnt + k)), 0.0), cnt


def scan_order3(codes, names, card, resid, hist_mask, eval_mask, denom, args):
    """순수 3차 상호작용을 훑는다.

    ANOVA 항등식으로 저차 성분을 전부 뺀다.

        dev3[a,b,c] = m[a,b,c] - m[a,b] - m[a,c] - m[b,c]
                                + m[a] + m[b] + m[c] - m[]

    주변부와 2차 평균은 쌍마다 재계산하면 낭비이므로 미리 만들어 둔다 —
    3중 조합은 수천 개인데 쌍은 수백 개뿐이다.
    """
    use = [n for n in names if card[n] <= args.max_card]
    print(f"3차 대상 피처 {len(use)}개 (수준 {args.max_card} 이하) | "
          f"조합 {len(use)*(len(use)-1)*(len(use)-2)//6:,}개", flush=True)

    rh, re_ = resid[hist_mask], resid[eval_mask]
    mu = float(rh.mean())

    # 주변부
    marg = {}
    for n in use:
        c = codes[n]
        marg[n] = group_mean(c[hist_mask], card[n], rh, args.k)[0]

    # 2차 — 미리 만들어 재사용
    t = time.time()
    pair = {}
    for a, b in itertools.combinations(use, 2):
        na, nb = card[a], card[b]
        idx = codes[a][hist_mask] * nb + codes[b][hist_mask]
        pair[(a, b)] = group_mean(idx, na * nb, rh, args.k)[0]
    print(f"2차 사전계산 {len(pair)}쌍 [{time.time()-t:.0f}s]", flush=True)

    rows = []
    t = time.time()
    for a, b, c in itertools.combinations(use, 3):
        na, nb, nc = card[a], card[b], card[c]
        ncell = na * nb * nc
        if ncell > args.max_cells:
            continue
        ha, hb, hc = codes[a][hist_mask], codes[b][hist_mask], codes[c][hist_mask]
        m3, cnt = group_mean((ha * nb + hb) * nc + hc, ncell, rh, args.k)

        # 저차 성분을 3차 격자 위로 펼쳐서 뺀다
        ai = np.repeat(np.arange(na), nb * nc)
        bi = np.tile(np.repeat(np.arange(nb), nc), na)
        ci = np.tile(np.arange(nc), na * nb)
        dev = (m3
               - pair[(a, b)][ai * nb + bi]
               - pair[(a, c)][ai * nc + ci]
               - pair[(b, c)][bi * nc + ci]
               + marg[a][ai] + marg[b][bi] + marg[c][ci] - mu)
        dev = np.where(cnt > 0, dev, 0.0)

        d = dev[(codes[a][eval_mask] * nb + codes[b][eval_mask]) * nc
                + codes[c][eval_mask]]
        gain = 100000.0 * np.mean(2 * d * re_ - d ** 2) / denom
        rows.append((a, b, c, ncell, gain))
    print(f"3차 평가 {len(rows):,}개 [{time.time()-t:.0f}s]\n")

    rows.sort(key=lambda r: -r[4])
    print(f"=== 순수 3차 상호작용 상위 {args.top} "
          f"(현재 모델이 못 잡는 것, 시점 분리) ===")
    print(f"{'A x B x C':>76}{'셀':>6}{'이득':>9}")
    print("-" * 92)
    for a, b, c, nc_, g in rows[:args.top]:
        print(f"{a + ' x ' + b + ' x ' + c:>76}{nc_:6d}{g:9.2f}")
    pos = sum(1 for r in rows if r[4] > 0)
    print(f"\n양수 {pos} / 전체 {len(rows):,}개")
    print(f"최대 {rows[0][4]:.2f}  (참고: 2차 최대는 +8.05, "
          f"채택 기준은 로컬 +15~20)")
    return rows


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

    for c in ["game_month", "game_dayofweek", "top_bottom", "game_type",
              "balls_before", "strikes_before", "outs_before",
              "runner_on_1b", "runner_on_2b", "runner_on_3b", "num_runners_on",
              "base_state", "pitcher_hand", "batter_hand",
              "pitcher_team_id", "batter_team_id"]:
        put(c, df[c])
    put("inning", np.clip(df["inning"], 1, 10))
    # 4-7 에서 채택한 파생. 이미 모델에 들어 있으므로 여기서는 0 이 나와야 한다
    put("same_hand", (df["pitcher_hand"] == df["batter_hand"]).astype(int))

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


def cell_dev(ca, cb, na, nb, resid, mask, k, pure=True):
    """셀별 잔차 편차. pure 면 주변부를 빼고 순수 상호작용만 남긴다."""
    ia, ib, r = ca[mask], cb[mask], resid[mask]
    ncell = na * nb
    idx = ia * nb + ib
    cnt = np.bincount(idx, minlength=ncell).astype(np.float64)
    tot = np.bincount(idx, weights=r, minlength=ncell)
    cell = np.divide(tot, cnt, out=np.zeros(ncell), where=cnt > 0)

    if pure:
        a_cnt = np.bincount(ia, minlength=na).astype(np.float64)
        a_tot = np.bincount(ia, weights=r, minlength=na)
        b_cnt = np.bincount(ib, minlength=nb).astype(np.float64)
        b_tot = np.bincount(ib, weights=r, minlength=nb)
        mu = r.mean()
        a_eff = np.divide(a_tot, a_cnt, out=np.full(na, mu), where=a_cnt > 0) - mu
        b_eff = np.divide(b_tot, b_cnt, out=np.full(nb, mu), where=b_cnt > 0) - mu
        cell = cell - mu - np.repeat(a_eff, nb) - np.tile(b_eff, na)

    return np.where(cnt > 0, cell * (cnt / (cnt + k)), 0.0)


def gain_on(dev, ca, cb, nb, resid, mask, denom):
    """이전 시즌에서 잰 dev 를 평가 시즌에 얹었을 때의 점수 이득."""
    d = dev[ca[mask] * nb + cb[mask]]
    red = np.mean(2 * d * resid[mask] - d ** 2)
    return 100000.0 * red / denom


def main():
    args = parse_args()
    if not os.path.exists(DATA):
        raise SystemExit(f"{DATA} 없음")

    print("로드 중 ...", flush=True)
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    y = df[TARGET].to_numpy(dtype=np.float64)
    season = df["season"].to_numpy()

    # 평가 폴드를 바꾸면 이력은 그 이전 시즌 전부가 된다. 예측 캐시는
    # extrap_test.py 가 만든 것(2020~2024, L10)을 쓴다.
    hist_f, eval_f, cache_name = HIST, EVAL, f"{args.model}_seed42"
    if args.eval_fold:
        eval_f = args.eval_fold
        hist_f = [Y for Y in [2020, 2021, 2022, 2023] if Y < eval_f]
        cache_name = "L10_seed42"
        globals()["CACHE"] = "./.extrapcache"
        if not hist_f:
            raise SystemExit("이력 폴드가 없다")

    # ---- 기준선 ----
    p0 = np.full(len(df), np.nan)
    if args.baseline == "model":
        for Y in hist_f + [eval_f]:
            path = os.path.join(CACHE, f"{Y}_{cache_name}.npy")
            if not os.path.exists(path):
                raise SystemExit(
                    f"{path} 없음 — 먼저 blend_test.py 를 돌려 예측 캐시를 만들 것")
            m = season == Y
            pred = np.load(path)
            if len(pred) != m.sum():
                raise SystemExit(f"{path} 길이 {len(pred)} != {Y} 시즌 행 {m.sum()}")
            p0[m] = pred
        print(f"기준선: 캐시 예측 {cache_name} (시즌 {hist_f + [eval_f]})")
    else:
        # 예전 방식 — 시즌 평균만. 쌍별 주변부는 pure 처리로 따로 뺀다.
        for Y in hist_f + [eval_f]:
            m = season == Y
            p0[m] = y[m].mean()
        print("기준선: 시즌 평균 (가법. 예전 방식과 같은 성격)")

    resid = y - p0
    hist_mask = np.isin(season, hist_f)
    eval_mask = season == eval_f
    r = y[eval_mask].mean()
    denom = r * (1 - r)
    print(f"이력 {hist_mask.sum():,} 행 (시즌 {hist_f}) -> "
          f"평가 {eval_mask.sum():,} 행 (시즌 {eval_f})")
    print(f"평가 시즌 잔차 평균 {resid[eval_mask].mean():+.5f} "
          f"(중심 편차) | 기준선 점수 "
          f"{max(0, 100000*(1-(resid[eval_mask]**2).mean()/denom)):.2f}\n")

    t = time.time()
    codes = build_codes(df)
    names = sorted(codes)
    card = {n: int(codes[n].max()) + 1 for n in names}
    print(f"피처 {len(names)}개 코드화 [{time.time()-t:.0f}s] | "
          f"쌍 {len(names)*(len(names)-1)//2}개", flush=True)

    if args.order == 3:
        scan_order3(codes, names, card, resid, hist_mask, eval_mask, denom, args)
        print("""
읽는 법. 2차가 전부 0 으로 보이는 자리는 탐욕적 트리가 경로 자체를 못 찾는다 —
루트에서도, 두 번째 분할에서도 이득이 없기 때문이다. 사각지대 논리(4-7)가
차수를 올리면 더 강해진다. 다만 3차 셀은 표본이 얇아 추정이 거칠고, 값은
**후보를 좁히는 데만** 쓴다. 채택은 interact_feat.py 로 확인할 것.""")
        return

    t = time.time()
    rows = []
    for a, b in itertools.combinations(names, 2):
        ca, cb, na, nb = codes[a], codes[b], card[a], card[b]
        if na * nb > args.max_cells:
            continue
        g_int = gain_on(cell_dev(ca, cb, na, nb, resid, hist_mask, args.k, True),
                        ca, cb, nb, resid, eval_mask, denom)
        g_all = gain_on(cell_dev(ca, cb, na, nb, resid, hist_mask, args.k, False),
                        ca, cb, nb, resid, eval_mask, denom)
        rows.append((a, b, na * nb, g_int, g_all))
    print(f"평가 완료 [{time.time()-t:.0f}s]\n")

    rows.sort(key=lambda r_: -r_[3])
    print(f"=== 현재 모델이 못 잡는 상호작용 상위 {args.top} "
          f"({hist_f} -> {eval_f}, 시점 분리) ===")
    print(f"{'A':>38} x {'B':<38}{'셀':>5}{'inter':>9}{'main':>9}")
    print("-" * 104)
    for a, b, nc, gi, ga in rows[:args.top]:
        print(f"{a:>38} x {b:<38}{nc:5d}{gi:9.2f}{ga:9.2f}")

    pos = sum(1 for x in rows if x[3] > 0)
    print(f"\ninter 양수 {pos} / 전체 {len(rows)}쌍")

    # ---- 자체 검증 ----
    # same_hand 는 이미 모델에 있으므로 0 근처여야 한다. 가법 기준선에서는
    # 이 쌍이 +63.56 으로 상위였다 (4-7 이전 도구).
    chk = [x for x in rows if {x[0], x[1]} == {"pitcher_hand", "batter_hand"}]
    if chk:
        a, b, nc, gi, ga = chk[0]
        rank = rows.index(chk[0]) + 1
        print(f"\n자체 검증  pitcher_hand x batter_hand : inter {gi:+.2f} "
              f"(전체 {len(rows)}쌍 중 {rank}위)")
        print(f"  이미 same_hand 로 모델에 넣었으므로 0 근처여야 한다. "
              f"예전 가법 기준선에서는 +63.56 으로 상위였다")

    print("""
읽는 법.
  inter  주변부를 뺀 순수 상호작용분. 4-7 의 사각지대 이론이 겨냥하는 값이다
  main   그 2차원 투영에서 모델이 놓친 것 전부 (주변부 miss 포함)
  둘 다 "이전 시즌에서 재고 평가 시즌에서 확인한" 값이라 표본 노이즈는 빠져 있다.
  다만 폴드 하나짜리이므로 크기는 거칠다 — 후보를 좁히는 데만 쓰고 채택은
  interact_feat.py 로 3폴드 x 2시드 확인할 것 (4-6).""")


if __name__ == "__main__":
    main()
