r"""`asof_*` 축소(경험 베이즈)와 최근 형태 편차 컬럼을 캐시에 덧붙인다.

## 왜

이 문제의 성질은 "표현력"이 아니라 **과신하지 않기**다 (4-1). `asof_*` 비율은
표본이 적을 때 극단으로 튀는데, 표본 수 컬럼(`asof_pitcher_n` 등)이 따로 있어도
트리는 축 평행 분기만 하므로 `rate x n` 의 곱 형태를 표현하기 어렵다. 축소된 값을
**직접 만들어 주면** 그 곱을 모델이 배우지 않아도 된다.

    r_shrunk = (n * r + k * mu) / (n + k)

표본이 많으면 r 그대로, 적으면 사전값 mu 로 당겨진다. `k` 는 "사전값이 몇 개의
관측만큼 무게를 갖는가"다.

두 번째로 **최근 형태 편차**를 만든다. `prev1/3/5_game_*_rate` 는 통산 비율과
따로 놓여 있는데, 진짜 정보는 "통산 대비 지금 얼마나 벗어났는가"다. 4-9 에서
채택된 것도 수준값이 아니라 **편차**였고, 4-18 에서 CTR 이 실패한 이유도 낡은
수준값을 실어 왔기 때문이다.

## 사전값의 미래 참조를 막는다

`mu` 를 전체 구간 평균으로 잡으면 fold2021 실험이 2022~2024 를 들여다보게 된다.
그래서 **2019~2020 두 시즌만으로 계산한다** — 모든 폴드(2021/2022/2024)의 학습
구간에 항상 포함되는 구간이라 어떤 폴드에서도 미래를 보지 않는다. 계산된 값은
`cols.json` 에 박아 제출 경로가 같은 상수를 쓰게 한다.

기존 캐시를 덮어쓰지 않고 컬럼을 뒤에 덧붙이며, `prod` 는 기존 목록 그대로 둔다.

    .\.venv\Scripts\python.exe exp\prep_shrink.py
"""
import json

import numpy as np

import prep as P

CACHE = P.CACHE

PRIOR_SEASONS = (2019, 2020)          # 모든 폴드의 학습 구간에 포함된다
K_VALUES = (200, 1000)

# (비율 컬럼, 분모 컬럼, 짧은 이름)
RATE_N = [
    ("asof_pitcher_success_rate",  "asof_pitcher_n",          "p_succ"),
    ("asof_pitcher_reverse_rate",  "asof_pitcher_n",          "p_rev"),
    ("asof_pitcher_middle_rate",   "asof_pitcher_n",          "p_mid"),
    ("asof_pitcher_ball_rate",     "asof_pitcher_n",          "p_ball"),
    ("asof_pitcher_strike_rate",   "asof_pitcher_n",          "p_str"),
    ("asof_batter_success_rate",   "asof_batter_n",           "b_succ"),
    ("asof_batter_middle_rate",    "asof_batter_n",           "b_mid"),
    ("asof_pitcher_fastball_rate", "asof_pitcher_pitchmix_n", "p_fb"),
    ("asof_pitcher_breaking_rate", "asof_pitcher_pitchmix_n", "p_bb"),
    ("asof_pitcher_offspeed_rate", "asof_pitcher_pitchmix_n", "p_os"),
]

# (최근 컬럼, 기준 컬럼, 짧은 이름)
DEV_PAIRS = [
    (f"asof_pitcher_prev{k}_game_{kind}_rate",
     f"asof_pitcher_{kind}_rate", f"p{k}_{kind[:3]}")
    for kind in ("success", "middle") for k in (1, 3, 5)
]


def main():
    X = np.load(f"{CACHE}/X.npy")
    season = np.load(f"{CACHE}/season.npy")
    meta = json.load(open(f"{CACHE}/cols.json"))
    cols = meta["cols"]
    ix = {c: i for i, c in enumerate(cols)}

    already = [c for c in cols if c.startswith(("sk", "dev_"))]
    if already:
        raise SystemExit(f"이미 붙어 있다 ({len(already)}개). 캐시를 다시 만들려면 "
                         f"exp/prep.py 부터 실행할 것: {already[:4]}")

    prior_mask = np.isin(season, PRIOR_SEASONS)
    print(f"사전값 구간 {PRIOR_SEASONS} n={prior_mask.sum():,} "
          f"(전체의 {prior_mask.mean():.1%})")

    new_cols, new_arrs, priors = [], [], {}
    for rate, ncol, short in RATE_N:
        r = X[:, ix[rate]].astype(np.float64)
        n = X[:, ix[ncol]].astype(np.float64)
        mu = float(np.nanmean(r[prior_mask]))
        priors[short] = mu
        for k in K_VALUES:
            # n 이 결측이면 관측이 없다는 뜻이므로 사전값 그대로 둔다.
            nn = np.where(np.isnan(n), 0.0, n)
            rr = np.where(np.isnan(r), mu, r)
            new_cols.append(f"sk{k}_{short}")
            new_arrs.append(((nn * rr + k * mu) / (nn + k)).astype(np.float32))
        print(f"  {rate:42s} mu={mu:.4f}  n 중앙값 "
              f"{np.nanmedian(n):8.0f}  결측 {np.isnan(r).mean():5.1%}")

    for recent, ref, short in DEV_PAIRS:
        d = (X[:, ix[recent]].astype(np.float64)
             - X[:, ix[ref]].astype(np.float64))
        new_cols.append(f"dev_{short}")
        new_arrs.append(d.astype(np.float32))
        print(f"  dev_{short:10s} = {recent.replace('asof_pitcher_', '')}"
              f" - {ref.replace('asof_pitcher_', '')}   결측 "
              f"{np.isnan(d).mean():5.1%}  표준편차 {np.nanstd(d):.4f}")

    X2 = np.hstack([X, np.stack(new_arrs, axis=1)])
    meta2 = {**meta, "cols": cols + new_cols,
             # prod 는 손대지 않는다. 새 컬럼은 features.add 로만 들어온다.
             "prod": meta["prod"],
             "shrink": new_cols,
             "shrink_prior": {"seasons": list(PRIOR_SEASONS), "mu": priors,
                              "k": list(K_VALUES)}}

    np.save(f"{CACHE}/X.npy", X2)
    with open(f"{CACHE}/cols.json", "w") as f:
        json.dump(meta2, f, indent=1)
    print(f"\n확장 완료: X {X2.shape} ({X2.nbytes/1e6:.0f}MB), "
          f"새 컬럼 {len(new_cols)}개")
    print(f"  prod 그대로 {len(meta['prod'])}개 (기존 캐시 영향 없음)")


if __name__ == "__main__":
    main()
