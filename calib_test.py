r"""캘리브레이션 3종 비교 — alpha / sigmoid(Platt) / isotonic.

왜 다시 재는가. 모델이 바뀌면 캘리브레이션 특성도 바뀐다. 이번 세션에서 두 번
확인됐다 — `same_hand` 추가로 최적 alpha 가 1.01 -> 1.1278 로 움직였고,
CatBoost 도입 후 **전 구간 양수인 alpha 가 처음 생겼다**(1.02). 주력이 HGB 에서
CatBoost 로 바뀌었으니 전부 다시 재야 한다.

**파라미터 수 가설을 검정한다.** 4-11 에서 이식 가능성이 파라미터 수에 따라
갈렸다.

    위치 (중심 이동, 1개)     +7.16          채택
    척도 (alpha, 1개)         +1~3           경계선
    모양 (등온, 20개+)        -125 ~ -804    적극적 유해

`sigmoid`(Platt) 는 **2개**(a·logit(p)+b)라 그 사이에 있어야 한다. 그게 맞으면
가설이 강화되고, 아니면 파라미터 수가 아닌 다른 요인이 있다는 뜻이다.

**시점 분리가 이 도구의 전부다.** 평가 폴드 자기 정답으로 맞춘 캘리브레이션은
항상 이득이 난다. 이전 시즌에서 배워 평가 시즌에 옮긴 값만 실전값이다.

    .\.venv\Scripts\python.exe calib_test.py
    .\.venv\Scripts\python.exe calib_test.py --cb-tag cb_d7_l210_it600_noid_seed42
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

DATA = "./data/train.csv"
CACHE = "./.blendcache"
TARGET = "control_success"
PREV1 = "asof_pitcher_prev1_game_success_rate"
FOLDS = [2021, 2022, 2024]
# 풀링용 — 클리핑 없는 비교가 가능하도록 2020·2023 도 쓴다. 예측 캐시는
# extrap_test.py 가 만든 L10 HGB 이고, 혼합 성분이 없어 근사다. 다만 곡선의
# **모양**을 배우는 데는 충분하다.
POOL_FOLDS = [2020, 2021, 2022, 2023]
POOL_CACHE = "./.extrapcache"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cb-tag", default="cb_d7_l210_it600_noid_seed42")
    p.add_argument("--w-cb", type=float, default=0.60)
    p.add_argument("--w-lr", type=float, default=0.10)
    p.add_argument("--lam", type=float, default=0.03)
    return p.parse_args()


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def main():
    args = parse_args()
    df = pd.read_csv(DATA, encoding="utf-8-sig",
                     usecols=["season", TARGET, PREV1])
    season = df["season"].to_numpy()
    y_all = df[TARGET].to_numpy(dtype=float)

    mix, ys, denoms = {}, {}, {}
    for Y in FOLDS:
        m = season == Y
        y = y_all[m]
        c = float(y_all[season < Y].mean())
        anc = df.loc[m, PREV1].fillna(c).to_numpy(dtype=float) - c
        p_h = np.load(os.path.join(CACHE, f"{Y}_hgb_seed42.npy"))
        p_l = np.load(os.path.join(CACHE, f"{Y}_lr_seed42.npy"))
        p_c = np.load(os.path.join(CACHE, f"{Y}_{args.cb_tag}.npy"))
        w_h = 1 - args.w_cb - args.w_lr
        # 중심 보정(4-9)까지 적용한 것이 현재 파이프라인의 최종 출력이다.
        # 캘리브레이션은 그 위에 얹는 것이므로 여기서부터 시작한다.
        mix[Y] = np.clip(w_h * p_h + args.w_cb * p_c + args.w_lr * p_l
                         + args.lam * anc, 0, 1)
        ys[Y], denoms[Y] = y, y.mean() * (1 - y.mean())

    def sc(Y, p):
        return max(0.0, 100000 * (1 - ((np.clip(p, 0, 1) - ys[Y]) ** 2).mean()
                                  / denoms[Y]))

    print(f"CatBoost: {args.cb_tag}")
    print(f"혼합: hgb {1-args.w_cb-args.w_lr:.2f} / cb {args.w_cb} / "
          f"lr {args.w_lr} / lam {args.lam}\n")
    # 풀링 등온 — 학습 시즌 전체의 OOF 예측을 합쳐 곡선 하나를 배운다.
    # 단일 시즌 이식은 그 해의 모양에 베팅하는 것이라 2023 같은 이상치가
    # 학습원이면 무너진다. 풀링은 국면 이동의 **평균 패턴**을 잡는다.
    pool_p, pool_y = [], []
    for Y in POOL_FOLDS:
        f = os.path.join(POOL_CACHE, f"{Y}_L10_seed42.npy")
        if os.path.exists(f):
            pool_p.append(np.load(f))
            pool_y.append(y_all[season == Y])
    iso_pool = None
    if pool_p:
        iso_pool = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso_pool.fit(np.concatenate(pool_p), np.concatenate(pool_y))
        print(f"풀링 등온: {POOL_FOLDS} 의 OOF 예측 "
              f"{sum(len(p) for p in pool_p):,} 행으로 곡선 1개 적합\n")

    print(f"{'평가':>6}{'학습원':>9}{'기준':>10}{'alpha':>10}{'sigmoid':>10}"
          f"{'iso(단일)':>11}{'iso(풀링)':>11}")
    print(f"{'':>6}{'':>9}{'':>10}{'1개':>10}{'2개':>10}{'20+개':>11}{'20+개':>11}")
    print("-" * 68)

    tot = {"alpha": [], "sigmoid": [], "isotonic": [], "iso_pool": []}
    for i, Y in enumerate(FOLDS):
        if i == 0:
            continue
        src = FOLDS[i - 1]
        ps, yv = mix[src], ys[src]
        pe = mix[Y]
        base = sc(Y, pe)

        # 1) alpha — 척도만. 중심은 학습 데이터 성공률(상수)
        c = float(y_all[season < Y].mean())
        d = ps - c
        a = 1.0 if (d ** 2).mean() <= 0 else float((d * (yv - c)).mean()
                                                   / (d ** 2).mean())
        s_alpha = sc(Y, c + a * (pe - c))

        # 2) sigmoid (Platt) — logit 공간의 1차식, 파라미터 2개
        lr = LogisticRegression(solver="lbfgs")
        lr.fit(logit(ps).reshape(-1, 1), yv)
        s_sig = sc(Y, lr.predict_proba(logit(pe).reshape(-1, 1))[:, 1])

        # 3) isotonic — 단조 자유형, 파라미터 20개+
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(ps, yv)
        s_iso = sc(Y, iso.predict(pe))

        # 4) 풀링 등온 — 학습 시즌 전체 OOF 에서 배운 곡선 하나
        s_pool = sc(Y, iso_pool.predict(pe)) if iso_pool is not None else base

        for k, v in (("alpha", s_alpha), ("sigmoid", s_sig),
                     ("isotonic", s_iso), ("iso_pool", s_pool)):
            tot[k].append(v - base)
        print(f"{Y:>6}{src:>9}{base:10.2f}"
              f"{s_alpha-base:+10.2f}{s_sig-base:+10.2f}"
              f"{s_iso-base:+11.2f}{s_pool-base:+11.2f}")

    print(f"{'평균':>6}{'':>9}{'':>10}"
          + f"{np.mean(tot['alpha']):+10.2f}{np.mean(tot['sigmoid']):+10.2f}"
          + f"{np.mean(tot['isotonic']):+11.2f}"
          + f"{np.mean(tot['iso_pool']):+11.2f}")
    print("""
읽는 법. 전부 **이전 시즌에서 배워 평가 시즌에 적용**한 값이다. 자기 폴드
정답으로 맞추면 셋 다 양수가 나오지만 그건 실전값이 아니다.

파라미터가 적을수록 이식이 잘 된다는 것이 4-11 의 관찰이었다. sigmoid(2개)가
alpha(1개)와 isotonic(20개+) 사이에 오면 그 가설이 강화된다.""")


if __name__ == "__main__":
    main()
