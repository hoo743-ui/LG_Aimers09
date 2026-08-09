r"""CatBoost 시드 수를 늘리면 얼마나 얻는가.

현재 확정 구성은 CatBoost **2시드** 평균이다 (4-14). HGB 는 5시드를 쓰는데
CatBoost 만 2개인 것은 학습 시간(폴드당 7분) 때문이지 측정 결과가 아니다.

시드 평균은 이 프로젝트에서 **실패 사례가 없는 유일한 조작**이다 — 예측의
퍼짐은 그대로 두고 분산만 깎으므로 Brier 가 거의 항상 내려간다 (README 5장).
다만 수확 체감이 있으므로 어디서 멈출지는 재봐야 한다.

관측된 CatBoost 시드 편차는 단독 기준 13점이었다 (698.55 vs 711.57).
HGB 의 ±15 와 비슷하므로 평균낼 값어치가 있다.

평가는 제출 파이프라인 그대로다 — hgb 0.30 / cb 0.60 / lr 0.10, lam 0.03.

    .\.venv\Scripts\python.exe cb_seeds.py
"""
import os

import numpy as np
import pandas as pd

DATA = "./data/train.csv"
CACHE = "./.blendcache"
TARGET = "control_success"
PREV1 = "asof_pitcher_prev1_game_success_rate"
FOLD = 2024
W_HGB, W_CB, W_LR, LAM = 0.30, 0.60, 0.10, 0.03
TAG = "cb_d6_l210_it1100_noid_seed"


def main():
    df = pd.read_csv(DATA, encoding="utf-8-sig",
                     usecols=["season", TARGET, PREV1])
    m = df["season"].to_numpy() == FOLD
    y = df.loc[m, TARGET].to_numpy(dtype=float)
    denom = y.mean() * (1 - y.mean())
    c = float(df.loc[df["season"].to_numpy() < FOLD, TARGET].mean())
    anc = df.loc[m, PREV1].fillna(c).to_numpy(dtype=float) - c
    p_h = np.load(os.path.join(CACHE, f"{FOLD}_hgb_seed42.npy"))
    p_l = np.load(os.path.join(CACHE, f"{FOLD}_lr_seed42.npy"))

    def sc(p):
        return max(0.0, 100000 * (1 - ((np.clip(p, 0, 1) - y) ** 2).mean() / denom))

    seeds, preds = [], []
    for s in range(42, 50):
        p = os.path.join(CACHE, f"{FOLD}_{TAG}{s}.npy")
        if os.path.exists(p):
            seeds.append(s)
            preds.append(np.load(p))
    if not preds:
        raise SystemExit("CatBoost 예측 캐시가 없다")
    print(f"사용 가능한 시드 {seeds}\n")

    print(f"{'시드수':>6}{'단독':>10}{'혼합':>10}{'2시드 대비':>11}")
    print("-" * 38)
    ref = None
    for k in range(1, len(preds) + 1):
        p_cb = np.mean(preds[:k], axis=0)
        solo = sc(p_cb)
        mix = sc(np.clip(W_HGB * p_h + W_CB * p_cb + W_LR * p_l + LAM * anc,
                         0, 1))
        if k == 2:
            ref = mix
        d = "" if ref is None or k == 2 else f"{mix - ref:+11.2f}"
        print(f"{k:6d}{solo:10.2f}{mix:10.2f}"
              + (f"{mix-ref:+11.2f}" if ref is not None and k != 2 else
                 ("     (기준)" if k == 2 else "")))

    print(f"\n시드별 단독: "
          + "  ".join(f"s{s} {sc(p):.2f}" for s, p in zip(seeds, preds)))
    print("""
시드 평균은 실패 사례가 없지만 수확 체감이 있다. 늘린 만큼 학습·추론 비용이
붙으므로(폴드당 7분, 추론도 비례) 이득이 평평해지는 지점에서 멈출 것.
그리고 제출 기회를 쓰려면 로컬 +25~30 이 필요하다 (1장) — 시드만으로는
대개 그에 못 미치므로 다른 개선과 합쳐서 판단할 것.""")


if __name__ == "__main__":
    main()
