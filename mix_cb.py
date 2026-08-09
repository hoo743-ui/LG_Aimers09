r"""CatBoost 를 포함한 최종 혼합 가중치 + 중심 보정 계수 동시 탐색.

배경. CatBoost(ID 를 범주형으로 선언하지 않은 것)가 2024 폴드 단독 698.55 로
HGB(660.95)를 **+37.6** 이겼다. 이 프로젝트에서 단일 모델이 주력을 이긴 것은
처음이다. 학습량 추세도 감소 방향이 아니다 — RF 가 실전에서 진 원인(4-8)이
여기엔 없다.

**lambda 도 같이 탐색한다.** 중심 보정 0.03 은 HGB 기반 혼합에 맞춰 고른 값인데
(4-9), CatBoost 는 중심 편차가 +0.0107 로 HGB(+0.0119)보다 작아 최적값이 다를
수 있다. 주력이 바뀌면 보정도 다시 골라야 한다.

기준은 현재 확정 구성이다 — `hgb 0.90 / lr 0.10 / lambda 0.03` (LB 834.58).

판정 (4-8 의 RF 실패 이후 올린 기준):
  1) 3폴드 부호 일치
  2) 2024 폴드 기여 +15 이상
  3) 학습량 추세가 감소 방향이 아닐 것

예측 캐시는 blend_test.py 와 catboost_test.py 가 만든다.

    .\.venv\Scripts\python.exe mix_cb.py
"""
import itertools
import os

import numpy as np
import pandas as pd

DATA = "./data/train.csv"
CACHE = "./.blendcache"
TARGET = "control_success"
PREV1 = "asof_pitcher_prev1_game_success_rate"
FOLDS = [2021, 2022, 2024]
CB_TAG = "cb_d6_l210_it1100_noid_seed42"
CUR_W_LR, CUR_LAM = 0.10, 0.03          # 현재 확정 구성


def load(Y, name):
    p = os.path.join(CACHE, f"{Y}_{name}.npy")
    if not os.path.exists(p):
        raise SystemExit(f"{p} 없음")
    return np.load(p)


def main():
    df = pd.read_csv(DATA, encoding="utf-8-sig",
                     usecols=["season", TARGET, PREV1])
    season = df["season"].to_numpy()
    y_all = df[TARGET].to_numpy(dtype=float)

    fold = {}
    for Y in FOLDS:
        m = season == Y
        y = y_all[m]
        c = float(y_all[season < Y].mean())
        fold[Y] = dict(
            y=y, denom=y.mean() * (1 - y.mean()),
            hgb=load(Y, "hgb_seed42"), lr=load(Y, "lr_seed42"),
            cb=load(Y, CB_TAG),
            anc=df.loc[m, PREV1].fillna(c).to_numpy(dtype=float) - c)

    def sc(Y, w_cb, w_lr, lam):
        d = fold[Y]
        p = ((1 - w_cb - w_lr) * d["hgb"] + w_cb * d["cb"] + w_lr * d["lr"])
        p = np.clip(p + lam * d["anc"], 0, 1)
        return max(0.0, 100000 * (1 - ((p - d["y"]) ** 2).mean() / d["denom"]))

    ref = {Y: sc(Y, 0.0, CUR_W_LR, CUR_LAM) for Y in FOLDS}
    print(f"기준 = 현재 확정 구성 (hgb .90 / lr {CUR_W_LR} / lam {CUR_LAM}) "
          f"— LB 834.58")
    print("  " + "  ".join(f"{Y} {ref[Y]:.2f}" for Y in FOLDS)
          + f"   평균 {np.mean(list(ref.values())):.2f}")
    print(f"\n참고: 폴드별 단독")
    for Y in FOLDS:
        d = fold[Y]
        f = lambda p: max(0.0, 100000*(1-((np.clip(p,0,1)-d["y"])**2).mean()/d["denom"]))
        print(f"  {Y}  hgb {f(d['hgb']):8.2f}   cb {f(d['cb']):8.2f}   "
              f"lr {f(d['lr']):8.2f}")

    steps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    lams = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]
    out = []
    for w_cb, w_lr, lam in itertools.product(steps, [0.0, 0.05, 0.10, 0.15], lams):
        if w_cb + w_lr > 1.0 + 1e-9:
            continue
        per = [sc(Y, w_cb, w_lr, lam) - ref[Y] for Y in FOLDS]
        out.append((w_cb, w_lr, lam, float(np.mean(per)), per,
                    all(x > 0 for x in per)))

    ok = [r for r in out if r[5]]
    print(f"\n부호 일치 조합 {len(ok)} / 전체 {len(out)}")

    print(f"\n=== 2024 폴드 기준 상위 (부호 일치만) ===")
    ok.sort(key=lambda r: -r[4][-1])
    print(f"{'hgb':>6}{'cb':>6}{'lr':>6}{'lam':>6}{'평균':>9}{'2024':>9}"
          f"   {'폴드별':>26}")
    print("-" * 68)
    for w_cb, w_lr, lam, mean, per, _ in ok[:14]:
        print(f"{1-w_cb-w_lr:6.2f}{w_cb:6.2f}{w_lr:6.2f}{lam:6.2f}"
              f"{mean:+9.2f}{per[-1]:+9.2f}   "
              + " ".join(f"{x:+8.2f}" for x in per))

    print(f"\n=== 3폴드 평균 기준 상위 (부호 일치만) ===")
    ok.sort(key=lambda r: -r[3])
    for w_cb, w_lr, lam, mean, per, _ in ok[:8]:
        print(f"{1-w_cb-w_lr:6.2f}{w_cb:6.2f}{w_lr:6.2f}{lam:6.2f}"
              f"{mean:+9.2f}{per[-1]:+9.2f}   "
              + " ".join(f"{x:+8.2f}" for x in per))

    print("""
격자에서 최고점을 그대로 쓰면 이 3폴드를 훔쳐보는 셈이다 (4-6). 두 정렬의
상위가 겹치는 지점, 그중 보수적인(주력 비중이 크고 lam 이 기존값에 가까운)
조합을 고를 것. 채택 기준은 2024 폴드 +15 이상이다.""")


if __name__ == "__main__":
    main()
