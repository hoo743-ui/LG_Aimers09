r"""축소계수 alpha 를 **작게 고정**해서 쓸 수 있는가.

배경. 2024 폴드 최적 a 는 1.128 이고 그때 이득이 +19.40 이다 — 남은 후보 중
상한이 가장 크다. 그런데 4-5 는 축소 보정을 기각했고, survey.py 의 LOO 이식도
-2.93 이었다.

**그 실패가 중심 보정(4-9)과 똑같은 모양이다.** 거기서도 LOO 로 lambda 를
자유롭게 맞추면 -1.06 이었는데, 작게 **고정**하니 +13.86 으로 살아났다. cap 이
큰 폴드가 계수를 끌어올려 이미 맞은 폴드를 망치는 구조였고, 처방은 축소였다.

alpha 에도 같은 처방을 시험한다. 이득 곡선이

    gain(a) = cap * (1 - (1 - (a-1)/(a*-1))^2)

이므로, a 를 1 에 가깝게 고정하면 a* 가 큰 폴드에서 이득을 가져오면서 a* 가
1 근처인 폴드(2022)의 손실은 작게 묶인다.

순서. script.py 와 같다 — 앙상블 평균 -> 계열 혼합 -> alpha -> 중심 보정.
여기서 alpha 만 바꿔가며 잰다. 순서를 바꾸면 실제 추론과 다른 값을 재게 된다.

예측 캐시는 blend_test.py 가 만든다 (`.blendcache/`).

    .\.venv\Scripts\python.exe alpha_test.py
"""
import os

import numpy as np
import pandas as pd

DATA = "./data/train.csv"
CACHE = "./.blendcache"
TARGET = "control_success"
PREV1 = "asof_pitcher_prev1_game_success_rate"
FOLDS = [2021, 2022, 2024]
W_LR = 0.10        # 4-8 확정
W_CB = 0.60        # 4-14 확정 — 주력이 CatBoost 로 바뀌었다
CB_TAG = "cb_d6_l210_it1100_noid_seed42"
LAM = 0.03         # 4-9 확정
# 주력이 바뀌면 캘리브레이션 특성도 바뀐다. CatBoost 는 예측 sd 가 좁고 중심
# 편차도 작아 최적 alpha 가 1 보다 작을 수도 있다 — 양쪽으로 훑는다.
ALPHAS = [0.90, 0.95, 0.98, 1.00, 1.02, 1.05, 1.08, 1.10, 1.13]


def load(Y, name, seeds=1):
    acc, n = None, 0
    for s in range(42, 42 + seeds):
        p = os.path.join(CACHE, f"{Y}_{name}_seed{s}.npy")
        if os.path.exists(p):
            v = np.load(p)
            acc = v if acc is None else acc + v
            n += 1
    if acc is None:
        raise SystemExit(f"{Y} {name} 캐시 없음")
    return acc / n


def main():
    df = pd.read_csv(DATA, encoding="utf-8-sig",
                     usecols=["season", TARGET, PREV1])
    season = df["season"].to_numpy()
    y_all = df[TARGET].to_numpy(dtype=float)

    print(f"순서: 앙상블 -> 혼합(hgb {1-W_LR-W_CB:.2f} / cb {W_CB} / lr {W_LR})"
          f" -> alpha -> 중심보정(lam {LAM})")
    print(f"{'폴드':>7}{'기준':>10}" + "".join(f"{a:9.2f}" for a in ALPHAS))
    print("-" * (17 + 9 * len(ALPHAS)))

    rows = []
    for Y in FOLDS:
        m = season == Y
        y = y_all[m]
        denom = y.mean() * (1 - y.mean())
        c = float(y_all[season < Y].mean())        # 학습 데이터 성공률 (상수)
        p_h = load(Y, "hgb")
        p_l = load(Y, "lr")
        p_c = load(Y, CB_TAG.replace("_seed42", ""))
        a_col = df.loc[m, PREV1].fillna(c).to_numpy(dtype=float) - c

        p_mix = (1 - W_LR - W_CB) * p_h + W_LR * p_l + W_CB * p_c
        cur = np.clip(p_mix + LAM * a_col, 0, 1)   # alpha=1 인 현재 구성
        base = max(0.0, 100000 * (1 - ((cur - y) ** 2).mean() / denom))

        row = []
        for a in ALPHAS:
            p = c + a * (p_mix - c)                # alpha 적용
            p = np.clip(p + LAM * a_col, 0, 1)     # 그 다음 중심 보정
            row.append(max(0.0, 100000 * (1 - ((p - y) ** 2).mean() / denom))
                       - base)
        rows.append(row)
        print(f"{Y:>7}{base:10.2f}" + "".join(f"{x:+9.2f}" for x in row))

    arr = np.array(rows)
    print(f"{'평균':>7}{'':>10}" + "".join(f"{arr[:, j].mean():+9.2f}"
                                          for j in range(len(ALPHAS))))
    print(f"{'최악':>7}{'':>10}" + "".join(f"{arr[:, j].min():+9.2f}"
                                          for j in range(len(ALPHAS))))
    ok = [j for j in range(len(ALPHAS)) if (arr[:, j] > -0.01).all()]
    if ok:
        j = max(ok, key=lambda j: arr[:, j].mean())
        print(f"\n세 폴드 모두 손해 없는 최고 alpha = {ALPHAS[j]:.2f}  "
              f"평균 {arr[:, j].mean():+.2f}  최악 {arr[:, j].min():+.2f}  "
              f"2024 {arr[-1, j]:+.2f}")
    else:
        print("\n세 폴드 모두 손해 없는 alpha 가 없다")
    print("""
읽는 법. alpha=1.00 이 현재 구성이므로 그 열은 0 이다. 1 보다 크면 예측을
벌린다(과소확신 교정). 2022 는 원래 잘 맞아서 벌리면 손해가 나므로, 세 폴드
모두 손해가 없는 지점을 고르는 것이 4-6 규약에 맞는다.""")


if __name__ == "__main__":
    main()
