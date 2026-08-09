r"""2023 형 붕괴의 정체 — 순위가 틀린 것인가, 수준이 무너진 것인가.

왜. extrap_test.py 로 2023 을 되살려 처음 재보니 클리핑 없는 BSS 가 **−1339.95**
다. 모델이 상수 예측보다 1340점만큼 나쁘다. 4-2 는 "중심을 맞춰도 0점"이라고만
적어뒀고 크기는 몰랐다.

무엇을 가르는가. Murphy 분해에서 **해상도는 항상 0 이상**이므로, BSS 가 음수인
것은 `신뢰도오차 > 해상도` 라는 뜻이다. 원인이 둘 중 어느 쪽이냐에 따라 대응이
완전히 달라진다.

  신뢰도오차가 크다  -> 순위는 멀쩡하고 **확률 수준만** 무너졌다. 원리적으로
                       되찾을 수 있는 손실이며, 2025 에 이 성분이 있다면 큰 건이다
  해상도가 낮다      -> 그 해에는 예측 가능한 신호 자체가 없었다. 손댈 방법이 없다

세 가지를 같이 낸다.

  BSS            실제 점수 (클리핑 없음)
  중심만 보정     예측 평균을 그 해 실제값에 맞췄을 때. 4-2 가 "그래도 0"이라 한 것
  완전 재보정     예측을 구간으로 묶어 각 구간을 그 해 실제 비율로 바꿨을 때.
                 = 해상도/불확실성. **순위 품질의 상한이자 캘리브레이션의 상한**

완전 재보정은 그 폴드의 정답을 쓰므로 실전값이 아니다. **순위가 살아 있는지**를
보는 진단이다. 살아 있다면 그 해의 손실은 원리적으로 회수 가능한 종류다.

예측 캐시는 extrap_test.py 가 만든다 (`.extrapcache/`).

    .\.venv\Scripts\python.exe regime_diag.py
"""
import argparse
import os

import numpy as np
import pandas as pd

DATA = "./data/train.csv"
CACHE = "./.extrapcache"
TARGET = "control_success"
FOLDS = [2020, 2021, 2022, 2023, 2024]
NBIN = 20


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="L10")
    p.add_argument("--bins", type=int, default=NBIN)
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(DATA, encoding="utf-8-sig",
                     usecols=["season", TARGET,
                              "asof_pitcher_prev1_game_success_rate"])
    season = df["season"].to_numpy()
    y_all = df[TARGET].to_numpy(dtype=float)

    print(f"구성 {args.config} | 구간 {args.bins}개 | 지표는 클리핑 없는 BSS x 100000\n")
    print(f"{'폴드':>6}{'실제r':>8}{'예측평균':>9}{'중심편차':>10}"
          f"{'BSS':>10}{'중심만보정':>11}{'완전재보정':>11}{'재보정이득':>11}")
    print("-" * 76)

    rows = []
    for Y in FOLDS:
        path = os.path.join(CACHE, f"{Y}_{args.config}_seed42.npy")
        if not os.path.exists(path):
            print(f"{Y:>6}  캐시 없음 — extrap_test.py 를 먼저 돌릴 것")
            continue
        m = season == Y
        y = y_all[m]
        p = np.load(path)
        if len(p) != m.sum():
            raise SystemExit(f"{Y}: 캐시 {len(p)} != 행 {m.sum()}")
        r = y.mean()
        unc = r * (1 - r)

        def bss(pp):
            return 100000.0 * (1 - ((np.clip(pp, 0, 1) - y) ** 2).mean() / unc)

        s_raw = bss(p)
        s_center = bss(p + (r - p.mean()))       # 중심만 맞춤 (정답을 씀)

        # 완전 재보정 — 예측 구간마다 그 해 실제 비율로 치환 (= 해상도/불확실성)
        q = pd.qcut(pd.Series(p), args.bins, labels=False,
                    duplicates="drop").to_numpy()
        p_cal = np.empty_like(p)
        res = 0.0
        for k in np.unique(q):
            sel = q == k
            p_cal[sel] = y[sel].mean()
            res += (sel.mean()) * (y[sel].mean() - r) ** 2
        s_full = bss(p_cal)

        rows.append((Y, r, p.mean(), p.mean() - r, s_raw, s_center, s_full))
        print(f"{Y:>6}{r:8.4f}{p.mean():9.4f}{p.mean()-r:+10.4f}"
              f"{s_raw:10.2f}{s_center:11.2f}{s_full:11.2f}"
              f"{s_full-s_raw:+11.2f}")

    print(f"""
읽는 법.
  중심만보정  예측 평균을 그 해 실제값에 맞췄을 때. 4-2 는 2023 에서 이래도
              0 이라고 했다 — 이 열이 그 주장을 수치로 확인한다
  완전재보정  구간별로 그 해 실제 비율로 바꿨을 때. **순위 품질의 상한**이다.
              이 값이 크면 순위는 살아 있고 손실은 캘리브레이션 쪽이다
  재보정이득  회수 가능한 총량의 상한. 2023 에서 이게 크면 국면 붕괴는
              "신호가 사라진 것"이 아니라 "수준이 어긋난 것"이다""")

    # ---- 등온 회귀 이식 ----
    # "완전재보정 716.51" 은 그 폴드의 **정답**으로 구간을 갈아끼운 상한이라
    # 만들 수 있는 모델이 아니다. 다만 그 모양 보정을 **학습 시즌에서 배워
    # 평가 시즌에 옮기는 것**은 합법이고 한 번도 안 해봤다.
    #
    # 지금까지 시험한 재보정은 파라미터 1개짜리 둘뿐이었다 —
    # 중심 이동(위치, +7.16 채택)과 alpha(척도, 기각). 등온 회귀는 모양 전체를
    # 배우므로 파라미터가 많고, 그만큼 이식이 어려울 수 있다.
    print("\n=== 등온 회귀 이식 (직전 폴드에서 배워 평가 폴드에 적용) ===")
    print("  '완전재보정' 은 정답을 쓴 상한이다. 이 열이 실제로 쓸 수 있는 값이다.")
    from sklearn.isotonic import IsotonicRegression

    print(f"{'평가':>6}{'학습원':>9}{'기준':>10}{'등온이식':>10}{'차이':>9}"
          f"{'(상한)':>10}")
    print("-" * 56)
    for i, Y in enumerate(FOLDS):
        if i == 0:
            continue
        src = FOLDS[i - 1]
        ps = os.path.join(CACHE, f"{src}_{args.config}_seed42.npy")
        pe = os.path.join(CACHE, f"{Y}_{args.config}_seed42.npy")
        if not (os.path.exists(ps) and os.path.exists(pe)):
            continue
        m_s, m_e = season == src, season == Y
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(np.load(ps), y_all[m_s])

        p = np.load(pe)
        y = y_all[m_e]
        r = y.mean()
        unc = r * (1 - r)
        base = 100000.0 * (1 - ((p - y) ** 2).mean() / unc)
        adj = np.clip(iso.predict(p), 0, 1)
        got = 100000.0 * (1 - ((adj - y) ** 2).mean() / unc)
        cap = next((r_[6] for r_ in rows if r_[0] == Y), float("nan"))
        print(f"{Y:>6}{src:>9}{base:10.2f}{got:10.2f}{got-base:+9.2f}"
              f"{cap:10.2f}")
    print("  파라미터가 20개 넘는 보정이라 중심 이동(1개)보다 이식이 어렵다.\n"
          "  이 열이 음수면 모양 보정은 시즌을 못 넘는다는 뜻이다.")

    # ---- 축소계수 보험 ----
    # 2023 은 과대확신(재보정이 1830 을 되찾는다), 2024 는 과소확신(alpha* 1.128).
    # **폴드마다 부호가 반대다.** 3폴드만 보면 전부 "벌리자"가 되는데 2023 을
    # 넣으면 뒤집힌다. 예측을 좁히는 것은 정상 해에 소폭 손해지만 붕괴 해에
    # 수백 점을 구하는 보험이다. 그 거래 조건을 정확히 잰다.
    print("\n=== 축소계수 보험 (alpha < 1 = 예측을 좁힌다) ===")
    print("  중심(c)은 학습 데이터 성공률. 평가 정답을 쓰지 않는다.")
    alphas = [1.00, 0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50]
    tab = []
    for Y in FOLDS:
        path = os.path.join(CACHE, f"{Y}_{args.config}_seed42.npy")
        if not os.path.exists(path):
            continue
        m = season == Y
        y = y_all[m]
        p = np.load(path)
        r = y.mean()
        unc = r * (1 - r)
        c = float(y_all[season < Y].mean())
        row = []
        for a in alphas:
            pp = np.clip(c + a * (p - c), 0, 1)
            row.append(100000.0 * (1 - ((pp - y) ** 2).mean() / unc))
        tab.append((Y, row))

    print(f"{'폴드':>6}" + "".join(f"{a:10.2f}" for a in alphas))
    print("-" * (6 + 10 * len(alphas)))
    for Y, row in tab:
        print(f"{Y:>6}" + "".join(f"{v:10.1f}" for v in row))
    arr = np.array([r_ for _, r_ in tab])
    print(f"{'평균':>6}" + "".join(f"{arr[:, j].mean():10.1f}"
                                   for j in range(len(alphas))))
    print(f"{'최악':>6}" + "".join(f"{arr[:, j].min():10.1f}"
                                   for j in range(len(alphas))))
    d = arr - arr[:, :1]
    print(f"{'vs 1.0':>6}" + "".join(f"{d[:, j].mean():+10.1f}"
                                     for j in range(len(alphas))))
    j = int(np.argmax(arr.mean(axis=0)))
    print(f"\n  5폴드 평균 최고 alpha = {alphas[j]:.2f}  "
          f"(평균 {arr[:, j].mean():.1f}, 1.00 대비 {d[:, j].mean():+.1f})")
    j3 = int(np.argmax(arr[[1, 2, 4]].mean(axis=0)))   # 2021/2022/2024 만
    print(f"  2023 제외 3폴드 기준 최고 alpha = {alphas[j3]:.2f} "
          f"— 2023 을 버리면 이 답이 나온다")

    if len(rows) >= 2:
        arr = np.array([(r_[6], r_[4]) for r_ in rows])
        print(f"\n순위 품질(완전재보정) 폴드별: "
              + "  ".join(f"{r_[0]} {r_[6]:.0f}" for r_ in rows))
        print(f"실제 점수(BSS)      폴드별: "
              + "  ".join(f"{r_[0]} {r_[4]:.0f}" for r_ in rows))


if __name__ == "__main__":
    main()
