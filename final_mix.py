r"""제출 파이프라인 그대로에서 혼합 가중치를 다시 고른다.

blend_test.py 의 격자는 **중심 보정(4-9) 이전**에 잰 것이다. RF 를 더할지
말지는 중심 보정 위에서 다시 봐야 한다 — 두 효과가 겹치면 RF 의 이득이
사라질 수 있다.

순서는 script.py 와 같다.

    앙상블 평균 -> 계열 혼합 -> alpha(=1) -> 중심 보정 -> clip

RF 를 넣으면 배선 비용이 붙는다. 200그루 직렬화는 HGB 와 형식이 달라 6-4
(낮은 numpy 에서 pkl 로드 실패)를 처음부터 재검증해야 하고, pkl 이 수 MB 늘고
추론 시간도 는다. **2024 폴드에서 확실히 남을 때만 값어치가 있다.**

    .\.venv\Scripts\python.exe final_mix.py
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
LAM = 0.03


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

    fold = {}
    for Y in FOLDS:
        m = season == Y
        y = y_all[m]
        c = float(y_all[season < Y].mean())
        fold[Y] = dict(
            y=y, denom=y.mean() * (1 - y.mean()), c=c,
            hgb=load(Y, "hgb"), rf=load(Y, "rf"), lr=load(Y, "lr"),
            anc=df.loc[m, PREV1].fillna(c).to_numpy(dtype=float) - c)

    def score(Y, w_rf, w_lr):
        d = fold[Y]
        p = (1 - w_rf - w_lr) * d["hgb"] + w_rf * d["rf"] + w_lr * d["lr"]
        p = np.clip(p + LAM * d["anc"], 0, 1)
        return max(0.0, 100000 * (1 - ((p - d["y"]) ** 2).mean() / d["denom"]))

    ref = {Y: score(Y, 0.0, 0.10) for Y in FOLDS}     # 현재 확정 구성
    print("현재 구성 (hgb 0.90 / lr 0.10 / 중심보정 0.03)")
    print("  " + "  ".join(f"{Y} {ref[Y]:.2f}" for Y in FOLDS)
          + f"   평균 {np.mean(list(ref.values())):.2f}\n")

    steps = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    out = []
    for w_rf, w_lr in itertools.product(steps, steps):
        if 1 - w_rf - w_lr < 0.55 - 1e-9:
            continue
        per = [score(Y, w_rf, w_lr) - ref[Y] for Y in FOLDS]
        out.append((w_rf, w_lr, float(np.mean(per)), per, all(x > 0 for x in per)))

    out.sort(key=lambda r: -r[3][-1])       # 2024 기준 정렬
    print(f"{'hgb':>6}{'rf':>6}{'lr':>6}{'평균':>9}{'2024':>9}"
          f"   {'폴드별 (현재 구성 대비)':>26}  부호")
    print("-" * 74)
    for w_rf, w_lr, mean, per, ok in out[:14]:
        print(f"{1-w_rf-w_lr:6.2f}{w_rf:6.2f}{w_lr:6.2f}{mean:+9.2f}"
              f"{per[-1]:+9.2f}   " + " ".join(f"{x:+8.2f}" for x in per)
              + ("  일치" if ok else "  ★엇갈림"))

    best = [r for r in out if r[4]]
    if best:
        b = max(best, key=lambda r: r[3][-1])
        print(f"\n세 폴드 모두 양수 중 2024 최고: "
              f"hgb {1-b[0]-b[1]:.2f} / rf {b[0]:.2f} / lr {b[1]:.2f}"
              f"  →  2024 {b[3][-1]:+.2f}  평균 {b[2]:+.2f}")
    else:
        print("\n현재 구성을 이기면서 세 폴드 모두 양수인 조합이 없다")
    print("""
RF 를 넣으려면 2024 에서 확실히 남아야 한다. 배선 비용이 붙기 때문이다 —
200그루 직렬화는 HGB 와 형식이 달라 6-4 를 처음부터 재검증해야 한다.""")


if __name__ == "__main__":
    main()
