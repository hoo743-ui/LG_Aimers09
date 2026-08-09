r"""잔차 상관 — 어떤 모델을 섞을 값어치가 있는지 먼저 가른다.

왜 이걸 먼저 보는가. 혼합 이득은 성능이 아니라 **오차의 독립성**에서 나온다.
로지스틱은 단독 164.59 로 참사인데 혼합에서 +8.24 를 냈다 (4-8). 반대로
잔차 상관이 0.98 이면 아무리 좋은 모델이라도 섞어서 얻을 게 거의 없다.

그러므로 XGBoost / LightGBM 을 튜닝하기 **전에** 상관부터 봐야 한다. 튜닝은
비싸고(라이브러리당 수십 분) 상관은 공짜다.

이론. 두 예측을 w 로 섞을 때 오차 분산은

    Var((1-w)e1 + w·e2) = (1-w)^2 V1 + w^2 V2 + 2w(1-w)·rho·sqrt(V1 V2)

rho 가 1 에 가까우면 어떤 w 로도 개선이 없다. rho 가 낮을수록 최적 w 가 커지고
이득도 커진다. 성능이 비슷한 두 모델이면 최적 이득은 대략 (1-rho)/2 에 비례한다.

**잔차로 재야 한다.** 예측값끼리의 상관은 공통 신호(중심, 카운트 효과 등) 때문에
항상 높게 나온다. 우리가 알고 싶은 건 **틀리는 방식이 다른가** 이므로 y-p 를 본다.

    .\.venv\Scripts\python.exe corr_test.py
"""
import os

import numpy as np
import pandas as pd

DATA = "./data/train.csv"
CACHE = "./.blendcache"
TARGET = "control_success"
FOLD = 2024

# (표시 이름, 캐시 파일 접미사)
CANDS = [
    ("hgb",      "hgb_seed42"),
    ("cb_d6",    "cb_d6_l210_it1100_noid_seed42"),
    ("cb_d7",    "cb_d7_l210_it600_noid_seed42"),
    ("cb_d7_s43","cb_d6_l210_it1100_noid_seed43"),   # 같은 계열 다른 시드
    ("rf",       "rf_seed42"),
    ("lr",       "lr_seed42"),
]


def main():
    df = pd.read_csv(DATA, encoding="utf-8-sig", usecols=["season", TARGET])
    m = df["season"].to_numpy() == FOLD
    y = df.loc[m, TARGET].to_numpy(dtype=float)
    denom = y.mean() * (1 - y.mean())

    names, preds, scores = [], [], []
    for label, suf in CANDS:
        p = os.path.join(CACHE, f"{FOLD}_{suf}.npy")
        if not os.path.exists(p):
            continue
        v = np.load(p)
        names.append(label)
        preds.append(v)
        scores.append(max(0.0, 100000 * (1 - ((v - y) ** 2).mean() / denom)))

    print(f"fold {FOLD} | 모델 {len(names)}개\n")
    print(f"{'모델':>11}{'단독점수':>10}{'예측 sd':>10}")
    for n, s, p in zip(names, scores, preds):
        print(f"{n:>11}{s:10.2f}{p.std():10.4f}")

    # 잔차 상관은 이 문제에서 무의미하다. y-p 는 기약 노이즈(분산 0.25)가
    # 모델 오차(분산 ~0.0018)보다 140배 커서 무엇을 넣어도 0.997 이 나온다.
    # **예측값끼리의 불일치**를 봐야 기약 노이즈가 빠지고 모델 차이만 남는다.
    print(f"\n=== 예측 상관 (기약 노이즈 제거됨) ===")
    print(f"{'':>11}" + "".join(f"{n:>11}" for n in names))
    R = np.corrcoef(np.vstack(preds))
    for i, n in enumerate(names):
        print(f"{n:>11}" + "".join(f"{R[i, j]:11.4f}" for j in range(len(names))))

    print(f"\n=== 불일치 크기  sd(p_i - p_j) / 평균 예측 sd ===")
    print(f"{'':>11}" + "".join(f"{n:>11}" for n in names))
    for i, n in enumerate(names):
        row = []
        for j in range(len(names)):
            if i == j:
                row.append("          -")
            else:
                d = (preds[i] - preds[j]).std()
                ref = 0.5 * (preds[i].std() + preds[j].std())
                row.append(f"{d/ref:11.3f}")
        print(f"{n:>11}" + "".join(row))

    # ---- 불일치가 **유용한 방향**인가 ----
    # 불일치만으로는 부족하다. RF 는 불일치가 0.47~0.55 로 컸는데 실전에서
    # -9.45 였다 (4-8). 그 다름이 CatBoost 가 틀린 쪽이어야 이득이 난다.
    #
    #   w*    = -E[(p_cb - y)(p_o - p_cb)] / E[(p_o - p_cb)^2]
    #   최대이득 = E[(p_cb - y)(p_o - p_cb)]^2 / E[(p_o - p_cb)^2]
    #
    # 분모가 불일치, 분자가 정렬도다. 이득 = 불일치 x 정렬도^2 로 분해된다.
    base = "cb_d7" if "cb_d7" in names else names[0]
    b = preds[names.index(base)]
    e = b - y
    print(f"\n=== {base} 에 섞었을 때 (닫힌 해, 이 폴드 상한) ===")
    print(f"{'상대':>11}{'불일치sd':>10}{'정렬도':>9}{'최적w':>9}{'최대이득':>10}")
    print("-" * 50)
    for n, p in zip(names, preds):
        if n == base:
            continue
        d = p - b
        den = float((d ** 2).mean())
        if den <= 0:
            continue
        num = -float((e * d).mean())
        w = num / den
        gain = 100000 * (num ** 2 / den) / denom
        # 정렬도 = 불일치와 오차의 상관. 부호까지 본다.
        align = -float(np.corrcoef(d, e)[0, 1])
        print(f"{n:>11}{np.sqrt(den):10.4f}{align:+9.3f}{w:+9.3f}{gain:10.2f}")

    print("""
읽는 법.
  **하한선**은 같은 계열 다른 시드의 불일치다(0.156). 시드 앙상블로 이미
  걷어내는 성분이므로 그 수준이면 계열을 바꾼 의미가 없다.

  그런데 불일치만으로는 부족하다 — RF 는 불일치가 컸는데 실전에서 -9.45 였다.
  **정렬도**가 그 다름이 유용한 방향인지를 말한다. 이득은 불일치 x 정렬도^2 에
  비례하므로 둘 다 필요하다.

  '최대이득'은 이 폴드에서 w 를 최적화한 **상한**이다. 실전값은 3폴드에서
  고정 w 로 재야 한다 (4-6).""")


if __name__ == "__main__":
    main()
