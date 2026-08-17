r"""후처리 **5번째 편차축** 후보 — 제출 후보 선정용. 학습 0회.

## 왜 이것인가

F-specific 은 생산 경로에서 3폴드 전부 음수로 기각됐다 (xf −0.7/−1.2/−4.9,
hf −1.9/−7.5/−1.8, xhf +0.0/−5.9/−1.5). 대체 후보를 근거로 고른다.

오늘 47열 x 쌍 감사에서 손익분기(+19.8)를 넘긴 오라클이 둘 나왔다.

    pitcher x 점수차      +20.5   어느 기존 축도 다루지 않는 승부 맥락
    pitcher x 카운트우위   +25.6   그러나 이것은 **기존 편차 2번축 그 자체**다
                                  (축2 = 플래툰 -> 플래툰x투수유리, w=0.825)

따라서 신규는 **점수차 축**이다. 카운트우위는 기존 축의 k/w 재최적이고 그 축은
이미 닫혔다(편차가중 재최적 — 현행이 최적 근처).

## 구성 — 기존 4축과 **같은 기계**를 쓴다

    dev = nested_dev(parent=플래툰, child=플래툰 x 점수차구간, y_train, k)
    예측 += w * dev(그 행의 키)

학습 구간 라벨로만 만들고 조회 키는 그 행 자신의 컬럼이다 (규정 4 안전).
**모델 재학습이 필요 없다** — Champion 모델을 그대로 쓰고 후처리만 는다.

## 선택 편향 차단

`(w, k)` 를 2024 에서 고르면 그 값이 부풀려진다. 그래서

    (w, k) 는 2022 와 2023 의 평균 이득으로 고르고,  2024 는 **평가에만** 쓴다.

    .\.venv\Scripts\python.exe -u exp\axis5.py
"""
import io
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "exp"))
import build_asof as ba                                    # noqa: E402
from path_alloc import build_df                            # noqa: E402
from resid_table import post_for                           # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

KS = [300, 800, 2000, 5000, 15000]
WS = [0.10, 0.20, 0.30, 0.45, 0.60, 0.80]
FOLDS = (2022, 2023, 2024)


def r2(p, y):
    return 1e5 * np.corrcoef(p, y)[0, 1] ** 2


def main():
    tr = build_df()
    season = tr["season"].to_numpy()
    y = tr["control_success"].to_numpy(np.float64)
    P = tr["pitcher_id"].to_numpy(np.int64)
    BH = tr["batter_hand"].to_numpy(np.int64)
    BB = tr["balls_before"].to_numpy(np.int64)
    SS = tr["strikes_before"].to_numpy(np.int64)
    PH = P * 10 + BH                                   # 플래툰 (기존 축의 부모)
    SD = np.clip(tr["score_diff_pitcher_team"].to_numpy(np.int64), -3, 3) + 3
    INN = np.clip(tr["inning"].to_numpy(np.int64), 1, 9)
    OB = (tr["num_runners_on"].to_numpy(np.int64) > 0).astype(np.int64)

    CAND = {
        "A 플래툰x점수차": (PH, PH * 10 + SD),
        "B 투수x점수차": (P, P * 10 + SD),
        "C 플래툰x점수차x주자": (PH * 10 + SD, (PH * 10 + SD) * 10 + OB),
        "D 플래툰x이닝": (PH, PH * 10 + INN),
    }

    pv, base = {}, {}
    for f in FOLDS:
        m = season == f
        Q = np.load(os.path.join(ROOT, "exp", f"prod_champ_{f}.npy"))
        pv[f] = Q[:3].mean(0) + post_for(tr, y, season < f, m)
        base[f] = r2(pv[f], y[m])
    print("기준선 " + "  ".join(f"{f} {base[f]:.1f}" for f in FOLDS))
    print("(원장 2475.8 / 211.8 / 943.8 — 3시드 구성)\n")

    out = {}
    for name, (par, ch) in CAND.items():
        print("=" * 76)
        print(f"{name}")
        print("=" * 76)
        G = {f: {} for f in FOLDS}
        for f in FOLDS:
            m_tr, m_va = season < f, season == f
            u, d = ba.nested_dev(par[m_tr], ch[m_tr], y[m_tr], 1)   # k 는 아래서
            for k in KS:
                uu, dd = ba.nested_dev(par[m_tr], ch[m_tr], y[m_tr], k)
                add = ba.look(uu, dd, ch[m_va])
                for w in WS:
                    G[f][(k, w)] = r2(pv[f] + w * add, y[m_va]) - base[f]
        # (w,k) 는 2022+2023 으로만 고른다
        keys = list(G[2022])
        sel = max(keys, key=lambda kw: (G[2022][kw] + G[2023][kw]) / 2)
        print(f"  선택 (2022+2023 기준)  k={sel[0]}  w={sel[1]}")
        print(f"  {'폴드':<8}{'이득':>9}")
        for f in FOLDS:
            print(f"  {f:<8}{G[f][sel]:>+9.1f}" + ("   <- 평가 전용" if f == 2024 else ""))
        best24 = max(keys, key=lambda kw: G[2024][kw])
        print(f"  참고: 2024 에서 고르면 k={best24[0]} w={best24[1]} 로 "
              f"{G[2024][best24]:+.1f} (선택 편향 규모)")
        out[name] = dict(sel_k=sel[0], sel_w=sel[1],
                         gains={str(f): G[f][sel] for f in FOLDS},
                         oracle_pick_2024=G[2024][best24])
        print()

    json.dump(out, io.open(os.path.join(ROOT, "exp", "axis5.json"), "w",
                           encoding="utf-8"), indent=1, ensure_ascii=False)
    print("판정 — 2024(평가 전용)가 양수이고 2022/2023 도 양수여야 후보다.")


if __name__ == "__main__":
    main()
