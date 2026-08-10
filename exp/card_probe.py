r"""범주형 인코딩을 정하기 전에 카디널리티와 범주당 표본 수를 먼저 센다.

규칙 5: "categorical encoding 은 cardinality 와 category 당 표본 수를 먼저
분석한 뒤 numeric/one-hot/CTR 을 비교한다."

지금 파이프라인은 범주형 3개만 OrdinalEncoder 로 정수화하고 나머지 ID 계열은
**숫자 그대로** 넣는다. 트리에게 ordinal 정수는 "순서가 있는 값"이라
`pitcher_id < 500` 같은 무의미한 분할을 만들 수 있다. 다만 4-5 는 ID 를 남기는
쪽이 +8.5 로 이득임을 확인했다 — 순서가 무의미해도 트리가 구간을 잘게 쪼개
개별 선수를 식별해내기 때문이다.

여기서 재는 것:
  - 카디널리티 (범주 수)
  - 범주당 중앙 표본 수 — CTR 을 쓸 수 있는지의 하한
  - 폴드별 **미지 범주 비율** — 학습에 없던 값이 검증에 얼마나 나오는가
  - 순서성 검정: 범주 인덱스와 타깃 성공률의 순위상관. 0 에 가까우면
    ordinal 정수는 트리에게 의미 없는 축이다

    .\.venv\Scripts\python.exe exp\card_probe.py
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "exp", "cache")

CANDS = ["top_bottom", "game_type", "base_state", "pitcher_hand", "batter_hand",
         "pitcher_team_id", "batter_team_id", "pitcher_id", "batter_id",
         "inning", "game_month", "game_dayofweek"]


def main():
    X = np.load(f"{CACHE}/X.npy", mmap_mode="r")
    y = np.load(f"{CACHE}/y.npy").astype(np.float64)
    season = np.load(f"{CACHE}/season.npy")
    meta = json.load(open(f"{CACHE}/cols.json"))
    ix = {c: i for i, c in enumerate(meta["cols"])}

    print(f"{'컬럼':18s} {'카디널':>7s} {'범주당중앙':>10s} {'최소':>6s} "
          f"{'순위상관':>8s} {'미지비율 2021/2022/2024':>24s}")
    print("-" * 88)
    for c in CANDS:
        if c not in ix:
            continue
        v = np.asarray(X[:, ix[c]])
        ok = ~np.isnan(v)
        vv = v[ok]
        vals, cnt = np.unique(vv, return_counts=True)
        # 범주별 성공률 -> 인덱스와의 순위상관
        rate = np.array([y[ok][vv == u].mean() for u in vals]) if len(vals) <= 2000 \
            else None
        rho = (spearmanr(vals, rate).statistic if rate is not None
               and len(vals) > 2 else np.nan)
        unk = []
        for Y in (2021, 2022, 2024):
            tr = set(np.unique(v[(season < Y) & ok]).tolist())
            va = v[(season == Y) & ok]
            unk.append(np.mean([x not in tr for x in va]) if len(va) else np.nan)
        print(f"{c:18s} {len(vals):>7,} {int(np.median(cnt)):>10,} "
              f"{int(cnt.min()):>6,} {rho:>8.3f}   "
              + "  ".join(f"{u:6.1%}" for u in unk))

    print("\n판단 기준")
    print("  - 순위상관 |rho| 이 0 근처면 ordinal 정수 축은 무의미하다.")
    print("    그래도 4-5 가 ID 유지 +8.5 를 쟀다 — 트리가 구간을 잘게 쪼개")
    print("    개체를 식별하기 때문이지 순서를 쓰는 게 아니다.")
    print("  - 미지 비율이 높으면 CTR/one-hot 은 그만큼 결측이 된다.")
    print("  - 범주당 표본이 수백 미만이면 CTR 은 잡음이다 (sit_probe 가 보인 것).")


if __name__ == "__main__":
    main()
