r"""정보 해상도 감사 — 47열이 **어느 수준에서 실제로 변하는가.** 학습 0회.

"컬럼이 모델에 들어갔다"와 "그 컬럼의 해상도를 다 썼다"는 다른 말이다.
후자를 재려면 각 컬럼의 분산이 어느 계층에서 생기는지 봐야 한다.

## 계층 복원

    투수            pitcher_id
    투수 x 시즌      + season
    투수 x 경기      + 경기 경계 (prev1_game_* 가 경기 안에서 상수라는 성질)
    투수 x 타석      + 타석 경계 (볼=0 & 스트라이크=0 에서 새 타석)
    투구            개별 행

경기·타석 복원은 **train 안에서만 쓰는 진단**이다 (평가셋에서는 행 간 참조가
규정 4 로 금지되므로 피처가 될 수 없다).

## 분산분해

    x = 투수 + (시즌|투수) + (경기|시즌) + (타석|경기) + (투구|타석)

각 성분은 계층 평균의 차의 분산으로 잡는다. 합이 총분산이 되도록 행 가중.

읽는 법 — 어떤 컬럼의 분산이 **투수 수준에 몰려 있으면** 그 컬럼은 사실상
선수 상수이고(4-4 정적 프로필 족보), **투구 수준에 몰려 있으면** 행마다
움직이는 진짜 고해상도 정보다.

    .\.venv\Scripts\python.exe -u exp\info_coverage.py
"""
import io
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "exp"))
from path_alloc import build_df                            # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def gmean(keys, x):
    """그룹 평균을 각 행에 되돌린다."""
    u, inv = np.unique(keys, return_inverse=True)
    n = np.bincount(inv)
    s = np.bincount(inv, weights=x)
    return (s / np.maximum(n, 1))[inv]


def main():
    tr = build_df()
    season = tr["season"].to_numpy()
    PID = tr["pitcher_id"].to_numpy(np.int64)
    an = tr["asof_pitcher_n"].to_numpy(np.float64)
    p1 = tr["asof_pitcher_prev1_game_success_rate"].to_numpy(np.float64)
    p1m = tr["asof_pitcher_prev1_game_middle_rate"].to_numpy(np.float64)
    bb = tr["balls_before"].to_numpy(np.int64)
    ss = tr["strikes_before"].to_numpy(np.int64)

    print("계층 복원 중...", flush=True)
    o = np.lexsort((an, PID))
    gid = np.zeros(len(o), np.int64)
    aid = np.zeros(len(o), np.int64)
    G = A = 0
    pp, pv, pv2 = -1, np.nan, np.nan
    for i in o:
        if pp != PID[i] or not (np.isclose(p1[i], pv, equal_nan=True)
                                and np.isclose(p1m[i], pv2, equal_nan=True)):
            G += 1
            A += 1
        elif bb[i] == 0 and ss[i] == 0:
            A += 1
        gid[i] = G
        aid[i] = A
        pp, pv, pv2 = PID[i], p1[i], p1m[i]
    K_P = PID
    K_S = PID.astype(np.int64) * 10000 + season
    K_G = gid
    K_A = aid
    print(f"  투수 {len(np.unique(K_P)):,} / 투수x시즌 {len(np.unique(K_S)):,}"
          f" / 경기 {len(np.unique(K_G)):,} / 타석 {len(np.unique(K_A)):,}")
    print(f"  경기당 타석 중앙값 "
          f"{np.median(pd.Series(K_A).groupby(K_G).nunique()):.0f}"
          f"   타석당 투구 평균 {len(K_A)/len(np.unique(K_A)):.2f}", flush=True)

    cols = [c for c in pd.read_csv(os.path.join(ROOT, "data", "test.csv"),
                                   encoding="utf-8-sig", nrows=0).columns
            if c != "row_id"]
    print(f"\n{'컬럼':<38}{'투수':>7}{'시즌':>7}{'경기':>7}{'타석':>7}{'투구':>7}")
    out = {}
    for c in cols:
        v = tr[c].to_numpy()
        x = (pd.factorize(v)[0].astype(np.float64) if v.dtype == object
             else v.astype(np.float64))
        x = np.nan_to_num(x, nan=np.nanmedian(x[np.isfinite(x)]) if
                          np.isfinite(x).any() else 0.0)
        tot = x.var()
        if tot <= 0:
            out[c] = dict(pitcher=1.0, season=0, game=0, ab=0, pitch=0)
            print(f"  {c:<36}{'상수':>7}")
            continue
        m1 = gmean(K_P, x)
        m2 = gmean(K_S, x)
        m3 = gmean(K_G, x)
        m4 = gmean(K_A, x)
        comp = np.array([m1.var(), (m2 - m1).var(), (m3 - m2).var(),
                         (m4 - m3).var(), (x - m4).var()])
        comp = comp / comp.sum()
        out[c] = dict(zip(("pitcher", "season", "game", "ab", "pitch"),
                          [float(v) for v in comp]))
        print(f"  {c:<36}" + "".join(f"{v:>7.2f}" for v in comp))

    json.dump(out, io.open(os.path.join(ROOT, "exp", "info_coverage.json"), "w",
                           encoding="utf-8"), indent=1, ensure_ascii=False)
    print("\n읽는 법 — 투수 열에 몰리면 사실상 선수 상수(정적 프로필 족보), "
          "투구 열에 몰리면 행마다 움직이는 고해상도 정보다.")


if __name__ == "__main__":
    main()
