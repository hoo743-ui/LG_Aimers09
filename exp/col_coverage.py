r"""47개 공식 컬럼 활용도 감사 — "넣었는가"가 아니라 "관계를 탐색했는가".

앞선 감사들이 `asof_*` 19열은 전수로 덮었지만(잔차·오라클·지속성·대수종속),
**나머지 28열은 잔차/오라클 칸이 비어 있었다.** 이 파일이 그 칸을 채운다.

    (1) 47열 전부의 잔차상관과 십분위/범주 오라클 (폴드 2024, 학습 0회)
    (2) 무작위 조합이 아니라 **야구·생성 의미가 있는 쌍**만 골라 오라클

쌍은 브루트포스로 만들지 않는다. 이미 닫힌 것(X/H1 의 cur x 맥락, K2 2스트라이크,
REGIME B/C 의 아웃·주자·이닝, game_type 계열, 투수x타자손, 팀·구장)은 빼고,
**메커니즘 설명이 붙는 것만** 남긴다. 각 쌍마다 왜 의미가 있는지 주석을 단다.

    .\.venv\Scripts\python.exe -u exp\col_coverage.py
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
from d_limits import cv2_gain                              # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def main():
    tr = build_df()
    season = tr["season"].to_numpy()
    y = tr["control_success"].to_numpy(np.float64)
    m24 = season == 2024
    P = np.load(os.path.join(ROOT, "exp", "prod_champ_2024.npy"))
    pred = P[:3].mean(0) + np.load(os.path.join(ROOT, "exp",
                                                "prod_post_2024.npy"))
    y24 = y[m24]
    base = 1e5 * np.corrcoef(pred, y24)[0, 1] ** 2
    resid = y24 - pred
    half = np.random.default_rng(0).random(int(m24.sum())) < 0.5
    se = 1 / np.sqrt(int(m24.sum()))
    cols = [c for c in pd.read_csv(os.path.join(ROOT, "data", "test.csv"),
                                   encoding="utf-8-sig", nrows=0).columns
            if c != "row_id"]
    print(f"공식 컬럼 {len(cols)}개   기준선 {base:.1f}   상관 SE {se:.4f}")
    out = {}

    def keyify(c):
        """범주형은 코드로, 연속형은 십분위로. 저카디널리티는 원값 그대로."""
        v = tr[c].to_numpy()
        if v.dtype == object:
            return pd.factorize(v)[0][m24].astype(np.int64), "범주"
        v = v.astype(np.float64)
        u = np.unique(v[np.isfinite(v)])
        if len(u) <= 15:
            return np.nan_to_num(v[m24], nan=-1).astype(np.int64), f"원값{len(u)}"
        q = np.nanquantile(v[m24], np.linspace(0, 1, 11)[1:-1])
        return np.searchsorted(q, np.nan_to_num(v[m24], nan=np.nanmedian(v[m24]))), "십분위"

    print("\n" + "=" * 92)
    print("1. 47열 전부 — 잔차상관과 그룹 오라클 (폴드 2024)")
    print("=" * 92)
    print(f"{'컬럼':<38}{'결측률':>8}{'잔차상관':>10}{'SE배수':>8}"
          f"{'셀':>6}{'오라클':>9}")
    for c in cols:
        k, kind = keyify(c)
        v = tr[c].to_numpy()
        miss = float(pd.isna(pd.Series(v[m24])).mean())
        if v.dtype == object:
            rc = 0.0
        else:
            x = v.astype(np.float64)[m24]
            o = np.isfinite(x)
            rc = (float(np.corrcoef(x[o], resid[o])[0, 1])
                  if o.sum() > 100 and x[o].std() > 0 else 0.0)
        g, nc, bk = cv2_gain(k, pred, y24, half)
        out[c] = dict(miss=miss, resid=rc, se_mult=rc / se, cells=nc,
                      oracle=float(g), kind=kind)
        print(f"  {c:<36}{miss:>8.1%}{rc:>+10.4f}{rc/se:>+8.1f}"
              f"{nc:>6}{g:>+9.1f}")

    print("\n" + "=" * 92)
    print("2. 의미가 붙는 미검증 쌍 — 브루트포스 아님 (각 쌍에 메커니즘 주석)")
    print("=" * 92)
    g = lambda c: tr[c].to_numpy(np.float64)[m24]   # 쌍은 폴드 2024 만 쓴다
    BB, SS = g("balls_before").astype(np.int64), g("strikes_before").astype(np.int64)
    CNT = BB * 4 + SS
    OUT = g("outs_before").astype(np.int64)
    R1 = g("runner_on_1b").astype(np.int64)
    R2 = g("runner_on_2b").astype(np.int64)
    R3 = g("runner_on_3b").astype(np.int64)
    PH_ = g("pitcher_hand").astype(np.int64)
    BH_ = g("batter_hand").astype(np.int64)
    INN = np.clip(g("inning").astype(np.int64), 1, 10)
    SD = np.clip(g("score_diff_pitcher_team").astype(np.int64), -6, 6)
    LI = np.digitize(g("li"), np.nanquantile(g("li"), [.25, .5, .75, .9]))
    MON = g("game_month").astype(np.int64)
    DOW = g("game_dayofweek").astype(np.int64)
    BN = np.digitize(g("asof_batter_n"),
                     np.nanquantile(g("asof_batter_n"), [.25, .5, .75]))
    SH = (PH_ == BH_).astype(np.int64)

    PAIRS = [
        ("2루주자 x 카운트", R2 * 100 + CNT,
         "2루 주자면 사인 노출 우려로 배합·세트가 바뀐다. base_state 단독은 잰 적 있으나 카운트와의 결합은 없다"),
        ("2루주자 x 같은손", R2 * 10 + SH,
         "사인 교체가 좌우 매치업과 겹치는가"),
        ("주자1루 x 투수손", R1 * 10 + PH_,
         "좌투수는 1루 견제 자세가 달라 세트포지션 부담이 다르다"),
        ("3루주자 x 아웃", R3 * 10 + OUT,
         "3루 주자 + 2아웃 미만은 폭투 위험 국면이라 낮은 코스를 피한다"),
        ("점수차 x 이닝", (SD + 6) * 100 + INN,
         "큰 점수차 후반은 승부를 걸고, 접전 후반은 유인구가 는다"),
        ("점수차 x 카운트", (SD + 6) * 100 + CNT,
         "블로아웃에서는 카운트 운영 자체를 포기한다"),
        ("li 4구간 x 카운트", LI * 100 + CNT,
         "중요도가 카운트 운영을 바꾸는가 (li 는 상태의 결정함수라 새 정보는 아니나 결합은 미검증)"),
        ("타자경험 x 카운트", BN * 100 + CNT,
         "베테랑 타자 상대로 카운트 운영이 달라지는가"),
        ("타자경험 x 같은손", BN * 10 + SH,
         "경험 x 플래툰"),
        ("월 x 카운트", MON * 100 + CNT,
         "시즌 초 제구 난조가 특정 카운트에 몰리는가"),
        ("요일 x 이닝", DOW * 100 + INN,
         "요일은 더블헤더·주간경기 대리. 이닝과 겹치면 피로 국면"),
        ("이닝 x 카운트", INN * 100 + CNT,
         "REGIME C 는 이닝 x cur_state 였고 이닝 x 카운트는 미검증"),
        ("아웃 x 카운트", OUT * 100 + CNT,
         "REGIME B 는 아웃 x cur_state 였고 아웃 x 카운트는 미검증"),
        ("투수손 x 타자손 x 카운트", (PH_ * 10 + BH_) * 100 + CNT,
         "플래툰은 후처리에 있으나 카운트까지 3단은 in-model 로 검증된 적 없다"),
    ]
    print(f"{'쌍':<28}{'셀':>7}{'오라클':>9}{'k':>8}   메커니즘")
    for name, key, why in PAIRS:
        v, nc, bk = cv2_gain(key, pred, y24, half)
        out[f"pair|{name}"] = dict(cells=nc, oracle=float(v), k=bk, why=why)
        print(f"  {name:<26}{nc:>7}{v:>+9.1f}{bk:>8}   {why[:44]}")

    print("\n  참고 — 같은 자 위의 기준점: pitcher x 타자손 +76.8 / "
          "pitcher +17.9 / 위약(pitcher x 무작위2) +9.4 / 손익분기 +19.8")
    json.dump(out, io.open(os.path.join(ROOT, "exp", "col_coverage.json"), "w",
                           encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
