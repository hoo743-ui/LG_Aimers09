r"""제출 전략 — 시드 앙상블 곡선과 아핀 손실을 생산 경로에서 잰다.

## 왜 아핀이 점수를 바꾸는가

`rho` 는 아핀 불변이지만 **점수는 최적 아핀에서 `1e5*rho^2` 를 천장으로 갖는
R^2 계열**이다 (원장 §8, 13회차가 천장에 소수 6자리까지 닿았다). 그래서
아핀은 `rho` 를 못 올리지만 **천장에서 얼마나 떨어졌는가**는 실제 점수다.

손실은 평균 오프셋의 **제곱**으로 커진다.

    손실 = 1e5 * (mean(q) - mean(y))^2 / Var(y)      (기울기가 맞을 때)

오프셋 0.003 이면 4점, 0.016 이면 102점이다. 21회차가 center 를 학습 평균으로
두고 31점을 흘린 것이 이 항이다.

## 무엇을 재는가

폴드 2024 생산 경로에서 시드를 늘려가며 (Champion 은 3시드를 싣는다):

    k=1..7 의 rho^2      -> 시드 앙상블이 아직 이득을 주는가
    아핀 손실 분해        -> 평균/기울기 중 어디서 새는가
    추론 시간            -> 10분 제한 (245,789행)

**평가셋 정답이나 LB 점수로 파라미터를 고르지 않는다.** 전부 학습 구간 안에서만
정한다.
"""
import io
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "exp"))
import build_asof as ba                                    # noqa: E402
from path_alloc import build_df                            # noqa: E402

ft, sc = ba.ft, ba.sc
SEEDS = [42, 43, 44, 45, 46, 47, 48]


def main():
    tr = build_df()
    tc = pd.read_csv(os.path.join(ft.DATA_DIR, "test.csv"),
                     encoding="utf-8-sig", nrows=0).columns
    allf = [c for c in tc if c != ft.ID]
    ctxf = [c for c in ft.COUNT_FEATS + ft.HAND_FEATS
            if c not in ("tmc_n", "tmh_n")]
    CHAMP = list(allf) + ctxf + sc.ASOF_COLS + sc.CTX_COLS + sc.LVL_COLS
    y = tr[ft.TARGET].to_numpy(np.float64)
    season = tr["season"].to_numpy()

    P = tr["pitcher_id"].to_numpy(np.int64)
    BH = tr["batter_hand"].to_numpy(np.int64)
    BB = tr["balls_before"].to_numpy(np.int64)
    SS = tr["strikes_before"].to_numpy(np.int64)
    OB = (tr["num_runners_on"].to_numpy(np.int64) > 0).astype(np.int64)
    PH = P * 10 + BH
    PHA = PH * 10 + (SS > BB).astype(np.int64)
    AX = [(P, PH), (PH, PHA), (PHA, PHA * 100 + (BB * 4 + SS)), (PH, PH * 10 + OB)]

    for f in (2024,):
        m_tr, m_va = season < f, season == f
        yv = y[m_va]
        post = np.column_stack([
            ba.look(*ba.nested_dev(p[m_tr], c[m_tr], y[m_tr], k), c[m_va])
            for (p, c), k in zip(AX, ba.KSH)]) @ ba.WPOST
        print(f"\n=== 폴드 {f}  학습 {int(m_tr.sum()):,}행 ===", flush=True)
        preds = []
        for sd in SEEDS:
            t = time.time()
            mm = ba.pipeline(CHAMP, sd)
            mm.fit(tr.loc[m_tr, CHAMP], y[m_tr].astype(int))
            t1 = time.time()
            pv = mm.predict_proba(tr.loc[m_va, CHAMP])[:, 1]
            inf = time.time() - t1
            preds.append(pv)
            print(f"  seed {sd}  학습 {t1-t:.0f}s  추론 {inf:.1f}s"
                  f"  ({int(m_va.sum()):,}행)  rho2 "
                  f"{1e5*np.corrcoef(pv+post,yv)[0,1]**2:.1f}", flush=True)
            del mm
        np.save(os.path.join(ROOT, "exp", f"prod_champ_{f}.npy"),
                np.array(preds, dtype=np.float32))
        np.save(os.path.join(ROOT, "exp", f"prod_post_{f}.npy"), post)
        print(f"\n  시드 앙상블 곡선 (후처리 포함, 아핀 이전)")
        print(f"  {'k':>3}{'rho2':>10}{'vs k=3':>10}")
        base = None
        for k in range(1, len(SEEDS) + 1):
            q = np.mean(preds[:k], 0) + post
            r = 1e5 * np.corrcoef(q, yv)[0, 1] ** 2
            if k == 3:
                base = r
            print(f"  {k:>3}{r:>10.1f}" + (f"{r-base:>+10.1f}" if base else ""))
    print("\n저장: exp/prod_champ_2024.npy (7시드), exp/prod_post_2024.npy")


if __name__ == "__main__":
    main()
