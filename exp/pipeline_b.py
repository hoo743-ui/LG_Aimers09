r"""PIPELINE A vs B — state-first 가 정보를 더 보존하는가. 새 피처 0개.

## 구조 감사에서 나온 실험 설계

D 는 **학습이 없는 결정론적 추정기**다.

    cur_rate = (asof_n * asof_rate - prior_events[선수]) / cur_n      (대수 항등식)

그래서 "추정기를 분리한다"는 것 자체로는 정보가 늘지 않는다. 실제로 달라질 수
있는 것은 **Stage 2 의 입력 구성**뿐이고, 거기에 미검증 지점이 하나 있다.

원시 `asof_rate` 는 정의상 **혼합**이다.

    asof_rate = w * prior_rate + (1-w) * cur_rate ,    w = prior_n / asof_n

현재 파이프라인 A 는 트리에게 **혼합값과 분해값을 동시에** 준다. 트리는
`cur_n`(혼합비의 재료)으로 분기해가며 그 혼합을 스스로 풀어야 한다. 그 일에
용량을 쓰고 있다면, **혼합값을 빼는 것**이 state-first 의 실체다.

    A   Champion 82p                     (혼합 + 분해 동시 제공)
    B1  −원시 asof 비율 16열 = 66p        (분해만. 카운터 3열은 유지)
    B2  −원시 asof 19열 전부 = 63p        (분해만. 카운터도 제거)

`asof` 감사에서 경력 성분(prior)의 2024 잔차상관이 |0.0044| (2.2SE, 잡음 대역)
이었으므로 **정보 손실은 0 에 가깝고**, 이득이 있다면 그것은 순수하게
**표현 분리**의 몫이다. 이것이 사용자 질문("하나의 CatBoost 가 상태추정과
확률생성을 동시에 하는 것이 비효율적인가")의 직접 검증이다.

## 누수

세 구성 모두 학습 라벨만 쓴다. `cur_*` 는 그 행 자신의 컬럼 + 학습 구간 상수로
만들어지고 Stage 1 에 라벨이 들어가지 않는다. 검증 행의 라벨은 어디에도 안 쓴다.

## 판정 (사용자 게이트)

    B >= +3.8%   -> 3폴드 x 4~6시드로 확장
    B < +1%      -> state-first family 종료

    .\.venv\Scripts\python.exe -u exp\pipeline_b.py
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
from resid_table import post_for                           # noqa: E402

ft, sc = ba.ft, ba.sc
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

FOLDS = (2022, 2023, 2024)
SEEDS = (42, 43)
ASOF_CNT = ["asof_pitcher_n", "asof_pitcher_pitchmix_n", "asof_batter_n"]


def main():
    tr = build_df()
    season = tr["season"].to_numpy()
    y = tr["control_success"].to_numpy(np.float64)
    tc = pd.read_csv(os.path.join(ft.DATA_DIR, "test.csv"),
                     encoding="utf-8-sig", nrows=0).columns
    allf = [c for c in tc if c != ft.ID]
    ctxf = [c for c in ft.COUNT_FEATS + ft.HAND_FEATS
            if c not in ("tmc_n", "tmh_n")]
    CHAMP = list(allf) + ctxf + sc.ASOF_COLS + sc.CTX_COLS + sc.LVL_COLS
    assert len(CHAMP) == 82
    ASOF_ALL = [c for c in allf if c.startswith("asof_")]
    ASOF_RATE = [c for c in ASOF_ALL if c not in ASOF_CNT]
    print(f"  원시 asof {len(ASOF_ALL)}열 = 카운터 {len(ASOF_CNT)} + 비율 {len(ASOF_RATE)}")
    B1 = [c for c in CHAMP if c not in ASOF_RATE]
    B2 = [c for c in CHAMP if c not in ASOF_ALL]
    cfgs = [("B1_no_rate", B1), ("B2_no_asof", B2)]
    for n, f in cfgs:
        print(f"  {n:<12}{len(f):>4}p")

    R = {n: {} for n, _ in cfgs}
    B = {}
    for f in FOLDS:
        m_tr, m_va = season < f, season == f
        yv = y[m_va]
        post = post_for(tr, y, m_tr, m_va)
        P = np.load(os.path.join(ROOT, "exp", f"prod_champ_{f}.npy"))
        base = 1e5 * np.corrcoef(P[:2].mean(0) + post, yv)[0, 1] ** 2
        per_b = [1e5 * np.corrcoef(P[i] + post, yv)[0, 1] ** 2 for i in range(2)]
        B[f] = dict(rho2=base, per=per_b)
        print(f"\n=== 폴드 {f}  학습 {int(m_tr.sum()):,}행   A(Champion) "
              f"{base:.1f} (시드별 {per_b[0]:.1f} {per_b[1]:.1f}) ===", flush=True)
        for n, fs in cfgs:
            t = time.time()
            acc = np.zeros(int(m_va.sum()))
            per = []
            for sd in SEEDS:
                mm = ba.pipeline(fs, sd)
                mm.fit(tr.loc[m_tr, fs], y[m_tr].astype(int))
                pv = mm.predict_proba(tr.loc[m_va, fs])[:, 1]
                per.append(1e5 * np.corrcoef(pv + post, yv)[0, 1] ** 2)
                acc += pv
                del mm
            r = 1e5 * np.corrcoef(acc / len(SEEDS) + post, yv)[0, 1] ** 2
            R[n][f] = dict(rho2=r, per=per)
            print(f"  {n:<12}{len(fs):>4}p  {r:>9.1f}  ({r-base:+.1f}, "
                  f"{(r-base)/base*100:+.2f}%)   시드별 "
                  + " ".join(f"{v:.1f}" for v in per)
                  + f"   {time.time()-t:.0f}s", flush=True)

    print(f"\n=== A(Champion) 대비 증분 — 생산 경로 3폴드 x 2시드 ===")
    print(f"{'파이프라인':<14}" + "".join(f"{f:>12}" for f in FOLDS)
          + f"{'2024 %':>10}{'최악':>9}{'시드폭2024':>12}")
    out = {}
    for n, _ in cfgs:
        d = [R[n][f]["rho2"] - B[f]["rho2"] for f in FOLDS]
        pct = d[2] / B[2024]["rho2"] * 100
        sv = max(R[n][2024]["per"]) - min(R[n][2024]["per"])
        out[n] = dict(delta={str(f): d[i] for i, f in enumerate(FOLDS)},
                      pct_2024=pct, seed_span_2024=sv)
        print(f"{n:<14}" + "".join(f"{x:>+12.1f}" for x in d)
              + f"{pct:>+10.2f}{min(d):>+9.1f}{sv:>12.1f}")
    print(f"\n  A 자체 2024 시드폭 {max(B[2024]['per'])-min(B[2024]['per']):.1f}")
    print(f"  게이트 — B >= +3.8% 면 확장, B < +1% 면 state-first 종료")
    json.dump({"cand": {n: {str(f): R[n][f] for f in FOLDS} for n, _ in cfgs},
               "base": {str(f): B[f] for f in FOLDS}, "summary": out},
              io.open(os.path.join(ROOT, "exp", "pipeline_b.json"), "w",
                      encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
