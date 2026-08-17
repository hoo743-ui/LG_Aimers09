r"""HAND-DIFF 생산 경로 검증 — 학습량이 늘면 효과가 흡수되는가.

표 제작 잔차는 **strictly out-of-fold** 다 (그 시즌을 학습하지 않은 모델의 잔차).
생산 모델의 in-sample 잔차는 쓰지 않는다.

흡수 검정 — 같은 표(2022 잔차)를 학습량이 다른 두 목표 모델에 적용한다.

    목표 A  2019~2021 학습 (728k)  -> 2024 예측
    목표 B  2019~2022 학습 (976k)  -> 2024 예측
    목표 C  2019~2023 학습 (1.22M) -> 2024 예측     <- 기존
    생산    2019~2024 학습 (1.47M) -> 2025          <- 여기서 얼마나 더 줄까
"""
import os, sys, time
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "exp"))
import build_asof as ba
from path_alloc import build_df
ft, sc = ba.ft, ba.sc
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
tr = build_df(); season = tr["season"].to_numpy()
y = tr["control_success"].to_numpy(np.float64)
tc = pd.read_csv(os.path.join(ft.DATA_DIR, "test.csv"), encoding="utf-8-sig", nrows=0).columns
allf = [c for c in tc if c != ft.ID]
ctxf = [c for c in ft.COUNT_FEATS + ft.HAND_FEATS if c not in ("tmc_n", "tmh_n")]
CHAMP = list(allf) + ctxf + sc.ASOF_COLS + sc.CTX_COLS + sc.LVL_COLS
m24 = season == 2024
for upto in (2021, 2022):
    out = os.path.join(ROOT, "exp", f"pred24_from{upto}.npy")
    if os.path.exists(out):
        print(f"  <= {upto} 모델 이미 있음"); continue
    m_tr = season <= upto
    acc = []
    for sd in (42, 43):
        t = time.time()
        mm = ba.pipeline(CHAMP, sd)
        mm.fit(tr.loc[m_tr, CHAMP], y[m_tr].astype(int))
        acc.append(mm.predict_proba(tr.loc[m24, CHAMP])[:, 1])
        print(f"  <= {upto} 학습({int(m_tr.sum()):,}행) seed {sd} {time.time()-t:.0f}s", flush=True)
        del mm
    np.save(out, np.asarray(acc)); print(f"  저장 {out}")
