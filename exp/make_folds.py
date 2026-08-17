r"""폴드 2020/2021 생산 경로 예측 생성 — 전이 연구용. 시드 42/43."""
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
for f in (2020, 2021):
    out = os.path.join(ROOT, "exp", f"prod_champ_{f}.npy")
    if os.path.exists(out):
        print(f"  {f} 이미 있음"); continue
    m_tr, m_va = season < f, season == f
    acc = []
    for sd in (42, 43):
        t = time.time()
        mm = ba.pipeline(CHAMP, sd)
        mm.fit(tr.loc[m_tr, CHAMP], y[m_tr].astype(int))
        acc.append(mm.predict_proba(tr.loc[m_va, CHAMP])[:, 1])
        print(f"  fold{f} seed {sd} {time.time()-t:.0f}s ({int(m_tr.sum()):,}행)", flush=True)
        del mm
    np.save(out, np.asarray(acc))
    print(f"  저장 {out}")
