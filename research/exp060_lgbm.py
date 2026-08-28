# -*- coding: utf-8 -*-
r"""EXP060 — 모델 계열 재검: LightGBM (2026-08-29, 고위험 트랙).

원장의 'LightGBM 3폴드 음수 -> 닫힘' 은 로컬 증거뿐이다 (EXP057: 로컬 부호율 50%).
같은 82피처·같은 후처리 위에서 LGBM 단독과 CatBoost 혼합을 폴드 2024 로 재본다.
로컬은 참사 확인용. 살아남으면 로터리 슬롯 후보.

    .\.venv\Scripts\python.exe -u research\exp060_lgbm.py
"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "exp"))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import build_asof as ba
from path_alloc import build_df

import zipfile, io, joblib
with zipfile.ZipFile(os.path.join(ROOT, "submissions", "cand_mir.zip")) as z:
    B = joblib.load(io.BytesIO(z.read("model/rf.pkl")))
FEAT = list(B["features"])

tr = build_df()
season = tr["season"].to_numpy(); y = tr["control_success"].to_numpy(np.float64)
P = tr["pitcher_id"].to_numpy(np.int64); BH = tr["batter_hand"].to_numpy(np.int64)
BB = tr["balls_before"].to_numpy(np.int64); SS = tr["strikes_before"].to_numpy(np.int64)
OB = (tr["num_runners_on"].to_numpy(np.int64) > 0).astype(np.int64)
PH = P*10 + BH; PHA = PH*10 + (SS > BB).astype(np.int64)
AX = [(P, PH), (PH, PHA), (PHA, PHA*100 + (BB*4+SS)), (PH, PH*10 + OB)]
mt, mv = season < 2024, season == 2024
post = np.column_stack([ba.look(*ba.nested_dev(p[mt], c[mt], y[mt], k), c[mv])
                        for (p, c), k in zip(AX, ba.KSH)]) @ ba.WPOST
Xtr, Xva, yv = tr.loc[mt, FEAT], tr.loc[mv, FEAT], y[mv]
del tr

def score(pred):
    return 1e5 * np.corrcoef(pred + post, yv)[0, 1]**2

res = {}
t0 = time.time()
m = ba.pipeline(FEAT, 42); m.fit(Xtr, y[mt].astype(int))
pc = m.predict_proba(Xva)[:, 1]
res["cat_base"] = score(pc)
print(f"  cat_base  {res['cat_base']:8.2f}   {time.time()-t0:.0f}s", flush=True)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
import lightgbm as lgb
ftmod = ba.ft
cat = [c for c in ftmod.CAT_COLS if c in FEAT]
num = [c for c in FEAT if c not in cat]
for nm, params in {
    "lgbm": dict(n_estimators=1200, learning_rate=0.02, num_leaves=63,
                 min_child_samples=200, subsample=0.9, subsample_freq=1,
                 colsample_bytree=0.9, reg_lambda=100.0, n_jobs=14, verbose=-1),
    "lgbm_d6": dict(n_estimators=1200, learning_rate=0.02, max_depth=6, num_leaves=63,
                    min_child_samples=200, reg_lambda=100.0, n_jobs=14, verbose=-1),
}.items():
    t0 = time.time()
    pre = ColumnTransformer([("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                                                    unknown_value=-1), cat),
                             ("num", "passthrough", num)])
    mdl = Pipeline([("pre", pre), ("m", lgb.LGBMClassifier(**params))])
    mdl.fit(Xtr, y[mt].astype(int))
    pl = mdl.predict_proba(Xva)[:, 1]
    res[nm] = score(pl)
    res[f"{nm}+cat50"] = score(0.5*pl + 0.5*pc)
    print(f"  {nm:9s} {res[nm]:8.2f}  ({res[nm]-res['cat_base']:+.2f})   "
          f"blend50 {res[f'{nm}+cat50']:8.2f}  ({res[f'{nm}+cat50']-res['cat_base']:+.2f})   "
          f"corr(lgbm,cat) {np.corrcoef(pl, pc)[0,1]:+.4f}   {time.time()-t0:.0f}s", flush=True)
json.dump(res, open(os.path.join(ROOT, "exp", "exp060_lgbm.json"), "w"), indent=1)
