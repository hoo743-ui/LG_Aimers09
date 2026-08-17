import sys, os, importlib.util, hashlib, zipfile
import numpy as np, pandas as pd, joblib
sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "exp"))
spec = importlib.util.spec_from_file_location("sc", os.path.join(ROOT, "script.py"))
sc = importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)
from path_alloc import build_df
from resid_table import post_for
from traj_probe import r2
tr = build_df(); season = tr["season"].to_numpy(); y = tr["control_success"].to_numpy(np.float64)
PID = tr["pitcher_id"].to_numpy(np.int64); PH = tr["pitcher_hand"].to_numpy(np.int64)
BH = tr["batter_hand"].to_numpy(np.int64); SS = tr["strikes_before"].to_numpy(np.int64)
NR = tr["num_runners_on"].to_numpy(np.int64)
m24 = season == 2024
b1 = joblib.load(os.path.join(ROOT, "model_cand", "cat_asof_xl.pkl"))
b3 = joblib.load(os.path.join(ROOT, "model_cand", "cat_submit_3.pkl"))
bf = joblib.load(os.path.join(ROOT, "model_cand", "cat_final.pkl"))
X = tr.loc[m24]
a1, a3, af = (sc.platoon_adjust(x, X) for x in (b1, b3, bf))
SRC = (2020,2021,2022,2023,2024)
res = {f: y[season==f] - (np.load(os.path.join(ROOT,"exp",f"prod_champ_{f}.npy"))[:2].mean(0)
                          + post_for(tr,y,season<f,season==f)) for f in SRC}
msrc = np.isin(season, SRC); rs = np.concatenate([res[f] for f in SRC])
def diff(ctx,k):
    gg=pd.DataFrame({"p":PID[msrc],"c":ctx[msrc],"r":rs}).groupby(["p","c"])["r"].agg(["mean","size"]).unstack()
    n0,n1=gg[("size",0)].fillna(0),gg[("size",1)].fillna(0)
    d=gg[("mean",1)]-gg[("mean",0)]; ne=(n0*n1)/(n0+n1).replace(0,np.nan)
    return (d*ne/(ne+k)).dropna()
SAME=(PH==BH).astype(int); TWO=(SS==2).astype(int); RUN=(NR>0).astype(int)
exp_add = sum(pd.Series(PID[m24]).map(diff(c,k)).fillna(0).to_numpy()*np.where(c[m24]==1,.5,-.5)
              for c,k in ((SAME,1000),(TWO,1000),(RUN,2000)))
print(f"추론−분석 최대차   {np.max(np.abs((af-a1)-exp_add)):.3e}")
print(f"FINAL−C3 보정 sd   {np.std(af-a3):.6f}")
Xs = X.iloc[:200]
print(f"행 독립성 (200행)   최대차 "
      f"{np.max(np.abs(sc.platoon_adjust(bf,Xs)-np.array([sc.platoon_adjust(bf,Xs.iloc[[i]])[0] for i in range(200)]))):.3e}")
p = os.path.join(ROOT,"submissions","cand_final.zip")
with zipfile.ZipFile(p) as z:
    mh = hashlib.sha256(z.read("model/rf.pkl")).hexdigest()
print(f"\nzip   {hashlib.sha256(open(p,'rb').read()).hexdigest()}")
print(f"model {mh[:32]}...  피처 {len(bf['features'])}  후처리 {len(bf['platoon'])}축")
print(f"asof_prior 항목 {sum(len(v) for v in bf['asof_prior'].values()):,}  "
      f"표 항목 {sum(len(s['table']) for s in bf['platoon'][4:]):,}")
