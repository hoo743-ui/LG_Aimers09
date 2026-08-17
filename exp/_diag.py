import sys, os, importlib.util
import numpy as np, pandas as pd, joblib
sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "exp"))
spec = importlib.util.spec_from_file_location("sc", os.path.join(ROOT, "script.py"))
sc = importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)
from path_alloc import build_df
from resid_table import post_for
tr = build_df(); season = tr["season"].to_numpy(); y = tr["control_success"].to_numpy(np.float64)
PID = tr["pitcher_id"].to_numpy(np.int64)
PH = tr["pitcher_hand"].to_numpy(np.int64); BH = tr["batter_hand"].to_numpy(np.int64)
m24 = season == 2024
b1 = joblib.load(os.path.join(ROOT,"model_cand","cat_asof_xl.pkl"))
b2 = joblib.load(os.path.join(ROOT,"model_cand","cat_submit_2.pkl"))
a = sc.platoon_adjust(b2, tr.loc[m24]) - sc.platoon_adjust(b1, tr.loc[m24])
res={}
for f in (2023,2024):
    m=season==f
    res[f]=y[m]-(np.load(os.path.join(ROOT,"exp",f"prod_champ_{f}.npy"))[:2].mean(0)+post_for(tr,y,season<f,m))
msrc=np.isin(season,(2023,2024)); rs=np.concatenate([res[2023],res[2024]])
SAME=(PH==BH).astype(int)
gg=pd.DataFrame({"p":PID[msrc],"c":SAME[msrc],"r":rs}).groupby(["p","c"])["r"].agg(["mean","size"]).unstack()
n0,n1=gg[("size",0)].fillna(0),gg[("size",1)].fillna(0)
d=(gg[("mean",1)]-gg[("mean",0)]); ne=(n0*n1)/(n0+n1).replace(0,np.nan)
ds=(d*ne/(ne+1000)).dropna()
hd=pd.Series(PID[m24]).map(ds).fillna(0).to_numpy()*np.where(SAME[m24]==1,.5,-.5)
bad=np.abs(a-hd)>1e-9
print(f"불일치 행 {bad.sum():,} / {len(a):,} ({bad.mean():.2%})")
if bad.sum():
    p=PID[m24][bad]; ph=PH[m24][bad]; bh=BH[m24][bad]
    u=pd.Series(p).value_counts()
    print("불일치 투수 수", len(u), "상위:", u.head(5).to_dict())
    pid0=int(u.index[0])
    sel=(PID[m24]==pid0)
    print(f"\n투수 {pid0}: 손 값들 {sorted(set(PH[PID==pid0].tolist()))}"
          f"  타자손 분포 {pd.Series(BH[m24][sel]).value_counts().to_dict()}")
    print(f"  표 d = {ds.get(pid0)}")
    print(f"  번들 항목: (pid,1)={b2['platoon'][-1]['table'].get((pid0,1))}"
          f"  (pid,2)={b2['platoon'][-1]['table'].get((pid0,2))}")
    i=np.flatnonzero(bad & sel)[:3]
    for j in i:
        print(f"   행 j={j} 타자손={BH[m24][j]} 투수손={PH[m24][j]} 아티팩트={a[j]:+.6f} 분석={hd[j]:+.6f}")
