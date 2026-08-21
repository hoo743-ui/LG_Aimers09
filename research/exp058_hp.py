r"""EXP058 — HP 동전 스크린. **승자 선별이 아니라 참사 제거용.**

원장의 `모델 적합 = 금지` 는 단조 제약 **n=1** 에서 나온 과잉일반화다. 지금 HP
(`depth6 · l2=100 · border32 · iter1200 · lr0.02`)는 **9회차**에 정해졌고 그때는
피처가 ~50열이고 D 가 없었다. 그 뒤 82열이 되고 D 가 +85.65 를 넣었는데 HP 는
한 번도 안 건드렸다.

EXP057 로 로컬 부호율이 50% 임이 밝혀졌으므로 여기서 **승자를 고르지 않는다.**
로컬은 참사(-10% 급)만 거른다. 하방이 0 이므로 나머지는 슬롯이 남는 대로 낸다.

    .\.venv\Scripts\python.exe -u research\exp058_hp.py
"""
import json, os, sys, time
import numpy as np

HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
sys.path.insert(0,os.path.join(ROOT,"exp"))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import build_asof as ba
from path_alloc import build_df

VAR={"base":{}, "bord254":{"border_count":254}, "bord128":{"border_count":128},
     "d7":{"depth":7}, "iter3k":{"iterations":3000,"learning_rate":0.008},
     "l2_30":{"l2_leaf_reg":30.0}, "rsm80":{"rsm":0.8}}
import zipfile, io, joblib
with zipfile.ZipFile(os.path.join(ROOT,"submissions","cand_mir.zip")) as z:
    B=joblib.load(io.BytesIO(z.read("model/rf.pkl")))
FEAT=list(B["features"]); W9=np.array([d["w"] for d in B["platoon"]])
tr=build_df(); season=tr["season"].to_numpy(); y=tr["control_success"].to_numpy(np.float64)
P=tr["pitcher_id"].to_numpy(np.int64); BH=tr["batter_hand"].to_numpy(np.int64)
BB=tr["balls_before"].to_numpy(np.int64); SS=tr["strikes_before"].to_numpy(np.int64)
OB=(tr["num_runners_on"].to_numpy(np.int64)>0).astype(np.int64)
PH=P*10+BH; PHA=PH*10+(SS>BB).astype(np.int64)
AX=[(P,PH),(PH,PHA),(PHA,PHA*100+(BB*4+SS)),(PH,PH*10+OB)]
mt,mv=season<2024,season==2024
post=np.column_stack([ba.look(*ba.nested_dev(p[mt],c[mt],y[mt],k),c[mv])
                      for (p,c),k in zip(AX,ba.KSH)])@ba.WPOST
Xtr,Xva,yv=tr.loc[mt,FEAT],tr.loc[mv,FEAT],y[mv]
del tr
res={}
for nm,hp in VAR.items():
    keep=ba.HP.copy(); ba.HP.update(hp)
    t0=time.time(); m=ba.pipeline(FEAT,42); m.fit(Xtr,y[mt].astype(int))
    s=1e5*np.corrcoef(m.predict_proba(Xva)[:,1]+post,yv)[0,1]**2
    ba.HP.clear(); ba.HP.update(keep)
    res[nm]=s
    d=s-res["base"] if "base" in res else 0.0
    print(f"  {nm:9s} {s:8.2f}  대조대비 {d:+7.2f} ({d/max(res.get('base',1),1)*100:+.2f}%)  "
          f"{time.time()-t0:.0f}s  {hp}",flush=True)
    json.dump(res,open(os.path.join(ROOT,"exp","exp058_hp.json"),"w"),indent=1)
b=res["base"]
print(f"\n{'판본':10s} {'점수':>9s} {'대조대비':>9s}  판정")
for nm,s in sorted(res.items(),key=lambda x:-x[1]):
    v="참사 — 제외" if (s-b)/b<-0.05 else ("대조" if nm=="base" else "빌드 대상")
    print(f"{nm:10s} {s:9.2f} {s-b:+9.2f}  {v}")
