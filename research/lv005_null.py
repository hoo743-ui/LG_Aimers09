r"""LV-005 — "36개 중 최대"의 **귀무분포**. LV-001/004 의 값을 이걸로 나눈다.

§5-c 4 와 EXP055 가 같은 것을 두 번 가르쳤다 — 여러 후보에서 고른 최댓값은
그 자체가 통계량이고, 위약 문턱을 넘지 못하면 신호가 아니다.

각 후보의 **키를 시즌 안에서 무작위 치환**한다. 셀 수·셀 크기 분포·행수가
전부 보존되고 오직 "그 키가 잔차와 갖는 관계"만 파괴된다.
"""
import io, json, os, sys, time
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
from lv001_level_sweep import (champion_pred, r2, level_table, lookup,   # noqa
                               gain_curve, build_keys, KS)
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

COLS = ["season","control_success","balls_before","strikes_before","pitcher_id",
        "batter_id","pitcher_team_id","batter_team_id","pitcher_hand","batter_hand",
        "inning","game_month","game_dayofweek","num_runners_on","outs_before",
        "game_type","top_bottom","base_state","score_diff_pitcher_team","li",
        "asof_batter_n"]
df=pd.read_csv(os.path.join(ROOT,"data","train.csv"),usecols=COLS)
df=df[df.season>=2022].reset_index(drop=True)
season=df.season.to_numpy(); y=df.control_success.to_numpy(float)
oof=np.load(os.path.join(ROOT,"exp","champ_oof.npz")); pv=champion_pred(df,oof,season)
K=build_keys(df); M={s:season==s for s in (2022,2023,2024)}
y24=y[M[2024]]; p24=pv[2024]
rtr=np.concatenate([y[M[2022]]-pv[2022], y[M[2023]]-pv[2023]])

def gain(ka_tr, ka_24):
    best=-1e9
    for k in KS+[0]:
        u,t,_=level_table(ka_tr,rtr,k)
        g=gain_curve(p24,lookup(u,t,ka_24),y24)["gain_opt"]
        best=max(best,g)
    return best

names=list(K)
print("실측 (LV-001 재계산)")
real={}
for nm in names:
    ka=K[nm]
    real[nm]=gain(np.concatenate([ka[M[2022]],ka[M[2023]]]), ka[M[2024]])
top=sorted(real.items(), key=lambda x:-x[1])[:8]
for nm,g in top: print(f"  {nm:<20}{g:>+8.2f}")
print(f"  실측 최대 = {top[0][1]:+.2f}")

R=12
print(f"\n귀무분포 — 키를 시즌 안에서 치환, {R}회")
t0=time.time(); mx=[]; per={nm:[] for nm in names}
rng=np.random.default_rng(7)
for r in range(R):
    best=-1e9
    for nm in names:
        ka=K[nm].copy()
        for s in (2022,2023,2024):
            m=M[s]; ka[m]=rng.permutation(ka[m])
        g=gain(np.concatenate([ka[M[2022]],ka[M[2023]]]), ka[M[2024]])
        per[nm].append(g); best=max(best,g)
    mx.append(best)
    print(f"  {r+1:2d}/{R}  최대 {best:+.2f}   ({time.time()-t0:.0f}s)", flush=True)

mx=np.array(mx)
print(f"\n귀무 '36개 중 최대'  평균 {mx.mean():+.2f}  sd {mx.std():.2f}  "
      f"최대 {mx.max():+.2f}")
print(f"실측 최대 {top[0][1]:+.2f}  ->  귀무 대비 z = "
      f"{(top[0][1]-mx.mean())/max(mx.std(),1e-9):+.1f}")
print(f"\n{'축':<20}{'실측':>8}{'귀무평균':>9}{'귀무sd':>8}{'z':>7}   판정")
out=[]
for nm,g in sorted(real.items(), key=lambda x:-x[1]):
    a=np.array(per[nm]); z=(g-a.mean())/max(a.std(),1e-9)
    v = "🔴 신호" if g>mx.mean()+2*mx.std() else ("⚠️ 경계" if g>mx.mean() else "위약 이하")
    out.append(dict(name=nm,real=g,null_mean=float(a.mean()),null_sd=float(a.std()),z=float(z),verdict=v))
    print(f"{nm:<20}{g:>+8.2f}{a.mean():>+9.2f}{a.std():>8.2f}{z:>+7.1f}   {v}")
json.dump(dict(null_max=dict(mean=float(mx.mean()),sd=float(mx.std()),
                             draws=[float(x) for x in mx]), axes=out),
          io.open(os.path.join(ROOT,"research","lv005_null.json"),"w",encoding="utf-8"),
          indent=1, ensure_ascii=False)
