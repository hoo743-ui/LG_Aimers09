r"""LV-003 — 카운트 수준축의 위약·중심화 검정.

의심 두 가지를 가른다.
  (1) 시즌 전체가 음수 잔차다 (리그 하락 드리프트). rho 는 상수 이동에 불변이지만
      **표가 상수를 실어 나르면** 다른 셀 비율 때문에 착시가 생길 수 있다 -> 중심화
  (2) 12셀이면 아무 분할이나 이득이 날 수 있다 -> **무작위 12셀 위약**
"""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
from lv001_level_sweep import champion_pred, r2, level_table, lookup, gain_curve  # noqa
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

df = pd.read_csv(os.path.join(ROOT,"data","train.csv"),
    usecols=["season","control_success","balls_before","strikes_before","pitcher_id",
             "batter_id","pitcher_hand","batter_hand","num_runners_on"])
df = df[df.season>=2022].reset_index(drop=True)
season=df.season.to_numpy(); y=df.control_success.to_numpy(float)
oof=np.load(os.path.join(ROOT,"exp","champ_oof.npz")); pv=champion_pred(df,oof,season)
cnt=df.balls_before.to_numpy(np.int64)*4+df.strikes_before.to_numpy(np.int64)
M={s:season==s for s in (2022,2023,2024)}
ktr=np.concatenate([cnt[M[2022]],cnt[M[2023]]])
rtr=np.concatenate([y[M[2022]]-pv[2022], y[M[2023]]-pv[2023]])
k24,p24,y24=cnt[M[2024]],pv[2024],y[M[2024]]
base=r2(p24,y24)

def rep(nm,v):
    g=gain_curve(p24,v,y24)
    print(f"  {nm:<34}{r2(p24+v,y24)-base:>+8.2f} (w=1)   w*={g['w_opt']:>6.2f} "
          f"이득 {g['gain_opt']:>+7.2f}")

u,t,n=level_table(ktr,rtr,0)
print(f"기준선 2024 = {base:.2f}\n")
print("① 원본 / 중심화 / 순위만")
rep("원본 표",                        lookup(u,t,k24))
rep("중심화 (셀평균 − 가중평균)",       lookup(u,t-(t*n).sum()/n.sum(),k24))
rk=(np.argsort(np.argsort(t))-5.5)/5.5*np.abs(t-(t*n).sum()/n.sum()).mean()
rep("순위만 (크기 버림)",               lookup(u,rk,k24))

print("\n② 위약 — 무작위 12셀 (같은 행수 분포)")
rng=np.random.default_rng(0); gs=[]
for i in range(20):
    fake_all=rng.permutation(np.concatenate([cnt[M[s]] for s in (2022,2023,2024)]))
    fa_tr=fake_all[:len(ktr)]; fa_24=fake_all[len(ktr):]
    u2,t2,_=level_table(fa_tr,rtr,0)
    gs.append(r2(p24+lookup(u2,t2,fa_24),y24)-base)
print(f"  20회 평균 {np.mean(gs):+.3f}   sd {np.std(gs):.3f}   최대 {np.max(gs):+.3f}")

print("\n③ 셀 하나씩 빼면 (기여 분해, w=1)")
full=lookup(u,t,k24); g_full=r2(p24+full,y24)-base
for c in sorted(np.unique(cnt)):
    t2=t.copy(); t2[u==c]=0.0
    print(f"  {c//4}-{c%4} 제거   {r2(p24+lookup(u,t2,k24),y24)-base:>+7.2f}"
          f"   (전체 {g_full:+.2f} 대비 {r2(p24+lookup(u,t2,k24),y24)-base-g_full:+.2f})")
