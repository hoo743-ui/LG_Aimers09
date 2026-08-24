r"""LV-004 — 수준 축 **탐욕 전진 선택**. 겹치는 축을 걷어내고 독립 집합을 남긴다.

LV-001 은 축마다 **단독** 이득을 냈다. 그런데 카운트·base_state·주자수·아웃수는
서로 같은 맥락의 재표현이라 단독 이득을 더하면 안 된다. 여기서는 매 단계
**현재까지 채택된 축을 전부 얹은 예측 위에서** 다음 축의 증분을 잰다.

⚠️ 이 선택은 2024 를 보고 한다 -> **승격 근거가 아니다.** 목적은 두 가지뿐이다.
   (1) 겹침 구조를 드러낸다  (2) LB 좌표로 열 축의 **순서**를 정한다
   각 단계마다 목표 2023 의 부호를 함께 찍어 3폴드 규율의 최소한을 지킨다.
"""
import io, json, os, sys
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
df = pd.read_csv(os.path.join(ROOT,"data","train.csv"), usecols=COLS)
df = df[df.season>=2022].reset_index(drop=True)
season=df.season.to_numpy(); y=df.control_success.to_numpy(float)
oof=np.load(os.path.join(ROOT,"exp","champ_oof.npz")); pv=champion_pred(df,oof,season)
K=build_keys(df); M={s:season==s for s in (2022,2023,2024)}
y24,y23=y[M[2024]],y[M[2023]]

def fit_apply(name, ktr, rtr, kta, pv_ta, y_ta):
    """k 격자에서 최선을 골라 (이득, 벡터, k, w) 를 낸다."""
    best=None
    for k in KS+[0]:
        u,t,_=level_table(ktr,rtr,k)
        v=lookup(u,t,kta); g=gain_curve(pv_ta,v,y_ta)
        if best is None or g["gain_opt"]>best[0]["gain_opt"]:
            best=(g,v,k)
    return best

cur24=pv[2024].copy(); cur23=pv[2023].copy()
chosen=[]; log=[]
pool=list(K)
print(f"기준선 2024 {r2(cur24,y24):.2f}   2023 {r2(cur23,y23):.2f}\n")
for step in range(1,9):
    cands=[]
    for nm in pool:
        ka=K[nm]
        ktr=np.concatenate([ka[M[2022]],ka[M[2023]]])
        rtr=np.concatenate([y[M[2022]]-pv[2022], y[M[2023]]-cur23])
        # 2024 증분 (표 = 2022+2023 잔차, 단 2023 잔차는 현재 누적 위에서)
        g,v,k = fit_apply(nm, ktr, rtr, ka[M[2024]], cur24, y24)
        # 2023 부호 (표 = 2022 만)
        g2,v2,_ = fit_apply(nm, ka[M[2022]], y[M[2022]]-pv[2022], ka[M[2023]], cur23, y23)
        cands.append((g["gain_opt"], nm, g, v, k, g2["gain_opt"], v2, g2["w_opt"]))
    cands.sort(key=lambda c:-c[0])
    gain, nm, g, v, k, g23, v23, w23 = cands[0]
    if gain < 0.15:
        print(f"[{step}] 최대 증분 {gain:+.2f} < 0.15 — 중단"); break
    cur24 = cur24 + g["w_opt"]*v
    cur23 = cur23 + w23*v23
    chosen.append(nm); pool.remove(nm)
    print(f"[{step}] {nm:<18} k={k:<7,} w*={g['w_opt']:>7.2f}  "
          f"2024 증분 {gain:>+6.2f} -> 누적 {r2(cur24,y24):8.2f}   "
          f"2023 증분 {g23:>+6.2f} {'✅' if g23>0 else '❌'}")
    log.append(dict(step=step,name=nm,k=k,w=g["w_opt"],gain24=gain,
                    cum24=float(r2(cur24,y24)),gain23=g23,pos23=bool(g23>0)))
    print("      다음 후보 3:", ", ".join(f"{c[1]} {c[0]:+.2f}" for c in cands[1:4]))

print(f"\n최종 2024 {r2(cur24,y24):.2f}  (기준선 대비 {r2(cur24,y24)-r2(pv[2024],y24):+.2f})")
print("채택 순서:", " -> ".join(chosen))
json.dump(log, io.open(os.path.join(ROOT,"research","lv004_greedy.json"),"w",
                       encoding="utf-8"), indent=1, ensure_ascii=False)
