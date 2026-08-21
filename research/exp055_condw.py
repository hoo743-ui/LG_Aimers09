r"""EXP055 — 🚩 **조건부 가중** 전수. 8개월간 후처리 가중은 전부 상수였다.

    지금까지   p = m + Σ w_i c_i                 w_i 는 상수
    이번       p = m + Σ (w_i + w'_i f) c_i      가중이 그 행의 피처 f 의 함수

기제 — `L_p` 는 **직전 2시즌**으로 만든 추정치인데 그 투수의 **평가 시즌 표본이
5구든 2,000구든** 같은 무게로 실렸다. 표본이 쌓이면 과거를 덜 믿어야 한다.
모델은 이 상호작용을 못 만든다 (`L_p` 는 후처리 표라 모델이 보지 못한다).

## 규율

조합도 가중도 **2022+2023 으로만** 고르고 2024 는 끝까지 숨긴다 (§5-a 조건 3).
위약(무작위 조건변수)을 같이 넣어 이 절차의 **잡음 바닥**을 측정한다.
`corr(선택 근거, 숨긴 결과)` 가 우연 수준이면 그 자체가 부류 기각이다.

    .\.venv\Scripts\python.exe -u research\exp055_condw.py
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "exp"))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from exp054_lvlshrink import components, W9                      # noqa: E402

AX = ["dev0_plat", "dev1_adv", "dev2_cnt", "dev3_run",
      "c_hand", "c_2S", "c_run", "L_pitch", "L_bat"]
WS = np.array([-3, -2.5, -2, -1.5, -1, -0.6, -0.3, 0.3, 0.6, 1, 1.5, 2, 3])
FOLDS = (2022, 2023, 2024)


def cond_cols(tr, m, mm, Cm):
    z = lambda v: (v - np.nanmean(v)) / (np.nanstd(v) + 1e-12)
    g = lambda c: tr[c].to_numpy(float)[m]
    d = {
        "logn_pitch": z(g("cur_logn_pitch")),
        "logn_bat": z(g("cur_logn_bat")),
        "log_asof_p": z(np.log1p(g("asof_pitcher_n"))),
        "log_asof_b": z(np.log1p(g("asof_batter_n"))),
        "cur_succ": z(np.nan_to_num(g("cur_succ"))),
        "cur_mid": z(np.nan_to_num(g("cur_mid"))),
        "strikes": z(g("strikes_before")),
        "balls": z(g("balls_before")),
        "onbase": z((g("num_runners_on") > 0).astype(float)),
        "inning": z(g("inning")),
        "month": z(g("game_month")),
        "li": z(np.nan_to_num(g("li"))),
        "model_p": z(mm),
        "career_rate": z(np.nan_to_num(g("asof_pitcher_success_rate"))),
    }
    rng = np.random.default_rng(7)
    for i in range(3):
        d[f"[위약{i+1}]"] = z(rng.standard_normal(len(mm)))
    return d


DATA = {}
from path_alloc import build_df                                   # noqa: E402
TR = build_df()
SEASON = TR["season"].to_numpy()
for f in FOLDS:
    mm, Cm, y, _ = components(f)
    m = SEASON == f
    cur = mm + Cm @ W9
    DATA[f] = dict(mm=mm, Cm=Cm, y=y, cur=cur, m=m,
                   base=1e5 * np.corrcoef(cur, y)[0, 1] ** 2,
                   cond=cond_cols(TR, m, mm, Cm))
    print(f"폴드 {f} 기준 {DATA[f]['base']:.2f}")
CONDS = list(DATA[2022]["cond"])
print(f"\n축 {len(AX)} x 조건 {len(CONDS)} = {len(AX)*len(CONDS)} 조합 x 가중 {len(WS)}")

rows = []
for ai, an in enumerate(AX):
    for cn in CONDS:
        gains = {}
        ok = True
        for f in FOLDS:
            D = DATA[f]
            v = D["Cm"][:, ai] * D["cond"][cn]
            s = v.std()
            if s == 0:
                ok = False
                break
            v = v / s * D["Cm"][:, 8].std()
            gains[f] = np.array([1e5 * np.corrcoef(D["cur"] + w * v, D["y"])[0, 1] ** 2
                                 - D["base"] for w in WS])
        if not ok:
            continue
        sel = (gains[2022] + gains[2023]) / 2          # **2024 를 보지 않고 고른다**
        j = int(np.argmax(sel))
        rows.append({"axis": an, "cond": cn, "w": float(WS[j]),
                     "sel_2223": float(sel[j]), "g22": float(gains[2022][j]),
                     "g23": float(gains[2023][j]), "hid24": float(gains[2024][j])})

D = pd.DataFrame(rows).sort_values("sel_2223", ascending=False)
pl = D[D.cond.str.startswith("[위약")]
print(f"\n=== 위약 {len(pl)}개 — 이 절차의 잡음 바닥 ===")
print(f"  선택근거(22+23) 평균 {pl.sel_2223.mean():+.2f} 최대 {pl.sel_2223.max():+.2f}")
print(f"  숨긴 2024        평균 {pl.hid24.mean():+.2f} 최대 {pl.hid24.max():+.2f}")
thr = pl.sel_2223.max()
print(f"\n=== 상위 20 (선택은 22+23 만, 2024 는 숨김) ===")
print(f"{'축':11s} {'조건':12s} {'w':>5s} {'선택22+23':>9s} {'2022':>7s} "
      f"{'2023':>7s} {'숨긴2024':>8s}")
for _, r in D.head(20).iterrows():
    mk = " ★" if (r.sel_2223 > thr and r.hid24 > 0) else ""
    print(f"{r.axis:11s} {r['cond']:12s} {r.w:+5g} {r.sel_2223:+9.2f} "
          f"{r.g22:+7.2f} {r.g23:+7.2f} {r.hid24:+8.2f}{mk}")

real = D[~D.cond.str.startswith("[위약")]
c = np.corrcoef(real.sel_2223, real.hid24)[0, 1]
print(f"\n🚩 corr(선택근거 22+23, 숨긴 2024) = {c:+.3f}   n={len(real)}")
print(f"   위약 문턱 {thr:+.2f} 초과 조합 {int((real.sel_2223>thr).sum())}개, "
      f"그중 2024 양수 {int(((real.sel_2223>thr)&(real.hid24>0)).sum())}개")
sel = real[real.sel_2223 > thr]
if len(sel):
    print(f"   그 조합들의 숨긴 2024 평균 {sel.hid24.mean():+.2f} "
          f"(양수비율 {(sel.hid24>0).mean():.0%})")
D.to_csv(os.path.join(ROOT, "exp", "exp055_condw.csv"), index=False, encoding="utf-8")
print("\n-> exp/exp055_condw.csv")
