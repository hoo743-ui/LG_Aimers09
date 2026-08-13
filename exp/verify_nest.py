r"""신규 기전 검증 — 투수유리(2진)를 볼/스트라이크 12칸으로 펼친 조회가
실험 경로와 같은 값을 내는가. script.py 의 platoon_adjust 를 그대로 부른다.

    .\.venv\Scripts\python.exe exp\verify_nest.py
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from script import platoon_adjust          # noqa: E402

CACHE = os.path.join(ROOT, "exp", "cache")
PKL = os.path.join(ROOT, "model_cand", "cat_nest_adv.pkl")


def nested_dev(parent, child, y, k):
    o = np.argsort(child, kind="stable")
    Ys, Ps, Cs = y[o], parent[o], child[o]
    u, s = np.unique(Cs, return_index=True)
    cnt = np.diff(np.append(s, len(Cs)))
    cell = np.add.reduceat(Ys, s) / cnt
    par = Ps[s]
    op = np.argsort(parent, kind="stable")
    Yp, Pp = y[op], parent[op]
    pu, ps = np.unique(Pp, return_index=True)
    pc = np.diff(np.append(ps, len(Pp)))
    pmean = np.add.reduceat(Yp, ps) / pc
    return u, cnt * (cell - pmean[np.searchsorted(pu, par)]) / (cnt + k)


def lookup(u, dev, keys):
    out = np.zeros(len(keys), dtype=np.float64)
    hit = np.zeros(len(keys), dtype=bool)
    ix = np.clip(np.searchsorted(u, keys), 0, len(u) - 1)
    ok = u[ix] == keys
    out[ok], hit[ok] = dev[ix[ok]], True
    return out, hit


meta = json.load(open(f"{CACHE}/cols.json"))
ixc = {c: i for i, c in enumerate(meta["cols"])}
X = np.load(f"{CACHE}/X.npy", mmap_mode="r")
y = np.load(f"{CACHE}/y.npy").astype(np.float64)
season = np.load(f"{CACHE}/season.npy")
col = lambda n: np.asarray(X[:, ixc[n]], dtype=np.float64)

P = col("pitcher_id").astype(np.int64)
BH = col("batter_hand").astype(np.int64)
BB, SS = col("balls_before").astype(np.int64), col("strikes_before").astype(np.int64)
NR = col("num_runners_on").astype(np.int64)
OB = (NR > 0).astype(np.int64)
PH, CNT = P * 10 + BH, BB * 4 + SS
ADV = (SS > BB).astype(np.int64)
PHA = PH * 10 + ADV

u1, t1 = nested_dev(P, PH, y, 300)
uC, tC = nested_dev(PH, PHA, y, 2000)
uN, tN = nested_dev(PHA, PHA * 100 + CNT, y, 800)
u3, t3 = nested_dev(PH, PH * 10 + OB, y, 2000)

va = season == 2024
e1, h1 = lookup(u1, t1, PH[va])
eC, hC = lookup(uC, tC, PHA[va])
eN, hN = lookup(uN, tN, (PHA * 100 + CNT)[va])
e3, h3 = lookup(u3, t3, (PH * 10 + OB)[va])

b = joblib.load(PKL)
W = [float(s["w"]) for s in b["platoon"]]
print(f"pkl 가중  플래툰 {W[0]}  거친 {W[1]}  카운트|거친 {W[2]}  주자유무 {W[3]}")
print(f"alpha={b['alpha']:.6f}  center={b['center']:.6f}")
for s in b["platoon"]:
    print(f"  {str(s['cols']):<70} {len(s['table']):>7,}칸")

exp_total = W[0] * e1 + W[1] * eC + W[2] * eN + W[3] * e3
Xdf = pd.DataFrame({"pitcher_id": P[va], "batter_hand": BH[va],
                    "balls_before": BB[va], "strikes_before": SS[va],
                    "num_runners_on": NR[va]})
inf_total = np.asarray(platoon_adjust(b, Xdf))
d = np.abs(inf_total - exp_total)
print(f"\n=== 추론 경로 vs 실험 경로 ({va.sum():,}행) ===")
print(f"  최대차 {d.max():.3e}  평균차 {d.mean():.3e}  "
      f"{'통과' if d.max() < 1e-12 else '불일치'}")

print(f"\n=== 커버리지 (2024 대역, 표는 전체 학습으로 제작) ===")
for nm, h in [("플래툰", h1), ("거친(투수유리)", hC),
              ("카운트|거친", hN), ("주자유무", h3)]:
    print(f"  {nm:<16} {100 * h.mean():5.1f}%")

print(f"\n=== 보류 커버리지 (표 2019~2023 -> 2024) ===")
tr = season <= 2023
for nm, par, ch, k in [("플래툰", P, PH, 300), ("거친(투수유리)", PH, PHA, 2000),
                       ("카운트|거친", PHA, PHA * 100 + CNT, 800),
                       ("주자유무", PH, PH * 10 + OB, 2000)]:
    u = np.unique(ch[tr])
    kk = ch[va]
    i = np.clip(np.searchsorted(u, kk), 0, len(u) - 1)
    print(f"  {nm:<16} {100 * (u[i] == kk).mean():5.1f}%")

print(f"\n=== 펼침 검사 — 같은 ADV 인 볼/스트라이크 칸이 모두 같은 값인가 ===")
tab = b["platoon"][1]["table"]
bad, chk = 0, 0
for (pid, hand, bl, st), v in list(tab.items())[:4000]:
    adv = int(st > bl)
    for b2 in range(4):
        for s2 in range(3):
            if int(s2 > b2) == adv and (pid, hand, b2, s2) in tab:
                chk += 1
                if abs(tab[(pid, hand, b2, s2)] - v) > 1e-15:
                    bad += 1
print(f"  검사 {chk:,}쌍 중 불일치 {bad}  {'통과' if bad == 0 else '불일치'}")

print(f"\n=== 두 층의 직교성 (2024) ===")
print(f"  corr(거친, 카운트|거친) = {np.corrcoef(eC, eN)[0, 1]:+.4f}")
