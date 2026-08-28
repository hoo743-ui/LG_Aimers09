# -*- coding: utf-8 -*-
r"""EXP059 — 타깃 채널 분해 (2026-08-29, 사용자 지시 고위험 트랙).

y = (1−middle)(1−reverse)(0.95−0.36·ball) 이 확정돼 있다 (합성 사건).
middle/reverse/ball 을 **각각** CatBoost 로 모델링해 공식으로 합성한다 —
스태킹(예측을 피처로 되먹임)이 아니라 같은 확률의 다른 추정량이다.

⚠️ 로컬은 **참사 확인용**이다 (EXP057: 로컬 부호율 50%). ft2 (2-인수, −5.83) 는
이것과 다른 구성이었다. §5-d 게이트(로컬 음수 제출 금지)를 사용자 지시로
명시적으로 우회한다 — 최고점 채점이라 하방 0.

    .\.venv\Scripts\python.exe -u research\exp059_chan.py
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

lab = np.load(os.path.join(ROOT, "exp", "cache", "pitch_labels.npz"), allow_pickle=True)
assert len(lab["season"]) == len(tr), (len(lab["season"]), len(tr))
# prod_df 는 원래 train.csv 순서가 아니다 — (pitcher_id, asof_pitcher_n) 유일 키로 조인
import pandas as pd
csv = pd.read_csv(os.path.join(ROOT, "data", "train.csv"),
                  usecols=["row_id", "pitcher_id", "asof_pitcher_n"])
rid_pos = {r: j for j, r in enumerate(lab["row_id"])}
csv_pos = np.array([rid_pos[r] for r in csv["row_id"]])       # train.csv 행 -> npz 행
key_csv = csv["pitcher_id"].to_numpy(np.int64) * 100000 + csv["asof_pitcher_n"].to_numpy(np.int64)
assert csv["asof_pitcher_n"].max() < 100000
key_map = dict(zip(key_csv, csv_pos))
key_tr = tr["pitcher_id"].to_numpy(np.int64) * 100000 + tr["asof_pitcher_n"].to_numpy(np.int64)
perm = np.array([key_map[k] for k in key_tr])
assert (lab["season"][perm] == season).all(), "조인 후에도 시즌 불일치"
MID, REV, BALL = (lab[k].astype(np.float64)[perm] for k in ("middle", "reverse", "ball"))
del csv, rid_pos, key_map

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
# 대조: 직접 y 모델 (exp058 base 944.05 재현 확인)
t0 = time.time()
m = ba.pipeline(FEAT, 42); m.fit(Xtr, y[mt].astype(int))
py = m.predict_proba(Xva)[:, 1]
res["base"] = score(py)
print(f"  base      {res['base']:8.2f}   {time.time()-t0:.0f}s", flush=True)

# 채널 3개
ch = {}
for nm, L in (("mid", MID), ("rev", REV), ("ball", BALL)):
    t0 = time.time()
    ok = mt & np.isfinite(L)
    m = ba.pipeline(FEAT, 42); m.fit(Xtr[ok[mt]], L[ok].astype(int))
    ch[nm] = m.predict_proba(Xva)[:, 1]
    auc_dir = float(np.corrcoef(ch[nm], np.nan_to_num(L[mv]))[0, 1])
    print(f"  ch_{nm:4s}  corr(pred, label_2024) {auc_dir:+.4f}   {time.time()-t0:.0f}s", flush=True)

pf = (1 - ch["mid"]) * (1 - ch["rev"]) * (0.95 - 0.36*ch["ball"])
res["chan3"] = score(pf)
res["blend50"] = score(0.5*py + 0.5*pf)
res["blend25"] = score(0.75*py + 0.25*pf)
# 참고: 채널 예측을 y 잔차 방향으로만 (직교 성분 크기)
c = np.corrcoef(pf, py)[0, 1]
print(f"\n  corr(chan3, base) = {c:+.4f}")
for k in ("base", "chan3", "blend50", "blend25"):
    print(f"  {k:8s} {res[k]:8.2f}   대조대비 {res[k]-res['base']:+7.2f}")
json.dump(res, open(os.path.join(ROOT, "exp", "exp059_chan.json"), "w"), indent=1)
