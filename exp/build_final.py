r"""FINAL 아티팩트 — C3 구조 그대로, 표만 가용 5시즌 OOF 잔차로 만든다.

변경점은 **하나**다: 표 원천을 2시즌(2023+2024) -> **5시즌(2020~2024)**.
k · 가중 · 구조는 전부 C3 와 동일 (과거 폴드에서 이미 검증된 값).

근거: 원천 시즌 수를 늘리면 폴드 2024 이득이 +14.4 -> +16.3 -> +19.1 로 오르고
폴드 2023 도 +8.2 -> +11.6 이다. 표본이 늘면 차등 추정 잡음이 줄기 때문이고,
가중/축소를 과거 폴드로 재선택하는 것(w=1.5 -> 2024 에서 +8.4 로 붕괴)과 달리
**하이퍼파라미터 튜닝이 아니라 가용 데이터를 다 쓰는 것**이다.
"""
import os, sys, hashlib, subprocess
import numpy as np, pandas as pd, joblib
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "exp"))
from path_alloc import build_df
from resid_table import post_for
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

tr = build_df(); season = tr["season"].to_numpy()
y = tr["control_success"].to_numpy(np.float64)
PID = tr["pitcher_id"].to_numpy(np.int64)
PH = tr["pitcher_hand"].to_numpy(np.int64); BH = tr["batter_hand"].to_numpy(np.int64)
SS = tr["strikes_before"].to_numpy(np.int64)
NR = tr["num_runners_on"].to_numpy(np.int64)
SRC = (2020, 2021, 2022, 2023, 2024)
res = {}
for f in SRC:
    m = season == f
    res[f] = y[m] - (np.load(os.path.join(ROOT, "exp", f"prod_champ_{f}.npy"))[:2].mean(0)
                     + post_for(tr, y, season < f, m))
msrc = np.isin(season, SRC)
rsrc = np.concatenate([res[f] for f in SRC])
print(f"  표 원천 {SRC}  총 {int(msrc.sum()):,}행")

def diff(ctx, k):
    gg = pd.DataFrame({"p": PID[msrc], "c": ctx[msrc], "r": rsrc}).groupby(
        ["p", "c"])["r"].agg(["mean", "size"]).unstack()
    n0, n1 = gg[("size", 0)].fillna(0), gg[("size", 1)].fillna(0)
    d = gg[("mean", 1)] - gg[("mean", 0)]
    ne = (n0 * n1) / (n0 + n1).replace(0, np.nan)
    return (d * ne / (ne + k)).dropna()

hand_d = diff((PH == BH).astype(int), 1000)
two_d = diff((SS == 2).astype(int), 1000)
run_d = diff((NR > 0).astype(int), 2000)
print(f"  손 {len(hand_d):,}명 |d|중앙 {hand_d.abs().median():.5f}   "
      f"2S {len(two_d):,} {two_d.abs().median():.5f}   "
      f"주자 {len(run_d):,} {run_d.abs().median():.5f}")

tab_hand = {(int(p), ph_, bh): float(0.5 * d if bh == ph_ else -0.5 * d)
            for p, d in hand_d.items() for ph_ in (1, 2) for bh in (1, 2)}
tab_two = {(int(p), s): float(0.5 * d if s == 2 else -0.5 * d)
           for p, d in two_d.items() for s in (0, 1, 2)}
tab_run = {(int(p), nr): float(0.5 * d if nr > 0 else -0.5 * d)
           for p, d in run_d.items() for nr in range(4)}
base = joblib.load(os.path.join(ROOT, "model_cand", "cat_asof_xl.pkl"))
b = dict(base)
b["platoon"] = list(base["platoon"]) + [
    {"w": 1.0, "cols": ["pitcher_id", "pitcher_hand", "batter_hand"],
     "table": tab_hand, "note": "잔차기반 손 차등 (OOF 2020~2024, k=1000)"},
    {"w": 1.0, "cols": ["pitcher_id", "strikes_before"], "table": tab_two,
     "note": "잔차기반 2스트라이크 차등 (OOF 2020~2024, k=1000)"},
    {"w": 1.0, "cols": ["pitcher_id", "num_runners_on"], "table": tab_run,
     "note": "잔차기반 주자유무 차등 (OOF 2020~2024, k=2000)"}]
b["note"] = str(base.get("note", "")) + " | FINAL: 잔차차등 3축, 표 원천 5시즌"
pkl = os.path.join(ROOT, "model_cand", "cat_final.pkl")
joblib.dump(b, pkl, compress=3)
zp = os.path.join(ROOT, "submissions", "cand_final.zip")
r = subprocess.run([sys.executable, os.path.join(ROOT, "make_submit.py"),
                    "--model", os.path.relpath(pkl, ROOT),
                    "--requirements", "requirements_cat.txt",
                    "--out", os.path.relpath(zp, ROOT)],
                   cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                   errors="replace")
print(r.stdout[-200:] if r.returncode == 0 else r.stdout[-100:] + r.stderr[-400:])
print(f"\n  cand_final.zip  {os.path.getsize(zp)/1e6:.2f} MB")
print(f"  sha256 {hashlib.sha256(open(zp,'rb').read()).hexdigest()}")
