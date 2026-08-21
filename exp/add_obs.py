r"""LB 실측 하나를 `exp/lb_obs.json` 에 추가한다. 가중은 zip 에서 직접 읽는다.

손으로 9개 숫자를 옮겨 적지 않는다 — 라운드마다 그러다 보면 반드시 틀린다.

    .\.venv\Scripts\python.exe exp\add_obs.py --zip submissions\cand_rob1.zip ^
        --lb 1076.1234567890 --round 44
"""
import argparse
import io
import json
import os
import sys
import zipfile

import joblib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(ROOT, "exp", "lb_obs.json")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ap = argparse.ArgumentParser()
ap.add_argument("--zip", required=True)
ap.add_argument("--lb", type=float, required=True)
ap.add_argument("--round", type=int, required=True)
a = ap.parse_args()

with zipfile.ZipFile(a.zip) as z:
    b = joblib.load(io.BytesIO(z.read("model/rf.pkl")))
w = [round(float(d["w"]), 6) for d in b["platoon"]]
cand = os.path.splitext(os.path.basename(a.zip))[0]

d = json.load(open(P, encoding="utf-8"))
assert not any(o["round"] == a.round for o in d["obs"]), f"{a.round}회차가 이미 있다"
d["obs"].append({"round": a.round, "cand": cand, "w": w, "lb": a.lb})
d["obs"].sort(key=lambda o: o["round"])
d.setdefault("built", {})[cand] = w
json.dump(d, open(P, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"{a.round}회차 {cand} = {a.lb:.10f}  추가 -> 관측 {len(d['obs'])}개")
print("  w =", w)
print("\n다음: .\.venv\Scripts\python.exe -u research\exp044_robust.py")
