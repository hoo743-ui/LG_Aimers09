"""아티팩트 추론 경로 == 분석 경로 인지 검증."""
import sys, os, importlib.util
import numpy as np, pandas as pd, joblib
sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "exp"))
spec = importlib.util.spec_from_file_location("sc", os.path.join(ROOT, "script.py"))
sc = importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)
from path_alloc import build_df
from resid_table import post_for
tr = build_df(); season = tr["season"].to_numpy()
y = tr["control_success"].to_numpy(np.float64)
PID = tr["pitcher_id"].to_numpy(np.int64)
m24 = season == 2024
X24 = tr.loc[m24]
b1 = joblib.load(os.path.join(ROOT, "model_cand", "cat_asof_xl.pkl"))
b2 = joblib.load(os.path.join(ROOT, "model_cand", "cat_submit_2.pkl"))
b3 = joblib.load(os.path.join(ROOT, "model_cand", "cat_submit_3.pkl"))
a1 = sc.platoon_adjust(b1, X24)
a2 = sc.platoon_adjust(b2, X24)
a3 = sc.platoon_adjust(b3, X24)
print("=== 아티팩트가 실제로 더하는 보정 (2024 행 253,507) ===")
print(f"  SUBMIT-2 − SUBMIT-1 : sd {np.std(a2-a1):.6f}  평균 {np.mean(a2-a1):+.6f}")
print(f"  SUBMIT-3 − SUBMIT-2 : sd {np.std(a3-a2):.6f}  평균 {np.mean(a3-a2):+.6f}")
# 분석 경로에서 만든 보정과 비교
res = {}
for f in (2023, 2024):
    m = season == f
    res[f] = y[m] - (np.load(os.path.join(ROOT, "exp", f"prod_champ_{f}.npy"))[:2].mean(0)
                     + post_for(tr, y, season < f, m))
msrc = np.isin(season, (2023, 2024)); rs = np.concatenate([res[2023], res[2024]])
def diff(ctx, k):
    gg = pd.DataFrame({"p": PID[msrc], "c": ctx[msrc], "r": rs}).groupby(
        ["p", "c"])["r"].agg(["mean", "size"]).unstack()
    n0, n1 = gg[("size", 0)].fillna(0), gg[("size", 1)].fillna(0)
    d = gg[("mean", 1)] - gg[("mean", 0)]
    ne = (n0 * n1) / (n0 + n1).replace(0, np.nan)
    return (d * ne / (ne + k)).dropna()
SAME = (tr["pitcher_hand"].to_numpy(np.int64) == tr["batter_hand"].to_numpy(np.int64)).astype(int)
TWO = (tr["strikes_before"].to_numpy(np.int64) == 2).astype(int)
hd = pd.Series(PID[m24]).map(diff(SAME, 1000)).fillna(0).to_numpy() * np.where(SAME[m24] == 1, .5, -.5)
ts = pd.Series(PID[m24]).map(diff(TWO, 1000)).fillna(0).to_numpy() * np.where(TWO[m24] == 1, .5, -.5)
print(f"\n  분석 경로 손차등과의 최대 절대차   {np.max(np.abs((a2-a1)-hd)):.3e}")
print(f"  분석 경로 2S차등과의 최대 절대차   {np.max(np.abs((a3-a2)-ts)):.3e}")
print(f"  -> 0 이면 아티팩트가 분석과 동일한 보정을 낸다")
te = pd.read_csv(os.path.join(ROOT, "data", "test.csv"), encoding="utf-8-sig")
print(f"\n=== 실제 test.csv 5행에서의 보정 ===")
for i, (p, bh, s) in enumerate(zip(te["pitcher_id"], te["batter_hand"], te["strikes_before"])):
    v2 = b2["platoon"][-1]["table"].get((int(p), int(bh)), None)
    v3 = b3["platoon"][-1]["table"].get((int(p), int(s)), None)
    print(f"  행{i} 투수 {p:>6} 타자손 {bh} 스트라이크 {s} | 손차등 "
          f"{'없음(0)' if v2 is None else f'{v2:+.5f}'}  2S차등 "
          f"{'없음(0)' if v3 is None else f'{v3:+.5f}'}")
