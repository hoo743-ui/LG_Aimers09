"""제출 직전 최종 관문 — 구성·표·추론·누수·예상밖 차이를 전부 검사."""
import sys, os, hashlib, importlib.util, zipfile
import numpy as np, pandas as pd, joblib
sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "exp"))
spec = importlib.util.spec_from_file_location("sc", os.path.join(ROOT, "script.py"))
sc = importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)
from path_alloc import build_df
from resid_table import post_for

def h(o):
    return hashlib.sha256(repr(o).encode()).hexdigest()[:16]
def fh(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()

B = {n: joblib.load(os.path.join(ROOT, "model_cand", f)) for n, f in
     (("C1", "cat_asof_xl.pkl"), ("C2", "cat_submit_2.pkl"), ("C3", "cat_submit_3.pkl"))}
print("=" * 96); print("1. 구성 확인"); print("=" * 96)
for n, b in B.items():
    ax = [s.get("note", "")[:46] for s in b["platoon"]]
    print(f"  {n}: 모델 {len(b['models'])}  피처 {len(b['features'])}  후처리 {len(b['platoon'])}축"
          f"  alpha {b['alpha']}  center {b['center']:.6f}")
    for i, a in enumerate(ax):
        print(f"       축{i+1} {a}")
print("\n" + "=" * 96); print("2. 표 3중 비교 — 학습 계산값 / 아티팩트 저장값 / 추론 적용값"); print("=" * 96)
tr = build_df(); season = tr["season"].to_numpy(); y = tr["control_success"].to_numpy(np.float64)
PID = tr["pitcher_id"].to_numpy(np.int64); PH = tr["pitcher_hand"].to_numpy(np.int64)
BH = tr["batter_hand"].to_numpy(np.int64); SS = tr["strikes_before"].to_numpy(np.int64)
RN = (tr["num_runners_on"].to_numpy(np.int64) > 0).astype(int)
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
    d = gg[("mean", 1)] - gg[("mean", 0)]; ne = (n0 * n1) / (n0 + n1).replace(0, np.nan)
    return (d * ne / (ne + k)).dropna()
SAME = (PH == BH).astype(int); TWO = (SS == 2).astype(int)
REF = {"hand": diff(SAME, 1000), "two": diff(TWO, 1000), "run": diff(RN, 2000)}
m24 = season == 2024; X24 = tr.loc[m24]
ANL = {"hand": pd.Series(PID[m24]).map(REF["hand"]).fillna(0).to_numpy() * np.where(SAME[m24] == 1, .5, -.5),
       "two": pd.Series(PID[m24]).map(REF["two"]).fillna(0).to_numpy() * np.where(TWO[m24] == 1, .5, -.5),
       "run": pd.Series(PID[m24]).map(REF["run"]).fillna(0).to_numpy() * np.where(RN[m24] == 1, .5, -.5)}
A = {n: sc.platoon_adjust(b, X24) for n, b in B.items()}
print(f"  {'축':<8}{'저장 표 항목':>12}{'체크섬':>18}{'추론−분석 최대차':>18}")
d21 = A["C2"] - A["C1"]; d32 = A["C3"] - A["C2"]
print(f"  {'hand':<8}{len(B['C2']['platoon'][4]['table']):>12}{h(sorted(B['C2']['platoon'][4]['table'].items())):>18}"
      f"{np.max(np.abs(d21 - ANL['hand'])):>18.3e}")
exp32 = ANL["two"] + ANL["run"]
print(f"  {'2S+run':<8}{len(B['C3']['platoon'][5]['table'])+len(B['C3']['platoon'][6]['table']):>12}"
      f"{h(sorted(B['C3']['platoon'][5]['table'].items())+sorted(B['C3']['platoon'][6]['table'].items())):>18}"
      f"{np.max(np.abs(d32 - exp32)):>18.3e}")
print("\n" + "=" * 96); print("3. 단계별 예측 분해 (2024 표본 5행)"); print("=" * 96)
Xf = sc.build_features(X24, B["C1"])
raw = None
for m in B["C1"]["models"]:
    p = m.predict_proba(Xf)[:, 1]; raw = p if raw is None else raw + p
raw = raw / len(B["C1"]["models"])
al, ce = float(B["C1"]["alpha"]), float(B["C1"]["center"])
idx = [0, 1000, 50000, 150000, 253000]
print(f"  {'행':>7}{'모델 raw':>10}{'기존4축':>10}{'hand':>10}{'2S':>9}{'run':>9}{'합':>10}{'아핀후 C3':>11}")
for i in idx:
    tot = A["C3"][i]
    fin = ce + al * (raw[i] + tot - ce)
    print(f"  {i:>7}{raw[i]:>10.5f}{A['C1'][i]:>+10.5f}{ANL['hand'][i]:>+10.5f}"
          f"{ANL['two'][i]:>+9.5f}{ANL['run'][i]:>+9.5f}{tot:>+10.5f}{fin:>11.5f}")
chk = np.max(np.abs((A["C1"] + ANL["hand"] + ANL["two"] + ANL["run"]) - A["C3"]))
print(f"\n  수식 일치 검사 max|C1축+hand+2S+run − C3축| = {chk:.3e}")
print("\n" + "=" * 96); print("4. 아티팩트 지문"); print("=" * 96)
for n, f in (("C1", "cand_submit_1.zip"), ("C2", "cand_submit_2.zip"), ("C3", "cand_submit_3.zip")):
    p = os.path.join(ROOT, "submissions", f)
    with zipfile.ZipFile(p) as z:
        mh = hashlib.sha256(z.read("model/rf.pkl")).hexdigest()
    b = B[n]
    fin = (ce + al * (raw + A[n] - ce)).clip(0, 1)
    print(f"  {n} {f}")
    print(f"     zip   {fh(p)}")
    print(f"     model {mh[:32]}...  피처 {len(b['features'])}  후처리 {len(b['platoon'])}축")
    print(f"     asof_prior 체크섬 {h(sorted((k, tuple(sorted(v.items()))[:1]) for k, v in b['asof_prior'].items()))}"
          f"   2024 예측 {len(fin):,}행  범위 {fin.min():.4f}~{fin.max():.4f}")
