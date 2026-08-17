import sys, os, importlib.util
import numpy as np, pandas as pd, joblib
sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "exp"))
spec = importlib.util.spec_from_file_location("sc", os.path.join(ROOT, "script.py"))
sc = importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)
from path_alloc import build_df
tr = build_df(); m = tr["season"].to_numpy() == 2024
X = tr.loc[m].iloc[:200].copy()
b = joblib.load(os.path.join(ROOT, "model_cand", "cat_submit_3.pkl"))
batch = sc.platoon_adjust(b, X)
solo = np.array([sc.platoon_adjust(b, X.iloc[[i]])[0] for i in range(len(X))])
print(f"  200행 배치 vs 1행씩 단독 — 최대 절대차 {np.max(np.abs(batch-solo)):.3e}")
print(f"  -> 0 이면 행 하나만 있어도 같은 값이다 (규정 4 준수)")
