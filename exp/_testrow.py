import sys, os, importlib.util
import numpy as np, pandas as pd, joblib
sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("sc", os.path.join(ROOT, "script.py"))
sc = importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)
te = pd.read_csv(os.path.join(ROOT, "data", "test.csv"), encoding="utf-8-sig")
out = {}
for n in ("cat_asof_xl", "cat_submit_2", "cat_submit_3"):
    b = joblib.load(os.path.join(ROOT, "model_cand", f"{n}.pkl"))
    X = sc.attach_asof_state(te.copy(), b)
    X = sc.attach_ctx_infer(X, b) if hasattr(sc, "attach_ctx_infer") else X
    try:
        out[n] = sc.predict(b, X)
    except Exception:
        feats = sc.build_features(X, b)
        out[n] = sc.predict(b, feats) if False else sc.predict_bundle(b, feats)
print(f"{'row_id':<14}{'투수':>7}{'손':>4}{'S':>3}{'SUBMIT-1':>11}{'SUBMIT-2':>11}{'SUBMIT-3':>11}{'2-1':>10}{'3-2':>10}")
for i in range(len(te)):
    p1, p2, p3 = out["cat_asof_xl"][i], out["cat_submit_2"][i], out["cat_submit_3"][i]
    print(f"{te['row_id'][i]:<14}{te['pitcher_id'][i]:>7}{te['batter_hand'][i]:>4}"
          f"{te['strikes_before'][i]:>3}{p1:>11.6f}{p2:>11.6f}{p3:>11.6f}"
          f"{p2-p1:>+10.6f}{p3-p2:>+10.6f}")
