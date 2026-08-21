r"""🚩 시드 앙상블 확대 — **수학적으로 이득이 보장되는** 유일한 변경.

    rho = cov(p,y)/(sd(p)·sd(y)),   p_k = μ(x) + ε_k,  var(ε_k)=σ²/k
    ->  rho_k = cov(μ,y)/sqrt(var(μ)+σ²/k)   **k 에 단조 증가. 음수 불가**

실측 σ²/var(p) = 0.0041 / 0.0048 / 0.0076 (폴드 2022/23/24)
    k=3 -> 30 이득  +1.32 / +1.56 / +2.46   3폴드 평균 **+1.78**

원장은 `시드 3→7 = +0.6, 기각` 이라 적혀 있으나, 그것은 시드 노이즈 ±7 인
로컬 측정으로 +2 짜리 효과를 재려다 못 본 것이다. **로컬이 안 보인다는 이유로
이론적으로 손해가 불가능한 변경을 버렸다.**

표·가중·아핀·asof_prior 는 Champion `cand_mir` 에서 그대로 가져온다.
바뀌는 것은 앙상블 크기뿐이다.

    .\venv_submit\Scripts\python.exe -u exp\build_seeds.py --seeds 30
"""
import argparse, gc, io, os, subprocess, sys, time, zipfile
import joblib, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "exp"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import build_asof as ba                                        # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tpl", default="submissions/cand_mir.zip")
ap.add_argument("--seeds", type=int, default=30)
ap.add_argument("--save-every", type=int, default=5)
ap.add_argument("--prefix", default="cand_s")
a = ap.parse_args()

with zipfile.ZipFile(os.path.join(ROOT, a.tpl)) as z:
    tpl = joblib.load(io.BytesIO(z.read("model/rf.pkl")))
FEAT = list(tpl["features"])
print(f"템플릿 {a.tpl}  피처 {len(FEAT)}열  alpha {tpl['alpha']} "
      f"center {tpl['center']:.6f}")
print(f"  가중 9개 = {[round(d['w'],6) for d in tpl['platoon']]}")

from path_alloc import build_df                                # noqa: E402
tr = build_df()
y = tr["control_success"].to_numpy(np.float64)
num = [c for c in FEAT if c not in ba.ft.CAT_COLS]
X = tr[FEAT].copy()
X[num] = X[num].astype(np.float32)
del tr; gc.collect()
print(f"  학습 행렬 {X.shape}")


def pack(models, name):
    b = dict(tpl)
    b["models"] = list(models)
    b["spec"] = [f"cat-s{42+i}" for i in range(len(models))]
    b["note"] = (tpl.get("note", "") +
                 f" | 시드 앙상블 {len(models)}개 (3 -> {len(models)}). "
                 f"rho 는 var(p) 분모에 시드잡음 sigma^2/k 를 담으므로 k 증가는 "
                 f"수학적으로 단조 이득. 실측 sigma^2/var(p)=0.0076(2024)")
    pkl = os.path.join(ROOT, "model_cand", f"{name}.pkl")
    zp = os.path.join(ROOT, "submissions", f"{name}.zip")
    if os.path.exists(zp):
        print(f"  (건너뜀, 이미 있음: {name})"); return
    joblib.dump(b, pkl, compress=3)
    r = subprocess.run([sys.executable, os.path.join(ROOT, "make_submit.py"),
                        "--model", os.path.relpath(pkl, ROOT),
                        "--requirements", "requirements_cat.txt",
                        "--out", os.path.relpath(zp, ROOT)],
                       cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace",
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    ok = "✅" if r.returncode == 0 else "❌"
    print(f"  {ok} {name}.zip  ({os.path.getsize(zp)/1e6:.1f} MB)"
          if r.returncode == 0 else f"  ❌ {name}\n{r.stderr[-600:]}", flush=True)


models = []
for i in range(a.seeds):
    t0 = time.time()
    m = ba.pipeline(FEAT, 42 + i)
    m.fit(X, y.astype(int))
    models.append(m)
    print(f"  seed {42+i} ({i+1}/{a.seeds}) {time.time()-t0:.0f}s", flush=True)
    gc.collect()
    if (i + 1) % a.save_every == 0 or (i + 1) == a.seeds:
        pack(models, f"{a.prefix}{i+1}")
print("\n완료")
