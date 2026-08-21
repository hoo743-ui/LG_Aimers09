r"""AUX 최소 판본 — 보조 라벨 **1개**(H = middle ∨ reverse)만 피처로.

46회차에서 나는 로컬 이득이 최대인 판본(aux8, +2.399%)을 냈고 전이가 -0.302 라
LB -7.79 였다. 로컬 이득이 라벨 수에 단조 증가하므로 **최소 판본이 최소 손실**
이면서 조회표 항목은 똑같이 얻는다. 이번엔 그 규칙대로 간다.

보조 모델은 `cand_aux8` 번들에서 재활용한다 — 반쪽 A 로 학습한 모델이 반쪽 B 를
예측하므로 **학습 없이** OOF 열이 복원된다. 주 모델 3시드만 새로 학습한다.

    .\venv_submit\Scripts\python.exe -u exp\build_aux1.py --out cand_aux1
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
ap.add_argument("--src", default="submissions/cand_aux8.zip")
ap.add_argument("--tpl", default="submissions/cand_rob2.zip")
ap.add_argument("--out", default="cand_aux1")
ap.add_argument("--keep", default="H")
ap.add_argument("--seeds", type=int, default=3)
a = ap.parse_args()
keep = [k for k in a.keep.split(",") if k]

out_pkl = os.path.join(ROOT, "model_cand", f"{a.out}.pkl")
out_zip = os.path.join(ROOT, "submissions", f"{a.out}.zip")
assert not os.path.exists(out_zip), f"이미 있다: {out_zip}"

with zipfile.ZipFile(os.path.join(ROOT, a.src)) as z:
    src = joblib.load(io.BytesIO(z.read("model/rf.pkl")))
with zipfile.ZipFile(os.path.join(ROOT, a.tpl)) as z:
    tpl = joblib.load(io.BytesIO(z.read("model/rf.pkl")))
BASE = list(src["aux"]["base"])
print(f"템플릿 {a.tpl} · 보조 모델 재활용 {a.src}")
print(f"  기본 {len(BASE)}열, 보조 라벨 {keep} (aux8 의 {list(src['aux']['models'])} 중)")

from path_alloc import build_df                                # noqa: E402
tr = build_df()
y = tr["control_success"].to_numpy(np.float64)
num = [c for c in BASE if c not in ba.ft.CAT_COLS]
X = tr[BASE].copy()
X[num] = X[num].astype(np.float32)
del tr; gc.collect()

half = np.arange(len(X)) % 2 == 0          # build_aux.py 와 동일한 분할
aux_models = {}
for nm in keep:
    ms = src["aux"]["models"][nm]           # ms[0]=half 로 학습, ms[1]=~half 로 학습
    o = np.full(len(X), np.nan, np.float32)
    t0 = time.time()
    o[~half] = ms[0].predict_proba(X[~half])[:, 1]
    o[half] = ms[1].predict_proba(X[half])[:, 1]
    X[f"aux_{nm}"] = o
    aux_models[nm] = ms
    print(f"  aux[{nm}] OOF 복원 {time.time()-t0:.0f}s (학습 없음)", flush=True)
    gc.collect()

feats = BASE + [f"aux_{nm}" for nm in keep]
print(f"주 모델 피처 {len(feats)}열")
models = []
for s in range(42, 42 + a.seeds):
    t0 = time.time()
    m = ba.pipeline(feats, s)
    m.fit(X, y.astype(int))
    models.append(m)
    print(f"  seed {s} 학습 {time.time()-t0:.0f}s", flush=True)
    gc.collect()
del X; gc.collect()

b = dict(tpl)
b["models"] = models
b["features"] = feats
b["spec"] = [f"cat-s{s}" for s in range(42, 42 + a.seeds)]
b["aux"] = {"base": BASE, "models": aux_models,
            "note": "학습셋 asof 증분으로 복원한 H=middle∨reverse 확률. 2겹 평균"}
b["note"] = (tpl.get("note", "") +
             f" | AUX 최소판본 {keep} — 로컬 2024 +0.643%. 46회차 aux8(+2.399%,"
             f" LB -7.79, 전이 -0.302) 의 최소 탐침")
joblib.dump(b, out_pkl, compress=3)
print(f"\n저장 {out_pkl} ({os.path.getsize(out_pkl)/1e6:.1f} MB)")
r = subprocess.run([sys.executable, os.path.join(ROOT, "make_submit.py"),
                    "--model", os.path.relpath(out_pkl, ROOT),
                    "--requirements", "requirements_cat.txt",
                    "--out", os.path.relpath(out_zip, ROOT)],
                   cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                   errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
print(r.stdout[-800:] if r.returncode == 0 else "FAILED\n"+r.stdout[-400:]+r.stderr[-1200:])
sys.exit(r.returncode)
