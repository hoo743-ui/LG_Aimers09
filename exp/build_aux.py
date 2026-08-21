r"""🚩 AUX 제출본 빌드 — 보조 라벨 확률 8열을 피처로 넣은 주 모델 (EXP049).

## 무엇을 하는가

    보조 타깃 8종  middle · reverse · ball · strike · fastball · breaking ·
                   offspeed · H(=middle∨reverse)
    각각 2겹으로 학습 -> 학습 행에는 OOF, 추론 행에는 두 모델 평균
    주 모델 3시드를 90열(82 + aux 8)로 학습

표(편차4 · 차등3 · 수준2) · 가중 9개 · alpha · center · asof_prior 는
**Champion `cand_rob2` 에서 그대로 가져온다.** 바뀌는 것은 모델과 피처뿐이다.

## 아핀을 재계산하지 않는 이유

`rho` 는 아핀 불변이고 clip 에 닿지 않으면 항등식이다 (원장 §16). rob2 의 확률
범위가 0.3771~0.5641 로 clip 에서 멀다. 빌드 후 범위를 실측해 확인한다.

## 규정 4

보조 모델은 학습 데이터만으로 만들어졌고 입력은 그 행 자신의 피처뿐이다.
라벨 복원(`asof_pitcher_n` 증분)은 **학습 단계에서만** 일어난다.
빌드 후 `exp/rowindep.py` 로 단독행 대 배치 동일성을 실측한다.

    .\venv_submit\Scripts\python.exe -u exp\build_aux.py --out cand_aux8
"""
import argparse
import gc
import io
import os
import subprocess
import sys
import time
import zipfile

import joblib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "exp"))
sys.path.insert(0, os.path.join(ROOT, "research"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import build_asof as ba                                       # noqa: E402
from exp046_pitchmix import pitch_events                      # noqa: E402

TARGETS = ["middle", "reverse", "ball", "strike", "fb", "br", "os", "H"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="submissions/cand_rob2.zip")
    ap.add_argument("--out", default="cand_aux8")
    ap.add_argument("--targets", default=",".join(TARGETS))
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    tg = [t for t in a.targets.split(",") if t]

    out_pkl = os.path.join(ROOT, "model_cand", f"{a.out}.pkl")
    out_zip = os.path.join(ROOT, "submissions", f"{a.out}.zip")
    assert not os.path.exists(out_zip), f"이미 있다: {out_zip} (덮어쓰지 않는다)"

    with zipfile.ZipFile(os.path.join(ROOT, a.src)) as z:
        tpl = joblib.load(io.BytesIO(z.read("model/rf.pkl")))
    BASE = list(tpl["features"])
    print(f"템플릿 {a.src}  기본 {len(BASE)}열  alpha {tpl['alpha']} "
          f"center {tpl['center']:.6f}")
    print("  가중 9개 =", [round(d["w"], 6) for d in tpl["platoon"]])

    ev = pitch_events()
    from path_alloc import build_df
    tr = build_df()
    y = tr["control_success"].to_numpy(np.float64)
    lab = {k: np.nan_to_num(ev[k]) for k in ("middle", "reverse", "ball",
                                             "strike", "fb", "br", "os")}
    lab["H"] = ((lab["middle"] > .5) | (lab["reverse"] > .5)).astype(float)
    good = ~np.isnan(ev["middle"])
    print(f"  생산 프레임 {tr.shape}  라벨 복원 가능 {good.sum():,} "
          f"({good.mean():.4%})")

    num = [c for c in BASE if c not in ba.ft.CAT_COLS]
    X = tr[BASE].copy()
    X[num] = X[num].astype(np.float32)
    del tr
    gc.collect()

    idx = np.arange(len(X))
    half = idx % 2 == 0                       # 실험과 동일한 2겹 분할
    aux_models, oof = {}, {}
    for nm in tg:
        t = lab[nm]
        ms, o = [], np.full(len(X), np.nan, np.float32)
        for hs in (half, ~half):
            fit = hs & good
            t0 = time.time()
            m = ba.pipeline(BASE, 42)
            m.fit(X[fit], t[fit].astype(int))
            o[~hs] = m.predict_proba(X[~hs])[:, 1]
            ms.append(m)
            print(f"  aux[{nm:7s}] 반쪽 {int(fit.sum()):,}행 "
                  f"{time.time()-t0:.0f}s", flush=True)
            gc.collect()
        aux_models[nm], oof[nm] = ms, o
    for nm in tg:
        X[f"aux_{nm}"] = oof[nm]
    feats = BASE + [f"aux_{nm}" for nm in tg]
    print(f"\n주 모델 피처 {len(feats)}열")

    models = []
    for s in range(42, 42 + a.seeds):
        t0 = time.time()
        m = ba.pipeline(feats, s)
        m.fit(X, y.astype(int))
        models.append(m)
        print(f"  seed {s} 학습 {time.time()-t0:.0f}s", flush=True)
        gc.collect()
    del X
    gc.collect()

    b = dict(tpl)
    b["models"] = models
    b["features"] = feats
    b["spec"] = [f"cat-s{s}" for s in range(42, 42 + a.seeds)]
    b["aux"] = {"base": BASE, "models": aux_models,
                "note": "학습셋 asof 증분으로 복원한 투구 단위 결과 확률. "
                        "2겹 평균. 입력은 그 행 자신의 피처뿐"}
    b["note"] = (tpl.get("note", "") +
                 f" | AUX {len(tg)}열 {tg} — 보조 라벨 확률을 피처로. "
                 f"EXP049 폴드2024 3시드 +22.53 (+2.399%), in-sample 상승률이 "
                 f"홀드아웃보다 낮아 과적합 서명 없음")
    os.makedirs(os.path.dirname(out_pkl), exist_ok=True)
    joblib.dump(b, out_pkl, compress=3)
    print(f"\n저장 {out_pkl} ({os.path.getsize(out_pkl)/1e6:.1f} MB)")

    r = subprocess.run(
        [sys.executable, os.path.join(ROOT, "make_submit.py"),
         "--model", os.path.relpath(out_pkl, ROOT),
         "--requirements", "requirements_cat.txt",
         "--out", os.path.relpath(out_zip, ROOT)],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
        errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    print(r.stdout[-1200:] if r.returncode == 0
          else "FAILED\n" + r.stdout[-600:] + r.stderr[-1500:])
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
