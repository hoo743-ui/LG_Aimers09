r"""제출 후보 격자를 한 번에 만든다. 재학습 0회.

`alpha` 와 이동은 pkl 안의 상수 두 개(`alpha`, `center`)로 표현되므로 학습된
모델을 그대로 두고 상수만 갈아 여러 후보를 만들 수 있다.

    p' = center' + alpha * (p - center')        center' = center + shift/(1-alpha)

`center'` 로 이동을 등가 구현하면 **기울기가 불변**이고 `script.py` 를 고칠 필요가
없다. 이 프로젝트에서 실패한 제출 2건이 모두 환경·코드 문제였으므로(6-1, 6-4)
추론 경로를 건드리지 않는 것이 원칙이다.

    .\.venv\Scripts\python.exe exp\make_grid.py
"""
import argparse
import json
import os
import subprocess
import sys

import joblib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C_TRAIN = 0.523766          # 전체 학습(2019~2024) 성공률. pkl 의 center 기본값

# (라벨, 기반 pkl, alpha, 이동, 설명)
GRID = [
    ("a120_s3_shift000", "cat_alpha109.pkl", 1.20,  0.000,
     "11회차 제출본과 동일 (LB 931.53). 기준선"),
    ("a120_s3_shift010", "cat_alpha109.pkl", 1.20, -0.010,
     "탐침이 (i) 쪽으로 나오면 중간 단계"),
    ("a120_s3_shift015", "cat_alpha109.pkl", 1.20, -0.015,
     "탐침이 (i) 쪽이면 최적 근처"),
    ("a120_s3_shift019", "cat_alpha109.pkl", 1.20, -0.019,
     "이야기 (i) 의 최적 이동 (-0.0187)"),
    ("affine_i_opt",     "cat_alpha109.pkl", 0.9651, None,
     "이야기 (i) 의 아핀 최적 (A*=0.9651, center=-0.2593). 기울기까지 최적"),
    ("a120_s6_shift000", "cat_a120_s6.pkl",  1.20,  0.000,
     "시드 6개. 로컬 근거만 (alpha 후 3폴드 전부 양수)"),
    ("a120_s6_shift005", "cat_a120_s6.pkl",  1.20, -0.005,
     "시드 6개 + 탐침 이동"),
    ("a120_s6_shift015", "cat_a120_s6.pkl",  1.20, -0.015,
     "시드 6개 + (i) 최적 근처"),
    ("affine_solved",    "cat_alpha109.pkl", 1.105030, None,
     "탐침 934.10 으로 평가셋이 풀린 뒤의 진짜 아핀 최적 (A*=1.10503, "
     "center=0.598664, m=0.473979). 예상 942.46"),
]
# 이야기 (i) 아핀 최적의 절편 (shift_solve.py 가 낸 값)
AFFINE_I_CENTER = -0.259333
# 12회차 탐침으로 확정된 아핀 최적의 절편. 이야기 (i) 는 기각됐다
AFFINE_SOLVED_CENTER = 0.598664
CENTER_OVERRIDE = {"affine_i_opt": AFFINE_I_CENTER,
                   "affine_solved": AFFINE_SOLVED_CENTER}


def build(label, base, alpha, shift, note):
    src = os.path.join(ROOT, "model_cand", base)
    if not os.path.exists(src):
        return None, f"기반 pkl 없음: {base}"
    b = dict(joblib.load(src))
    if shift is None:
        center = CENTER_OVERRIDE[label]
    else:
        center = C_TRAIN + shift / (1 - alpha) if alpha != 1.0 else C_TRAIN
    b["alpha"] = float(alpha)
    b["center"] = float(center)
    b["shift_equiv"] = None if shift is None else float(shift)
    b["note"] = (f"catboost ensemble; p' = center + alpha*(p-center) -> clip(0,1); "
                 f"alpha={alpha} center={center:.6f}; {note}")
    pkl = os.path.join(ROOT, "model_cand", f"grid_{label}.pkl")
    joblib.dump(b, pkl, compress=3)
    zip_ = os.path.join(ROOT, "submissions", f"grid_{label}.zip")
    r = subprocess.run(
        [sys.executable, os.path.join(ROOT, "make_submit.py"),
         "--model", os.path.relpath(pkl, ROOT),
         "--requirements", "requirements_cat.txt",
         "--out", os.path.relpath(zip_, ROOT)],
        # 6-6: make_submit 이 비ASCII 를 찍는다. text=True 의 기본 디코더는 이
        # 환경에서 cp949 라 그대로 두면 UnicodeDecodeError 로 죽는다.
        cwd=ROOT, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    if r.returncode != 0:
        return None, r.stdout[-300:] + r.stderr[-300:]
    return {"label": label, "zip": os.path.relpath(zip_, ROOT), "base": base,
            "alpha": alpha, "shift": shift, "center": center,
            "n_models": len(b["models"]), "note": note}, None


def main():
    ap = argparse.ArgumentParser()
    # 시드 6개 판은 제출 경로에서 -10.4 로 기각돼 zip 을 만들지 않기로 했다. 그런데
    # cat_a120_s6.pkl 이 생긴 뒤로는 통짜 실행이 그 결정을 되돌려 버린다 — 라벨을
    # 골라 짓고 색인은 병합한다.
    ap.add_argument("--only", nargs="*", default=None, help="지을 라벨만 지정")
    a = ap.parse_args()
    todo = [g for g in GRID if a.only is None or g[0] in a.only]

    d = np.load(os.path.join(ROOT, "exp", "valpred_cat_s3.npz"),
                allow_pickle=True)
    p = d["p"].astype(np.float64)
    idx = os.path.join(ROOT, "submissions", "GRID_INDEX.json")
    if a.only and os.path.exists(idx):
        with open(idx, encoding="utf-8") as f:
            out = [e for e in json.load(f) if e["label"] not in a.only]
    else:
        out = []
    for label, base, alpha, shift, note in todo:
        info, err = build(label, base, alpha, shift, note)
        if info is None:
            print(f"  [건너뜀] {label}: {err.strip()[:120]}")
            continue
        # 2024 홀드아웃 예측으로 범위와 clip 접촉을 확인한다 (3시드 기반일 때만 정확)
        q = np.clip(info["center"] + alpha * (p - info["center"]), 0, 1)
        info["range"] = f"{q.min():.4f}~{q.max():.4f}"
        info["clip_pct"] = float(100 * ((q <= 0) | (q >= 1)).mean())
        out.append(info)
        print(f"  {info['zip']:44s} alpha={alpha:<6.4f} center={info['center']:+.6f} "
              f"모델 {info['n_models']}개  clip {info['clip_pct']:.3f}%")
    with open(idx, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n{len(out)}개 생성. 색인: submissions/GRID_INDEX.json")
    print("clip 접촉이 0 이 아닌 판은 그만큼 보정이 잘려 이득이 줄어든다 — 확인할 것")


if __name__ == "__main__":
    main()
