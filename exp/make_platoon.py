r"""플래툰 편차를 얹은 제출 후보를 만든다. 재학습 0회.

기존 pkl(3시드 CatBoost)을 그대로 두고 두 가지만 넣는다.

  1. `platoon` — 학습 전체 구간(2019~2024)에서 만든 투수 x 타자손 편차표와 가중 w
  2. `alpha`  — 예측이 바뀌었으므로 최적 기울기도 바뀐다. 비율로 옮긴다

## alpha 를 왜 비율로 옮기는가

평가셋의 최적 아핀은 `A = C/s^2` 인데 `C`,`s^2` 는 예측이 바뀌면 다시 미지수다
(4-25). 그러나 **로컬에서 잰 A 의 변화 비율**은 옮길 수 있다 — 수준이 아니라 비율은
전이가 잘 된다. 그리고 A 는 최적 근처에서 2차로 평평해서(0.05 틀려도 1.93점) 위험이
작다. `center` 는 편차의 평균이 0 이라 예측 평균이 -0.000042 밖에 안 움직여 그대로 둔다.

    .\.venv\Scripts\python.exe exp\make_platoon.py
"""
import json
import os
import subprocess
import sys

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import platoon_probe as PP                                # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "exp", "cache")

BASE_PKL = os.path.join(ROOT, "model_cand", "grid_affine_solved.pkl")
OUT_PKL = os.path.join(ROOT, "model_cand", "cat_platoon.pkl")
OUT_ZIP = os.path.join(ROOT, "submissions", "cand_platoon.zip")

A_EVAL = 1.105030          # 13회차로 확정된 평가셋 최적 기울기
CENTER = 0.598664          # 같은 해의 절편
# 워크포워드 계수 (폴드 2021/2022/2024 에서 각각 맞춘 값의 평균)
W = float(np.mean([0.3664, 0.3369, 0.4679]))


def main():
    meta = json.load(open(f"{CACHE}/cols.json"))
    ix = {c: i for i, c in enumerate(meta["cols"])}
    X = np.load(f"{CACHE}/X.npy", mmap_mode="r")
    y = np.load(f"{CACHE}/y.npy")
    season = np.load(f"{CACHE}/season.npy")
    pid = np.asarray(X[:, ix["pitcher_id"]], dtype=np.int64)
    hnd = np.asarray(X[:, ix["batter_hand"]], dtype=np.int64)

    # --- alpha 이전 비율은 2024 홀드아웃에서 잰다 (표는 2024 이전만 써서) ---
    tab_ho = PP.platoon_table(pid[season < 2024], hnd[season < 2024],
                              y[season < 2024].astype(np.float64))
    d = np.load(os.path.join(ROOT, "exp", "valpred_cat_s3.npz"),
                allow_pickle=True)
    p, yv = d["p"].astype(np.float64), d["y"].astype(np.float64)
    f_ho = PP.lookup(tab_ho, pid[season == 2024], hnd[season == 2024])
    q = p + W * f_ho
    a_old = np.cov(p, yv, bias=True)[0, 1] / p.var()
    a_new = np.cov(q, yv, bias=True)[0, 1] / q.var()
    alpha = A_EVAL * (a_new / a_old)
    r0 = 1e5 * np.corrcoef(p, yv)[0, 1] ** 2
    r1 = 1e5 * np.corrcoef(q, yv)[0, 1] ** 2
    print(f"w = {W:.4f}   로컬 2024 rho^2 {r0:.2f} -> {r1:.2f} "
          f"({r1-r0:+.2f}, 상대 {100*(r1/r0-1):+.2f}%)")
    print(f"alpha  {A_EVAL:.6f} -> {alpha:.6f}  (비율 {a_new/a_old:.6f})")
    print(f"center {CENTER:.6f} 유지 (예측 평균 이동 {q.mean()-p.mean():+.6f})")

    # --- 제출본 표는 학습 전체 구간에서 만든다 ---
    tab = PP.platoon_table(pid, hnd, y.astype(np.float64))
    tab = {(int(k[0]), int(k[1])): float(v) for k, v in tab.items()}
    print(f"\n제출본 표: 전체 {len(y):,}행에서 {len(tab):,}칸 "
          f"(투수 {len({k[0] for k in tab}):,}명)")

    b = dict(joblib.load(BASE_PKL))
    b["alpha"] = float(alpha)
    b["center"] = float(CENTER)
    b["platoon"] = {"w": W, "table": tab,
                    "note": "학습 구간 2019~2024 의 투수 x 타자손 성공률 편차 "
                            "(그 투수 자신의 전체 성공률 대비). n/(n+300) 축소"}
    b["note"] = (f"catboost ensemble; p += {W:.4f}*platoon(pitcher,batter_hand) "
                 f"-> center+{alpha:.6f}*(p-center) -> clip(0,1); "
                 f"center={CENTER:.6f}")
    joblib.dump(b, OUT_PKL, compress=3)
    print(f"저장: {OUT_PKL} ({os.path.getsize(OUT_PKL)/1e6:.1f} MB)")

    r = subprocess.run(
        [sys.executable, os.path.join(ROOT, "make_submit.py"),
         "--model", os.path.relpath(OUT_PKL, ROOT),
         "--requirements", "requirements_cat.txt",
         "--out", os.path.relpath(OUT_ZIP, ROOT)],
        cwd=ROOT, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    print(r.stdout[-600:] if r.returncode == 0 else
          r.stdout[-400:] + r.stderr[-400:])


if __name__ == "__main__":
    main()
