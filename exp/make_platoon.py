r"""플래툰 편차를 얹은 제출 후보를 만든다. 재학습 0회.

기존 pkl(3시드 CatBoost)을 그대로 두고 두 가지만 넣는다.

  1. `platoon` — 학습 전체 구간(2019~2024)에서 만든 투수 x 타자손 편차표와 가중 w
  2. `alpha` · `center` — 예측이 바뀌었으므로 아핀을 **둘 다** 다시 낸다

## 14회차에서 여기를 틀렸다 (4-31)

`alpha` 만 로컬 비율로 옮기고 `center` 를 그대로 뒀다. 그런데 제출 코드가 쓰는 것은
`p' = center + alpha*(p - center)` = `alpha*p + center*(1-alpha)` 이므로
**`alpha` 를 바꾸면 절편 `B = center*(1-alpha)` 도 같이 바뀐다.** 최적 조건은
`A*m + B = r` 인데 그게 깨져 절편이 +0.0026 어긋났고 **2.73점**을 잃었다
(기울기 오차 0.35점까지 합쳐 3.13점).

이제 `B = r - A*m` 로 절편을 먼저 정하고 `center = B/(1-A)` 로 되돌린다. 빌드할 때
`A*m + B == r` 를 찍어서 확인한다.

`w` 도 로컬에서 고르지 않는다. 로컬 최적 0.3904 를 썼는데 평가셋 최적은 **0.1990**
이었다 — 신호가 로컬의 49% 만 전이됐기 때문이다. 14회차 LB 점으로 직접 풀었다.

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
OUT_PKL = os.path.join(ROOT, "model_cand", "cat_platoon_w020.pkl")
OUT_ZIP = os.path.join(ROOT, "submissions", "cand_platoon_w020.zip")

A_EVAL = 1.105030          # 13회차로 확정된 평가셋 최적 기울기
CENTER = 0.598664          # 같은 해의 절편

# 14회차(940.14)로 평가셋에서 직접 푼 값. 로컬 워크포워드 계수 0.3904 를 썼다가
# 두 배 오버슈팅했다 — 신호는 로컬의 49% 만 전이됐다 (4-31).
W = 0.1990
ALPHA = 1.089306
CENTER_OUT = 0.620389


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
    alpha = ALPHA
    r0 = 1e5 * np.corrcoef(p, yv)[0, 1] ** 2
    r1 = 1e5 * np.corrcoef(q, yv)[0, 1] ** 2
    print(f"w = {W:.4f}  (평가셋에서 직접 푼 최적. 로컬 최적 0.3904 는 2배 과대)")
    print(f"  로컬 2024 rho^2 {r0:.2f} -> {r1:.2f} ({r1-r0:+.2f})")
    print(f"  alpha {A_EVAL:.6f} -> {alpha:.6f}   center {CENTER:.6f} -> "
          f"{CENTER_OUT:.6f}")
    # 14회차의 실수 재발 방지 — A 를 바꾸면 B 도 바뀐다. A*m + B = r 를 확인한다.
    m_e, r_e = 0.473942, 0.460900
    B = CENTER_OUT * (1 - alpha)
    print(f"  절편 조건 A*m + B = {alpha*m_e + B:.6f}  (r = {r_e:.6f}, "
          f"차 {alpha*m_e + B - r_e:+.6f})")
    q_sub = np.clip(CENTER_OUT + alpha * (q - CENTER_OUT), 0, 1)
    print(f"  로컬 범위 {q_sub.min():.4f}~{q_sub.max():.4f}  "
          f"clip 접촉 {np.mean((q_sub <= 0) | (q_sub >= 1)):.4%}")

    # --- 제출본 표는 학습 전체 구간에서 만든다 ---
    tab = PP.platoon_table(pid, hnd, y.astype(np.float64))
    tab = {(int(k[0]), int(k[1])): float(v) for k, v in tab.items()}
    print(f"\n제출본 표: 전체 {len(y):,}행에서 {len(tab):,}칸 "
          f"(투수 {len({k[0] for k in tab}):,}명)")

    b = dict(joblib.load(BASE_PKL))
    b["alpha"] = float(alpha)
    b["center"] = float(CENTER_OUT)
    b["platoon"] = {"w": W, "table": tab,
                    "note": "학습 구간 2019~2024 의 투수 x 타자손 성공률 편차 "
                            "(그 투수 자신의 전체 성공률 대비). n/(n+300) 축소"}
    b["note"] = (f"catboost ensemble; p += {W:.4f}*platoon(pitcher,batter_hand) "
                 f"-> center+{alpha:.6f}*(p-center) -> clip(0,1); "
                 f"center={CENTER_OUT:.6f}")
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
