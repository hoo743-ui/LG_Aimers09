r"""마지막 제출 후보 — 플래툰 + (플래툰 x 카운트) 중첩 편차. 재학습 0회.

## 구성

    p += w1 * dev(투수 x 타자손 | 부모=투수)          w1 = 0.20   (15회차로 실측 확정)
    p += w2 * dev(투수 x 타자손 x 카운트 | 부모=플래툰) w2 = 0.20   (전이율 미측정)
    p' = center + alpha*(p - center) -> clip

**부모를 플래툰 셀로 둔 것이 핵심이다.** 부모를 투수로 두면 플래툰과 겹쳐서 증분이
줄어든다 — 현행 후보 위에서 잰 증분이 부모=투수 +2.97 vs 부모=플래툰 **+4.92** 다.

## 위험

`w1` 은 LB 두 점(14·15회차)으로 정확히 풀렸다. `w2` 는 **전이율을 못 쟀다** —
플래툰의 0.7685 를 빌려 쓴 추정이다. 셀이 잘아(중앙값 27행) 축소를 강하게(k=800)
걸었고 그만큼 가중이 커야 하는데, 신호가 안 넘어오면 분산만 늘어 손해다.

    k2 = 0.00 -> 943.24   k2 = 0.40 -> 946.14   k2 = 0.77 -> 948.82   k2 = 1.00 -> 950.50

기준선은 15회차 실측 **946.3826** 이다.

    .\.venv\Scripts\python.exe exp\make_final.py
"""
import json
import os
import subprocess
import sys

import joblib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "exp", "cache")
BASE_PKL = os.path.join(ROOT, "model_cand", "grid_affine_solved.pkl")
OUT_PKL = os.path.join(ROOT, "model_cand", "cat_plat_cnt.pkl")
OUT_ZIP = os.path.join(ROOT, "submissions", "cand_plat_cnt.zip")

W1, K1 = 0.20, 300          # 플래툰
W2, K2 = 0.20, 800          # 플래툰 x 카운트 (부모 = 플래툰 셀)
ALPHA, CENTER = 1.090437, 0.620268     # 중앙 추정으로 푼 아핀
R_EVAL, M_EVAL = 0.460900, 0.473942


def nested_dev(parent, child, y, k):
    """child 셀 평균 - parent 셀 평균, n/(n+k) 축소. 학습 구간 전체로 만든다."""
    o = np.argsort(child, kind="stable")
    Ys, Ps, Cs = y[o], parent[o], child[o]
    u, s = np.unique(Cs, return_index=True)
    cnt = np.diff(np.append(s, len(Cs)))
    cell = np.add.reduceat(Ys, s) / cnt
    par = Ps[s]
    op = np.argsort(parent, kind="stable")
    Yp, Pp = y[op], parent[op]
    pu, ps = np.unique(Pp, return_index=True)
    pc = np.diff(np.append(ps, len(Pp)))
    pmean = np.add.reduceat(Yp, ps) / pc
    dev = cnt * (cell - pmean[np.searchsorted(pu, par)]) / (cnt + k)
    return u, dev, cnt


def main():
    meta = json.load(open(f"{CACHE}/cols.json"))
    ix = {c: i for i, c in enumerate(meta["cols"])}
    X = np.load(f"{CACHE}/X.npy", mmap_mode="r")
    y = np.load(f"{CACHE}/y.npy").astype(np.float64)

    def col(n):
        return np.asarray(X[:, ix[n]], dtype=np.float64).astype(np.int64)

    P, BH = col("pitcher_id"), col("batter_hand")
    CNT = col("balls_before") * 4 + col("strikes_before")
    PH = P * 10 + BH

    u1, d1, c1 = nested_dev(P, PH, y, K1)
    tab1 = {(int(k // 10), int(k % 10)): float(v) for k, v in zip(u1, d1)}
    u2, d2, c2 = nested_dev(PH, PH * 100 + CNT, y, K2)
    # 조회 키는 test.csv 에 실제로 있는 컬럼이어야 한다. cnt12 는 파생이므로
    # balls/strikes 로 되돌려 4개 키로 담는다.
    tab2 = {(int(k // 1000), int((k // 100) % 10),
             int((k % 100) // 4), int(k % 4)): float(v)
            for k, v in zip(u2, d2)}
    print(f"플래툰       {len(tab1):,}칸  중앙 표본 {int(np.median(c1)):,}  "
          f"편차폭 {d1.min():+.4f}~{d1.max():+.4f}")
    print(f"플래툰x카운트 {len(tab2):,}칸  중앙 표본 {int(np.median(c2)):,}  "
          f"편차폭 {d2.min():+.4f}~{d2.max():+.4f}")

    B = CENTER * (1 - ALPHA)
    print(f"\nalpha={ALPHA:.6f} center={CENTER:.6f}  "
          f"절편 조건 A*m+B={ALPHA*M_EVAL+B:.6f} (r={R_EVAL:.6f}, "
          f"차 {ALPHA*M_EVAL+B-R_EVAL:+.6f})")

    b = dict(joblib.load(BASE_PKL))
    b["alpha"] = float(ALPHA)
    b["center"] = float(CENTER)
    b["platoon"] = [
        {"w": W1, "cols": ["pitcher_id", "batter_hand"], "table": tab1,
         "note": f"dev(투수x타자손 | 부모=투수), n/(n+{K1})"},
        {"w": W2, "cols": ["pitcher_id", "batter_hand", "balls_before",
                  "strikes_before"],
         "table": tab2,
         "note": f"dev(투수x타자손x카운트 | 부모=플래툰셀), n/(n+{K2})"},
    ]
    b["note"] = (f"catboost ensemble; p += {W1}*plat + {W2}*plat_x_count "
                 f"-> center+{ALPHA:.6f}*(p-center) -> clip(0,1)")
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
    print(r.stdout[-500:] if r.returncode == 0 else
          r.stdout[-400:] + r.stderr[-400:])


if __name__ == "__main__":
    main()
