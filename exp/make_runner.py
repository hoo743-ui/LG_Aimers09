r"""18회차 후보 — 플래툰 + 플래툰x카운트 + **플래툰x주자유무**. 재학습 0회.

## 무엇이 바뀌었나

17회차(952.4232)로 `w2` 곡선에 세 번째 점이 생겨 전이율 4개를 동시에 풀었다
(`solve_k2.py`). 3단 중첩의 실측 전이율이 `k2`=0.9651 로 **빌려온 0.9340 보다
높다.** 그 값으로 원장 5-b 가 접었던 축들을 다시 재니(`axis_scan.py`) 주자유무가
살아난다.

## 왜 주자유무인가

`risk_table.py` 의 폴드 검사 — 표를 `<f` 로 만들고 `f` 에서 `cov(d,y)` 를 쟀다.

    2021 +1.93e-6   2022 +4.79e-6   2023 +1.05e-5   2024 +1.10e-5   4/4 단조증가

`주자상태`(8단)보다 `주자유무`(2진)가 낫다 — 잘게 쪼개면 신호는 그대로인데 분산만
는다. 축소는 n/(n+2000).

## 왜 w3 = 0.30 인가

`k3` 는 이 축에서 미측정이고 카운트 축에서 빌려온다. 축 간 차용은 14·16 회차에서
두 번 다 빗나갔으므로 **고정 `w3` 위험표**로 고른다.

    진짜 k3      w3=0.15   w3=0.30   w3=0.45
    0.00          -0.62     -1.55     -2.79
    0.50          +0.10     -0.10     -0.61
    0.7689 (2단)  +0.49     +0.68     +0.56
    0.9651 (3단)  +0.78     +1.25     +1.41

0.45 는 상방 +0.16 더 얻자고 하방을 5배 키운다 — 14회차에서 데인 형태다.

## 조회 키

`num_runners_on` 은 원본 컬럼(0~3)이고 모델 features 에 있다. 2진 편차를 1/2/3 에
같은 값으로 펼쳐 담으면 **원본 컬럼만으로 행 독립 조회**가 된다.

    .\.venv\Scripts\python.exe exp\make_runner.py
"""
import json
import os
import subprocess
import sys

import joblib
import numpy as np
from scipy.optimize import fsolve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "exp", "cache")
BASE_PKL = os.path.join(ROOT, "model_cand", "grid_affine_solved.pkl")
OUT_PKL = os.path.join(ROOT, "model_cand", "cat_plat_cnt_runner.pkl")
OUT_ZIP = os.path.join(ROOT, "submissions", "cand_plat_cnt_runner.zip")

W1, KSH1 = 0.20, 300        # 플래툰            (14/15회차로 실측 확정)
KSH2 = 800                  # 플래툰x카운트      (16/17회차로 실측 확정)
W3, KSH3 = 0.30, 2000       # 플래툰x주자유무    (신규, 위험표로 고름)

R_EVAL = 0.460900
BASE = R_EVAL * (1 - R_EVAL)
A13, C13, S13 = 1.105030, 0.598664, 942.4577639361
OBS = [(0.3904, 0.0, 1.083674, 0.598664, 940.1357117095),
       (0.1990, 0.0, 1.089306, 0.620389, 946.3826029949),
       (0.2000, 0.2000, 1.090437, 0.620268, 950.0112119476),
       (0.2000, 0.5785, 1.089294, 0.622802, 952.4231549068)]


def nested_dev(parent, child, y, k):
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
    return u, cnt * (cell - pmean[np.searchsorted(pu, par)]) / (cnt + k), cnt


def lookup(u, dev, keys):
    out = np.zeros(len(keys), dtype=np.float64)
    ix = np.clip(np.searchsorted(u, keys), 0, len(u) - 1)
    ok = u[ix] == keys
    out[ok] = dev[ix[ok]]
    return out


def main():
    meta = json.load(open(f"{CACHE}/cols.json"))
    ixc = {c: i for i, c in enumerate(meta["cols"])}
    X = np.load(f"{CACHE}/X.npy", mmap_mode="r")
    y = np.load(f"{CACHE}/y.npy").astype(np.float64)
    season = np.load(f"{CACHE}/season.npy")

    def col(n):
        return np.asarray(X[:, ixc[n]], dtype=np.float64)

    P = col("pitcher_id").astype(np.int64)
    BH = col("batter_hand").astype(np.int64)
    CNT = (col("balls_before") * 4 + col("strikes_before")).astype(np.int64)
    NR = col("num_runners_on").astype(np.int64)
    OB = (NR > 0).astype(np.int64)
    PH, PHC, PHO = P * 10 + BH, (P * 10 + BH) * 100 + CNT, (P * 10 + BH) * 10 + OB

    # ---------- 1) 전이율 동시 해 (로컬 = 2019~2023 표 -> 2024 폴드) ----------
    tr, va = season <= 2023, season == 2024
    yv = y[va]
    pm = np.load(os.path.join(ROOT, "exp",
                              "valpred_cat_s3.npz"))["p"].astype(np.float64)

    def dev24(parent, child, k):
        u, d, _ = nested_dev(parent[tr], child[tr], y[tr], k)
        return lookup(u, d, child[va])

    d1, d2, d3 = (dev24(P, PH, KSH1), dev24(PH, PHC, KSH2),
                  dev24(PH, PHO, KSH3))
    C_M = S13 * BASE / (1e5 * A13)
    S2_M, M_M = C_M / A13, (R_EVAL - C13 * (1 - A13)) / A13
    D = [d1, d2, d3]
    v = [float(np.var(x)) for x in D]
    cm = [float(np.cov(pm, x, ddof=0)[0, 1]) for x in D]
    cy = [float(np.cov(x, yv, ddof=0)[0, 1]) for x in D]
    mu = [float(np.mean(x)) for x in D]
    cc = [[float(np.cov(a, b, ddof=0)[0, 1]) for b in D] for a in D]

    def sc(w1, w2, A, B, k1, l1, k2, l2):
        lx = np.sqrt(max(l1 * l2, 1e-12))
        V = (S2_M + 2 * w1 * l1 * cm[0] + w1 * w1 * l1 * v[0]
             + 2 * w2 * l2 * cm[1] + w2 * w2 * l2 * v[1]
             + 2 * w1 * w2 * lx * cc[0][1])
        C = C_M + w1 * k1 * cy[0] + w2 * k2 * cy[1]
        m = M_M + w1 * mu[0] + w2 * mu[1]
        return 1e5 * (1 - (A * A * V - 2 * A * C + BASE
                           + (A * m + B - R_EVAL) ** 2) / BASE)

    k1, l1, k2, l2 = fsolve(
        lambda t: [sc(a, b, A, c * (1 - A), *t) - s for a, b, A, c, s in OBS],
        [0.7685, 0.9725, 0.9340, 0.9725])
    print(f"실측 전이율  2단 k={k1:.4f} lam={l1:.4f} | "
          f"3단 k={k2:.4f} lam={l2:.4f}")

    # ---------- 2) w3 고정, w2 와 아핀을 다시 푼다 ----------
    def model(w1, w2, w3, k3):
        W, K, L = [w1, w2, w3], [k1, k2, k3], [l1, l2, l2]
        V = S2_M
        for i in range(3):
            V += 2 * W[i] * L[i] * cm[i] + W[i] ** 2 * L[i] * v[i]
            for j in range(i + 1, 3):
                V += 2 * W[i] * W[j] * np.sqrt(L[i] * L[j]) * cc[i][j]
        C = C_M + sum(W[i] * K[i] * cy[i] for i in range(3))
        m = M_M + sum(W[i] * mu[i] for i in range(3))
        A = C / V
        return 1e5 * C * C / (V * BASE), A, R_EVAL - A * m, m

    g = np.linspace(0.30, 0.80, 501)
    w2 = float(g[int(np.argmax([model(W1, b, W3, k2)[0] for b in g]))])
    s_exp, ALPHA, B_OPT, m_exp = model(W1, w2, W3, k2)
    CENTER = B_OPT / (1 - ALPHA)
    cur = OBS[-1][-1]
    print(f"w2 재최적화 {w2:.4f} (17회차 0.5785)   w3 고정 {W3}")
    print(f"기대 {s_exp:.4f}  (17회차 실측 {cur:.4f} 대비 {s_exp - cur:+.4f})")
    print(f"alpha={ALPHA:.6f}  center={CENTER:.6f}  "
          f"절편 A*m+B-r = {ALPHA * m_exp + B_OPT - R_EVAL:+.2e}")

    print("\n위험표 — w3 고정, 진짜 k3 에 따른 17회차 대비 증분")
    for k3 in (0.0, 0.25, 0.50, k1, k2, 1.20):
        tag = ("  <- 2단 실측" if abs(k3 - k1) < 1e-9 else
               "  <- 3단 실측 (기대값)" if abs(k3 - k2) < 1e-9 else "")
        print(f"  k3={k3:6.4f}   {model(W1, w2, W3, k3)[0] - cur:+8.3f}{tag}")

    # ---------- 3) 제출 표는 학습 전체(2019~2024)로 만든다 ----------
    u1, t1, c1 = nested_dev(P, PH, y, KSH1)
    u2, t2, c2 = nested_dev(PH, PHC, y, KSH2)
    u3, t3, c3 = nested_dev(PH, PHO, y, KSH3)
    tab1 = {(int(k // 10), int(k % 10)): float(x) for k, x in zip(u1, t1)}
    tab2 = {(int(k // 1000), int((k // 100) % 10),
             int((k % 100) // 4), int(k % 4)): float(x) for k, x in zip(u2, t2)}
    # 2진 편차를 num_runners_on 0/1/2/3 으로 펼친다 (1,2,3 은 같은 값)
    tab3 = {}
    for k, x in zip(u3, t3):
        pid, hand, ob = int(k // 100), int((k // 10) % 10), int(k % 10)
        for nr in ([0] if ob == 0 else [1, 2, 3]):
            tab3[(pid, hand, nr)] = float(x)
    print(f"\n플래툰        {len(tab1):,}칸 중앙n {int(np.median(c1)):,}")
    print(f"플래툰x카운트  {len(tab2):,}칸 중앙n {int(np.median(c2)):,}")
    print(f"플래툰x주자유무 {len(tab3):,}칸 중앙n {int(np.median(c3)):,}  "
          f"(2진 {len(u3):,}칸을 0/1/2/3 으로 펼침)")

    b = dict(joblib.load(BASE_PKL))
    b["alpha"], b["center"] = float(ALPHA), float(CENTER)
    b["platoon"] = [
        {"w": W1, "cols": ["pitcher_id", "batter_hand"], "table": tab1,
         "note": f"dev(투수x타자손 | 부모=투수), n/(n+{KSH1})"},
        {"w": float(w2),
         "cols": ["pitcher_id", "batter_hand", "balls_before", "strikes_before"],
         "table": tab2,
         "note": f"dev(투수x타자손x카운트 | 부모=플래툰), n/(n+{KSH2})"},
        {"w": W3, "cols": ["pitcher_id", "batter_hand", "num_runners_on"],
         "table": tab3,
         "note": f"dev(투수x타자손x주자유무 | 부모=플래툰), n/(n+{KSH3}). "
                 f"2진 편차를 주자수 1/2/3 에 동일 적용"},
    ]
    b["note"] = (f"catboost ensemble; p += {W1}*plat + {w2:.4f}*plat_x_count "
                 f"+ {W3}*plat_x_onbase -> center+{ALPHA:.6f}*(p-center) "
                 f"-> clip(0,1)")
    joblib.dump(b, OUT_PKL, compress=3)
    print(f"저장: {OUT_PKL} ({os.path.getsize(OUT_PKL) / 1e6:.1f} MB)")

    r = subprocess.run(
        [sys.executable, os.path.join(ROOT, "make_submit.py"),
         "--model", os.path.relpath(OUT_PKL, ROOT),
         "--requirements", "requirements_cat.txt",
         "--out", os.path.relpath(OUT_ZIP, ROOT)],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
        errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    print(r.stdout[-600:] if r.returncode == 0
          else r.stdout[-400:] + r.stderr[-600:])


if __name__ == "__main__":
    main()
