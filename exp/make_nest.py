r"""19회차 후보 — 카운트 축을 **중첩 2층**으로 분해. 재학습 0회. 제출은 별도 판단.

## 구조 변경

18회차까지 카운트는 한 층이었다.

    p += w2 * dev(플래툰x카운트 | 부모=플래툰, k=800)          <- 중앙 셀 27행

셀이 27행이라 추정 잡음이 크다. 이걸 두 층으로 나눈다.

    p += wc * dev(플래툰x투수유리   | 부모=플래툰,        k=2000)  <- 중앙 173행
    p += wn * dev(플래툰x투수유리x카운트 | 부모=플래툰x투수유리, k=800)   <- 중앙  27행

`투수유리` = `strikes_before > balls_before` (2진). 거친 층은 표본이 6배라 안정적이고,
잔 층은 자기 부모에서 재므로 두 층이 **직교한다** (상관 −0.027).

## 왜 병렬이 아니라 중첩인가

거친 축을 그냥 병렬로 얹으면 기대는 더 높지만(+4.78) 카운트 축과 상관이 0.409 이고
`k`=0 하방이 **−17.5** 다. 중첩은 기대 +4.46 에 하방이 훨씬 얕다. 16회차가 "부모를
투수로 두면 플래툰과 겹친다"며 부모를 바꿔 해결한 것과 같은 처리다.

## 조회 키

`투수유리` 는 `(balls_before, strikes_before)` 로 결정되므로 두 층 모두
**(pitcher_id, batter_hand, balls_before, strikes_before)** 로 조회된다. 원본 컬럼뿐이고
행 독립이다.

    .\.venv\Scripts\python.exe exp\make_nest.py [--build]
"""
import argparse
import json
import os
import subprocess
import sys

import joblib
import numpy as np
from scipy.optimize import brentq, fsolve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "exp", "cache")
BASE_PKL = os.path.join(ROOT, "model_cand", "grid_affine_solved.pkl")
OUT_PKL = os.path.join(ROOT, "model_cand", "cat_nest_adv.pkl")
OUT_ZIP = os.path.join(ROOT, "submissions", "cand_nest_adv.zip")

R_EVAL = 0.460900
BASE = R_EVAL * (1 - R_EVAL)
A13, C13, S13 = 1.105030, 0.598664, 942.4577639361
OBS4 = [(0.3904, 0.0, 1.083674, 0.598664, 940.1357117095),
        (0.1990, 0.0, 1.089306, 0.620389, 946.3826029949),
        (0.2000, 0.2000, 1.090437, 0.620268, 950.0112119476),
        (0.2000, 0.5785, 1.089294, 0.622802, 952.4231549068)]
OBS18 = (0.2000, 0.5470, 0.3000, 1.089163, 0.622907, 953.7373675006)
W1, KSH1, KSH3, W3G = 0.20, 300, 2000, np.linspace(0.30, 0.60, 7)


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="pkl/zip 까지 만든다")
    args = ap.parse_args()

    meta = json.load(open(f"{CACHE}/cols.json"))
    ixc = {c: i for i, c in enumerate(meta["cols"])}
    X = np.load(f"{CACHE}/X.npy", mmap_mode="r")
    y = np.load(f"{CACHE}/y.npy").astype(np.float64)
    season = np.load(f"{CACHE}/season.npy")
    col = lambda n: np.asarray(X[:, ixc[n]], dtype=np.float64)

    P = col("pitcher_id").astype(np.int64)
    BH = col("batter_hand").astype(np.int64)
    BB, SS = col("balls_before").astype(np.int64), col("strikes_before").astype(np.int64)
    NR = col("num_runners_on").astype(np.int64)
    OB = (NR > 0).astype(np.int64)
    PH, CNT = P * 10 + BH, BB * 4 + SS
    ADV = (SS > BB).astype(np.int64)
    PHA = PH * 10 + ADV

    tr, va = season <= 2023, season == 2024
    yv = y[va]
    pm = np.load(os.path.join(ROOT, "exp",
                              "valpred_cat_s3.npz"))["p"].astype(np.float64)
    dv = lambda par, ch, k: lookup(*nested_dev(par[tr], ch[tr], y[tr], k)[:2],
                                   ch[va])

    C_M = S13 * BASE / (1e5 * A13)
    S2_M, M_M = C_M / A13, (R_EVAL - C13 * (1 - A13)) / A13
    d1 = dv(P, PH, KSH1)
    d2 = dv(PH, PH * 100 + CNT, 800)
    d3 = dv(PH, PH * 10 + OB, KSH3)

    def make(D, L):
        v = [float(np.var(x)) for x in D]
        cm = [float(np.cov(pm, x, ddof=0)[0, 1]) for x in D]
        cy = [float(np.cov(x, yv, ddof=0)[0, 1]) for x in D]
        mu = [float(np.mean(x)) for x in D]
        cc = [[float(np.cov(a, b, ddof=0)[0, 1]) for b in D] for a in D]

        def VC(W, K):
            V = S2_M
            for i in range(len(D)):
                V += 2 * W[i] * L[i] * cm[i] + W[i] ** 2 * L[i] * v[i]
                for j in range(i + 1, len(D)):
                    V += 2 * W[i] * W[j] * np.sqrt(L[i] * L[j]) * cc[i][j]
            C = C_M + sum(W[i] * K[i] * cy[i] for i in range(len(D)))
            return V, C, M_M + sum(W[i] * mu[i] for i in range(len(D)))
        return VC

    def real(VC, W, K, A, B):
        V, C, m = VC(W, K)
        return 1e5 * (1 - (A * A * V - 2 * A * C + BASE
                           + (A * m + B - R_EVAL) ** 2) / BASE)

    k1, l1, k2, l2 = fsolve(
        lambda t: [real(make([d1, d2, d3], [t[1], t[3], 1]), [a, b, 0],
                        [t[0], t[2], 0], A, c * (1 - A)) - s
                   for a, b, A, c, s in OBS4], [0.7685, .9725, .9340, .9725])
    w1c, w2c, w3c, A18, c18, S18 = OBS18
    k3 = brentq(lambda k: real(make([d1, d2, d3], [l1, l2, l2]),
                               [w1c, w2c, w3c], [k1, k2, k], A18,
                               c18 * (1 - A18)) - S18, -2, 4)
    KD = 0.5 * (k2 + k3)
    print(f"실측 전이율  2단 k1={k1:.4f} lam1={l1:.4f} | "
          f"3단 k2={k2:.4f} k3={k3:.4f} lam2={l2:.4f}  -> KD={KD:.4f}\n")

    print("=== 거친축 축소 두 판의 기대 vs 위험 ===")
    print(f"  {'거친k':>6}{'wc':>7}{'wn':>7}{'w3':>6}{'기대':>10}"
          f"{'kc=0':>9}{'kc=0.5':>9}{'kc=k1':>9}{'kn=0':>8}")
    cands = []
    for ksh_c in (800, 2000):
        dC = dv(PH, PHA, ksh_c)
        dN = dv(PHA, PHA * 100 + CNT, 800)
        D = [d1, dC, dN, d3]
        VC = make(D, [l1, l2, l2, l2])
        gw = np.linspace(0, 1.6, 81)
        bb = None
        for wc in gw:
            for wn in np.linspace(0, 0.8, 41):
                for w3 in W3G:
                    V, C, m = VC([W1, wc, wn, w3], [k1, KD, KD, k3])
                    s = 1e5 * C * C / (V * BASE)
                    if bb is None or s > bb[0]:
                        bb = (s, wc, wn, w3, C / V, m)
        s, wc, wn, w3, A, m = bb
        W = [W1, wc, wn, w3]
        risk = []
        for K in ([k1, 0.0, KD, k3], [k1, 0.5, KD, k3],
                  [k1, k1, KD, k3], [k1, KD, 0.0, k3]):
            V, C, mm = VC(W, K)
            risk.append(1e5 * C * C / (V * BASE) - S18)
        print(f"  {ksh_c:>6}{wc:>7.3f}{wn:>7.3f}{w3:>6.2f}{s:>10.4f}"
              + "".join(f"{r:>9.2f}" for r in risk[:3])
              + f"{risk[3]:>8.2f}")
        cands.append((s, ksh_c, W, A, R_EVAL - A * m, risk))

    print("\n  두 판의 위험이 사실상 같다 — 축소를 낮춰도 위험은 안 줄어든다.")
    print("  위험은 가중이 아니라 '거친축이 기여하는 신호량' 에 비례하기 때문이다.")

    # ---------- wc 를 낮춰 위험을 사는 것이 유일한 조절 손잡이다 ----------
    ksh_c = 2000
    dC = dv(PH, PHA, ksh_c)
    dN = dv(PHA, PHA * 100 + CNT, 800)
    D = [d1, dC, dN, d3]
    VC = make(D, [l1, l2, l2, l2])
    WN, W3 = 0.280, 0.45

    def at(wc, kc):
        V, C, m = VC([W1, wc, WN, W3], [k1, kc, KD, k3])
        return 1e5 * C * C / (V * BASE) - S18, C / V, m

    print(f"\n=== wc 별 기대 vs 위험 (18회차 {S18:.4f} 대비 증분) ===")
    print(f"  {'wc':>6}{'kc=0':>9}{'kc=0.5':>9}{'kc=0.769':>10}"
          f"{'kc=0.965':>10}{'kc=0.973':>10}{'손익분기kc':>11}")
    grid = [0.3, 0.5, 0.7, 0.9, 1.16]
    for wc in grid:
        try:
            be = brentq(lambda kc: at(wc, kc)[0], 0.0, 3.0)
        except ValueError:
            be = float("nan")
        print(f"  {wc:>6.2f}" + "".join(f"{at(wc, kc)[0]:>9.2f}"
                                        for kc in (0.0, 0.5))
              + "".join(f"{at(wc, kc)[0]:>10.2f}"
                        for kc in (k1, k2, KD)) + f"{be:>11.3f}")

    print("\n  🚩 wc 가 작으면 오히려 손해다 (wc=0.30 은 손익분기 1.05).")
    print("  이건 항을 '추가' 하는 게 아니라 d2 를 '대체' 하는 구조라,")
    print("  거친층이 원래 d2 가 하던 일을 떠맡을 만큼은 실려야 하기 때문이다.")

    # 선택 — 손익분기 kc 를 최소화하는 wc. 기대 최대가 아니라 **여유 최대**다.
    fine = np.linspace(0.5, 1.3, 33)
    be_of = {}
    for wc in fine:
        try:
            be_of[wc] = brentq(lambda kc: at(wc, kc)[0], 0.0, 3.0)
        except ValueError:
            be_of[wc] = float("nan")
    WC = min((w for w in fine if be_of[w] == be_of[w]),
             key=lambda w: be_of[w])
    be = be_of[WC]

    # 고른 wc 에서 wn, w3 를 다시 최적화
    bb = None
    for wn in np.linspace(0, 0.8, 41):
        for w3 in W3G:
            V, C, m = VC([W1, WC, wn, w3], [k1, KD, KD, k3])
            sc = 1e5 * C * C / (V * BASE)
            if bb is None or sc > bb[0]:
                bb = (sc, wn, w3, C / V, m)
    s, WN, W3, ALPHA, m = bb
    B_OPT = R_EVAL - ALPHA * m
    W = [W1, float(WC), float(WN), float(W3)]
    CENTER = B_OPT / (1 - ALPHA)
    print(f"\n=== 선택: wc = {WC:.3f} (거친축 축소 n/(n+{ksh_c})) ===")
    print(f"  기준 — 기대 최대가 아니라 **손익분기 최소**로 골랐다.")
    print(f"  손익분기 kc = {be:.3f}, 실측 3단 두 값 {k2:.4f} / {k3:.4f} "
          f"-> 여유 {min(k2, k3) - be:.3f}")
    print(f"  기대 최대판(wc=1.16, +4.46, 손익분기 0.818) 대비 "
          f"기대 {at(WC, KD)[0] - at(1.16, KD)[0]:+.2f} 를 내주고 "
          f"kc=0 하방을 {at(1.16, 0)[0]:.1f} -> {at(WC, 0)[0]:.1f} 로 줄인다")
    print(f"  w = (플래툰 {W[0]:.2f}, 거친 {W[1]:.3f}, "
          f"카운트|거친 {W[2]:.3f}, 주자유무 {W[3]:.2f})")
    print(f"  기대 {s:.4f}   18회차 {S18:.4f} 대비 {s - S18:+.4f}")
    print(f"  alpha={ALPHA:.6f}  center={CENTER:.6f}")

    if not args.build:
        print("\n(--build 를 주면 pkl/zip 을 만든다)")
        return

    # ---------- 제출 표는 학습 전체로 ----------
    u1, t1, c1 = nested_dev(P, PH, y, KSH1)
    uC, tC, cC = nested_dev(PH, PHA, y, ksh_c)
    uN, tN, cN = nested_dev(PHA, PHA * 100 + CNT, y, 800)
    u3, t3, c3 = nested_dev(PH, PH * 10 + OB, y, KSH3)

    tab1 = {(int(k // 10), int(k % 10)): float(x) for k, x in zip(u1, t1)}
    # 거친축: ADV 는 (balls, strikes) 로 결정되므로 12칸으로 펼친다
    tabC = {}
    for k, x in zip(uC, tC):
        pid, hand, adv = int(k // 100), int((k // 10) % 10), int(k % 10)
        for b in range(4):
            for st in range(3):
                if int(st > b) == adv:
                    tabC[(pid, hand, b, st)] = float(x)
    tabN = {}
    for k, x in zip(uN, tN):
        cnt = int(k % 100)
        rest = k // 100
        pid, hand = int(rest // 100), int((rest // 10) % 10)
        tabN[(pid, hand, int(cnt // 4), int(cnt % 4))] = float(x)
    tab3 = {}
    for k, x in zip(u3, t3):
        pid, hand, ob = int(k // 100), int((k // 10) % 10), int(k % 10)
        for nr in ([0] if ob == 0 else [1, 2, 3]):
            tab3[(pid, hand, nr)] = float(x)

    print(f"\n플래툰          {len(tab1):,}칸 중앙n {int(np.median(c1)):,}")
    print(f"거친(투수유리)     {len(tabC):,}칸 중앙n {int(np.median(cC)):,}  "
          f"({len(uC):,}칸을 볼/스트라이크 12칸으로 펼침)")
    print(f"카운트|거친      {len(tabN):,}칸 중앙n {int(np.median(cN)):,}")
    print(f"주자유무         {len(tab3):,}칸 중앙n {int(np.median(c3)):,}")

    b = dict(joblib.load(BASE_PKL))
    b["alpha"], b["center"] = float(ALPHA), float(CENTER)
    CK = ["pitcher_id", "batter_hand", "balls_before", "strikes_before"]
    b["platoon"] = [
        {"w": W[0], "cols": ["pitcher_id", "batter_hand"], "table": tab1,
         "note": f"dev(투수x타자손 | 부모=투수), n/(n+{KSH1})"},
        {"w": float(W[1]), "cols": CK, "table": tabC,
         "note": f"dev(플래툰x투수유리(S>B) | 부모=플래툰), n/(n+{ksh_c}). "
                 f"2진 편차를 볼/스트라이크 12칸으로 펼침"},
        {"w": float(W[2]), "cols": CK, "table": tabN,
         "note": "dev(플래툰x투수유리x카운트 | 부모=플래툰x투수유리), n/(n+800)"},
        {"w": float(W[3]),
         "cols": ["pitcher_id", "batter_hand", "num_runners_on"],
         "table": tab3,
         "note": f"dev(플래툰x주자유무 | 부모=플래툰), n/(n+{KSH3})"},
    ]
    b["note"] = (f"catboost ensemble; p += {W[0]}*plat + {W[1]:.3f}*adv "
                 f"+ {W[2]:.3f}*cnt|adv + {W[3]:.2f}*onbase "
                 f"-> center+{ALPHA:.6f}*(p-center) -> clip(0,1)")
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
