r"""EXP040 — 감쇠 하에서 **편차 4축의 가중과 축소 상수까지** 재최적화. 학습 0회.

## 왜 여기가 남았는가

EXP038 에서 나온 따름 정리 — **표의 정밀도와 최적 가중은 함께 움직인다.**
대비 축은 표가 좋아지자 최적 가중이 0.65 에서 0.80 으로 올라갔다.

그런데 나는 감쇠를 걸면서 **대비 가중만** 재조정했다. 나머지는 전부 옛 정밀도
기준의 값 그대로다.

    편차 4축   w = [0.20, 0.825, 0.28, 0.45]   k = [300, 2000, 800, 2000]
               둘 다 하드/균등 시절에 잡힌 값이고, 로컬 기여는 **+22.7** 로 최대다
    축소 k     "표가 얼마나 잡음인가"의 함수다. 표가 정확해졌으면 k 도 내려가야 한다

## 사슬

편차를 바꾸면 잔차가 바뀌고 그 위의 대비·수준 표가 전부 바뀐다. 그래서 매
평가마다 전체를 다시 세운다. 한 평가가 비싸므로 좌표하강을 쓴다.

    선택   2022 와 2024 의 평균 (2023 은 퇴화 폴드)
    제약   3폴드 전부 양수인 점만 후보로 남긴다

    .\.venv\Scripts\python.exe -u research\exp040_full.py
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "exp"))
import engine as E                                          # noqa: E402
import build_asof as ba                                     # noqa: E402
from path_alloc import build_df                             # noqa: E402
from traj_probe import r2                                   # noqa: E402
from exp038_decay_all import nested_dev_w                    # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

EXP = "EXP040"
SRC = {2022: (2020, 2021), 2023: (2020, 2021, 2022), 2024: (2020, 2021, 2022, 2023)}
FOLDS = (2022, 2023, 2024)


def main():
    E.start_experiment(EXP, "FULL", "python research/exp040_full.py", "load")
    tr = build_df()
    season = tr["season"].to_numpy()
    y = tr["control_success"].to_numpy(np.float64)
    P = tr["pitcher_id"].to_numpy(np.int64)
    B = tr["batter_id"].to_numpy(np.int64)
    BH = tr["batter_hand"].to_numpy(np.int64)
    BB = tr["balls_before"].to_numpy(np.int64)
    SS = tr["strikes_before"].to_numpy(np.int64)
    OB = (tr["num_runners_on"].to_numpy(np.int64) > 0).astype(np.int64)
    PH = P * 10 + BH
    PHA = PH * 10 + (SS > BB).astype(np.int64)
    AXD = [(P, PH), (PH, PHA), (PHA, PHA * 100 + (BB * 4 + SS)), (PH, PH * 10 + OB)]

    def g(c):
        return tr[c].to_numpy(np.float64)

    SAME = (g("pitcher_hand") == g("batter_hand")).astype(int)
    TWO = (SS == 2).astype(int)
    AXC = [(SAME, 1000.0), (TWO, 1000.0), (OB, 2000.0)]
    MODEL = {f: np.load(os.path.join(ROOT, "exp", f"prod_champ_{f}.npy"))[:2].mean(0)
             for f in (2020, 2021, 2022, 2023, 2024)}

    DEVCACHE = {}

    def devmat(f, gd, ksh):
        key = (f, gd, tuple(np.round(ksh, 3)))
        if key in DEVCACHE:
            return DEVCACHE[key]
        m_tr, m_va = season < f, season == f
        w = (np.ones(int(m_tr.sum())) if gd is None
             else gd ** (f - 1 - season[m_tr].astype(float)))
        out = np.column_stack([
            ba.look(*nested_dev_w(p[m_tr], c[m_tr], y[m_tr], w, k), c[m_va])
            for (p, c), k in zip(AXD, ksh)])
        DEVCACHE[key] = out
        return out

    def chain(th):
        """th = dict(gd, ksh[4], wpost[4], gc, kc_mul, wc, gb, kp, kb, wp, wb)"""
        DEV = {f: devmat(f, th["gd"], th["ksh"]) for f in (2020, 2021, 2022, 2023, 2024)}
        res = {f: y[season == f] - (MODEL[f] + DEV[f] @ np.array(th["wpost"]))
               for f in (2020, 2021, 2022, 2023, 2024)}

        def wt(key, ctx, f, k, gamma, contrast, hard2=False):
            src = (f - 2, f - 1) if hard2 else SRC[f]
            T = max(src)
            parts = []
            for s in src:
                m = season == s
                w = 1.0 if (gamma is None or hard2) else gamma ** (T - s)
                d = {"k": key[m], "sr": res[s] * w, "n": np.full(int(m.sum()), w)}
                if contrast:
                    d["c"] = ctx[m]
                parts.append(pd.DataFrame(d))
            D = pd.concat(parts)
            if not contrast:
                q = D.groupby("k")[["sr", "n"]].sum()
                return (q["sr"] / q["n"]) * q["n"] / (q["n"] + k)
            q = D.groupby(["k", "c"])[["sr", "n"]].sum().unstack()
            n0, n1 = q[("n", 0)].fillna(0), q[("n", 1)].fillna(0)
            m0 = q[("sr", 0)] / n0.replace(0, np.nan)
            m1 = q[("sr", 1)] / n1.replace(0, np.nan)
            ne = (n0 * n1) / (n0 + n1).replace(0, np.nan)
            return ((m1 - m0) * ne / (ne + k)).dropna()

        out = {}
        for f in FOLDS:
            m = season == f
            v = MODEL[f] + DEV[f] @ np.array(th["wpost"])
            for ctx, k0 in AXC:
                t = wt(P, ctx, f, k0 * th["kc_mul"], th["gc"], True)
                v = v + th["wc"] * (pd.Series(P[m]).map(t).fillna(0.).to_numpy()
                                    * np.where(ctx[m] == 1, .5, -.5))
            v = v + th["wp"] * pd.Series(P[m]).map(
                wt(P, None, f, th["kp"], None, False, hard2=True)).fillna(0.).to_numpy()
            v = v + th["wb"] * pd.Series(B[m]).map(
                wt(B, None, f, th["kb"], th["gb"], False)).fillna(0.).to_numpy()
            out[f] = r2(v, y[m])
        return out

    TH0 = dict(gd=None, ksh=list(ba.KSH), wpost=list(ba.WPOST), gc=None, kc_mul=1.0,
               wc=0.65, gb=None, kp=50000.0, kb=20000.0, wp=1.0, wb=2.5)
    cur = chain(TH0)
    print(f"현행       2022 {cur[2022]:.1f}  2023 {cur[2023]:.1f}  2024 {cur[2024]:.1f}")
    TH = dict(gd=0.85, ksh=list(ba.KSH), wpost=list(ba.WPOST), gc=0.85, kc_mul=1.0,
              wc=0.80, gb=0.5, kp=50000.0, kb=20000.0, wp=2.0, wb=1.75)
    st = chain(TH)
    print(f"dc85 출발  " + "  ".join(f"{f} {st[f]-cur[f]:+.1f}" for f in FOLDS))
    E.beat("출발점")

    def sel(th):
        o = chain(th)
        gs = [o[f] - cur[f] for f in FOLDS]
        return np.mean([gs[0], gs[2]]), gs

    # 좌표하강 — 편차 가중 4 · 편차 k 4 · 대비 k 배수 · 수준 k 2
    KNOBS = ([("wpost", i) for i in range(4)] + [("ksh", i) for i in range(4)]
             + [("kc_mul", None), ("kp", None), ("kb", None), ("wc", None),
                ("wp", None), ("wb", None)])
    GRID = (0.6, 0.8, 1.0, 1.25, 1.6)
    best_s, best_g = sel(TH)
    print(f"\n좌표하강 시작 (선택 {best_s:+.2f})")
    t0 = time.time()
    for it in range(3):
        moved = False
        for name, idx in KNOBS:
            base_v = TH[name][idx] if idx is not None else TH[name]
            cand = None
            for mul in GRID:
                if abs(mul - 1.0) < 1e-9:
                    continue
                th2 = {k: (list(v) if isinstance(v, list) else v) for k, v in TH.items()}
                if idx is not None:
                    th2[name][idx] = base_v * mul
                else:
                    th2[name] = base_v * mul
                s, gs = sel(th2)
                if s > best_s + 5e-3 and (cand is None or s > cand[0]):
                    cand = (s, base_v * mul, gs)
            if cand:
                if idx is not None:
                    TH[name][idx] = cand[1]
                else:
                    TH[name] = cand[1]
                best_s, best_g = cand[0], cand[2]
                moved = True
            E.beat(f"{name}{idx if idx is not None else ''}")
        print(f"  pass {it+1}  2022 {best_g[0]:+.1f}  2023 {best_g[1]:+.1f}  "
              f"2024 {best_g[2]:+.1f}   선택 {best_s:+.2f}   ({time.time()-t0:.0f}s)")
        if not moved:
            break

    print(f"\n{'모수':<12}{'dc85':>12}{'최적':>12}")
    D0 = dict(gd=0.85, ksh=list(ba.KSH), wpost=list(ba.WPOST), gc=0.85, kc_mul=1.0,
              wc=0.80, gb=0.5, kp=50000.0, kb=20000.0, wp=2.0, wb=1.75)
    for k in ("wpost", "ksh"):
        for i in range(4):
            print(f"{k+str(i):<12}{D0[k][i]:>12.4g}{TH[k][i]:>12.4g}")
    for k in ("kc_mul", "wc", "kp", "kb", "wp", "wb"):
        print(f"{k:<12}{D0[k]:>12.4g}{TH[k]:>12.4g}")
    print(f"\n최적  2022 {best_g[0]:+.1f}  2023 {best_g[1]:+.1f}  2024 {best_g[2]:+.1f}"
          + ("   ★3폴드 양수" if all(x > 0 for x in best_g) else ""))

    E.finish_experiment(dict(
        experiment_id=EXP, hypothesis_id="FULL", type="B", level=3,
        started_at=E.read(E.CKPT)["start_time"],
        theta={k: (list(v) if isinstance(v, list) else v) for k, v in TH.items()},
        gains=[round(x, 2) for x in best_g], decision="측정",
        what_we_learned="감쇠 하에서 편차 가중·축소 상수까지 재최적화"))
    json.dump({k: (list(v) if isinstance(v, list) else v) for k, v in TH.items()}
              | {"gains": [float(x) for x in best_g]},
              open(os.path.join(ROOT, "exp", "exp040.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
