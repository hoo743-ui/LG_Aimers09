r"""EXP052 — **10번째 후처리 축** 전수 탐색. 재학습 0회.

EXP003 은 243개 이진 국면을 **오라클**로 쟀다. 여기서는 다른 것을 잰다 —
**현행 9축 예측 위에 그 축을 얹었을 때의 폴드 2024 직접 이득**이다.
기준선이 다르므로 결론이 같을 이유가 없다 (그때 기준선에는 수준 2축이 없었다).

후처리 축은 모델을 안 바꾸므로 축 하나당 수십 밀리초다. 전수가 가능하다.

표 = 직전 2시즌(2022·2023)의 out-of-fold 잔차, 키 = (투수, 국면).
현행 3개 대비 축(같은손·2스트라이크·주자유무)이 **0 근처로 나와야** 정상이다.

    .\.venv\Scripts\python.exe -u research\exp052_axis10.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "exp"))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from exp003_sweep import build_contexts                       # noqa: E402

W9 = np.array([0.099765, 0.411532, 0.139671, 0.224472,
               0.703699, 0.775302, 0.775315, 1.984998, 2.105])
KS = (300., 1000., 3000., 10000.)
WS = (0.3, 0.5, 0.65, 1.0, 1.5)


def main():
    z = np.load(os.path.join(ROOT, "exp", "cache", "exp043_comp.npz"))
    mm, Cm, y = z["mm"], z["Cm"], z["y"]
    from path_alloc import build_df
    tr = build_df()
    season = tr["season"].to_numpy()
    m = season == 2024
    ya = tr["control_success"].to_numpy(np.float64)
    cur = mm + Cm @ W9
    base = 1e5 * np.corrcoef(cur, y)[0, 1] ** 2
    print(f"현행 9축 폴드 2024 = {base:.2f}")

    RES = {f: ya[season == f] -
           np.load(os.path.join(ROOT, "exp", f"prod_champ_{f}.npy")).mean(0)
           for f in (2022, 2023)}
    P = tr["pitcher_id"].to_numpy(np.int64)
    CTX = build_contexts(tr, season)
    print(f"이진 국면 {len(CTX)}개 x k {len(KS)} x w {len(WS)}  (재학습 0회)\n")

    src = np.isin(season, (2022, 2023))
    rs = np.concatenate([RES[2022], RES[2023]])
    Ps = P[src]
    rows = []
    for i, (nm, ctx) in enumerate(sorted(CTX.items())):
        cs = ctx[src]
        ok = cs >= 0
        if ok.sum() < 20000:
            continue
        d = pd.DataFrame({"p": Ps[ok], "c": cs[ok], "r": rs[ok]})
        g = d.groupby(["p", "c"])["r"].agg(["sum", "size"]).unstack()
        n0, n1 = g[("size", 0)].fillna(0), g[("size", 1)].fillna(0)
        s0, s1 = g[("sum", 0)].fillna(0), g[("sum", 1)].fillna(0)
        m0 = s0 / n0.replace(0, np.nan)
        m1 = s1 / n1.replace(0, np.nan)
        ne = (n0 * n1) / (n0 + n1).replace(0, np.nan)
        cf = ctx[m]
        okf = cf >= 0
        best = (-9.9, None, None)
        for k in KS:
            t = ((m1 - m0) * ne / (ne + k)).dropna()
            v = pd.Series(P[m]).map(t).fillna(0.).to_numpy()
            v = np.where(okf, v * np.where(cf == 1, .5, -.5), 0.0)
            if v.std() == 0:
                continue
            for w in WS:
                s = 1e5 * np.corrcoef(cur + w * v, y)[0, 1] ** 2 - base
                if s > best[0]:
                    best = (s, w, k)
        rows.append({"axis": nm, "gain": best[0], "w": best[1], "k": best[2],
                     "cover": float(okf.mean())})
        if (i + 1) % 60 == 0:
            print(f"  {i+1}/{len(CTX)}", flush=True)

    D = pd.DataFrame(rows).sort_values("gain", ascending=False)
    print(f"\n=== 상위 25 (현행 9축 위의 직접 이득, 폴드 2024) ===")
    print(f"{'축':38s} {'이득':>8s} {'w':>5s} {'k':>7s} {'적용률':>7s}")
    for _, r in D.head(25).iterrows():
        print(f"{r.axis[:38]:38s} {r.gain:+8.2f} {r.w:5g} {r.k:7g} {r.cover:7.2%}")
    print(f"\n[대조] 이미 실려 있는 3축 — 0 근처여야 정상")
    for pat in ("batter_hand", "strikes_before==2", "num_runners_on==0"):
        for _, r in D[D.axis.str.contains(pat, regex=False)].head(3).iterrows():
            print(f"{r.axis[:38]:38s} {r.gain:+8.2f} {r.w:5g} {r.k:7g}")
    D.to_csv(os.path.join(ROOT, "exp", "exp052_axis10.csv"), index=False,
             encoding="utf-8")
    print(f"\n양수 축 {int((D.gain > 0).sum())}/{len(D)}   "
          f"+2 초과 {int((D.gain > 2).sum())}   -> exp/exp052_axis10.csv")


if __name__ == "__main__":
    main()
