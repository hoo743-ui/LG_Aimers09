r"""CONTINUOUS CONTEXT SENSITIVITY — 투수별 기울기. C3 기준선. 학습 0회.

## 이진 차등과 무엇이 다른가

```
기존   d_p = 잔차평균(맥락=1) - 잔차평균(맥락=0)        두 집단의 차
이번   b_p = d E[잔차] / d 맥락                        연속 반응의 기울기
```

보정은 `b_p x (맥락 - 그 투수의 맥락 평균)` 으로 넣는다 — **수준은 빼고
기울기만** 싣는다. 그래야 기존 축(수준 차이)의 재표현이 아니다.

## 축소

기울기의 분산은 `sigma^2 / (n * var(ctx))` 이므로 정보량은 `n * var(ctx)` 다.
그래서 `w = n*var / (n*var + k)` 로 축소한다 (이진 차등의 `n_eff` 와 같은 논리).
`k` 는 **과거 두 전이에서만** 고른다.

## 규약

기준선은 **C3(7축)**, 오라클은 **경기 단위 분할** + 위약 대조.
표는 목표 폴드 직전 두 시즌의 out-of-fold 잔차로만 만든다.

    .\.venv\Scripts\python.exe -u exp\slope_axis.py
"""
import io
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "exp"))
from path_alloc import build_df                            # noqa: E402
from resid_table import post_for                           # noqa: E402
from traj_probe import r2                                  # noqa: E402
from game_decomp import games                              # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

KGRID = [200, 500, 1000, 2000, 5000, 20000]
PREV2 = {2022: (2020, 2021), 2023: (2021, 2022), 2024: (2022, 2023)}


def slopes(pid, x, r):
    """투수별 (기울기, 정보량 n*var, 맥락 평균)."""
    d = pd.DataFrame({"p": pid, "x": x, "r": r})
    g = d.groupby("p")
    n = g.size()
    mx = g["x"].mean()
    vx = g["x"].var(ddof=0)
    cov = g.apply(lambda t: float(np.mean((t["x"] - t["x"].mean())
                                          * (t["r"] - t["r"].mean()))),
                  include_groups=False)
    b = cov / vx.replace(0, np.nan)
    return pd.DataFrame({"b": b, "info": n * vx, "mx": mx, "n": n}).dropna()


def main():
    tr = build_df()
    season = tr["season"].to_numpy()
    y = tr["control_success"].to_numpy(np.float64)
    P = tr["pitcher_id"].to_numpy(np.int64)
    g = lambda c: tr[c].to_numpy(np.float64)
    BB, SS = g("balls_before"), g("strikes_before")
    SAME = (g("pitcher_hand") == g("batter_hand")).astype(int)
    TWO = (SS.astype(int) == 2).astype(int)
    RUN = (g("num_runners_on") > 0).astype(int)

    pv0, res0 = {}, {}
    for f in (2020, 2021, 2022, 2023, 2024):
        m = season == f
        pv0[f] = (np.load(os.path.join(ROOT, "exp", f"prod_champ_{f}.npy"))[:2].mean(0)
                  + post_for(tr, y, season < f, m))
        res0[f] = y[m] - pv0[f]

    def dtab(src, ctx, k):
        p = np.concatenate([P[season == f] for f in src])
        c = np.concatenate([ctx[season == f] for f in src])
        r = np.concatenate([res0[f] for f in src])
        gg = pd.DataFrame({"p": p, "c": c, "r": r}).groupby(["p", "c"])["r"].agg(
            ["mean", "size"]).unstack()
        n0, n1 = gg[("size", 0)].fillna(0), gg[("size", 1)].fillna(0)
        d = gg[("mean", 1)] - gg[("mean", 0)]
        ne = (n0 * n1) / (n0 + n1).replace(0, np.nan)
        return (d * ne / (ne + k)).dropna()

    def ap(t, ctx, m):
        return (pd.Series(P[m]).map(t).fillna(0.0).to_numpy()
                * np.where(ctx[m] == 1, .5, -.5))

    C3, resC = {}, {}
    for f in (2022, 2023, 2024):
        m = season == f
        s = PREV2[f]
        C3[f] = (pv0[f] + ap(dtab(s, SAME, 1000), SAME, m)
                 + ap(dtab(s, TWO, 1000), TWO, m) + ap(dtab(s, RUN, 2000), RUN, m))
        resC[f] = y[m] - C3[f]
    m24 = season == 2024
    base = r2(C3[2024], y[m24])
    print(f"  C3 기준선 (폴드 2024) = {base:.1f}\n")

    CTX = {"count margin (S−B)": (SS - BB),
           "runner load (0~3)": g("num_runners_on"),
           "out state (0~2)": g("outs_before"),
           "score pressure (클립 ±4)": np.clip(g("score_diff_pitcher_team"), -4, 4)}

    GID = games(P[m24], g("asof_pitcher_n")[m24],
                g("asof_pitcher_prev1_game_success_rate")[m24],
                g("asof_pitcher_prev1_game_middle_rate")[m24])
    rng = np.random.default_rng(0)
    u, inv = np.unique(GID, return_inverse=True)
    half = (rng.random(len(u)) < 0.5)[inv]
    HD = ap(dtab((2022, 2023), SAME, 1000), SAME, m24)
    TS = ap(dtab((2022, 2023), TWO, 1000), TWO, m24)
    RN = ap(dtab((2022, 2023), RUN, 2000), RUN, m24)

    def apply_slope(sl, ctx, m, k):
        w = sl["info"] / (sl["info"] + k)
        bb = pd.Series(P[m]).map(sl["b"] * w).fillna(0.0).to_numpy()
        cc = pd.Series(P[m]).map(sl["mx"]).fillna(np.nanmean(ctx[m])).to_numpy()
        return bb * (ctx[m] - cc)

    print("=" * 108)
    print(f"{'맥락':<24}{'전체기울기':>10}{'투수sd':>9}{'위약sd':>9}{'오라클':>8}"
          f"{'위약오라클':>10}{'21→22':>8}{'22→23':>8}{'23→24':>8}{'k':>6}{'C3증분':>8}")
    print("=" * 108)
    out = {}
    for name, ctx in CTX.items():
        # 표본 구조 + 전체 기울기 + 투수별 산포
        sl24 = slopes(P[m24], ctx[m24], resC[2024])
        allb = float(np.cov(ctx[m24], resC[2024])[0, 1] / np.var(ctx[m24]))
        plac_p = rng.permutation(P[m24])
        slp = slopes(plac_p, ctx[m24], resC[2024])
        # 오라클 — 경기 분할 교차적합
        bo = -1e9
        for k in KGRID:
            add = np.zeros(int(m24.sum()))
            for m in (half, ~half):
                s = slopes(P[m24][m], ctx[m24][m], resC[2024][m])
                w = s["info"] / (s["info"] + k)
                bb = pd.Series(P[m24][~m]).map(s["b"] * w).fillna(0.0).to_numpy()
                cc = pd.Series(P[m24][~m]).map(s["mx"]).fillna(
                    np.nanmean(ctx[m24])).to_numpy()
                add[~m] = bb * (ctx[m24][~m] - cc)
            bo = max(bo, r2(C3[2024] + add, y[m24]) - base)
        bp = -1e9
        for k in KGRID:
            add = np.zeros(int(m24.sum()))
            for m in (half, ~half):
                s = slopes(plac_p[m], ctx[m24][m], resC[2024][m])
                w = s["info"] / (s["info"] + k)
                bb = pd.Series(plac_p[~m]).map(s["b"] * w).fillna(0.0).to_numpy()
                cc = pd.Series(plac_p[~m]).map(s["mx"]).fillna(
                    np.nanmean(ctx[m24])).to_numpy()
                add[~m] = bb * (ctx[m24][~m] - cc)
            bp = max(bp, r2(C3[2024] + add, y[m24]) - base)
        # 전이
        G = {k: [] for k in KGRID}
        for a, b in ((2021, 2022), (2022, 2023), (2023, 2024)):
            mb = season == b
            bs = r2(C3[b], y[mb])
            ma = season == a
            sa = slopes(P[ma], ctx[ma], res0[a])
            for k in KGRID:
                G[k].append(r2(C3[b] + apply_slope(sa, ctx, mb, k), y[mb]) - bs)
        kb = max(G, key=lambda k: np.mean(G[k][:2]))
        msrc = np.isin(season, (2022, 2023))
        ssrc = slopes(P[msrc], ctx[msrc], np.concatenate([res0[2022], res0[2023]]))
        add24 = apply_slope(ssrc, ctx, m24, kb)
        inc = r2(C3[2024] + add24, y[m24]) - base
        cs = [float(np.corrcoef(add24, v)[0, 1]) if add24.std() > 0 else 0.0
              for v in (HD, TS, RN, resC[2024])]
        out[name] = dict(all_slope=allb, sd=float(sl24["b"].std()),
                         placebo_sd=float(slp["b"].std()), oracle=bo, placebo=bp,
                         gains=G[kb], k=kb, inc=inc, corr=cs)
        print(f"{name:<24}{allb:>+10.5f}{sl24['b'].std():>9.4f}"
              f"{slp['b'].std():>9.4f}{bo:>8.1f}{bp:>10.1f}"
              + "".join(f"{v:>+8.1f}" for v in G[kb]) + f"{kb:>6}{inc:>+8.1f}")
    print("\n  상관 (보정 벡터 기준)")
    for name in CTX:
        c = out[name]["corr"]
        print(f"  {name:<24} hand {c[0]:+.2f}  2S {c[1]:+.2f}  runner {c[2]:+.2f}"
              f"  C3잔차 {c[3]:+.4f}")
    json.dump(out, io.open(os.path.join(ROOT, "exp", "slope_axis.json"), "w",
                           encoding="utf-8"), indent=1, ensure_ascii=False,
              default=float)


if __name__ == "__main__":
    main()
