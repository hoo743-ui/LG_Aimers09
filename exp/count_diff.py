r"""PITCHER x COUNT-ADVANTAGE 차등 — C3 위에서. 학습 0회.

## 기준선이 바뀌었다

Champion = C3 = 기존 4축 + 잔차차등 3축(손 k1000 / 2스트라이크 k1000 / 주자 k2000).
**모든 측정은 C3 위에서** 한다. 폴드 f 의 C3 는 f 직전 두 시즌 잔차로 만든다.

## K2 와 무엇이 다른가

```
K2        cur_{ball,rev} x {2S, 2S&b<=1}      **in-model 피처**, 전역 곱, LB -4.72
이번 후보  투수별 [우위 - 열세] 잔차 차등        **후처리 표**, 투수마다 다른 값
```

K2 는 "2스트라이크에서 볼 비율이 어떻게 읽히는가"를 **모든 투수에 공통**으로
줬다. 여기서는 **그 투수가 우위 카운트에서 평소보다 얼마나 다른가**를 준다.
게다가 C3 는 이미 2스트라이크 축을 갖고 있으므로, 겹치지 않는 정의도 함께 본다.

## 정의 4종

    adv      strikes > balls
    adv_x2s  strikes > balls **이면서 2스트라이크가 아님**   <- 기존 축과 겹치지 않는 몫
    margin   strikes - balls 를 상/하위로 (>=1 vs <=-1, 동수 제외)
    disadv   balls - strikes >= 2 (3볼 국면 포함)

    .\.venv\Scripts\python.exe -u exp\count_diff.py
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
from traj_probe import cells, look, r2                     # noqa: E402
from game_decomp import games                              # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

KGRID = [500, 1000, 2000, 3000, 5000, 10000]
PREV2 = {2022: (2020, 2021), 2023: (2021, 2022), 2024: (2022, 2023)}


def main():
    tr = build_df()
    season = tr["season"].to_numpy()
    y = tr["control_success"].to_numpy(np.float64)
    P = tr["pitcher_id"].to_numpy(np.int64)
    g = lambda c: tr[c].to_numpy(np.float64)
    BB, SS = g("balls_before").astype(int), g("strikes_before").astype(int)
    SAME = (g("pitcher_hand") == g("batter_hand")).astype(int)
    TWO = (SS == 2).astype(int)
    RUN = (g("num_runners_on") > 0).astype(int)

    pv0, res0 = {}, {}
    for f in (2020, 2021, 2022, 2023, 2024):
        m = season == f
        pv0[f] = (np.load(os.path.join(ROOT, "exp", f"prod_champ_{f}.npy"))[:2].mean(0)
                  + post_for(tr, y, season < f, m))
        res0[f] = y[m] - pv0[f]

    def dtab(src, ctx, k, resmap):
        p = np.concatenate([P[season == f] for f in src])
        c = np.concatenate([ctx[season == f] for f in src])
        r = np.concatenate([resmap[f] for f in src])
        ok = np.isin(c, (0, 1))
        gg = pd.DataFrame({"p": p[ok], "c": c[ok], "r": r[ok]}).groupby(
            ["p", "c"])["r"].agg(["mean", "size"]).unstack()
        if ("size", 0) not in gg or ("size", 1) not in gg:
            return None
        n0, n1 = gg[("size", 0)].fillna(0), gg[("size", 1)].fillna(0)
        d = gg[("mean", 1)] - gg[("mean", 0)]
        ne = (n0 * n1) / (n0 + n1).replace(0, np.nan)
        return (d * ne / (ne + k)).dropna()

    def apply(tabl, ctx, m):
        if tabl is None:
            return np.zeros(int(m.sum()))
        h = np.where(ctx[m] == 1, 0.5, np.where(ctx[m] == 0, -0.5, 0.0))
        return pd.Series(P[m]).map(tabl).fillna(0.0).to_numpy() * h

    # --- C3 기준선 (폴드마다 직전 두 시즌 표) ---
    C3 = {}
    for f in (2022, 2023, 2024):
        m = season == f
        src = PREV2[f]
        C3[f] = (pv0[f] + apply(dtab(src, SAME, 1000, res0), SAME, m)
                 + apply(dtab(src, TWO, 1000, res0), TWO, m)
                 + apply(dtab(src, RUN, 2000, res0), RUN, m))
    resC = {f: y[season == f] - C3[f] for f in (2022, 2023, 2024)}
    for f in (2022, 2023, 2024):
        m = season == f
        print(f"  폴드 {f}: 기존 {r2(pv0[f], y[m]):.1f} -> C3 {r2(C3[f], y[m]):.1f}"
              f"  ({r2(C3[f], y[m]) - r2(pv0[f], y[m]):+.1f})", flush=True)
    m24 = season == 2024
    base = r2(C3[2024], y[m24])
    print(f"\n  C3 기준선 (폴드 2024) = {base:.1f}\n")

    CTX = {
        "adv (S>B)": (SS > BB).astype(int),
        "adv_x2s (S>B & S<2)": np.where(SS == 2, -1, (SS > BB).astype(int)),
        "margin (>=1 vs <=-1)": np.where(SS - BB >= 1, 1,
                                         np.where(SS - BB <= -1, 0, -1)),
        "disadv (B-S>=2)": (BB - SS >= 2).astype(int),
    }

    GID24 = games(P[m24], g("asof_pitcher_n")[m24],
                  g("asof_pitcher_prev1_game_success_rate")[m24],
                  g("asof_pitcher_prev1_game_middle_rate")[m24])
    rng = np.random.default_rng(0)
    u, inv = np.unique(GID24, return_inverse=True)
    half = (rng.random(len(u)) < 0.5)[inv]

    print("=" * 104)
    print(f"{'정의':<24}{'오라클':>8}{'위약':>7}{'21→22':>8}{'22→23':>8}{'23→24':>8}"
          f"{'최적k':>7}{'C3증분':>8}{'hand':>7}{'2S':>7}{'run':>7}")
    print("=" * 104)
    HD = apply(dtab((2022, 2023), SAME, 1000, res0), SAME, m24)
    TS = apply(dtab((2022, 2023), TWO, 1000, res0), TWO, m24)
    RN = apply(dtab((2022, 2023), RUN, 2000, res0), RUN, m24)
    out = {}
    for name, ctx in CTX.items():
        key = P * 10 + np.clip(ctx, 0, 1)
        bo = -1e9
        for k in (0, 150, 500, 2000, 10000):
            add = np.zeros(int(m24.sum()))
            for m in (half, ~half):
                uu, tb, _ = cells(key[m24][m], resC[2024][m], k)
                add[~m] = look(uu, tb, key[m24][~m])
            bo = max(bo, r2(C3[2024] + add, y[m24]) - base)
        pl = P[m24] * 10 + rng.integers(0, 2, int(m24.sum()))
        bp = -1e9
        for k in (0, 150, 500, 2000, 10000):
            add = np.zeros(int(m24.sum()))
            for m in (half, ~half):
                uu, tb, _ = cells(pl[m], resC[2024][m], k)
                add[~m] = look(uu, tb, pl[~m])
            bp = max(bp, r2(C3[2024] + add, y[m24]) - base)
        gains = {k: [] for k in KGRID}
        for a, b in ((2021, 2022), (2022, 2023), (2023, 2024)):
            mb = season == b
            bs = r2(C3[b], y[mb]) if b in C3 else r2(pv0[b], y[mb])
            ref = C3[b] if b in C3 else pv0[b]
            rr = resC[b] if b in resC else res0[b]
            for k in KGRID:
                gains[k].append(r2(ref + apply(dtab((a,), ctx, k, res0), ctx, mb),
                                   y[mb]) - bs)
        past = {k: np.mean(gains[k][:2]) for k in KGRID}
        kb = max(past, key=past.get)
        add24 = apply(dtab((2022, 2023), ctx, kb, res0), ctx, m24)
        inc = r2(C3[2024] + add24, y[m24]) - base
        c = [float(np.corrcoef(add24, v)[0, 1]) if add24.std() > 0 else 0.0
             for v in (HD, TS, RN)]
        out[name] = dict(oracle=bo, placebo=bp, k=kb, gains=gains[kb], inc=inc,
                         corr=c)
        print(f"{name:<24}{bo:>8.1f}{bp:>7.1f}"
              + "".join(f"{v:>+8.1f}" for v in gains[kb])
              + f"{kb:>7}{inc:>+8.1f}{c[0]:>+7.2f}{c[1]:>+7.2f}{c[2]:>+7.2f}")

    json.dump(out, io.open(os.path.join(ROOT, "exp", "count_diff.json"), "w",
                           encoding="utf-8"), indent=1, ensure_ascii=False,
              default=float)
    print("\n  오라클/위약 = 경기 단위 분할. k 는 과거 2전이로만 선택.")
    print("  C3증분 = 현행 Champion(7축) 위에 더했을 때의 순증분")


if __name__ == "__main__":
    main()
