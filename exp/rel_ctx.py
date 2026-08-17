r"""RELATIVE CONTEXT — 같은 행 안의 **비교**를 국면으로 쓴다. C3 기준. 학습 0회.

## 왜 새로운가

지금까지 쓴 맥락은 전부 **경기 상황**이었다 (손·카운트·주자·이닝·점수).
비교 자체를 국면으로 쓴 적은 없다.

    기존 국면   "지금 2스트라이크인가"          -> 상황의 분류
    이번 국면   "지금 투수가 타자보다 나은가"     -> 두 값의 **상대 위치**

트리는 축평행 분할이라 `a > b` 같은 **대각 경계**를 원리적으로 못 만든다
(X/H1 이 통한 이유와 같은 구조인데, 그때는 곱이었고 이번엔 비교다).

## 후보 6종과 메커니즘

    R1 cur_succ > cur_bsucc      이 매치업에서 투수 상태가 타자보다 좋은가
    R2 cur_mid  < cur_bmid       가운데 성향 비교 (실투 위험의 상대 위치)
    R3 asof_pitcher_n > asof_batter_n   경력 우위 (베테랑 투수 vs 신인 타자)
    R4 prev1 결측                 그 투수의 첫 등판 국면 (직전 경기가 없다)
    R5 cur_succ > 통산 성공률       자기 통산 대비 호조인가 (d_succ 의 이진화)
    R6 cur_ball > cur_str         볼이 스트라이크보다 많은 상태

R5 는 `d_succ`(연속) 이 잔차상관 +0.0001 로 죽은 축이지만 **이진 국면으로는
미검증**이다. `이산 전환은 통하고 매끄러운 값은 안 통한다`는 이 프로젝트의
반복된 관찰이라 따로 본다.

## 절차 (이전과 동일)

기준선 C3(7축) · 오라클은 **경기 단위 분할 + 위약** · k 는 **과거 2전이로만**.

    .\.venv\Scripts\python.exe -u exp\rel_ctx.py
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

KGRID = [500, 1000, 2000, 5000, 20000]
PREV2 = {2022: (2020, 2021), 2023: (2021, 2022), 2024: (2022, 2023)}


def main():
    tr = build_df()
    season = tr["season"].to_numpy()
    y = tr["control_success"].to_numpy(np.float64)
    P = tr["pitcher_id"].to_numpy(np.int64)
    g = lambda c: tr[c].to_numpy(np.float64)
    SAME = (g("pitcher_hand") == g("batter_hand")).astype(int)
    TWO = (g("strikes_before").astype(int) == 2).astype(int)
    RUN = (g("num_runners_on") > 0).astype(int)

    pv0, res0 = {}, {}
    for f in (2020, 2021, 2022, 2023, 2024):
        m = season == f
        pv0[f] = (np.load(os.path.join(ROOT, "exp", f"prod_champ_{f}.npy"))[:2].mean(0)
                  + post_for(tr, y, season < f, m))
        res0[f] = y[m] - pv0[f]

    def dtab(ctx, src, k, rmap):
        p = np.concatenate([P[season == f] for f in src])
        c = np.concatenate([ctx[season == f] for f in src])
        r = np.concatenate([rmap[f] for f in src])
        ok = np.isin(c, (0, 1))
        gg = pd.DataFrame({"p": p[ok], "c": c[ok], "r": r[ok]}).groupby(
            ["p", "c"])["r"].agg(["mean", "size"]).unstack()
        if ("size", 0) not in gg or ("size", 1) not in gg:
            return None
        n0, n1 = gg[("size", 0)].fillna(0), gg[("size", 1)].fillna(0)
        d = gg[("mean", 1)] - gg[("mean", 0)]
        ne = (n0 * n1) / (n0 + n1).replace(0, np.nan)
        return (d * ne / (ne + k)).dropna()

    def ap(t, ctx, m):
        if t is None:
            return np.zeros(int(m.sum()))
        h = np.where(ctx[m] == 1, .5, np.where(ctx[m] == 0, -.5, 0.))
        return pd.Series(P[m]).map(t).fillna(0.0).to_numpy() * h

    C3, resC = {}, {}
    for f in (2022, 2023, 2024):
        m = season == f
        s = PREV2[f]
        C3[f] = (pv0[f] + ap(dtab(SAME, s, 1000, res0), SAME, m)
                 + ap(dtab(TWO, s, 1000, res0), TWO, m)
                 + ap(dtab(RUN, s, 2000, res0), RUN, m))
        resC[f] = y[m] - C3[f]
    m24 = season == 2024
    base = r2(C3[2024], y[m24])
    print(f"  C3 기준선 (폴드 2024) = {base:.1f}")

    prior_succ = np.where(
        np.expm1(g("cur_logn_pitch")) < g("asof_pitcher_n"),
        (g("asof_pitcher_n") * g("asof_pitcher_success_rate")
         - np.expm1(g("cur_logn_pitch")) * g("cur_succ"))
        / np.maximum(g("asof_pitcher_n") - np.expm1(g("cur_logn_pitch")), 1e-9),
        np.nan)
    R = {
        "R1 cur_succ > cur_bsucc": (g("cur_succ") > g("cur_bsucc")).astype(int),
        "R2 cur_mid < cur_bmid": (g("cur_mid") < g("cur_bmid")).astype(int),
        "R3 투수경력 > 타자경력": (g("asof_pitcher_n") > g("asof_batter_n")).astype(int),
        "R4 prev1 결측(첫 등판)": np.isnan(
            g("asof_pitcher_prev1_game_success_rate")).astype(int),
        "R5 cur_succ > 통산 성공률": np.where(
            np.isnan(prior_succ), -1, (g("cur_succ") > prior_succ).astype(int)),
        "R6 cur_ball > cur_str": (g("cur_ball") > g("cur_str")).astype(int),
    }
    for n, c in R.items():
        v = c[m24]
        print(f"    {n:<26} 조건=1 비율 {np.mean(v == 1):.1%}"
              f"  (제외 {np.mean(v == -1):.1%})")

    GID = games(P[m24], g("asof_pitcher_n")[m24],
                g("asof_pitcher_prev1_game_success_rate")[m24],
                g("asof_pitcher_prev1_game_middle_rate")[m24])
    rng = np.random.default_rng(0)
    u, inv = np.unique(GID, return_inverse=True)
    half = (rng.random(len(u)) < 0.5)[inv]
    HD = ap(dtab(SAME, (2022, 2023), 1000, res0), SAME, m24)
    TS = ap(dtab(TWO, (2022, 2023), 1000, res0), TWO, m24)
    RN = ap(dtab(RUN, (2022, 2023), 2000, res0), RUN, m24)

    print("\n" + "=" * 104)
    print(f"{'후보':<26}{'잔차상관':>10}{'오라클':>8}{'위약':>7}{'21→22':>8}"
          f"{'22→23':>8}{'23→24':>8}{'k':>7}{'C3증분':>8}{'hand/2S/run 상관':>18}")
    print("=" * 104)
    out = {}
    for n, ctx in R.items():
        cc = ctx[m24]
        ok = cc >= 0
        rc = float(np.corrcoef(cc[ok], resC[2024][ok])[0, 1])
        key = P[m24] * 10 + np.clip(cc, 0, 1)
        bo = -1e9
        for k in (0, 150, 500, 2000, 10000):
            add = np.zeros(int(m24.sum()))
            for m in (half, ~half):
                uu, tb, _ = cells(key[m], resC[2024][m], k)
                add[~m] = look(uu, tb, key[~m])
            bo = max(bo, r2(C3[2024] + add, y[m24]) - base)
        pl = P[m24] * 10 + rng.integers(0, 2, int(m24.sum()))
        bp = -1e9
        for k in (0, 150, 500, 2000, 10000):
            add = np.zeros(int(m24.sum()))
            for m in (half, ~half):
                uu, tb, _ = cells(pl[m], resC[2024][m], k)
                add[~m] = look(uu, tb, pl[~m])
            bp = max(bp, r2(C3[2024] + add, y[m24]) - base)
        G = {k: [] for k in KGRID}
        for a, b in ((2021, 2022), (2022, 2023), (2023, 2024)):
            mb = season == b
            bs = r2(C3[b], y[mb])
            for k in KGRID:
                G[k].append(r2(C3[b] + ap(dtab(ctx, (a,), k, res0), ctx, mb),
                               y[mb]) - bs)
        kb = max(G, key=lambda k: np.mean(G[k][:2]))
        add24 = ap(dtab(ctx, (2022, 2023), kb, res0), ctx, m24)
        inc = r2(C3[2024] + add24, y[m24]) - base
        ov = [float(np.corrcoef(add24, v)[0, 1]) if add24.std() > 0 else 0.0
              for v in (HD, TS, RN)]
        out[n] = dict(resid_corr=rc, oracle=bo, placebo=bp, k=kb,
                      gains=G[kb], inc=inc, overlap=ov)
        print(f"{n:<26}{rc:>+10.4f}{bo:>8.1f}{bp:>7.1f}"
              + "".join(f"{v:>+8.1f}" for v in G[kb]) + f"{kb:>7}{inc:>+8.1f}"
              + f"   {ov[0]:+.2f}/{ov[1]:+.2f}/{ov[2]:+.2f}")

    json.dump(out, io.open(os.path.join(ROOT, "exp", "rel_ctx.json"), "w",
                           encoding="utf-8"), indent=1, ensure_ascii=False,
              default=float)
    print("\n  잔차상관의 잡음 대역은 |0.002| (2024 253,507행 기준 1SE)")


if __name__ == "__main__":
    main()
