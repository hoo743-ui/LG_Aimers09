r"""EXP017 — `_bs` 계열의 지속성이 진짜인가. 적대적 검사 먼저.

## 왜 이 후보가 올라왔는가

EXP016 이 이득 대신 **d_p 의 시즌 간 상관**으로 246개를 재측정했다. 손의
재표현을 빼면 `_bs`(볼−스트라이크) 계열 5종이 최상위였고, 감쇠보정 지속성이
0.38~0.47 로 손(0.243)의 약 2배였다.

## 반대 가설 세 개를 먼저 검사한다

    H1 시간 교란   분위 기반 국면이 시즌 초에 몰리면 d_p 는 '초반 대 후반'을
                  재고, 그건 매 시즌 반복되지만 모델이 cur_logn 으로 이미 안다
    H2 불균형      투수별 ctx=1 비율이 극단이면 d_p 가 불안정해지고, 그 불안정
                  자체가 투수 수준 상수와 상관해 가짜 지속성을 만든다
    H3 자기 흡수   지속성이 진짜여도 모델이 이미 그 구조를 학습했으면 이득 0

H1 은 국면과 `cur_logn_pitch` 의 상관으로, H2 는 투수별 비율의 산포로,
H3 은 전이 이득으로 각각 잰다. **셋을 통과해야 후보다.**

    .\.venv\Scripts\python.exe -u research\exp017_bs.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "exp"))
import engine as E                                          # noqa: E402
from path_alloc import build_df                             # noqa: E402
from resid_table import post_for                            # noqa: E402
from traj_probe import cells, look, r2                      # noqa: E402
from game_decomp import games                               # noqa: E402
from exp003_sweep import build_contexts                     # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

EXP = "EXP017"
KGRID = [500, 1000, 2000, 5000, 20000]
PREV2 = {2022: (2020, 2021), 2023: (2021, 2022), 2024: (2022, 2023)}
TARGET = ["dx_mid_bs 상위25%", "lx_rev_bs 상위25%", "lx_str_bs 상위25%",
          "lx_ball_bs 상위25%", "dx_succ_bs 상위25%"]


def main():
    E.start_experiment(EXP, "L0-R", "python research/exp017_bs.py", "load")
    tr = build_df()
    season = tr["season"].to_numpy()
    y = tr["control_success"].to_numpy(np.float64)
    P = tr["pitcher_id"].to_numpy(np.int64)

    def g(c):
        return tr[c].to_numpy(np.float64)

    AX = {"hand": ((g("pitcher_hand") == g("batter_hand")).astype(int), 1000),
          "2S": ((g("strikes_before").astype(int) == 2).astype(int), 1000),
          "runner": ((g("num_runners_on") > 0).astype(int), 2000)}
    pv0, res0 = {}, {}
    for f in (2020, 2021, 2022, 2023, 2024):
        m = season == f
        pv0[f] = (np.load(os.path.join(ROOT, "exp", f"prod_champ_{f}.npy"))[:2].mean(0)
                  + post_for(tr, y, season < f, m))
        res0[f] = y[m] - pv0[f]

    def dvec(ctx, src, k):
        p = np.concatenate([P[season == f] for f in src])
        c = np.concatenate([ctx[season == f] for f in src])
        r = np.concatenate([res0[f] for f in src])
        ok = np.isin(c, (0, 1))
        gg = pd.DataFrame({"p": p[ok], "c": c[ok], "r": r[ok]}).groupby(
            ["p", "c"])["r"].agg(["mean", "size"]).unstack()
        if ("size", 0) not in gg or ("size", 1) not in gg:
            return None
        n0, n1 = gg[("size", 0)].fillna(0), gg[("size", 1)].fillna(0)
        ne = (n0 * n1) / (n0 + n1).replace(0, np.nan)
        return ((gg[("mean", 1)] - gg[("mean", 0)]) * ne / (ne + k)).dropna()

    def ap(t, ctx, m):
        if t is None:
            return np.zeros(int(m.sum()))
        h = np.where(ctx[m] == 1, .5, np.where(ctx[m] == 0, -.5, 0.))
        return pd.Series(P[m]).map(t).fillna(0.).to_numpy() * h

    C3, resC = {}, {}
    for f in (2022, 2023, 2024):
        m = season == f
        v = pv0[f].copy()
        for a, (ctx, k) in AX.items():
            v += ap(dvec(ctx, PREV2[f], k), ctx, m)
        C3[f], resC[f] = v, y[m] - v
    m24 = season == 2024
    base = r2(C3[2024], y[m24])
    print(f"C3 기준선 2024 = {base:.1f}\n")

    CTX = build_contexts(tr, season)
    CTX["[대조] 같은손"] = AX["hand"][0]
    CTX["[대조] 2스트라이크"] = AX["2S"][0]
    cand = {n: CTX[n] for n in TARGET if n in CTX}
    cand["[대조] 같은손"] = CTX["[대조] 같은손"]
    cand["[대조] 2스트라이크"] = CTX["[대조] 2스트라이크"]
    print(f"검사 대상 {len(cand)}개\n")
    E.beat("H1/H2 진단")

    logn = g("cur_logn_pitch")
    print("=" * 92)
    print("H1 시간 교란 · H2 불균형 — 국면이 무엇과 섞여 있는가")
    print("=" * 92)
    print(f"{'국면':<26}{'=1비율':>8}{'corr(ctx,cur_logn)':>20}"
          f"{'투수별 비율 sd':>15}{'양쪽 가진 투수':>14}")
    diag = {}
    for nm, ctx in cand.items():
        c24 = ctx[m24]
        ok = c24 >= 0
        rc = float(np.corrcoef(c24[ok], logn[m24][ok])[0, 1])
        d = pd.DataFrame({"p": P[m24][ok], "c": c24[ok]}).groupby("p")["c"].agg(
            ["mean", "size"])
        d = d[d["size"] >= 50]
        both = int(((d["mean"] > 0.02) & (d["mean"] < 0.98)).sum())
        diag[nm] = dict(frac=float(np.mean(c24 == 1)), corr_logn=rc,
                        sd_ratio=float(d["mean"].std()), both=both, npit=len(d))
        print(f"{nm:<26}{np.mean(c24==1):>8.1%}{rc:>+20.4f}"
              f"{d['mean'].std():>15.3f}{both:>8} / {len(d)}")

    print("\n" + "=" * 92)
    print("H3 자기 흡수 — 오라클(경기분할)과 전이 이득")
    print("=" * 92)
    GID = games(P[m24], g("asof_pitcher_n")[m24],
                g("asof_pitcher_prev1_game_success_rate")[m24],
                g("asof_pitcher_prev1_game_middle_rate")[m24])
    rng = np.random.default_rng(0)
    u, inv = np.unique(GID, return_inverse=True)
    half = (rng.random(len(u)) < 0.5)[inv]

    def oracle(key):
        b = -1e9
        for k in (0, 150, 500, 2000, 10000):
            add = np.zeros(int(m24.sum()))
            for m in (half, ~half):
                uu, tb, _ = cells(key[m], resC[2024][m], k)
                add[~m] = look(uu, tb, key[~m])
            b = max(b, r2(C3[2024] + add, y[m24]) - base)
        return b

    plac = oracle(P[m24] * 10 + rng.integers(0, 2, int(m24.sum())))
    print(f"{'국면':<26}{'오라클':>9}{'위약':>8}{'21→22':>9}{'22→23':>9}"
          f"{'23→24':>9}{'k':>7}{'C3증분':>9}")
    out = {}
    for nm, ctx in cand.items():
        E.beat(f"H3 {nm}")
        orc = oracle(P[m24] * 10 + np.clip(ctx[m24], 0, 1))
        G = {k: [] for k in KGRID}
        for a, b in ((2021, 2022), (2022, 2023), (2023, 2024)):
            mb = season == b
            bs = r2(C3[b], y[mb])
            for k in KGRID:
                G[k].append(r2(C3[b] + ap(dvec(ctx, (a,), k), ctx, mb), y[mb]) - bs)
        kb = max(G, key=lambda k: np.mean(G[k][:2]))
        inc = r2(C3[2024] + ap(dvec(ctx, (2022, 2023), kb), ctx, m24), y[m24]) - base
        out[nm] = dict(oracle=orc, gains=G[kb], k=kb, inc=inc, **diag[nm])
        print(f"{nm:<26}{orc:>9.1f}{plac:>8.1f}"
              + "".join(f"{v:>+9.1f}" for v in G[kb]) + f"{kb:>7}{inc:>+9.1f}")

    new = [n for n in TARGET if n in out and out[n]["inc"] >= 4.8
           and all(v > 0 for v in out[n]["gains"])]
    print("\n" + "=" * 92)
    print("판정")
    print("=" * 92)
    for nm in TARGET:
        if nm not in out:
            continue
        d = out[nm]
        h1 = "탈락(시간교란)" if abs(d["corr_logn"]) > 0.15 else "통과"
        h2 = "탈락(불균형)" if d["both"] < 0.5 * d["npit"] else "통과"
        h3 = "탈락(흡수)" if d["inc"] < 4.8 else "통과"
        print(f"  {nm:<26} H1 {h1:<14} H2 {h2:<12} H3 {h3}")
    dec = "PROMISING" if new else "REJECTED"
    E.set_hypothesis_status("L0-R", "PROMISING" if new else "CLOSED", level=0,
                            hypothesis="_bs(볼−스트라이크) 계열 지속성",
                            result=len(new))
    E.finish_experiment(dict(
        experiment_id=EXP, hypothesis_id="L0-R", type="A", level=0,
        started_at=E.read(E.CKPT)["start_time"], placebo=round(plac, 1),
        results={k: {kk: (round(vv, 4) if isinstance(vv, float)
                          else ([round(x, 2) for x in vv] if isinstance(vv, list) else vv))
                     for kk, vv in v.items()} for k, v in out.items()},
        decision=dec, artifact=None,
        what_we_learned=("EXP016 이 올린 _bs 계열의 지속성을 세 반대가설로 검사. "
                         f"통과 {len(new)}개")))
    json.dump(out, open(os.path.join(ROOT, "exp", "exp017_bs.json"), "w"),
              indent=1, default=float)
    print(f"\nDECISION = {dec}")


if __name__ == "__main__":
    main()
