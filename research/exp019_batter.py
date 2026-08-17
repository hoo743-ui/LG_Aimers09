r"""EXP019 — 주체를 타자로. 지속성 통계량으로 재측정. 학습 0회.

## REOPEN_JUSTIFICATION

타자 주체 차등은 `오라클 <= 위약` 과 전이 이득으로 닫혔다. 둘 다 **저검정력**
통계량이다 — EXP016 에서 LB 로 검증된 손 차등조차 z=1.91 이었다.

지속성(`corr_b(d_b(s), d_b(s+1))`)은 타자 수백 명에 걸친 상관이라 자유도가
훨씬 크다. **같은 축을 같은 잣대로 반복하는 것이 아니라 잣대를 바꾼다.**

기제도 있다 — 타자의 좌우 대응 감도는 실재하는 형질이고, 우리가 채택한 유일한
축(투수 x 타자손)의 **거울상**이다. 투수 쪽에서 지속되는 것이 타자 쪽에서
지속되지 않을 이유가 선험적으로 없다.

## EXP018 이 가르친 것을 반영한다

지속성만으로는 부족하다. 구성 교란이면 정보 0 에서도 지속성이 나온다.
그래서 셋을 **함께** 본다.

    지속성      d_b 의 시즌 간 상관 (감쇠보정 포함)
    오라클      경기 단위 분할 교차적합 (위약 대조)
    구성차      국면이 다른 축을 몰래 싣고 있는가

    .\.venv\Scripts\python.exe -u research\exp019_batter.py
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
from exp016_persist import wcorr                            # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

EXP = "EXP019"
SEASONS = (2020, 2021, 2022, 2023, 2024)
PAIRS = [(2020, 2021), (2021, 2022), (2022, 2023), (2023, 2024)]
PREV2 = {2022: (2020, 2021), 2023: (2021, 2022), 2024: (2022, 2023)}
KGRID = [500, 1000, 2000, 5000, 20000]
MIN_NE = 20.0


def main():
    E.start_experiment(EXP, "L0-T", "python research/exp019_batter.py", "load")
    tr = build_df()
    season = tr["season"].to_numpy()
    y = tr["control_success"].to_numpy(np.float64)
    P = tr["pitcher_id"].to_numpy(np.int64)
    B = tr["batter_id"].to_numpy(np.int64)

    def g(c):
        return tr[c].to_numpy(np.float64)

    SAME = (g("pitcher_hand") == g("batter_hand")).astype(int)
    TWO = (g("strikes_before").astype(int) == 2).astype(int)
    RUN = (g("num_runners_on") > 0).astype(int)
    BS = g("balls_before") - g("strikes_before")
    AX = {"hand": (SAME, 1000), "2S": (TWO, 1000), "runner": (RUN, 2000)}

    pv0, res0 = {}, {}
    for f in SEASONS:
        m = season == f
        pv0[f] = (np.load(os.path.join(ROOT, "exp", f"prod_champ_{f}.npy"))[:2].mean(0)
                  + post_for(tr, y, season < f, m))
        res0[f] = y[m] - pv0[f]

    def dtab(subj, ctx, src, k, ret_raw=False):
        s = np.concatenate([subj[season == f] for f in src])
        c = np.concatenate([ctx[season == f] for f in src])
        r = np.concatenate([res0[f] for f in src])
        ok = np.isin(c, (0, 1))
        gg = pd.DataFrame({"s": s[ok], "c": c[ok], "r": r[ok]}).groupby(
            ["s", "c"])["r"].agg(["mean", "size"]).unstack()
        if ("size", 0) not in gg or ("size", 1) not in gg:
            return None
        n0, n1 = gg[("size", 0)].fillna(0), gg[("size", 1)].fillna(0)
        ne = (n0 * n1) / (n0 + n1).replace(0, np.nan)
        d = gg[("mean", 1)] - gg[("mean", 0)]
        if ret_raw:
            return pd.DataFrame({"d": d, "ne": ne}).dropna()
        return (d * ne / (ne + k)).dropna()

    def ap(t, subj, ctx, m):
        if t is None:
            return np.zeros(int(m.sum()))
        h = np.where(ctx[m] == 1, .5, np.where(ctx[m] == 0, -.5, 0.))
        return pd.Series(subj[m]).map(t).fillna(0.).to_numpy() * h

    C3, resC = {}, {}
    for f in (2022, 2023, 2024):
        m = season == f
        v = pv0[f].copy()
        for a, (ctx, k) in AX.items():
            v += ap(dtab(P, ctx, PREV2[f], k), P, ctx, m)
        C3[f], resC[f] = v, y[m] - v
    m24 = season == 2024
    base = r2(C3[2024], y[m24])
    print(f"C3 기준선 2024 = {base:.1f}")
    print(f"투수 {len(np.unique(P)):,}명  타자 {len(np.unique(B)):,}명\n")
    E.beat("기준선")

    def persist(subj, ctx, shuffle=False):
        per = {}
        for f in SEASONS:
            m = season == f
            s = subj[m]
            if shuffle:
                s = np.random.default_rng(f).permutation(s)
            c, r = ctx[m], res0[f]
            ok = np.isin(c, (0, 1))
            gg = pd.DataFrame({"s": s[ok], "c": c[ok], "r": r[ok]}).groupby(
                ["s", "c"])["r"].agg(["mean", "size"]).unstack()
            if ("size", 0) not in gg or ("size", 1) not in gg:
                continue
            n0, n1 = gg[("size", 0)].fillna(0), gg[("size", 1)].fillna(0)
            ne = (n0 * n1) / (n0 + n1).replace(0, np.nan)
            per[f] = pd.DataFrame({"d": gg[("mean", 1)] - gg[("mean", 0)],
                                   "ne": ne}).dropna()
        vals, ns = [], []
        for a, b in PAIRS:
            if a not in per or b not in per:
                continue
            j = per[a].join(per[b], lsuffix="a", rsuffix="b", how="inner")
            j = j[(j["nea"] >= MIN_NE) & (j["neb"] >= MIN_NE)]
            if len(j) < 40:
                continue
            vals.append(wcorr(j["da"].to_numpy(), j["db"].to_numpy(),
                              (j["nea"] * j["neb"] / (j["nea"] + j["neb"])).to_numpy()))
            ns.append(len(j))
        if not vals:
            return None
        cr, n = float(np.nanmean(vals)), int(np.mean(ns))
        return dict(cross=cr, n=n, z=cr * np.sqrt(max(n - 3, 1)))

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

    CASES = [
        ("타자 x 상대 투수손(플래툰)", B, SAME),
        ("타자 x 2스트라이크", B, TWO),
        ("타자 x 주자유무", B, RUN),
        ("타자 x 카운트우위", B, (BS > 0).astype(int)),
        ("[대조] 투수 x 타자손", P, SAME),
        ("[대조] 투수 x 2스트라이크", P, TWO),
    ]
    print("=" * 104)
    print(f"{'축':<28}{'주체수':>8}{'지속성':>10}{'z':>7}{'섞기 z':>9}"
          f"{'오라클':>9}{'위약':>8}{'구성차':>9}")
    print("=" * 104)
    plac = oracle(B[m24] * 10 + rng.integers(0, 2, int(m24.sum())))
    out = {}
    for nm, subj, ctx in CASES:
        E.beat(nm)
        pr = persist(subj, ctx)
        sh = persist(subj, ctx, shuffle=True)
        orc = oracle(subj[m24] * 10 + np.clip(ctx[m24], 0, 1))
        c24, s24 = ctx[m24], subj[m24]
        ok = c24 >= 0
        d = pd.DataFrame({"s": s24[ok], "c": c24[ok], "b": BS[m24][ok]}).groupby(
            ["s", "c"])["b"].mean().unstack()
        comp = float((d[1] - d[0]).mean()) if 0 in d and 1 in d else float("nan")
        out[nm] = dict(persist=pr["cross"], z=pr["z"], shuffle_z=sh["z"] if sh else None,
                       n=pr["n"], oracle=orc, placebo=plac, comp=comp)
        print(f"{nm:<28}{pr['n']:>8,}{pr['cross']:>+10.4f}{pr['z']:>7.2f}"
              f"{(sh['z'] if sh else float('nan')):>9.2f}{orc:>9.1f}{plac:>8.1f}"
              f"{comp:>+9.3f}")

    # 지속성이 대조축을 넘는 것만 전이까지 간다
    ctrl_z = out["[대조] 투수 x 타자손"]["z"]
    live = [nm for nm, subj, ctx in CASES
            if not nm.startswith("[대조]") and out[nm]["z"] >= 0.7 * ctrl_z]
    print(f"\n대조(투수x타자손) z={ctrl_z:.2f} 의 70% 이상인 타자 축 {len(live)}개")
    if live:
        print("\n" + "=" * 84)
        print(f"{'축':<28}{'21→22':>10}{'22→23':>10}{'23→24':>10}{'k':>8}{'C3증분':>10}")
        print("=" * 84)
        for nm in live:
            subj, ctx = [(s, c) for n2, s, c in CASES if n2 == nm][0]
            G = {k: [] for k in KGRID}
            for a, b in ((2021, 2022), (2022, 2023), (2023, 2024)):
                mb = season == b
                bs_ = r2(C3[b], y[mb])
                for k in KGRID:
                    G[k].append(r2(C3[b] + ap(dtab(subj, ctx, (a,), k), subj, ctx, mb),
                                   y[mb]) - bs_)
            kb = max(G, key=lambda k: np.mean(G[k][:2]))
            inc = r2(C3[2024] + ap(dtab(subj, ctx, (2022, 2023), kb), subj, ctx, m24),
                     y[m24]) - base
            out[nm]["gains"], out[nm]["k"], out[nm]["inc"] = G[kb], kb, inc
            print(f"{nm:<28}" + "".join(f"{v:>+10.1f}" for v in G[kb])
                  + f"{kb:>8}{inc:>+10.1f}")

    new = [nm for nm in live if out[nm].get("inc", -9) >= 4.8
           and all(v > 0 for v in out[nm].get("gains", [-1]))]
    dec = "PROMISING" if new else "REJECTED"
    E.set_hypothesis_status("L0-T", "PROMISING" if new else "CLOSED", level=0,
                            hypothesis="타자 주체 지속성 재측정", result=len(new))
    E.finish_experiment(dict(
        experiment_id=EXP, hypothesis_id="L0-T", type="A", level=0,
        started_at=E.read(E.CKPT)["start_time"],
        reopen_justification="저검정력 통계량(오라클·이득)으로 닫힌 축을 고검정력 지속성으로 재측정",
        results={k: {kk: (round(vv, 4) if isinstance(vv, float)
                          else ([round(x, 2) for x in vv] if isinstance(vv, list) else vv))
                     for kk, vv in v.items()} for k, v in out.items()},
        decision=dec, artifact=None,
        what_we_learned=(f"타자 주체 4축의 지속성을 대조(투수x타자손 z="
                         f"{ctrl_z:.2f})와 나란히 측정. 통과 {len(new)}개")))
    json.dump(out, open(os.path.join(ROOT, "exp", "exp019_batter.json"), "w"),
              indent=1, default=float)
    print(f"\nDECISION = {dec}")


if __name__ == "__main__":
    main()
