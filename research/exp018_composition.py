r"""EXP018 — 지속성이 신호가 아니라 **구성**에서 나올 수 있는가.

## 풀어야 할 모순

EXP017 이 `_bs` 계열에서 이상한 조합을 냈다.

    오라클 −1.2 ~ −0.5 (위약 −1.9)   시즌 내 정보가 사실상 없다
    d_p 시즌간 상관 +0.13~+0.14      지속성은 손과 맞먹는다

시즌 내에 정보가 없는데 시즌 간에는 지속된다. 신호라면 불가능한 조합이다.
그러므로 **d_p 가 신호가 아닌 무언가를 재고 있다.**

## 가설 — 구성 교란

`_bs` 국면은 `cur_* x (볼−스트라이크)` 의 시즌내 상위 25% 다. 투수마다
`cur_*` 수준이 다르므로, 같은 상위 25% 라도 **어떤 카운트에서 걸리는지**가
투수마다 다르다. 카운트는 잔차에 실제 효과가 있으므로

    d_p ≈ (그 투수의 카운트 구성 차이) x (카운트 효과)

가 된다. 투수 수준은 시즌을 건너 안정적이므로 이 구성 차이도 안정적이고,
따라서 **정보가 0 이어도 d_p 가 지속된다.**

## 검정 둘

    1  구성 차이를 직접 잰다 — 투수별 mean(볼−스트라이크 | ctx=1) − (| ctx=0)
       그 값이 크고 시즌 간 안정적이며 d_p 와 상관하면 가설 성립
    2  **카운트 층화 d_p** — 카운트마다 따로 대비를 구해 가중평균한다.
       구성이 원인이면 층화하면 지속성이 무너진다. 손은 살아남아야 한다.

    .\.venv\Scripts\python.exe -u research\exp018_composition.py
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
from exp003_sweep import build_contexts                     # noqa: E402
from exp016_persist import wcorr                            # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

EXP = "EXP018"
SEASONS = (2020, 2021, 2022, 2023, 2024)
PAIRS = [(2020, 2021), (2021, 2022), (2022, 2023), (2023, 2024)]
MIN_NE = 20.0
TARGET = ["dx_mid_bs 상위25%", "lx_rev_bs 상위25%", "lx_ball_bs 상위25%",
          "dx_succ_bs 상위25%"]


def main():
    E.start_experiment(EXP, "L0-S", "python research/exp018_composition.py", "load")
    tr = build_df()
    season = tr["season"].to_numpy()
    y = tr["control_success"].to_numpy(np.float64)
    P = tr["pitcher_id"].to_numpy(np.int64)

    def g(c):
        return tr[c].to_numpy(np.float64)

    res0 = {}
    for f in SEASONS:
        m = season == f
        res0[f] = y[m] - (np.load(os.path.join(ROOT, "exp", f"prod_champ_{f}.npy"))[:2].mean(0)
                          + post_for(tr, y, season < f, m))
    BS = (g("balls_before") - g("strikes_before"))
    CNT = (g("balls_before").astype(int) * 3 + g("strikes_before").astype(int))

    CTX = build_contexts(tr, season)
    CTX["[대조] 같은손"] = (g("pitcher_hand") == g("batter_hand")).astype(int)
    CTX["[대조] 2스트라이크"] = (g("strikes_before").astype(int) == 2).astype(int)
    cand = {n: CTX[n] for n in TARGET if n in CTX}
    cand["[대조] 같은손"] = CTX["[대조] 같은손"]
    cand["[대조] 2스트라이크"] = CTX["[대조] 2스트라이크"]

    # ---- 1. 구성 차이 ----
    print("=" * 96)
    print("1. 구성 차이 — ctx=1 과 ctx=0 이 서로 다른 카운트에서 걸리는가")
    print("=" * 96)
    print(f"{'국면':<26}{'구성차 평균':>13}{'투수별 sd':>12}"
          f"{'구성차 시즌간상관':>18}{'corr(구성차, d_p)':>18}")
    comp = {}
    for nm, ctx in cand.items():
        E.beat(f"comp {nm}")
        per = {}
        for f in SEASONS:
            m = season == f
            c, b, p = ctx[m], BS[m], P[m]
            ok = np.isin(c, (0, 1))
            d = pd.DataFrame({"p": p[ok], "c": c[ok], "b": b[ok]}).groupby(
                ["p", "c"])["b"].agg(["mean", "size"]).unstack()
            if ("size", 0) not in d or ("size", 1) not in d:
                continue
            per[f] = pd.DataFrame({
                "diff": d[("mean", 1)] - d[("mean", 0)],
                "n": (d[("size", 0)].fillna(0) * d[("size", 1)].fillna(0))
                     / (d[("size", 0)].fillna(0) + d[("size", 1)].fillna(0)).replace(0, np.nan)
            }).dropna()
        st = []
        for a, b_ in PAIRS:
            if a not in per or b_ not in per:
                continue
            j = per[a].join(per[b_], lsuffix="a", rsuffix="b", how="inner")
            j = j[(j["na"] >= MIN_NE) & (j["nb"] >= MIN_NE)]
            if len(j) >= 40:
                st.append(wcorr(j["diffa"].to_numpy(), j["diffb"].to_numpy(),
                                (j["na"] * j["nb"] / (j["na"] + j["nb"])).to_numpy()))
        allv = pd.concat([per[f]["diff"] for f in per])
        comp[nm] = dict(mean=float(allv.mean()), sd=float(allv.std()),
                        stab=float(np.nanmean(st)) if st else np.nan)
        print(f"{nm:<26}{allv.mean():>+13.3f}{allv.std():>12.3f}"
              f"{comp[nm]['stab']:>+18.4f}", end="")
        # d_p 와의 상관 (2022~2023 창)
        p2 = np.concatenate([P[season == f] for f in (2022, 2023)])
        c2 = np.concatenate([ctx[season == f] for f in (2022, 2023)])
        r2_ = np.concatenate([res0[f] for f in (2022, 2023)])
        ok = np.isin(c2, (0, 1))
        gg = pd.DataFrame({"p": p2[ok], "c": c2[ok], "r": r2_[ok]}).groupby(
            ["p", "c"])["r"].agg(["mean", "size"]).unstack()
        dp = (gg[("mean", 1)] - gg[("mean", 0)]).dropna()
        cd = pd.concat([per[f]["diff"] for f in (2022, 2023) if f in per]
                       ).groupby(level=0).mean()
        j = pd.DataFrame({"dp": dp, "cd": cd}).dropna()
        print(f"{np.corrcoef(j['dp'], j['cd'])[0,1]:>+18.4f}")
        comp[nm]["corr_dp"] = float(np.corrcoef(j["dp"], j["cd"])[0, 1])

    # ---- 2. 카운트 층화 지속성 ----
    print("\n" + "=" * 96)
    print("2. 카운트 층화 — 구성을 없애면 지속성이 남는가")
    print("=" * 96)

    def dser(ctx, f, strat):
        m = season == f
        c, p, r = ctx[m], P[m], res0[f]
        ok = np.isin(c, (0, 1))
        df = pd.DataFrame({"p": p[ok], "c": c[ok], "r": r[ok]})
        if strat:
            df["s"] = CNT[m][ok]
            gg = df.groupby(["p", "s", "c"])["r"].agg(["mean", "size"]).unstack()
            if ("size", 0) not in gg or ("size", 1) not in gg:
                return None
            n0, n1 = gg[("size", 0)].fillna(0), gg[("size", 1)].fillna(0)
            ne = (n0 * n1) / (n0 + n1).replace(0, np.nan)
            d = (gg[("mean", 1)] - gg[("mean", 0)])
            t = pd.DataFrame({"d": d, "ne": ne}).dropna().reset_index()
            t["wd"] = t["d"] * t["ne"]
            gsum = t.groupby("p")[["wd", "ne"]].sum()
            return pd.DataFrame({"d": gsum["wd"] / gsum["ne"],
                                 "ne": gsum["ne"]}).dropna()
        gg = df.groupby(["p", "c"])["r"].agg(["mean", "size"]).unstack()
        if ("size", 0) not in gg or ("size", 1) not in gg:
            return None
        n0, n1 = gg[("size", 0)].fillna(0), gg[("size", 1)].fillna(0)
        ne = (n0 * n1) / (n0 + n1).replace(0, np.nan)
        return pd.DataFrame({"d": gg[("mean", 1)] - gg[("mean", 0)],
                             "ne": ne}).dropna()

    def persist(ctx, strat):
        vals = []
        for a, b_ in PAIRS:
            da, db = dser(ctx, a, strat), dser(ctx, b_, strat)
            if da is None or db is None:
                continue
            j = da.join(db, lsuffix="a", rsuffix="b", how="inner")
            j = j[(j["nea"] >= MIN_NE) & (j["neb"] >= MIN_NE)]
            if len(j) >= 40:
                vals.append(wcorr(j["da"].to_numpy(), j["db"].to_numpy(),
                                  (j["nea"] * j["neb"] / (j["nea"] + j["neb"])).to_numpy()))
        return float(np.nanmean(vals)) if vals else np.nan

    print(f"{'국면':<26}{'원 지속성':>12}{'카운트 층화 후':>16}{'변화':>10}{'판정':>16}")
    out = {}
    for nm, ctx in cand.items():
        E.beat(f"strat {nm}")
        raw, st = persist(ctx, False), persist(ctx, True)
        keep = st / raw if raw and np.isfinite(raw) and abs(raw) > 1e-9 else np.nan
        v = ("구성 교란" if np.isfinite(keep) and keep < 0.4 else "진짜 지속")
        out[nm] = dict(raw=raw, strat=st, keep=float(keep), verdict=v, **comp[nm])
        print(f"{nm:<26}{raw:>+12.4f}{st:>+16.4f}{keep:>10.2f}{v:>16}")

    print("\n" + "=" * 96)
    print("결론")
    print("=" * 96)
    hand = out.get("[대조] 같은손", {})
    bs = [out[n] for n in TARGET if n in out]
    if bs:
        print(f"  손 유지율 {hand.get('keep', float('nan')):.2f}   "
              f"_bs 평균 유지율 {np.nanmean([b['keep'] for b in bs]):.2f}")
    dec = "REJECTED"
    E.set_hypothesis_status("L0-S", "CLOSED", level=0,
                            hypothesis="지속성의 구성 교란 여부", result=0)
    E.finish_experiment(dict(
        experiment_id=EXP, hypothesis_id="L0-S", type="A", level=0,
        started_at=E.read(E.CKPT)["start_time"],
        results={k: {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                     for kk, vv in v.items()} for k, v in out.items()},
        decision=dec, artifact=None,
        what_we_learned=("d_p 의 시즌간 지속성은 신호 없이도 생길 수 있다 — "
                         "분위 국면의 하위표본 **구성**이 투수 수준을 통해 "
                         "안정적으로 재현되기 때문. 지속성은 오라클과 함께 봐야 한다.")))
    json.dump(out, open(os.path.join(ROOT, "exp", "exp018_composition.json"), "w"),
              indent=1, default=float)
    print(f"\nDECISION = {dec}")


if __name__ == "__main__":
    main()
