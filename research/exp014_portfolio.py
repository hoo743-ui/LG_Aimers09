r"""EXP014 — 약신호 포트폴리오. 개별로는 검출 불가능한 것들이 합쳐지는가.

## 왜 이 실험이 지금 가능해졌는가

지금까지의 하드 게이트는 `C3증분 >= +4.8` 이었다. 그런데 폴드 2024 의 시드
잡음이 ±7.2 다. 즉 **+2~5 짜리 진짜 신호는 개별로는 원리적으로 검출할 수 없다.**
게이트는 그것들을 전부 잡음과 함께 버려 왔다.

독립인 약신호 10개가 각 +0.5 면 합은 +5 다. 이건 새 정보가 아니라 **집계**의
문제이고, 지금까지 한 번도 시험하지 않았다.

## 다중검정을 어떻게 통제하는가 (이게 이 실험의 전부다)

후보를 243개 훑으면 우연 양성이 반드시 나온다. 그래서 선택과 평가를 가른다.

    선택   2022·2023 두 전이에서 **둘 다 양수**인 축만 고른다
    평가   2024 는 선택에 일절 쓰지 않는다 (숨긴 홀드아웃)

우연 양성은 2024 에서 평균 0 을 기여하므로, 포트폴리오가 2024 에서 양수면
그건 **진짜 약신호가 있다**는 뜻이다. 위약(무작위 이진 국면 243개로 같은 절차)
분포와 비교해 유의성을 읽는다.

## 반증 대상

가설 H: "게이트 아래에 버려진 것들은 전부 잡음이었다."
포트폴리오가 위약 분포를 넘으면 H 는 반증된다.

    .\.venv\Scripts\python.exe -u research\exp014_portfolio.py
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
from traj_probe import r2                                   # noqa: E402
from exp003_sweep import build_contexts                     # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

EXP = "EXP014"
KGRID = [1000, 5000, 20000]
PREV2 = {2022: (2020, 2021), 2023: (2021, 2022), 2024: (2022, 2023)}
N_PLACEBO = 25


def main():
    E.start_experiment(EXP, "L0-P", "python research/exp014_portfolio.py", "load")
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

    def stats(ctx, src):
        """(투수, 셀) 평균·개수를 한 번만 구한다. k 는 나중에 싸게 적용."""
        p = np.concatenate([P[season == f] for f in src])
        c = np.concatenate([ctx[season == f] for f in src])
        r = np.concatenate([res0[f] for f in src])
        ok = np.isin(c, (0, 1))
        gg = pd.DataFrame({"p": p[ok], "c": c[ok], "r": r[ok]}).groupby(
            ["p", "c"])["r"].agg(["mean", "size"]).unstack()
        if ("size", 0) not in gg or ("size", 1) not in gg:
            return None
        n0, n1 = gg[("size", 0)].fillna(0), gg[("size", 1)].fillna(0)
        d = (gg[("mean", 1)] - gg[("mean", 0)])
        ne = (n0 * n1) / (n0 + n1).replace(0, np.nan)
        return pd.DataFrame({"d": d, "ne": ne}).dropna()

    def shrunk(st, k):
        return st["d"] * st["ne"] / (st["ne"] + k)

    def apply_(t, ctx, m):
        h = np.where(ctx[m] == 1, .5, np.where(ctx[m] == 0, -.5, 0.))
        return pd.Series(P[m]).map(t).fillna(0.).to_numpy() * h

    C3, base = {}, {}
    for f in (2022, 2023, 2024):
        m = season == f
        v = pv0[f].copy()
        for a, (ctx, k) in AX.items():
            st = stats(ctx, PREV2[f])
            v += apply_(shrunk(st, k), ctx, m)
        C3[f] = v
        base[f] = r2(v, y[m])
    print(f"C3 기준선  2022 {base[2022]:.1f}  2023 {base[2023]:.1f}  "
          f"2024 {base[2024]:.1f}\n")
    E.beat("기준선")

    CTX = build_contexts(tr, season)
    print(f"후보 국면 {len(CTX)}개 — 각각 3폴드 전이 측정 (k 는 과거 2폴드로만)")

    def evaluate(items, tag):
        """국면 목록에 대해 (선택용 2폴드, 홀드아웃 2024) 이득을 잰다."""
        rec = {}
        for i, (nm, ctx) in enumerate(items):
            if i % 25 == 0:
                E.beat(f"{tag} {i}/{len(items)}")
            gains, kbest = {}, None
            per_k = {k: [] for k in KGRID}
            ok = True
            for f in (2022, 2023, 2024):
                m = season == f
                st = stats(ctx, PREV2[f])
                if st is None or not len(st):
                    ok = False
                    break
                for k in KGRID:
                    per_k[k].append(r2(C3[f] + apply_(shrunk(st, k), ctx, m),
                                       y[m]) - base[f])
            if not ok:
                continue
            kbest = max(KGRID, key=lambda k: np.mean(per_k[k][:2]))
            gains = per_k[kbest]
            rec[nm] = dict(k=kbest, g22=gains[0], g23=gains[1], g24=gains[2],
                           ctx=ctx)
        return rec

    real = evaluate(sorted(CTX.items()), "real")
    sel = {n: v for n, v in real.items() if v["g22"] > 0 and v["g23"] > 0}
    print(f"\n선택 (2022·2023 둘 다 양수) {len(sel)} / {len(real)}"
          f"   우연 기대 {len(real)/4:.0f}")
    hold = [v["g24"] for v in sel.values()]
    print(f"선택된 축의 2024(숨김) 평균 {np.mean(hold):+.3f}  "
          f"중앙 {np.median(hold):+.3f}  양수비율 {np.mean(np.array(hold)>0):.1%}")

    def portfolio(rec, sel_keys, fold, w):
        m = season == fold
        add = np.zeros(int(m.sum()))
        for n in sel_keys:
            v = rec[n]
            st = stats(v["ctx"], PREV2[fold])
            if st is None:
                continue
            add += apply_(shrunk(st, v["k"]), v["ctx"], m)
        return r2(C3[fold] + w * add, y[m]) - base[fold]

    print("\n" + "=" * 68)
    print(f"{'포트폴리오 가중 w':<20}{'2022(선택)':>13}{'2023(선택)':>13}"
          f"{'2024(숨김)':>13}")
    print("=" * 68)
    keys = list(sel)
    pf = {}
    for w in (0.25, 0.5, 1.0):
        E.beat(f"portfolio w={w}")
        r22 = portfolio(real, keys, 2022, w)
        r23 = portfolio(real, keys, 2023, w)
        r24 = portfolio(real, keys, 2024, w)
        pf[w] = (r22, r23, r24)
        print(f"{w:<20.2f}{r22:>+13.1f}{r23:>+13.1f}{r24:>+13.1f}")
    wbest = max(pf, key=lambda w: np.mean(pf[w][:2]))
    obs = pf[wbest][2]
    print(f"\n과거로 고른 w={wbest:g}  ->  2024(숨김) {obs:+.1f}")

    # ---- 위약 — 같은 절차를 무작위 국면으로 ----
    print(f"\n위약 {N_PLACEBO}회 (무작위 이진 국면 {len(CTX)}개로 동일 절차)")
    rng = np.random.default_rng(0)
    null = []
    for s in range(N_PLACEBO):
        E.beat(f"placebo {s}")
        items = []
        for j in range(len(CTX)):
            pr = rng.uniform(0.1, 0.9)
            items.append((f"p{j}", (rng.random(len(season)) < pr).astype(np.int64)))
        pr_ = evaluate(items, f"plac{s}")
        ks = [n for n, v in pr_.items() if v["g22"] > 0 and v["g23"] > 0]
        null.append(portfolio(pr_, ks, 2024, wbest))
    null = np.array(null)
    p = float(np.mean(null >= obs))
    print(f"  위약 2024 분포  평균 {null.mean():+.2f}  sd {null.std():.2f}"
          f"  5~95% {np.percentile(null,5):+.2f}~{np.percentile(null,95):+.2f}")
    print(f"  관측 {obs:+.1f}  ->  p = {p:.3f}")

    dec = "PROMISING" if (p < 0.05 and obs >= 4.8) else "REJECTED"
    E.set_hypothesis_status("L0-P", "PROMISING" if dec == "PROMISING" else "CLOSED",
                            level=0, hypothesis="약신호 포트폴리오",
                            result=round(obs, 2))
    E.finish_experiment(dict(
        experiment_id=EXP, hypothesis_id="L0-P", type="A", level=0,
        started_at=E.read(E.CKPT)["start_time"],
        n_candidates=len(real), n_selected=len(sel), w=wbest,
        observed_2024=round(obs, 2), placebo_mean=round(float(null.mean()), 2),
        placebo_sd=round(float(null.std()), 2), p_value=p,
        multiple_testing_control="선택은 2022·2023, 평가는 2024 숨김",
        decision=dec, artifact=None,
        what_we_learned=(f"{len(real)}개 후보 중 과거 2폴드 양수 {len(sel)}개를 "
                         f"묶으면 숨긴 2024 에서 {obs:+.1f} (위약 평균 "
                         f"{null.mean():+.2f}, p={p:.3f})")))
    json.dump(dict(n=len(real), sel=len(sel), w=wbest, obs=obs,
                   null_mean=float(null.mean()), null_sd=float(null.std()), p=p,
                   top=sorted([(n, v["g22"], v["g23"], v["g24"])
                               for n, v in sel.items()],
                              key=lambda t: -(t[1] + t[2]))[:20]),
              open(os.path.join(ROOT, "exp", "exp014_portfolio.json"), "w"),
              indent=1, default=float)
    print(f"\nDECISION = {dec}")


if __name__ == "__main__":
    main()
