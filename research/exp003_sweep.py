r"""EXP003 — 국면 전수 소진 테스트 (EXHAUSTION SWEEP). TYPE A.

## 왜 전수인가

국면을 하나씩 추측해 26가지를 쟀고 3개만 살았다. 개별 추측을 계속하는 대신
**정의 가능한 이진 국면을 기계적으로 전부 열거**해서 "남은 것이 있는가"를
한 번에 판정한다. 이 실험이 음성이면 `투수 x 이진 국면` family 는 소진이다.

## 열거 규칙 (합법성)

그 행 **자신의 컬럼**만 쓴다. 평가셋 다른 행·집계·미래 정보를 쓰지 않는다.

    저기수 정수/범주   각 수준을 1 vs 나머지 로 (수준당 1개)
    연속               그 시즌 중앙값 / 상·하위 25% 분할
    이미 채택된 3축     대조군으로 함께 넣는다 (반드시 살아나와야 정상)

## 판정

오라클(경기 단위 분할, 교차적합)에서 **같은 셀 수의 위약**을 뺀 값을 본다.
`오라클 − 위약 <= +3` 이면 그 국면에는 투수별 차등 정보가 **없다**.
살아남은 것만 전이(3폴드)를 마저 잰다.

    .\.venv\Scripts\python.exe -u research\exp003_sweep.py
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

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

EXP = "EXP003"
KGRID = [500, 1000, 2000, 5000, 20000]
PREV2 = {2022: (2020, 2021), 2023: (2021, 2022), 2024: (2022, 2023)}
GATE = 3.0            # 오라클 − 위약 이 이보다 커야 전이를 잰다
SKIP = {"control_success", "season", "pitcher_id", "batter_id",
        "pitcher_team_id", "batter_team_id", "game_id", "game_date"}


def build_contexts(tr, season):
    """정의 가능한 이진 국면을 기계적으로 열거한다. 결측/제외는 -1."""
    ctx = {}
    for c in tr.columns:
        if c in SKIP:
            continue
        s = tr[c]
        if s.dtype == object or str(s.dtype) == "category":
            u = pd.unique(s.dropna())
            if len(u) > 8:
                continue
            for lv in sorted(map(str, u)):
                v = np.where(s.isna().to_numpy(), -1,
                             (s.astype(str).to_numpy() == lv).astype(np.int64))
                if 0.02 < np.mean(v == 1) < 0.98:
                    ctx[f"{c}=={lv}"] = v
            continue
        v = s.to_numpy(np.float64)
        u = np.unique(v[~np.isnan(v)])
        if len(u) <= 1:
            continue
        if len(u) <= 8:                                      # 저기수
            for lv in u:
                b = np.where(np.isnan(v), -1, (v == lv).astype(np.int64))
                if 0.02 < np.mean(b == 1) < 0.98:
                    ctx[f"{c}=={lv:g}"] = b
        else:                                                # 연속 — 시즌내 분위
            for tag, (lo, hi) in (("상위50%", (50, None)), ("상위25%", (75, None)),
                                  ("하위25%", (None, 25))):
                b = np.full(len(v), -1, np.int64)
                for f in np.unique(season):
                    m = season == f
                    vv = v[m]
                    ok = ~np.isnan(vv)
                    if ok.sum() < 1000:
                        continue
                    o = b[m]
                    if hi is None:
                        o[ok] = (vv[ok] >= np.percentile(vv[ok], lo)).astype(np.int64)
                    else:
                        o[ok] = (vv[ok] <= np.percentile(vv[ok], hi)).astype(np.int64)
                    b[m] = o
                if 0.02 < np.mean(b == 1) < 0.98:
                    ctx[f"{c} {tag}"] = b
    return ctx


def main():
    E.start_experiment(EXP, "H013", "python research/exp003_sweep.py", "load")
    tr = build_df()
    season = tr["season"].to_numpy()
    y = tr["control_success"].to_numpy(np.float64)
    P = tr["pitcher_id"].to_numpy(np.int64)

    def g(c):
        return tr[c].to_numpy(np.float64)

    SAME = (g("pitcher_hand") == g("batter_hand")).astype(int)
    TWO = (g("strikes_before").astype(int) == 2).astype(int)
    RUN = (g("num_runners_on") > 0).astype(int)

    pv0, res0 = {}, {}
    for f in (2020, 2021, 2022, 2023, 2024):
        m = season == f
        pv0[f] = (np.load(os.path.join(ROOT, "exp", f"prod_champ_{f}.npy"))[:2].mean(0)
                  + post_for(tr, y, season < f, m))
        res0[f] = y[m] - pv0[f]

    def dtab(ctx, src, k):
        p = np.concatenate([P[season == f] for f in src])
        c = np.concatenate([ctx[season == f] for f in src])
        r = np.concatenate([res0[f] for f in src])
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
        return pd.Series(P[m]).map(t).fillna(0.).to_numpy() * h

    C3, resC = {}, {}
    for f in (2022, 2023, 2024):
        m = season == f
        s = PREV2[f]
        C3[f] = (pv0[f] + ap(dtab(SAME, s, 1000), SAME, m)
                 + ap(dtab(TWO, s, 1000), TWO, m) + ap(dtab(RUN, s, 2000), RUN, m))
        resC[f] = y[m] - C3[f]
    m24 = season == 2024
    base = r2(C3[2024], y[m24])

    CTX = build_contexts(tr, season)
    CTX["[대조] 같은손"] = SAME
    CTX["[대조] 2스트라이크"] = TWO
    CTX["[대조] 주자있음"] = RUN
    print(f"C3 기준선 2024 = {base:.1f}   열거된 국면 {len(CTX)}개\n")

    GID = games(P[m24], g("asof_pitcher_n")[m24],
                g("asof_pitcher_prev1_game_success_rate")[m24],
                g("asof_pitcher_prev1_game_middle_rate")[m24])
    rng = np.random.default_rng(0)
    u, inv = np.unique(GID, return_inverse=True)
    half = (rng.random(len(u)) < 0.5)[inv]
    n24 = int(m24.sum())

    def oracle(key):
        best = -1e9
        for k in (0, 150, 500, 2000, 10000):
            add = np.zeros(n24)
            for m in (half, ~half):
                uu, tb, _ = cells(key[m], resC[2024][m], k)
                add[~m] = look(uu, tb, key[~m])
            best = max(best, r2(C3[2024] + add, y[m24]) - base)
        return best

    # 위약은 셀 수만 같은 무작위 국면. 3회 평균으로 안정화한다.
    plac = float(np.mean([oracle(P[m24] * 10 + np.random.default_rng(s).integers(0, 2, n24))
                          for s in (1, 2, 3)]))
    print(f"위약 기준선 (무작위 이진 국면 3회 평균) = {plac:+.1f}")
    print(f"통과선 = 위약 + {GATE:.0f} = {plac + GATE:+.1f}\n")

    rows, survivors = [], []
    for i, (name, ctx) in enumerate(sorted(CTX.items())):
        if i % 20 == 0:
            E.beat(f"sweep {i}/{len(CTX)}")
        cc = ctx[m24]
        if not (0.02 < np.mean(cc == 1) < 0.98):
            continue
        orc = oracle(P[m24] * 10 + np.clip(cc, 0, 1))
        rows.append((name, float(np.mean(cc == 1)), orc, orc - plac))
        if orc - plac > GATE:
            survivors.append((name, ctx, orc))

    rows.sort(key=lambda t: -t[2])
    print("=" * 74)
    print(f"{'국면 (오라클 상위 20)':<44}{'=1비율':>9}{'오라클':>9}{'−위약':>9}")
    print("=" * 74)
    for name, fr, orc, dv in rows[:20]:
        mark = "  ✅" if dv > GATE else ""
        print(f"{name:<44}{fr:>9.1%}{orc:>9.1f}{dv:>+9.1f}{mark}")
    print(f"\n열거 {len(rows)}개 중 통과선 초과 {len(survivors)}개")

    out = {"base": base, "placebo": plac, "gate": GATE,
           "all": [{"ctx": n, "frac": f, "oracle": o, "minus_placebo": d}
                   for n, f, o, d in rows]}
    res = []
    if survivors:
        print("\n" + "=" * 92)
        print(f"{'통과 국면 — 전이 측정':<40}{'21→22':>9}{'22→23':>9}{'23→24':>9}"
              f"{'k':>8}{'C3증분':>9}")
        print("=" * 92)
        for name, ctx, orc in survivors:
            E.beat(f"transfer {name}")
            G = {k: [] for k in KGRID}
            for a, b in ((2021, 2022), (2022, 2023), (2023, 2024)):
                mb = season == b
                bs = r2(C3[b], y[mb])
                for k in KGRID:
                    G[k].append(r2(C3[b] + ap(dtab(ctx, (a,), k), ctx, mb), y[mb]) - bs)
            kb = max(G, key=lambda k: np.mean(G[k][:2]))
            inc = r2(C3[2024] + ap(dtab(ctx, (2022, 2023), kb), ctx, m24), y[m24]) - base
            res.append(dict(ctx=name, oracle=orc, k=kb, gains=G[kb], inc=inc))
            print(f"{name:<40}" + "".join(f"{v:>+9.1f}" for v in G[kb])
                  + f"{kb:>8}{inc:>+9.1f}")
    out["survivors"] = res
    new = [r for r in res if not r["ctx"].startswith("[대조]")
           and r["inc"] >= 4.8 and all(v > 0 for v in r["gains"])]
    dec = "PROMISING" if new else "REJECTED"
    E.set_hypothesis_status("H013", "CLOSED" if not new else "PROMISING",
                            family="투수 x 이진 국면 전수", type="A",
                            hypothesis="정의 가능한 모든 이진 국면의 투수 차등",
                            result=len(new))
    E.finish_experiment(dict(
        experiment_id=EXP, hypothesis_id="H013", type="A",
        started_at=E.read(E.CKPT)["start_time"],
        n_contexts=len(rows), placebo=round(plac, 1),
        n_survivors=len(survivors), decision=dec,
        survivors=[{k: (round(v, 2) if isinstance(v, float) else v)
                    for k, v in r.items()} for r in res],
        what_we_learned=(f"47열에서 열거 가능한 이진 국면 {len(rows)}개 전수. "
                         f"위약({plac:+.1f}) + {GATE:.0f} 초과 {len(survivors)}개, "
                         f"그중 신규 채택 후보 {len(new)}개")))
    json.dump(out, open(os.path.join(ROOT, "exp", "exp003_sweep.json"), "w"),
              indent=1, default=float)
    print(f"\nDECISION = {dec}   신규 채택 후보 {len(new)}개")


if __name__ == "__main__":
    main()
