r"""EXP002 — 새 국면 정의 (투수 주체 이진 차등). TYPE A.

살아남은 셋(좌우 대응·결정구·세트포지션)의 공통점은 **투수의 행동이 실제로
바뀌는 국면**이다. 그 기준으로 미검증 국면을 고른다.

    N1 상대 타자 수준     좋은 타자 상대로 신중해지는가 (제구 성향 변화)
    N2 상대 타자 성향     가운데를 잘 치는 타자 상대로 코너를 노리는가
    N3 이닝 선두타자      주자0·아웃0 — 이닝 시작 국면 (출루 허용 비용이 다르다)
    N4 자기 시즌 누적     시즌 내 워크로드 상하위 (컨디션/피로)
    N5 주자 배치 세분     주자 있는 행 안에서 1루만 vs 2·3루 포함
    N6 타자 좌우 자체     투수 정체성 없는 순수 타자손 (양성 대조)

N6 은 이미 손 차등에 흡수됐어야 하므로 **틀이 제대로 작동하는지 확인하는
양성 대조**다 (겹침 상관이 높게 나와야 정상이다).

기준선 C3 · 오라클 경기 단위 분할 + 위약 · k 는 과거 2전이로만.

    .\.venv\Scripts\python.exe -u research\exp002_newctx.py
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

EXP = "EXP002"
KGRID = [500, 1000, 2000, 5000, 20000]
PREV2 = {2022: (2020, 2021), 2023: (2021, 2022), 2024: (2022, 2023)}
HYP = {"N1 상대 타자 수준": "H007", "N2 상대 타자 중앙성향": "H008",
       "N3 이닝 선두타자": "H009", "N4 시즌 누적 상위": "H010",
       "N5 주자중 2·3루 포함": "H011", "N6 타자 좌우(대조)": "H012"}


def main():
    E.start_experiment(EXP, "H007-H012", "python research/exp002_newctx.py", "load")
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
    E.beat("C3 기준선")

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
    print(f"C3 기준선 2024 = {base:.1f}\n")

    def med_split(col, minn=30):
        """시즌 안에서 중앙값 이진화. 결측은 -1(제외). 행 자신의 컬럼만 쓴다."""
        v = g(col)
        out = np.full(len(v), -1, np.int64)
        for f in np.unique(season):
            m = season == f
            vv = v[m]
            ok = ~np.isnan(vv)
            if ok.sum() < minn:
                continue
            o = out[m]
            o[ok] = (vv[ok] >= np.median(vv[ok])).astype(np.int64)
            out[m] = o
        return out

    curn = np.expm1(g("cur_logn_pitch"))
    work = np.full(len(curn), -1, np.int64)
    for f in np.unique(season):
        m = season == f
        work[m] = (curn[m] >= np.median(curn[m])).astype(np.int64)
    scor = np.where(RUN == 1,
                    ((g("runner_on_2b") > 0) | (g("runner_on_3b") > 0)).astype(np.int64),
                    -1)

    CAND = {
        "N1 상대 타자 수준": med_split("asof_batter_success_rate"),
        "N2 상대 타자 중앙성향": med_split("asof_batter_middle_rate"),
        "N3 이닝 선두타자": ((g("num_runners_on") == 0)
                        & (g("outs_before") == 0)).astype(np.int64),
        "N4 시즌 누적 상위": work,
        "N5 주자중 2·3루 포함": scor,
        "N6 타자 좌우(대조)": (g("batter_hand").astype(int) == 2).astype(np.int64),
    }
    for n, h in HYP.items():
        E.set_hypothesis_status(h, "TESTING", family="투수 x 새 국면", type="A",
                                hypothesis=n, legality="LEGAL (행 자신의 컬럼)")

    GID = games(P[m24], g("asof_pitcher_n")[m24],
                g("asof_pitcher_prev1_game_success_rate")[m24],
                g("asof_pitcher_prev1_game_middle_rate")[m24])
    rng = np.random.default_rng(0)
    u, inv = np.unique(GID, return_inverse=True)
    half = (rng.random(len(u)) < 0.5)[inv]
    HD = ap(dtab(SAME, (2022, 2023), 1000), SAME, m24)
    TS = ap(dtab(TWO, (2022, 2023), 1000), TWO, m24)
    RN = ap(dtab(RUN, (2022, 2023), 2000), RUN, m24)

    def oracle(key):
        best = -1e9
        for k in (0, 150, 500, 2000, 10000):
            add = np.zeros(int(m24.sum()))
            for m in (half, ~half):
                uu, tb, _ = cells(key[m], resC[2024][m], k)
                add[~m] = look(uu, tb, key[~m])
            best = max(best, r2(C3[2024] + add, y[m24]) - base)
        return best

    print("=" * 108)
    print(f"{'국면':<22}{'=1비율':>8}{'잔차상관':>10}{'오라클':>8}{'위약':>7}"
          f"{'21→22':>8}{'22→23':>8}{'23→24':>8}{'k':>7}{'C3증분':>8}"
          f"{'hand/2S/run':>17}")
    print("=" * 108)
    out = {}
    for name, ctx in CAND.items():
        E.beat(f"probe {name}")
        cc = ctx[m24]
        ok = cc >= 0
        rc = float(np.corrcoef(cc[ok], resC[2024][ok])[0, 1])
        bo = oracle(P[m24] * 10 + np.clip(cc, 0, 1))
        bp = oracle(P[m24] * 10 + rng.integers(0, 2, int(m24.sum())))
        G = {k: [] for k in KGRID}
        for a, b in ((2021, 2022), (2022, 2023), (2023, 2024)):
            mb = season == b
            bs = r2(C3[b], y[mb])
            for k in KGRID:
                G[k].append(r2(C3[b] + ap(dtab(ctx, (a,), k), ctx, mb), y[mb]) - bs)
        kb = max(G, key=lambda k: np.mean(G[k][:2]))
        add24 = ap(dtab(ctx, (2022, 2023), kb), ctx, m24)
        inc = r2(C3[2024] + add24, y[m24]) - base
        ov = [float(np.corrcoef(add24, v)[0, 1]) if add24.std() > 0 else 0.
              for v in (HD, TS, RN)]
        out[name] = dict(oracle=bo, placebo=bp, k=kb, gains=G[kb], inc=inc,
                         overlap=ov, resid_corr=rc)
        print(f"{name:<22}{np.mean(cc == 1):>8.1%}{rc:>+10.4f}{bo:>8.1f}{bp:>7.1f}"
              + "".join(f"{v:>+8.1f}" for v in G[kb])
              + f"{kb:>7}{inc:>+8.1f}    {ov[0]:+.2f}/{ov[1]:+.2f}/{ov[2]:+.2f}")

    real = {k: v for k, v in out.items() if not k.startswith("N6")}
    best = max(real, key=lambda n: real[n]["inc"])
    bb = real[best]
    dec = ("PROMISING" if (bb["inc"] >= 4.8 and all(v > 0 for v in bb["gains"]))
           else "REJECTED")
    for n, h in HYP.items():
        v = out[n]
        E.set_hypothesis_status(
            h, "PROMISING" if (n == best and dec == "PROMISING") else "CLOSED",
            result=round(v["inc"], 2), oracle=round(v["oracle"], 1),
            transfer=[round(x, 1) for x in v["gains"]])
    E.finish_experiment(dict(
        experiment_id=EXP, hypothesis_id="H007-H012", type="A",
        started_at=E.read(E.CKPT)["start_time"],
        local_result={k: round(v["inc"], 2) for k, v in out.items()},
        transfer_result={k: [round(x, 1) for x in v["gains"]] for k, v in out.items()},
        oracle={k: round(v["oracle"], 1) for k, v in out.items()},
        placebo={k: round(v["placebo"], 1) for k, v in out.items()},
        decision=dec, artifact=None,
        what_we_learned=(f"최선 {best} C3증분 {bb['inc']:+.1f}, "
                         f"전이 {[round(x, 1) for x in bb['gains']]}")))
    json.dump(out, open(os.path.join(ROOT, "exp", "exp002_newctx.json"), "w"),
              indent=1, default=float)
    print(f"\nDECISION = {dec}   (best {best} {bb['inc']:+.1f})")
    print("  N6 은 양성 대조 — 겹침 상관이 높아야 틀이 정상이다")


if __name__ == "__main__":
    main()
