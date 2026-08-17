r"""EXP006 — EXP005 의 +39.3 합법성 분해. 학습 0회.

## 왜 이 실험이 필요한가

EXP005 의 L2(대비/센터링)가 C3 잔차에서 +39.3 을 냈다. 채택 후보가 아니라
**누수 의심**이다. 센터링 8열은

    v - mean(v | 투수)      투수의 그 시즌 평균
    v - mean(v | 카운트)     카운트 그룹의 그 시즌 평균

인데 두 평균 모두 **2024 안에서** 계산됐다. 2025 에서 같은 값을 만들려면
평가셋의 다른 행이 필요하다 — 규정 4 위반이다. 과거에 같은 자리에서
"투수 평균 피처 +46.3" 을 잡은 적이 있고 이번 값도 그 근처다.

## 분해

같은 열을 두 판본으로 만든다.

    ILLEGAL   그룹 평균을 목표 시즌(2024) 안에서 계산   <- EXP005 가 한 것
    LEGAL     그룹 평균을 **과거 시즌에서만** 계산해 상수 조회로 적용
              (투수 평균은 2022~2023, 카운트 평균도 2022~2023)

LEGAL 판본은 그 행 자신의 값과 과거 train 상수만 쓰므로 행 독립이다.
평가셋에 행이 하나만 있어도 같은 값이 나온다.

**LEGAL 이 0 근처면 EXP005 의 이득은 전부 누수였다는 뜻이다.**

    .\.venv\Scripts\python.exe -u research\exp006_legal.py
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
from game_decomp import games                               # noqa: E402
from exp005_geom import ridge_cv                            # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

EXP = "EXP006"
ALPHAS = [1e2, 1e3, 1e4, 1e5, 1e6]
CUR = ("cur_succ", "cur_mid", "cur_ball", "cur_str")


def main():
    E.start_experiment(EXP, "L1-G", "python research/exp006_legal.py", "legality")
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
        gg = pd.DataFrame({"p": p, "c": c, "r": r}).groupby(["p", "c"])["r"].agg(
            ["mean", "size"]).unstack()
        n0, n1 = gg[("size", 0)].fillna(0), gg[("size", 1)].fillna(0)
        ne = (n0 * n1) / (n0 + n1).replace(0, np.nan)
        return ((gg[("mean", 1)] - gg[("mean", 0)]) * ne / (ne + k)).dropna()

    m24 = season == 2024
    C3 = pv0[2024].copy()
    for a, (ctx, k) in AX.items():
        C3 += (pd.Series(P[m24]).map(dvec(ctx, (2022, 2023), k)).fillna(0.).to_numpy()
               * np.where(ctx[m24] == 1, .5, -.5))
    r24 = y[m24] - C3
    base = r2(C3, y[m24])
    GID = games(P[m24], g("asof_pitcher_n")[m24],
                g("asof_pitcher_prev1_game_success_rate")[m24],
                g("asof_pitcher_prev1_game_middle_rate")[m24])
    rng = np.random.default_rng(0)
    u, inv = np.unique(GID, return_inverse=True)
    half = (rng.random(len(u)) < 0.5)[inv]
    print(f"C3 기준선 2024 = {base:.1f}\n")
    E.beat("기준선")

    cnt_all = (g("balls_before").astype(int) * 3 + g("strikes_before").astype(int))
    cnt = cnt_all[m24]
    msrc = np.isin(season, (2022, 2023))

    def col(c):
        v = g(c)
        return np.nan_to_num(v, nan=float(np.nanmean(v)))

    ill_p, ill_c, leg_p, leg_c = [], [], [], []
    for c in CUR:
        if c not in tr.columns:
            continue
        v = col(c)
        v24 = v[m24]
        # ILLEGAL — 그룹 평균을 2024 안에서
        ill_p.append(v24 - pd.Series(v24).groupby(P[m24]).transform("mean").to_numpy())
        ill_c.append(v24 - pd.Series(v24).groupby(cnt).transform("mean").to_numpy())
        # LEGAL — 그룹 평균을 과거 시즌(2022~2023)에서만
        pm = pd.Series(v[msrc]).groupby(P[msrc]).mean()
        gm = float(v[msrc].mean())
        leg_p.append(v24 - pd.Series(P[m24]).map(pm).fillna(gm).to_numpy())
        cm = pd.Series(v[msrc]).groupby(cnt_all[msrc]).mean()
        leg_c.append(v24 - pd.Series(cnt).map(cm).fillna(gm).to_numpy())

    def std(cols):
        X = np.column_stack(cols)
        return (X - X.mean(0)) / (X.std(0) + 1e-12)

    BLK = {"투수 센터링 (ILLEGAL, 2024 내)": std(ill_p),
           "카운트 센터링 (ILLEGAL, 2024 내)": std(ill_c),
           "투수 센터링 (LEGAL, 과거 상수)": std(leg_p),
           "카운트 센터링 (LEGAL, 과거 상수)": std(leg_c),
           "LEGAL 둘 다": std(leg_p + leg_c),
           "ILLEGAL 둘 다": std(ill_p + ill_c)}

    print("=" * 84)
    print(f"{'표현 블록':<36}{'열':>5}{'잔차상관':>11}{'C3증분':>10}{'합법':>10}")
    print("=" * 84)
    out = {}
    for nm, X in BLK.items():
        E.beat(nm)
        pred, cr = ridge_cv(X, r24, half, ALPHAS)
        inc = r2(C3 + pred, y[m24]) - base
        lab = "❌ 규정4" if "ILLEGAL" in nm else "✅"
        out[nm] = dict(cols=int(X.shape[1]), corr=cr, inc=inc,
                       legal=("ILLEGAL" not in nm))
        print(f"{nm:<36}{X.shape[1]:>5}{cr:>+11.4f}{inc:>+10.1f}{lab:>10}")

    legal_best = max(v["inc"] for v in out.values() if v["legal"])
    ill_best = max(v["inc"] for v in out.values() if not v["legal"])
    print(f"\n합법 최선 {legal_best:+.1f}   불법 최선 {ill_best:+.1f}"
          f"   격차 {ill_best - legal_best:+.1f}")
    print("  격차가 곧 '평가셋 다른 행을 보는 값'이다 — 규정 4 가 막는 몫")

    dec = "PROMISING" if legal_best >= 4.8 else "REJECTED"
    E.set_hypothesis_status("L1-G", "CLOSED" if dec == "REJECTED" else "PROMISING",
                            level=1, hypothesis="대비/센터링 표현 (합법 판본)",
                            result=round(legal_best, 2))
    E.finish_experiment(dict(
        experiment_id=EXP, hypothesis_id="L1-G", type="REPRESENTATION", level=1,
        started_at=E.read(E.CKPT)["start_time"],
        results={n: {k: (round(v, 4) if isinstance(v, float) else v)
                     for k, v in d.items()} for n, d in out.items()},
        legal_best=round(legal_best, 2), illegal_best=round(ill_best, 2),
        decision=dec, artifact=None,
        what_we_learned=(f"EXP005 의 +39.3 분해: 합법 판본 {legal_best:+.1f}, "
                         f"불법(2024 내 그룹평균) {ill_best:+.1f}. "
                         f"격차 {ill_best - legal_best:+.1f} 이 규정 4 가 막는 몫")))
    json.dump(out, open(os.path.join(ROOT, "exp", "exp006_legal.json"), "w"),
              indent=1, default=float)
    print(f"DECISION = {dec}")


if __name__ == "__main__":
    main()
