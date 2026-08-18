r"""EXP024 — 투수 x 타자 매치업 축 재개. TYPE A · 후처리. 학습 0회.

## REOPEN_JUSTIFICATION

`CLAUDE.md` §2 에 이렇게 적혀 있다.

    투수 x 타자 매치업   오라클 경기분할 +4.0(중앙 +2.4)   전이 +3.9 = +0.41%

그리고 기각 사유는 **`+3.8%` 게이트 미달**이었다. 그 게이트는 로컬 +36점을
요구하던 기준인데, 우리가 실제로 채택한 차등 3축은 로컬 **+14.6** 으로 LB
**+10.4** 를 냈다. **게이트가 틀렸다.**

게다가 오늘 LB 앵커 3개로 전이 위계를 확정했다 — 후처리 보정은 전이율 +0.84 로
유일하게 신뢰할 수 있는 등급이다. 매치업은 그 등급에 속한다.

    TYPE          A (새 정보축 — 투수 x 타자 상호작용)
    등급          후처리 (모델 재학습 없음)
    조회 키        pitcher_id · batter_id — **둘 다 그 행의 컬럼** (규정 4 안전)
    표 원천        직전 2시즌 strictly OOF 잔차 (30회차 교훈으로 고정)

## 설계 — 주효과를 뺀 **상호작용만** 싣는다

투수 단위 잔차 표는 이미 기각됐다(전이 +0.7%, 손익분기 아래). 그래서 쌍 평균을
그대로 쓰면 안 되고, **그 투수 자신의 평균에서의 편차**만 쓴다.

    d_pair = mean(resid | 투수,타자) − mean(resid | 투수)

축소는 쌍 표본수로 하고, `k` 는 **과거 두 전이로만** 고른다. 5-b 가 경고하듯
적률법 k(=89)는 −46.9 로 참사였고 전이 최적은 500 이상이었다.

    .\.venv\Scripts\python.exe -u research\exp024_matchup.py
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

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

EXP = "EXP024"
PREV2 = {2022: (2020, 2021), 2023: (2021, 2022), 2024: (2022, 2023)}
KGRID = [200, 500, 1000, 2000, 5000, 20000]
WGRID = [0.25, 0.5, 0.65, 0.8, 1.0]
WD = 0.65


def main():
    E.start_experiment(EXP, "A-MU", "python research/exp024_matchup.py", "load")
    tr = build_df()
    season = tr["season"].to_numpy()
    y = tr["control_success"].to_numpy(np.float64)
    P = tr["pitcher_id"].to_numpy(np.int64)
    B = tr["batter_id"].to_numpy(np.int64)

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

    def apc(t, ctx, m):
        return (pd.Series(P[m]).map(t).fillna(0.).to_numpy()
                * np.where(ctx[m] == 1, .5, -.5))

    C3, base = {}, {}
    for f in (2022, 2023, 2024):
        m = season == f
        v = pv0[f].copy()
        for a, (ctx, k) in AX.items():
            v += WD * apc(dvec(ctx, PREV2[f], k), ctx, m)
        C3[f], base[f] = v, r2(v, y[m])
    print(f"Champion(w=0.65)  2022 {base[2022]:.1f}  2023 {base[2023]:.1f}  "
          f"2024 {base[2024]:.1f}\n")
    E.beat("기준선")

    def pair_tab(src, k):
        """쌍 편차 = mean(resid|투수,타자) − mean(resid|투수). 쌍 표본수로 축소."""
        p = np.concatenate([P[season == f] for f in src])
        b = np.concatenate([B[season == f] for f in src])
        r = np.concatenate([res0[f] for f in src])
        d = pd.DataFrame({"p": p, "b": b, "r": r})
        pm = d.groupby("p")["r"].mean()
        gg = d.groupby(["p", "b"])["r"].agg(["mean", "size"])
        dev = gg["mean"] - gg.index.get_level_values(0).map(pm)
        n = gg["size"]
        return (dev * n / (n + k)).dropna(), n

    def apply_pair(t, m):
        key = pd.MultiIndex.from_arrays([P[m], B[m]])
        return pd.Series(key.map(t)).fillna(0.).to_numpy()

    print("=" * 84)
    print("1. 쌍 축소 상수 k — 과거 두 전이로만 고른다")
    print("=" * 84)
    print(f"{'k':>8}{'21→22':>10}{'22→23':>10}{'23→24(숨김)':>14}{'쌍 수':>10}"
          f"{'적용률':>9}")
    rows = {}
    for k in KGRID:
        E.beat(f"k={k}")
        gains = []
        for a, b_ in ((2021, 2022), (2022, 2023), (2023, 2024)):
            mb = season == b_
            t, _ = pair_tab((a,), k)
            gains.append(r2(C3[b_] + apply_pair(t, mb), y[mb]) - base[b_])
        t24, n24 = pair_tab((2022, 2023), k)
        m24 = season == 2024
        v24 = apply_pair(t24, m24)
        rows[k] = dict(gains=gains, npair=len(t24), cover=float((v24 != 0).mean()))
        print(f"{k:>8}" + "".join(f"{v:>+10.1f}" for v in gains[:2])
              + f"{gains[2]:>+14.1f}{len(t24):>10,}{(v24 != 0).mean():>9.1%}")
    kb = max(KGRID, key=lambda k: np.mean(rows[k]["gains"][:2]))
    print(f"\n  과거 2폴드로 고른 k = {kb}")

    print("\n" + "=" * 84)
    print(f"2. 전역 가중 w (k={kb} 고정) — 후처리 등급이라 LB 로 최적화 가능")
    print("=" * 84)
    m24 = season == 2024
    t24, _ = pair_tab((2022, 2023), kb)
    v24 = apply_pair(t24, m24)
    HD = WD * apc(dvec(AX["hand"][0], (2022, 2023), 1000), AX["hand"][0], m24)
    TS = WD * apc(dvec(AX["2S"][0], (2022, 2023), 1000), AX["2S"][0], m24)
    RN = WD * apc(dvec(AX["runner"][0], (2022, 2023), 2000), AX["runner"][0], m24)
    print(f"{'w':>8}{'2022':>10}{'2023':>10}{'2024':>10}")
    wres = {}
    for w in WGRID:
        gs = []
        for f in (2022, 2023, 2024):
            mf = season == f
            tf, _ = pair_tab(PREV2[f], kb)
            gs.append(r2(C3[f] + w * apply_pair(tf, mf), y[mf]) - base[f])
        wres[w] = gs
        print(f"{w:>8.2f}" + "".join(f"{v:>+10.1f}" for v in gs))
    ov = [float(np.corrcoef(v24, x)[0, 1]) if v24.std() > 0 else 0.
          for x in (HD, TS, RN)]
    print(f"\n기존 3축과의 겹침  hand {ov[0]:+.3f}  2S {ov[1]:+.3f}  runner {ov[2]:+.3f}")
    print(f"보정 벡터 sd {v24.std():.6f}  (차등 3축 합 sd {(HD+TS+RN).std():.6f})")

    wbest = max(WGRID, key=lambda w: np.mean(wres[w][:2]))
    g24 = wres[wbest][2]
    print(f"\n과거로 고른 w={wbest:g}  ->  2024 {g24:+.1f}")
    print(f"  전이 위계상 후처리는 로컬->LB 감쇠 0.84 이므로 LB 기대 {g24*0.71:+.1f} ~ "
          f"{g24*0.84:+.1f}")
    dec = "PROMISING" if (g24 > 1.5 and all(v > 0 for v in wres[wbest][:2])) else "REJECTED"
    E.set_hypothesis_status("A-MU", "PROMISING" if dec == "PROMISING" else "CLOSED",
                            level=0, hypothesis="투수x타자 매치업 후처리 축",
                            result=round(g24, 2))
    E.finish_experiment(dict(
        experiment_id=EXP, hypothesis_id="A-MU", type="A", level=0,
        started_at=E.read(E.CKPT)["start_time"],
        reopen_justification="+3.8% 게이트가 틀렸음이 LB 로 확인됨 (채택 3축은 로컬 +14.6 -> LB +10.4)",
        k=kb, w=wbest, k_curve={str(k): [round(x, 2) for x in v["gains"]]
                                for k, v in rows.items()},
        w_curve={str(w): [round(x, 2) for x in v] for w, v in wres.items()},
        overlap=[round(x, 3) for x in ov], gain_2024=round(g24, 2),
        decision=dec, artifact=None,
        what_we_learned=(f"쌍 편차 축: k={kb}, w={wbest}, 2024 {g24:+.1f}, "
                         f"기존 3축과 겹침 최대 {max(abs(x) for x in ov):.3f}")))
    json.dump(dict(k=kb, w=wbest, rows={str(k): v for k, v in rows.items()},
                   wres={str(w): v for w, v in wres.items()}, overlap=ov),
              open(os.path.join(ROOT, "exp", "exp024_matchup.json"), "w"),
              indent=1, default=float)
    print(f"\nDECISION = {dec}")


if __name__ == "__main__":
    main()
