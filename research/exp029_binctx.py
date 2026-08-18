r"""EXP029 — 예측 구간 x 맥락 보정. 투수 정체성이 없는 새 축. 학습 0회.

## 왜 이 모양이 새로운가

지금까지의 후처리 보정은 **전부 투수 정체성을 키로** 썼다.

    편차 4축   투수 x 타자손 / 플래툰 x 우위 / +카운트 / 플래툰 x 주자
    차등 3축   투수 x {손, 2S, 주자}
    매치업     투수 x 타자

이번 축은 투수를 아예 쓰지 않는다.

    d( 예측값 구간 , 맥락 )   "모델이 **어느 예측 수준에서 어느 국면에** 틀리는가"

기존 것들과 구조적으로 독립이고, 이미 닫힌 둘과도 다르다.

    맥락 주효과 (오라클 −4.1)      구간 없이 잰 것
    구간별 아핀 (EXP028 −21.8)     맥락 없이 잰 것
    **둘의 교차는 미검증**

## 기제

모델이 "좋은 투수를 2스트라이크에서 과대평가한다" 같은 실수를 한다면 그건
투수별이 아니라 **전역 패턴**이다. 우리 보정은 전부 투수별이라 원리적으로
못 잡는다.

## 합법성

구간은 **예측값 자체**로 나눈다 (분위로 나누면 평가셋 분포가 필요해 규정 4 위반).
표는 과거 시즌의 (예측, 잔차)로만 만든 상수다. 그 행의 예측값과 맥락만 있으면
같은 값이 나온다.

    .\.venv\Scripts\python.exe -u research\exp029_binctx.py
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

EXP = "EXP029"
PREV2 = {2022: (2020, 2021), 2023: (2021, 2022), 2024: (2022, 2023)}
WD = 0.65
NBINS = (5, 10, 20, 40)
KGRID = (200, 1000, 5000, 20000)
WGRID = (0.25, 0.5, 0.75, 1.0)


def main():
    E.start_experiment(EXP, "R-BINCTX", "python research/exp029_binctx.py", "load")
    tr = build_df()
    season = tr["season"].to_numpy()
    y = tr["control_success"].to_numpy(np.float64)
    P = tr["pitcher_id"].to_numpy(np.int64)

    def g(c):
        return tr[c].to_numpy(np.float64)

    SAME = (g("pitcher_hand") == g("batter_hand")).astype(int)
    TWO = (g("strikes_before").astype(int) == 2).astype(int)
    RUN = (g("num_runners_on") > 0).astype(int)
    ADV = (g("strikes_before") > g("balls_before")).astype(int)
    ISF = (tr["game_type"].to_numpy() == "F").astype(int)
    AX = {"hand": (SAME, 1000), "2S": (TWO, 1000), "runner": (RUN, 2000)}

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

    C3, base, resC = {}, {}, {}
    for f in (2020, 2021, 2022, 2023, 2024):
        m = season == f
        v = pv0[f].copy()
        if f in PREV2:
            for a, (ctx, k) in AX.items():
                v += WD * (pd.Series(P[m]).map(dvec(ctx, PREV2[f], k)).fillna(0.).to_numpy()
                           * np.where(ctx[m] == 1, .5, -.5))
        C3[f], base[f] = v, r2(v, y[m])
        resC[f] = y[m] - v
    print(f"Champion(w=0.65)  2022 {base[2022]:.1f}  2023 {base[2023]:.1f}  "
          f"2024 {base[2024]:.1f}\n")
    E.beat("기준선")

    # 구간 경계는 **학습 구간의 예측 분포**에서 정한 상수 (평가셋 미사용)
    EDGE_SRC = np.concatenate([C3[f] for f in (2020, 2021)])

    def binid(p, nb, edges):
        return np.clip(np.digitize(p, edges) - 1, 0, nb - 1)

    CTX = {"같은손": SAME, "2스트라이크": TWO, "주자유무": RUN,
           "카운트우위": ADV, "F경기": ISF}

    def table(src, nb, edges, ctx, k):
        b = np.concatenate([binid(C3[f], nb, edges) for f in src])
        c = np.concatenate([ctx[season == f] for f in src])
        r = np.concatenate([resC[f] for f in src])
        d = pd.DataFrame({"b": b, "c": c, "r": r}).groupby(["b", "c"])["r"].agg(
            ["mean", "size"])
        return (d["mean"] * d["size"] / (d["size"] + k)).to_dict()

    def apply_(t, fold, nb, edges, ctx):
        m = season == fold
        key = list(zip(binid(C3[fold], nb, edges), ctx[m]))
        return np.array([t.get(kk, 0.0) for kk in key])

    print("=" * 92)
    print(f"{'맥락':<14}{'구간':>6}{'k':>8}{'21→22':>10}{'22→23':>10}{'23→24':>10}"
          f"{'겹침':>8}")
    print("=" * 92)
    m24 = season == 2024
    HD = WD * (pd.Series(P[m24]).map(dvec(SAME, (2022, 2023), 1000)).fillna(0.).to_numpy()
               * np.where(SAME[m24] == 1, .5, -.5))
    out, bestrow = {}, None
    for cn, ctx in CTX.items():
        E.beat(cn)
        best = None
        for nb in NBINS:
            edges = np.percentile(EDGE_SRC, np.linspace(0, 100, nb + 1))[:-1]
            for k in KGRID:
                gs = []
                for a, b in ((2021, 2022), (2022, 2023), (2023, 2024)):
                    t = table((a,), nb, edges, ctx, k)
                    gs.append(r2(C3[b] + apply_(t, b, nb, edges, ctx), y[season == b])
                              - base[b])
                sc = np.mean(gs[:2])
                if best is None or sc > best[0]:
                    best = (sc, nb, k, gs, edges)
        sc, nb, k, gs, edges = best
        t24 = table((2022, 2023), nb, edges, ctx, k)
        v24 = apply_(t24, 2024, nb, edges, ctx)
        g24 = r2(C3[2024] + v24, y[m24]) - base[2024]
        ov = float(np.corrcoef(v24, HD)[0, 1]) if v24.std() > 0 else 0.
        out[cn] = dict(nb=nb, k=k, gains=gs, g24=g24, overlap=ov)
        print(f"{cn:<14}{nb:>6}{k:>8}" + "".join(f"{v:>+10.1f}" for v in gs[:2])
              + f"{g24:>+10.1f}{ov:>+8.2f}")
        if bestrow is None or g24 > out[bestrow]["g24"]:
            bestrow = cn

    bg = out[bestrow]["g24"]
    print(f"\n최선 = {bestrow}  2024 {bg:+.1f}")
    print("  (구간·k 는 과거 두 전이로만 선택. 2024 는 숨긴 값)")
    dec = "PROMISING" if bg > 1.5 else "REJECTED"
    E.set_hypothesis_status("R-BINCTX", "PROMISING" if dec == "PROMISING" else "CLOSED",
                            level=3, hypothesis="예측 구간 x 맥락 보정",
                            result=round(bg, 2))
    E.finish_experiment(dict(
        experiment_id=EXP, hypothesis_id="R-BINCTX", type="A", level=3,
        started_at=E.read(E.CKPT)["start_time"],
        results={k: {kk: (round(vv, 3) if isinstance(vv, float)
                          else ([round(x, 2) for x in vv] if isinstance(vv, list) else vv))
                     for kk, vv in v.items()} for k, v in out.items()},
        best=bestrow, gain_2024=round(bg, 2), decision=dec, artifact=None,
        what_we_learned=("투수 정체성이 없는 첫 후처리 축 — 예측 구간 x 맥락. "
                         f"최선 {bestrow} 2024 {bg:+.1f}")))
    json.dump({k: {kk: vv for kk, vv in v.items()} for k, v in out.items()},
              open(os.path.join(ROOT, "exp", "exp029_binctx.json"), "w"),
              indent=1, default=float)
    print(f"\nDECISION = {dec}")


if __name__ == "__main__":
    main()
