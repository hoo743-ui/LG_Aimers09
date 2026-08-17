r"""EXP021 — 보정 가중 곡선. 제출 슬롯이 흔해졌으므로 LB 로 잴 준비를 한다.

## 정책 전환의 근거

`+3.8% 게이트`와 TYPE B 동결은 **제출이 희소하다**는 전제 위에 있었다.
로컬로 판정 못 하는 것은 건드리지 않는다는 규율이었다. 제출이 5회/일 갱신이면
그 전제가 사라진다 — 로컬이 못 재는 것을 **LB 로 직접 재면 된다.**

LB 는 245,789행이고 우리 로컬 폴드는 시드 잡음 ±7.2 다. 측정기의 정밀도가
반대다.

## 이론

보정 벡터 `c` 를 `w` 배로 실으면, 그 시즌에서의 실제 유효 강도를 `b` 라 할 때

    이득(w) ≈ A(2bw − w²)        w* = b 에서 최대,  최대값 A·b²

LB 앵커가 이미 둘 있다.

    w=0    1049.9225979712   (cand_submit_1)
    w=1.0  1057.3394030999   (cand_submit_3)   ->  이득 +7.4168

`b=1` 이면 이득이 `A` 이고, 전이율 0.46 은 `b<1` 을 시사한다. 한 점만 더
찍으면 `A`, `b` 가 풀린다.

## 이 스크립트가 하는 일

로컬 2024 에서 같은 곡선을 그려 **모양과 곡률**을 확인하고, 어느 `w` 를
제출해야 정보가 가장 큰지 고른다. 학습 0회.

    .\.venv\Scripts\python.exe -u research\exp021_wcurve.py
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

EXP = "EXP021"
PREV2 = {2022: (2020, 2021), 2023: (2021, 2022), 2024: (2022, 2023)}
LB0, LB1 = 1049.9225979712, 1057.3394030999


def main():
    E.start_experiment(EXP, "S-1", "python research/exp021_wcurve.py", "load")
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

    def ap(t, ctx, m):
        return (pd.Series(P[m]).map(t).fillna(0.).to_numpy()
                * np.where(ctx[m] == 1, .5, -.5))

    print("=" * 78)
    print("1. 로컬 가중 곡선 — 세 차등축에 전역 배수 w")
    print("=" * 78)
    curves = {}
    WS = [0.0, 0.25, 0.5, 0.6, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0]
    for f in (2022, 2023, 2024):
        m = season == f
        diff = np.zeros(int(m.sum()))
        for a, (ctx, k) in AX.items():
            diff += ap(dvec(ctx, PREV2[f], k), ctx, m)
        b0 = r2(pv0[f], y[m])
        curves[f] = {w: r2(pv0[f] + w * diff, y[m]) - b0 for w in WS}
        E.beat(f"curve {f}")
    print(f"{'w':>7}" + "".join(f"{f:>11}" for f in (2022, 2023, 2024)))
    for w in WS:
        mark = "  <- 현행" if w == 1.0 else ""
        print(f"{w:>7.2f}" + "".join(f"{curves[f][w]:>+11.1f}"
                                    for f in (2022, 2023, 2024)) + mark)
    for f in (2022, 2023, 2024):
        best = max(WS, key=lambda w: curves[f][w])
        print(f"  폴드 {f} 로컬 최적 w = {best:.2f}   "
              f"(w=1.0 대비 {curves[f][best] - curves[f][1.0]:+.1f})")

    print("\n" + "=" * 78)
    print("2. LB 두 점으로 b 를 풀면 — 세 번째 점이 왜 필요한가")
    print("=" * 78)
    gain1 = LB1 - LB0
    print(f"  LB 이득(w=1.0) = {gain1:+.4f}")
    print(f"  모형 이득(w) = A(2bw − w²)   ->   A(2b−1) = {gain1:.4f}")
    print(f"\n  {'가정한 b':>10}{'A':>10}{'최적 w':>9}{'최적 이득':>11}"
          f"{'현행 대비':>11}{'w=0.6 이득':>12}")
    rows = []
    for b in (0.6, 0.7, 0.8, 0.9, 1.0, 1.2):
        if abs(2 * b - 1) < 1e-9:
            continue
        A = gain1 / (2 * b - 1)
        gbest = A * b * b
        g06 = A * (2 * b * 0.6 - 0.36)
        rows.append((b, A, b, gbest, gbest - gain1, g06))
        print(f"  {b:>10.2f}{A:>10.2f}{b:>9.2f}{gbest:>11.2f}"
              f"{gbest - gain1:>+11.2f}{g06:>12.2f}")
    print("\n  b 마다 w=0.6 의 예측 이득이 다르다 -> **한 점 찍으면 b 가 결정된다**")
    print("  b<1 이면 현행 w=1.0 은 과잉이고, 낮출수록 이득이 커진다")

    print("\n" + "=" * 78)
    print("3. 후처리 편차 4축에도 같은 질문 — 전역 배수 s")
    print("=" * 78)
    m24 = season == 2024
    post24 = post_for(tr, y, season < 2024, m24)
    basem = np.load(os.path.join(ROOT, "exp", "prod_champ_2024.npy"))[:2].mean(0)
    diff24 = np.zeros(int(m24.sum()))
    for a, (ctx, k) in AX.items():
        diff24 += ap(dvec(ctx, (2022, 2023), k), ctx, m24)
    b_no = r2(basem, y[m24])
    print(f"{'s':>7}{'2024 점수':>12}{'s=1 대비':>11}")
    sc = {}
    for s in (0.0, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5):
        v = r2(basem + s * post24 + diff24, y[m24])
        sc[s] = v
        print(f"{s:>7.2f}{v:>12.1f}{v - sc.get(1.0, v):>+11.1f}"
              + ("  <- 현행" if s == 1.0 else ""))
    bs = max(sc, key=sc.get)
    print(f"  로컬 최적 s = {bs:.2f}  (기여 총량 {sc[1.0] - b_no:+.1f})")

    out = dict(curves={str(f): {str(w): v for w, v in c.items()}
                       for f, c in curves.items()},
               lb_gain=gain1, post_scale={str(k): v for k, v in sc.items()})
    E.finish_experiment(dict(
        experiment_id=EXP, hypothesis_id="S-1", type="B", level=3,
        started_at=E.read(E.CKPT)["start_time"], results=out,
        decision="SUBMIT-READY (LB 측정 설계)", artifact=None,
        what_we_learned=("제출이 5회/일 갱신이면 TYPE B 동결의 전제가 사라진다. "
                         "보정 가중 w 는 LB 두 점(w=0, w=1)이 이미 있어 "
                         "한 점만 더 찍으면 최적 w 가 결정된다.")))
    json.dump(out, open(os.path.join(ROOT, "exp", "exp021_wcurve.json"), "w"),
              indent=1, default=float)


if __name__ == "__main__":
    main()
