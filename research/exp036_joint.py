r"""EXP036 — 후처리 9개 가중의 **결합 최적화**. 학습 0회.

## 왜 필요한가

지금까지 나는 가중을 **좌표별로 따로** 훑었다.

    편차 4축   각각 다른 셋을 현행에 고정한 채
    대비 3축   전역 배수 하나로만
    수준 2축   wp 볼 때 wb=2.5 고정, wb 볼 때 wp=1.0 고정

마지막 것에서 문제가 드러났다 — 결합 스캔을 하니 (wp=2.0, wb=2.0)이 (1.0, 2.5)보다
2024 에서 +1.0 이었다. **wb 를 낮추면 wp 를 올릴 여지가 생긴다.** 좌표별 스캔은
그런 상충을 못 본다.

9개를 한 번도 함께 본 적이 없으므로 여기서 한다.

## 절차

성분 벡터 9개를 폴드마다 미리 계산해두면 채점이 가중합이라 평가가 싸다.
다중 패스 좌표하강을 수렴까지 돌린다.

    선택 기준   **2022 와 2024 의 평균** (2023 은 퇴화 폴드라 제외)
    제약        마지막에 3폴드 전부 양수인지 확인한다
    출발점      현행 [0.20, 0.825, 0.28, 0.45, 0.65, 0.65, 0.65, 1.0, 2.5]

## 과적합 통제

9개 모수를 두 폴드에 맞추므로 과적합 위험이 있다. 그래서 (1) 이동 폭을 곱셈
격자로 제한하고 (2) 3폴드 부호를 확인하며 (3) 최종 판정은 LB 가 한다.

    .\.venv\Scripts\python.exe -u research\exp036_joint.py
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
import build_asof as ba                                     # noqa: E402
from path_alloc import build_df                             # noqa: E402
from traj_probe import r2                                   # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

EXP = "EXP036"
PREV2 = {2022: (2020, 2021), 2023: (2021, 2022), 2024: (2022, 2023)}
NAMES = ["dev0 투수x타자손", "dev1 플래툰x우위", "dev2 +카운트", "dev3 플래툰x주자",
         "c_hand 손 차등", "c_2S 2S 차등", "c_run 주자 차등",
         "L_pit 투수 주효과", "L_bat 타자 주효과"]
W0 = [0.20, 0.825, 0.28, 0.45, 0.65, 0.65, 0.65, 1.0, 2.5]
GRID = (0.7, 0.85, 1.0, 1.18, 1.4)          # 곱셈 격자 — 한 번에 크게 안 움직인다


def main():
    E.start_experiment(EXP, "J-9", "python research/exp036_joint.py", "load")
    tr = build_df()
    season = tr["season"].to_numpy()
    y = tr["control_success"].to_numpy(np.float64)
    P = tr["pitcher_id"].to_numpy(np.int64)
    B = tr["batter_id"].to_numpy(np.int64)
    BH = tr["batter_hand"].to_numpy(np.int64)
    BB = tr["balls_before"].to_numpy(np.int64)
    SS = tr["strikes_before"].to_numpy(np.int64)
    OB = (tr["num_runners_on"].to_numpy(np.int64) > 0).astype(np.int64)
    PH = P * 10 + BH
    PHA = PH * 10 + (SS > BB).astype(np.int64)
    AXD = [(P, PH), (PH, PHA), (PHA, PHA * 100 + (BB * 4 + SS)), (PH, PH * 10 + OB)]

    def g(c):
        return tr[c].to_numpy(np.float64)

    SAME = (g("pitcher_hand") == g("batter_hand")).astype(int)
    TWO = (SS == 2).astype(int)
    AXC = [("hand", SAME, 1000), ("2S", TWO, 1000), ("runner", OB, 2000)]

    MODEL = {f: np.load(os.path.join(ROOT, "exp", f"prod_champ_{f}.npy"))[:2].mean(0)
             for f in (2020, 2021, 2022, 2023, 2024)}
    DEV = {}
    for f in (2020, 2021, 2022, 2023, 2024):
        m_tr, m_va = season < f, season == f
        DEV[f] = np.column_stack([
            ba.look(*ba.nested_dev(p[m_tr], c[m_tr], y[m_tr], k), c[m_va])
            for (p, c), k in zip(AXD, ba.KSH)])
    res0 = {f: y[season == f] - (MODEL[f] + DEV[f] @ np.array(W0[:4]))
            for f in (2020, 2021, 2022, 2023, 2024)}
    E.beat("성분 준비")

    def ctab(ctx, src, k):
        p = np.concatenate([P[season == f] for f in src])
        c = np.concatenate([ctx[season == f] for f in src])
        r = np.concatenate([res0[f] for f in src])
        gg = pd.DataFrame({"p": p, "c": c, "r": r}).groupby(["p", "c"])["r"].agg(
            ["mean", "size"]).unstack()
        n0, n1 = gg[("size", 0)].fillna(0), gg[("size", 1)].fillna(0)
        ne = (n0 * n1) / (n0 + n1).replace(0, np.nan)
        return ((gg[("mean", 1)] - gg[("mean", 0)]) * ne / (ne + k)).dropna()

    def mtab(key, src, k):
        kk = np.concatenate([key[season == f] for f in src])
        r = np.concatenate([res0[f] for f in src])
        d = pd.DataFrame({"k": kk, "r": r}).groupby("k")["r"].agg(["mean", "size"])
        return (d["mean"] * d["size"] / (d["size"] + k))

    # 폴드별 9개 성분 벡터 (가중 1 기준)
    COMP, YY = {}, {}
    for f in (2022, 2023, 2024):
        m = season == f
        cols = [DEV[f][:, j] for j in range(4)]
        for nm, ctx, k in AXC:
            cols.append(pd.Series(P[m]).map(ctab(ctx, PREV2[f], k)).fillna(0.).to_numpy()
                        * np.where(ctx[m] == 1, .5, -.5))
        cols.append(pd.Series(P[m]).map(mtab(P, PREV2[f], 50000)).fillna(0.).to_numpy())
        cols.append(pd.Series(B[m]).map(mtab(B, PREV2[f], 20000)).fillna(0.).to_numpy())
        COMP[f] = np.column_stack(cols)
        YY[f] = y[m]
        E.beat(f"성분 {f}")

    def score(f, w):
        return r2(MODEL[f] + COMP[f] @ np.array(w), YY[f])

    b0 = {f: score(f, W0) for f in (2022, 2023, 2024)}
    print(f"현행  2022 {b0[2022]:.1f}  2023 {b0[2023]:.1f}  2024 {b0[2024]:.1f}")
    print(f"현행 가중 {[round(x, 3) for x in W0]}\n")

    def sel(w):
        return np.mean([score(2022, w) - b0[2022], score(2024, w) - b0[2024]])

    w = list(W0)
    print("좌표하강 (선택 = 2022·2024 평균 이득)")
    for it in range(6):
        moved = False
        for j in range(9):
            best = (sel(w), w[j])
            for mul in GRID:
                if abs(mul - 1.0) < 1e-9:
                    continue
                w2 = list(w)
                w2[j] = w[j] * mul
                s = sel(w2)
                if s > best[0] + 1e-4:
                    best = (s, w2[j])
            if abs(best[1] - w[j]) > 1e-9:
                w[j] = best[1]
                moved = True
        gs = [score(f, w) - b0[f] for f in (2022, 2023, 2024)]
        print(f"  pass {it+1}  2022 {gs[0]:+.1f}  2023 {gs[1]:+.1f}  2024 {gs[2]:+.1f}"
              f"   선택 {sel(w):+.2f}")
        E.beat(f"pass {it+1}")
        if not moved:
            break

    gs = [score(f, w) - b0[f] for f in (2022, 2023, 2024)]
    print(f"\n{'축':<20}{'현행':>9}{'최적':>9}{'배수':>8}")
    for nm, a, b in zip(NAMES, W0, w):
        print(f"{nm:<20}{a:>9.3f}{b:>9.3f}{b/a:>8.2f}")
    print(f"\n결합 최적  2022 {gs[0]:+.1f}  2023 {gs[1]:+.1f}  2024 {gs[2]:+.1f}"
          + ("   ★3폴드 양수" if all(x > 0 for x in gs) else "   (2023 음수)"))

    # 이동 폭을 절반으로 줄인 보수판 (과적합 완화)
    wh = [a + (b - a) * 0.5 for a, b in zip(W0, w)]
    gh = [score(f, wh) - b0[f] for f in (2022, 2023, 2024)]
    print(f"보수판(이동 50%)  2022 {gh[0]:+.1f}  2023 {gh[1]:+.1f}  2024 {gh[2]:+.1f}"
          + ("   ★" if all(x > 0 for x in gh) else ""))

    E.finish_experiment(dict(
        experiment_id=EXP, hypothesis_id="J-9", type="B", level=3,
        started_at=E.read(E.CKPT)["start_time"],
        w0=W0, w_opt=[round(x, 4) for x in w], gains=[round(x, 2) for x in gs],
        w_half=[round(x, 4) for x in wh], gains_half=[round(x, 2) for x in gh],
        decision="PROMISING" if all(x > 0 for x in gs) else "MIXED",
        what_we_learned=("후처리 9개 가중을 처음으로 결합 최적화. 좌표별 스캔은 "
                         "상충하는 국소값에 갇힌다는 것이 (wp,wb)에서 이미 드러났다.")))
    json.dump({"w0": W0, "w_opt": w, "gains": gs, "w_half": wh, "gains_half": gh},
              open(os.path.join(ROOT, "exp", "exp036_joint.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
