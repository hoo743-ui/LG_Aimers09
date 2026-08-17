r"""EXP005 — LEVEL 1 입력 기하. 트리가 못 만드는 방향이 남아 있는가. 학습 0회.

## 무엇을 묻는가

EXP004 는 **보정**의 표현을 쟀고 국소 최적이었다. 이번엔 **입력**의 표현이다.

부스팅 트리는 축평행 분할만 한다. 그러므로 원리적으로 못 만드는 것이 있다.

    회전/대각      a*x + b*z 같은 방향 (X/H1 이 통한 이유가 정확히 이것)
    전역 선형 성분  많은 열에 얇게 퍼진 신호 (분할로 모으려면 깊이가 폭발한다)
    대비/센터링     x - E[x | 국면] 같은 상대 위치

이것들이 남아 있다면 **C3 잔차를 선형으로 설명할 수 있어야 한다.**
없다면 입력 회전·PCA·대비 표현은 전부 죽은 것이고, 학습을 한 번도 돌리지
않고 LEVEL 1 입력면을 닫을 수 있다.

## 탐침

C3 잔차를 특징행렬에 능형회귀한다. **경기 단위로 절반을 갈라 교차적합**하므로
자기 자신을 맞히는 부풀림이 없다 (투구 단위 분할은 같은 경기를 양쪽에 넣어
+수십점의 가짜 이득을 만든다 — 이 프로젝트에서 반복 확인됨).

    L0 선형        원 특징 전체
    L1 회전        원 특징 + 상위 분산 15열의 모든 쌍곱 (대각 경계)
    L2 대비        + 센터링/상대위치 표현 (그 행의 값 − 국면 기대값)
    위약           같은 열 수의 무작위 행렬 (과적합 바닥)

    .\.venv\Scripts\python.exe -u research\exp005_geom.py
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

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

EXP = "EXP005"
PREV2 = {2022: (2020, 2021), 2023: (2021, 2022), 2024: (2022, 2023)}
ALPHAS = [1e2, 1e3, 1e4, 1e5, 1e6]
SKIP = {"control_success", "season", "pitcher_id", "batter_id",
        "pitcher_team_id", "batter_team_id", "game_id", "game_date"}


def ridge_cv(X, r, half, alphas):
    """경기 분할 교차적합 능형회귀. 최선 alpha 의 적합값을 돌려준다."""
    X = np.column_stack([np.ones(len(X)), X])
    best, bv = None, -1e18
    for al in alphas:
        pred = np.zeros(len(r))
        for m in (half, ~half):
            A = X[m].T @ X[m] + al * np.eye(X.shape[1])
            A[0, 0] -= al
            beta = np.linalg.solve(A, X[m].T @ r[m])
            pred[~m] = X[~m] @ beta
        v = float(np.corrcoef(pred, r)[0, 1]) if pred.std() > 1e-12 else 0.0
        if v > bv:
            bv, best = v, pred
    return best, bv


def main():
    E.start_experiment(EXP, "L1-F..H", "python research/exp005_geom.py", "load")
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
    print(f"C3 기준선 2024 = {base:.1f}   잔차 sd = {r24.std():.4f}\n")
    E.beat("기준선")

    # ---- 특징행렬 ----
    num = [c for c in tr.columns if c not in SKIP
           and pd.api.types.is_numeric_dtype(tr[c])]
    X0 = np.column_stack([g(c) for c in num])[m24]
    X0 = np.nan_to_num(X0, nan=0.0, posinf=0.0, neginf=0.0)
    mu, sd = X0.mean(0), X0.std(0)
    keep = sd > 1e-9
    X0 = (X0[:, keep] - mu[keep]) / sd[keep]
    names = [c for c, k_ in zip(num, keep) if k_]
    print(f"수치 특징 {X0.shape[1]}열")

    # 회전 — 분산 상위 15열의 모든 쌍곱 (대각 경계)
    top = np.argsort(-np.abs(np.corrcoef(np.column_stack([X0, r24]),
                                         rowvar=False)[-1, :-1]))[:15]
    prods = []
    for i in range(len(top)):
        for j in range(i, len(top)):
            prods.append(X0[:, top[i]] * X0[:, top[j]])
    XP = np.column_stack(prods)
    XP = (XP - XP.mean(0)) / (XP.std(0) + 1e-12)
    print(f"쌍곱 {XP.shape[1]}열 (잔차상관 상위 15열의 조합)")

    # 대비/센터링 — 그 행의 값이 국면 기대값에서 얼마나 떨어져 있는가
    cnt = (g("balls_before").astype(int) * 3 + g("strikes_before").astype(int))[m24]
    CON = []
    for c in ("cur_succ", "cur_mid", "cur_ball", "cur_str"):
        if c not in tr.columns:
            continue
        v = g(c)[m24]
        v = np.nan_to_num(v, nan=float(np.nanmean(v)))
        CON.append(v - pd.Series(v).groupby(cnt).transform("mean").to_numpy())
        CON.append(v - pd.Series(v).groupby(P[m24]).transform("mean").to_numpy())
    XC = np.column_stack(CON) if CON else np.zeros((len(r24), 0))
    if XC.shape[1]:
        XC = (XC - XC.mean(0)) / (XC.std(0) + 1e-12)
    print(f"대비/센터링 {XC.shape[1]}열\n")

    GID = games(P[m24], g("asof_pitcher_n")[m24],
                g("asof_pitcher_prev1_game_success_rate")[m24],
                g("asof_pitcher_prev1_game_middle_rate")[m24])
    rng = np.random.default_rng(0)
    u, inv = np.unique(GID, return_inverse=True)
    half = (rng.random(len(u)) < 0.5)[inv]

    TESTS = [("L0 선형 (원 특징)", X0),
             ("L1 회전 (원 + 쌍곱)", np.column_stack([X0, XP])),
             ("L2 대비 (원 + 쌍곱 + 센터링)", np.column_stack([X0, XP, XC])),
             ("위약 (무작위 같은 열수)",
              rng.standard_normal((len(r24), X0.shape[1] + XP.shape[1])))]
    print("=" * 82)
    print(f"{'표현':<32}{'열':>6}{'잔차상관':>11}{'C3증분':>10}{'판정':>14}")
    print("=" * 82)
    out = {}
    for nm, X in TESTS:
        E.beat(f"ridge {nm}")
        pred, cr = ridge_cv(X, r24, half, ALPHAS)
        inc = r2(C3 + pred, y[m24]) - base
        verdict = "신호" if inc > 5 else ("잡음" if inc < 2 else "경계")
        out[nm] = dict(cols=int(X.shape[1]), corr=cr, inc=inc)
        print(f"{nm:<32}{X.shape[1]:>6}{cr:>+11.4f}{inc:>+10.1f}{verdict:>14}")

    real = max(v["inc"] for n, v in out.items() if not n.startswith("위약"))
    plac = out["위약 (무작위 같은 열수)"]["inc"]
    dec = "PROMISING" if real - plac > 5 else "REJECTED"
    print(f"\n최선 실제 {real:+.1f}  위약 {plac:+.1f}  차이 {real - plac:+.1f}")
    E.set_hypothesis_status("L1-F", "CLOSED" if dec == "REJECTED" else "PROMISING",
                            level=1, hypothesis="입력 회전/대각 표현",
                            result=round(real - plac, 2))
    E.finish_experiment(dict(
        experiment_id=EXP, hypothesis_id="L1-F..H", type="REPRESENTATION", level=1,
        started_at=E.read(E.CKPT)["start_time"],
        results={n: {k: round(v, 4) if isinstance(v, float) else v
                     for k, v in d.items()} for n, d in out.items()},
        decision=dec, artifact=None,
        what_we_learned=(f"C3 잔차에 남은 선형/회전/대비 구조: 실제 최선 {real:+.1f} "
                         f"vs 위약 {plac:+.1f}")))
    json.dump(out, open(os.path.join(ROOT, "exp", "exp005_geom.json"), "w"),
              indent=1, default=float)
    print(f"DECISION = {dec}")


if __name__ == "__main__":
    main()
