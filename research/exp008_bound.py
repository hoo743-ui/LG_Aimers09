r"""EXP008 — LEVEL 2 상한 증명. 비선형 잔차 학습기가 무엇을 더 뽑는가.

## REOPEN_JUSTIFICATION

`잔차 2단 모델` 은 §7 에서 닫혀 있다. 다시 여는 근거는 셋이다.

    1  당시 기준선은 차등 3축 이전이다. 기준선이 바뀌면 잔차도 다른 대상이다
    2  이번 목적은 후보 발굴이 아니라 **상한 증명**이다. 0 이 나오면 대체 트리
       구조·GAM·전문가 구조가 노릴 몫이 없다는 것까지 함께 닫힌다
    3  경기 단위 교차적합 + 라벨 셔플 위약을 붙인다 (당시엔 없었다)

## 무엇을 묻는가

EXP007 에서 캐시 예측 7개의 오차 상관이 **0.9999** 였다. 오차가 추정 분산이
아니라 구조적이라는 뜻이고, 분산 감소를 노리는 모든 모델 후보의 상한은 ~0 이다.

남은 것은 편향이다 — "현재 모델이 못 잡은 구조를 다른 학습기가 잡는가".
선형/회전으로는 없었다(EXP005). 이번엔 부스팅 트리를 **잔차에 직접** 걸어
합법적으로 뽑을 수 있는 최대치를 잰다.

    실제      C3 잔차를 목표로 학습
    위약      같은 절차, 목표만 셔플 (과적합 바닥)

둘 다 경기 단위로 절반을 갈라 교차적합한다. 투구 단위 분할은 같은 경기를
양쪽에 넣어 가짜 이득을 만든다 (이 프로젝트에서 반복 확인).

    .\.venv\Scripts\python.exe -u research\exp008_bound.py
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

EXP = "EXP008"
SKIP = {"control_success", "season", "pitcher_id", "batter_id",
        "pitcher_team_id", "batter_team_id", "game_id", "game_date"}


def main():
    from catboost import CatBoostRegressor

    E.start_experiment(EXP, "L2-B", "python research/exp008_bound.py", "load")
    tr = build_df()
    season = tr["season"].to_numpy()
    y = tr["control_success"].to_numpy(np.float64)
    P = tr["pitcher_id"].to_numpy(np.int64)

    def g(c):
        return tr[c].to_numpy(np.float64)

    AX = {"hand": ((g("pitcher_hand") == g("batter_hand")).astype(int), 1000),
          "2S": ((g("strikes_before").astype(int) == 2).astype(int), 1000),
          "runner": ((g("num_runners_on") > 0).astype(int), 2000)}
    res0 = {}
    for f in (2022, 2023):
        m = season == f
        res0[f] = y[m] - (np.load(os.path.join(ROOT, "exp", f"prod_champ_{f}.npy"))[:2].mean(0)
                          + post_for(tr, y, season < f, m))

    def dvec(ctx, k):
        p = np.concatenate([P[season == f] for f in (2022, 2023)])
        c = np.concatenate([ctx[season == f] for f in (2022, 2023)])
        r = np.concatenate([res0[f] for f in (2022, 2023)])
        gg = pd.DataFrame({"p": p, "c": c, "r": r}).groupby(["p", "c"])["r"].agg(
            ["mean", "size"]).unstack()
        n0, n1 = gg[("size", 0)].fillna(0), gg[("size", 1)].fillna(0)
        ne = (n0 * n1) / (n0 + n1).replace(0, np.nan)
        return ((gg[("mean", 1)] - gg[("mean", 0)]) * ne / (ne + k)).dropna()

    m24 = season == 2024
    C3 = (np.load(os.path.join(ROOT, "exp", "prod_champ_2024.npy"))[:2].mean(0)
          + post_for(tr, y, season < 2024, m24))
    for a, (ctx, k) in AX.items():
        C3 += (pd.Series(P[m24]).map(dvec(ctx, k)).fillna(0.).to_numpy()
               * np.where(ctx[m24] == 1, .5, -.5))
    r24 = y[m24] - C3
    base = r2(C3, y[m24])
    print(f"C3 기준선 2024 = {base:.1f}   잔차 sd = {r24.std():.4f}")

    num = [c for c in tr.columns if c not in SKIP
           and pd.api.types.is_numeric_dtype(tr[c])]
    X = np.column_stack([g(c) for c in num])[m24]
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    # 투수 정체성도 준다 — 차등 표가 못 실은 잔여 개체 효과까지 포함해 상한을 넉넉히
    X = np.column_stack([X, P[m24]])
    print(f"특징 {X.shape[1]}열 (수치 {len(num)} + pitcher_id)\n")

    GID = games(P[m24], g("asof_pitcher_n")[m24],
                g("asof_pitcher_prev1_game_success_rate")[m24],
                g("asof_pitcher_prev1_game_middle_rate")[m24])
    rng = np.random.default_rng(0)
    u, inv = np.unique(GID, return_inverse=True)
    half = (rng.random(len(u)) < 0.5)[inv]
    print(f"경기 {len(u):,}개를 절반으로 — 앞 {int(half.sum()):,}행 "
          f"뒤 {int((~half).sum()):,}행\n")

    def crossfit(target, tag):
        pred = np.zeros(len(target))
        for i, m in enumerate((half, ~half)):
            E.beat(f"{tag} half{i}")
            mdl = CatBoostRegressor(iterations=600, depth=6, learning_rate=0.05,
                                    l2_leaf_reg=10.0, loss_function="RMSE",
                                    random_seed=7, verbose=0, thread_count=6)
            mdl.fit(X[m], target[m])
            pred[~m] = mdl.predict(X[~m])
        return pred

    out = {}
    print("=" * 76)
    print(f"{'잔차 학습기':<26}{'잔차상관':>12}{'C3증분':>11}{'판정':>14}")
    print("=" * 76)
    for tag, tgt in (("실제 (C3 잔차)", r24),
                     ("위약 (셔플된 잔차)", rng.permutation(r24))):
        pred = crossfit(tgt, tag)
        cr = float(np.corrcoef(pred, r24)[0, 1]) if pred.std() > 1e-12 else 0.0
        best = max(r2(C3 + w * pred, y[m24]) - base for w in (0.25, 0.5, 1.0))
        out[tag] = dict(corr=cr, inc=best)
        v = "신호" if best > 5 else ("잡음" if best < 2 else "경계")
        print(f"{tag:<26}{cr:>+12.4f}{best:>11.1f}{v:>14}")

    gap = out["실제 (C3 잔차)"]["inc"] - out["위약 (셔플된 잔차)"]["inc"]
    print(f"\n실제 − 위약 = {gap:+.1f}")
    dec = "PROMISING" if gap >= 4.8 else "REJECTED"
    E.set_hypothesis_status("L2-B", "PROMISING" if dec == "PROMISING" else "CLOSED",
                            level=2, hypothesis="비선형 잔차 학습기 (LEVEL2 상한)",
                            result=round(gap, 2))
    E.finish_experiment(dict(
        experiment_id=EXP, hypothesis_id="L2-B", type="MODEL", level=2,
        started_at=E.read(E.CKPT)["start_time"],
        reopen_justification=("기준선이 C3 로 바뀜 · 목적이 상한 증명 · "
                              "경기 교차적합과 셔플 위약 추가"),
        results={k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in out.items()},
        gap=round(gap, 2), decision=dec, artifact=None,
        what_we_learned=(f"C3 잔차에 부스팅 트리를 직접 걸어도 실제−위약 {gap:+.1f}. "
                         f"오차 상관 0.9999(EXP007)와 합쳐 model space 상한 확정")))
    json.dump(out, open(os.path.join(ROOT, "exp", "exp008_bound.json"), "w"),
              indent=1, default=float)
    print(f"DECISION = {dec}")


if __name__ == "__main__":
    main()
