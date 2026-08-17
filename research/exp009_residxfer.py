r"""EXP009 — EXP008 의 +4.4 를 가른다. 시즌 내 개체효과인가, 전이되는 구조인가.

## 적대적 자기검사

EXP008 은 C3 잔차에 부스팅 트리를 걸어 **+4.4** (잔차상관 +0.0090 = 4.5SE)를
얻었다. 승격 전에 반대 가설을 먼저 검사한다.

    반대가설 1  pitcher_id 를 줬고 같은 투수가 양쪽 절반에 있다.
                절반 A 에서 투수 잔차평균을 배워 B 에 쓰면 그건 시즌 내
                개체효과다 — 이미 "전이 +0.7%, 손익분기 아래"로 닫힌 것
    반대가설 2  2024 한 시즌에서만 쟀다. 체제 효과일 수 있다
    반대가설 3  교차적합이라도 목표 시즌 안에서 적합했다. 생산은 그럴 수 없다

## 설계

    A  pitcher_id 제거, 2024 내 교차적합    -> 개체효과를 뺀 구조가 남는가
    B  **과거 시즌에서 학습해 목표에 적용**   -> 생산과 동일한 진짜 전이
       B 는 2023(2021~22 학습)과 2024(2022~23 학습) 두 점에서 잰다

B 가 0 이면 EXP008 의 +4.4 는 시즌 내에만 존재하는 것이고, 2025 에서는
규정 4 로 접근할 수 없다 — 도달 불가 신호로 확정된다.

    .\.venv\Scripts\python.exe -u research\exp009_residxfer.py
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

EXP = "EXP009"
PREV2 = {2022: (2020, 2021), 2023: (2021, 2022), 2024: (2022, 2023)}
SKIP = {"control_success", "season", "pitcher_id", "batter_id",
        "pitcher_team_id", "batter_team_id", "game_id", "game_date"}
PAR = dict(iterations=600, depth=6, learning_rate=0.05, l2_leaf_reg=10.0,
           loss_function="RMSE", random_seed=7, verbose=0, thread_count=6)


def main():
    from catboost import CatBoostRegressor

    E.start_experiment(EXP, "L2-B2", "python research/exp009_residxfer.py", "load")
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

    C3, resC = {}, {}
    for f in (2022, 2023, 2024):
        m = season == f
        v = pv0[f].copy()
        for a, (ctx, k) in AX.items():
            v += (pd.Series(P[m]).map(dvec(ctx, PREV2[f], k)).fillna(0.).to_numpy()
                  * np.where(ctx[m] == 1, .5, -.5))
        C3[f], resC[f] = v, y[m] - v
    E.beat("C3 준비")

    num = [c for c in tr.columns if c not in SKIP
           and pd.api.types.is_numeric_dtype(tr[c])]
    XA = np.nan_to_num(np.column_stack([g(c) for c in num]),
                       nan=0.0, posinf=0.0, neginf=0.0)
    print(f"수치 특징 {XA.shape[1]}열\n")

    out = {}
    # ---- A. pitcher_id 제거, 2024 내 교차적합 ----
    m24 = season == 2024
    GID = games(P[m24], g("asof_pitcher_n")[m24],
                g("asof_pitcher_prev1_game_success_rate")[m24],
                g("asof_pitcher_prev1_game_middle_rate")[m24])
    rng = np.random.default_rng(0)
    u, inv = np.unique(GID, return_inverse=True)
    half = (rng.random(len(u)) < 0.5)[inv]
    base24 = r2(C3[2024], y[m24])
    print("=" * 80)
    print("A. 2024 내 교차적합 — pitcher_id 유무로 개체효과를 가른다")
    print("=" * 80)
    for tag, X in (("pitcher_id 포함 (EXP008 재현)",
                    np.column_stack([XA[m24], P[m24]])),
                   ("pitcher_id 제거", XA[m24])):
        E.beat(f"A {tag}")
        pred = np.zeros(int(m24.sum()))
        for m in (half, ~half):
            mdl = CatBoostRegressor(**PAR)
            mdl.fit(X[m], resC[2024][m])
            pred[~m] = mdl.predict(X[~m])
        inc = max(r2(C3[2024] + w * pred, y[m24]) - base24 for w in (0.25, 0.5, 1.0))
        cr = float(np.corrcoef(pred, resC[2024])[0, 1])
        out[f"A {tag}"] = dict(corr=cr, inc=inc)
        print(f"  {tag:<34} 잔차상관 {cr:>+8.4f}   C3증분 {inc:>+7.1f}")

    # ---- B. 과거 시즌 학습 -> 목표 시즌 적용 (진짜 전이) ----
    print("\n" + "=" * 80)
    print("B. 과거 시즌 학습 -> 목표 적용 — 생산과 동일한 조건")
    print("=" * 80)
    for tgt in (2023, 2024):
        src = PREV2[tgt]
        msrc = np.isin(season, src)
        mt = season == tgt
        bs = r2(C3[tgt], y[mt])
        rsrc = np.concatenate([resC[f] if f in resC else res0[f] for f in src])
        for tag, Xs, Xt in (("pitcher_id 포함",
                             np.column_stack([XA[msrc], P[msrc]]),
                             np.column_stack([XA[mt], P[mt]])),
                            ("pitcher_id 제거", XA[msrc], XA[mt])):
            E.beat(f"B {tgt} {tag}")
            mdl = CatBoostRegressor(**PAR)
            mdl.fit(Xs, rsrc)
            pred = mdl.predict(Xt)
            incs = {w: r2(C3[tgt] + w * pred, y[mt]) - bs for w in (0.25, 0.5, 1.0)}
            cr = float(np.corrcoef(pred, resC[tgt])[0, 1])
            out[f"B {tgt} {tag}"] = dict(corr=cr, inc=incs)
            print(f"  {tgt} {tag:<20} 잔차상관 {cr:>+8.4f}   "
                  + "  ".join(f"w={w:g} {v:>+6.1f}" for w, v in incs.items()))

    b23 = out["B 2023 pitcher_id 포함"]["inc"][0.5]
    b24 = out["B 2024 pitcher_id 포함"]["inc"][0.5]
    a_id = out["A pitcher_id 포함 (EXP008 재현)"]["inc"]
    a_no = out["A pitcher_id 제거"]["inc"]
    print(f"\n{'='*80}")
    print("해석")
    print("=" * 80)
    print(f"  시즌 내 (A)   개체 포함 {a_id:+.1f}   개체 제거 {a_no:+.1f}"
          f"   -> 개체효과 몫 {a_id - a_no:+.1f}")
    print(f"  전이   (B)   2023 {b23:+.1f}   2024 {b24:+.1f}")
    dec = "PROMISING" if (b23 > 0 and b24 >= 4.8) else "REJECTED"
    E.set_hypothesis_status("L2-B2", "PROMISING" if dec == "PROMISING" else "CLOSED",
                            level=2, hypothesis="잔차 학습기의 전이성",
                            result=round(b24, 2))
    E.finish_experiment(dict(
        experiment_id=EXP, hypothesis_id="L2-B2", type="MODEL", level=2,
        started_at=E.read(E.CKPT)["start_time"],
        results={k: {kk: (round(vv, 4) if isinstance(vv, float)
                          else {str(w): round(x, 2) for w, x in vv.items()})
                     for kk, vv in v.items()} for k, v in out.items()},
        within_season_identity_share=round(a_id - a_no, 2),
        transfer_2023=round(b23, 2), transfer_2024=round(b24, 2),
        decision=dec, artifact=None,
        what_we_learned=(f"EXP008 +4.4 의 분해: 시즌내 개체효과 몫 {a_id-a_no:+.1f}, "
                         f"과거학습 전이 2023 {b23:+.1f} / 2024 {b24:+.1f}")))
    json.dump(out, open(os.path.join(ROOT, "exp", "exp009_residxfer.json"), "w"),
              indent=1, default=str)
    print(f"\nDECISION = {dec}")


if __name__ == "__main__":
    main()
