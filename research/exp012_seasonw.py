r"""EXP012 — LEVEL 3 마지막 미검증 상자. 시즌 표본가중. TYPE B.

## 왜 이것만 남았는가

    창 자르기   D 의 cur_n = asof_n - prior_n[투수] 가 학습 창 끝에 묶여 있어
                창을 바꾸면 피처 정의가 바뀐다 (EXP010 에서 확인, 958->252)
    표본가중    D 를 보존한 채 최신성만 바꾼다. **이것만 유효한 검정이다**

## 기제

2023 년에 `game_type` F 의 라벨 정의가 바뀌었다 (F 성공률 .709 -> .473,
같은 투수 92명 중 98.9% 가 동일 방향). 2025 는 신체제다. 구체제 시즌
(2019~2022)을 낮추면 좋아질 수 있다.

## 사전 확률은 낮다 — 그리고 그걸 명시한다

같은 기제를 **표적해서** 이미 실패했다.

    F 행 전량 제외      2024 -6.9
    구체제 F 만 제외     2024 -9.7

표본가중은 그보다 무딘 도구다. 그럼에도 돌리는 이유는 LEVEL 3 의 마지막
상자를 **논증이 아니라 측정으로** 닫기 위해서다.

## 채택 문턱 (TYPE B 이므로 높다)

`CLAUDE.md` 5-a 의 다섯 조건을 모두 넘어야 한다. 로컬 양수만으로는 채택하지
않는다 — 30회차가 정확히 그렇게 실패했다(로컬 +0.50%, LB -0.94%).

    .\.venv\Scripts\python.exe -u research\exp012_seasonw.py
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "exp"))
import engine as E                                          # noqa: E402
import build_asof as ba                                     # noqa: E402
from path_alloc import build_df                             # noqa: E402
from resid_table import post_for                            # noqa: E402
from traj_probe import r2                                   # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

EXP = "EXP012"
GAMMAS = [1.0, 0.7, 0.5]          # w = gamma ** (2024 - season)
SEED = 42


def main():
    E.start_experiment(EXP, "L3-B", "python research/exp012_seasonw.py", "load")
    ft, sc = ba.ft, ba.sc
    tr = build_df()
    season = tr["season"].to_numpy()
    y = tr["control_success"].to_numpy(np.float64)
    P = tr["pitcher_id"].to_numpy(np.int64)

    def g(c):
        return tr[c].to_numpy(np.float64)

    tc = pd.read_csv(os.path.join(ft.DATA_DIR, "test.csv"),
                     encoding="utf-8-sig", nrows=0).columns
    allf = [c for c in tc if c != ft.ID]
    ctxf = [c for c in ft.COUNT_FEATS + ft.HAND_FEATS if c not in ("tmc_n", "tmh_n")]
    CHAMP = list(allf) + ctxf + sc.ASOF_COLS + sc.CTX_COLS + sc.LVL_COLS

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
    m_tr = season < 2024
    add = post_for(tr, y, m_tr, m24)
    for a, (ctx, k) in AX.items():
        add += (pd.Series(P[m24]).map(dvec(ctx, k)).fillna(0.).to_numpy()
                * np.where(ctx[m24] == 1, .5, -.5))

    ref = np.load(os.path.join(ROOT, "exp", "prod_champ_2024.npy"))
    print(f"참조 — 현행 2시드 평균 {r2(ref[:2].mean(0) + add, y[m24]):.1f}")
    print(f"     시드 {SEED} 단독(캐시 [0]) {r2(ref[0] + add, y[m24]):.1f}")
    print(f"학습셋 {int(m_tr.sum()):,}행  특징 {len(CHAMP)}열\n")

    print("=" * 76)
    print(f"{'감쇠 gamma':<14}{'2019 가중':>11}{'유효표본':>12}{'2024 점수':>12}"
          f"{'gamma=1 대비':>14}")
    print("=" * 76)
    out, ref1 = {}, None
    for gm in GAMMAS:
        E.beat(f"gamma={gm}")
        t0 = time.time()
        w = gm ** (2024 - season[m_tr].astype(float))
        mdl = ba.pipeline(CHAMP, SEED)
        mdl.fit(tr.loc[m_tr, CHAMP], y[m_tr].astype(int), clf__sample_weight=w)
        pv = mdl.predict_proba(tr.loc[m24, CHAMP])[:, 1]
        sc_ = r2(pv + add, y[m24])
        if ref1 is None:
            ref1 = sc_
        neff = w.sum() ** 2 / (w ** 2).sum()
        out[gm] = dict(score=sc_, neff=float(neff), secs=time.time() - t0)
        print(f"{gm:<14.2f}{gm**5:>11.3f}{neff:>12,.0f}{sc_:>12.1f}"
              f"{sc_ - ref1:>+14.1f}   ({time.time()-t0:.0f}s)")
        np.save(os.path.join(ROOT, "exp", f"pred24_gamma{gm}.npy"), pv)
        del mdl

    best = max(out, key=lambda k: out[k]["score"])
    gain = out[best]["score"] - ref1
    dec = "PROMISING" if (best != 1.0 and gain >= 4.8) else "REJECTED"
    print(f"\n최선 gamma={best}  gamma=1 대비 {gain:+.1f}")
    print("  TYPE B 이므로 로컬 양수만으로는 채택하지 않는다 (5-a 다섯 조건)")
    E.set_hypothesis_status("L3-B", "PROMISING" if dec == "PROMISING" else "CLOSED",
                            level=3, result=round(gain, 2))
    E.finish_experiment(dict(
        experiment_id=EXP, hypothesis_id="L3-B", type="B", level=3,
        started_at=E.read(E.CKPT)["start_time"],
        results={str(k): {kk: round(vv, 2) for kk, vv in v.items()}
                 for k, v in out.items()},
        best_gamma=best, gain=round(gain, 2), decision=dec, artifact=None,
        what_we_learned=(f"시즌 지수감쇠 표본가중: 최선 gamma={best}, "
                         f"gamma=1 대비 {gain:+.1f}. LEVEL 3 마지막 상자를 "
                         f"측정으로 닫음")))
    json.dump(out, open(os.path.join(ROOT, "exp", "exp012_seasonw.json"), "w"),
              indent=1, default=float)
    print(f"DECISION = {dec}")


if __name__ == "__main__":
    main()
