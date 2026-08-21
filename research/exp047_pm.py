r"""EXP047 — PM(맥락별 구종 성향) in-model 후보를 **생산 경로**로 잰다.

캐시 경로는 착시 전력이 있다 (F-specific X/H1 — 캐시 +8.7, 생산 -4.9). 그래서
`path_alloc.build_df` 의 생산 프레임 위에서 `build_asof.py` 와 같은 절차로 잰다.

    대조   Champion 82열
    pm4    + pm_xsucc_{cnt,sh,two,onb}                      (합성 4열)
    pm8    + pm_xsucc_{cnt,sh} + pm_{cnt,sh}_{fb,br,os}     (8열)
    pm16   + 4국면 x (배합편차 3 + 합성 1)                    (16열)

표는 **직전 2시즌**(원장 고정값)에서, 조회 키는 그 행 자신의 `pitcher_id` 와
국면 컬럼뿐이다. 행 독립성 위반 없음.

    .\.venv\Scripts\python.exe -u research\exp047_pm.py --folds 2024 --seeds 42
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "exp"))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import build_asof as ba                                         # noqa: E402
from exp046_pitchmix import pitch_events, ctx_dev, look, TYPES  # noqa: E402

CTXN = ["cnt", "sh", "two", "onb"]
K_MIX = 300.0


def pm_columns(tr, ev, season, fold, K=K_MIX, win=2):
    """폴드 `fold` 의 행에 대한 PM 열. 표는 직전 `win` 시즌으로만 만든다."""
    src = (season < fold) & (season >= fold - win)
    m = season == fold
    cold = not src.any()          # 첫 시즌 — 표를 만들 과거가 없다. 전부 0
    n = int(m.sum())
    P = tr["pitcher_id"].to_numpy(np.int64)
    BB = tr["balls_before"].to_numpy(np.int64)
    SS = tr["strikes_before"].to_numpy(np.int64)
    SH = (tr["pitcher_hand"].to_numpy() == tr["batter_hand"].to_numpy()).astype(np.int64)
    OB = (tr["num_runners_on"].to_numpy(np.int64) > 0).astype(np.int64)
    CTX = {"cnt": BB * 4 + SS, "sh": SH,
           "two": (SS == 2).astype(np.int64), "onb": OB}
    y = tr["control_success"].to_numpy(float)
    out = {}
    if cold:
        # 과거가 없는 시즌은 편차가 **0 인 것이 아니라 알 수 없는 것**이다.
        # 0 으로 채우면 트리가 'PM=0 = 그 시즌' 을 학습한다 (season 이 in-model).
        for cn in CTXN:
            for t in TYPES:
                out[f"pm_{cn}_{t}"] = np.full(n, np.nan)
            out[f"pm_xsucc_{cn}"] = np.full(n, np.nan)
        return out
    for cn in CTXN:
        ctx = CTX[cn]
        tot = np.zeros(n)
        for t in TYPES:
            ok = ~np.isnan(ev[t])
            v = np.nan_to_num(ev[t])
            tb, _ = ctx_dev(P, ctx, v, src & ok, K)                # 배합 편차
            dmix = look(tb, None, P[m], ctx[m])
            out[f"pm_{cn}_{t}"] = dmix
            st, _ = ctx_dev(P, v.astype(np.int64), y, src & ok, K)  # 구종별 성공 편차
            tot += dmix * look(st, None, P[m], np.ones(n, np.int64))
        out[f"pm_xsucc_{cn}"] = tot
    return out


TMCOLS = ["tmc_fastball_dev", "tmc_breaking_dev", "tmc_offspeed_dev", "tmc_speed_dev",
          "tmh_fastball_dev", "tmh_breaking_dev", "tmh_offspeed_dev", "tmh_speed_dev"]
# 판본별로 **빼는** 열. 교체판은 오염된 TrackMan 8열을 PM 으로 갈아끼운다.
DROPS = {"pmswap": TMCOLS, "tmdrop": TMCOLS}

VARIANTS = {
    "pm4":  [f"pm_xsucc_{c}" for c in CTXN],
    "pm8":  [f"pm_xsucc_{c}" for c in ("cnt", "sh")] +
            [f"pm_{c}_{t}" for c in ("cnt", "sh") for t in TYPES],
    "pm16": [f"pm_{c}_{t}" for c in CTXN for t in TYPES] +
            [f"pm_xsucc_{c}" for c in CTXN],
}
VARIANTS["pmswap"] = VARIANTS["pm16"]      # 같은 16열 + TrackMan 8열 제거
VARIANTS["tmdrop"] = []                    # 대조군 — TrackMan 만 빼고 PM 없음


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="2024")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--variants", default="pm4,pm8,pm16")
    ap.add_argument("--win", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(ROOT, "exp", "exp047_pm.json"))
    a = ap.parse_args()
    folds = [int(x) for x in a.folds.split(",")]
    seeds = [int(x) for x in a.seeds.split(",")]
    vs = [v for v in a.variants.split(",") if v]

    ev = pitch_events()
    from path_alloc import build_df
    tr = build_df()
    import zipfile, io, joblib
    with zipfile.ZipFile(os.path.join(ROOT, "submissions", "cand_h1.zip")) as z:
        BASE = joblib.load(io.BytesIO(z.read("model/rf.pkl")))["features"]
    print(f"기본 {len(BASE)}열, 생산 프레임 {tr.shape}")

    season = tr["season"].to_numpy()
    y = tr["control_success"].to_numpy(np.float64)
    P = tr["pitcher_id"].to_numpy(np.int64)
    BH = tr["batter_hand"].to_numpy(np.int64)
    BB = tr["balls_before"].to_numpy(np.int64)
    SS = tr["strikes_before"].to_numpy(np.int64)
    OB = (tr["num_runners_on"].to_numpy(np.int64) > 0).astype(np.int64)
    PH = P * 10 + BH
    PHA = PH * 10 + (SS > BB).astype(np.int64)
    AX = [(P, PH), (PH, PHA), (PHA, PHA * 100 + (BB * 4 + SS)), (PH, PH * 10 + OB)]

    res = {}
    for fold in folds:
        mt, mv = season < fold, season == fold
        post = np.column_stack([
            ba.look(*ba.nested_dev(p[mt], c[mt], y[mt], k), c[mv])
            for (p, c), k in zip(AX, ba.KSH)]) @ ba.WPOST
        pm = pm_columns(tr, ev, season, fold, win=a.win)
        pm_tr = {f2: pm_columns(tr, ev, season, f2, win=a.win)
                 for f2 in sorted(set(season[mt]))}      # 학습 창을 **자르지 않는다**
        use = sorted(pm_tr)
        mt2 = mt
        assert np.array_equal(np.sort(season[mt2]), np.sort(np.concatenate(
            [season[season == f2] for f2 in use]))), "학습 창 불일치"

        print(f"\n=== 폴드 {fold} ===  학습 {mt2.sum():,}행 (시즌 {use})  "
              f"검증 {mv.sum():,}행", flush=True)

        for v in ["base"] + vs:
            cols = [] if v == "base" else VARIANTS[v]
            keep = [c for c in BASE if c not in DROPS.get(v, [])]
            Xtr = tr.loc[mt2, keep].copy()
            Xva = tr.loc[mv, keep].copy()
            for c in cols:
                Xtr[c] = np.concatenate([pm_tr[f2][c] for f2 in use])
                Xva[c] = pm[c]
            feats = keep + cols
            sc_ = []
            for s in seeds:
                t0 = time.time()
                mm = ba.pipeline(feats, s)
                mm.fit(Xtr, y[mt2].astype(int))
                pv = mm.predict_proba(Xva)[:, 1] + post
                r = 1e5 * np.corrcoef(pv, y[mv])[0, 1] ** 2
                sc_.append(r)
                print(f"  {v:5s} seed{s} {len(feats):3d}열  {r:8.2f}  "
                      f"({time.time()-t0:.0f}s)", flush=True)
                json.dump(res, open(a.out, "w"), indent=1)
            res[f"{fold}:{v}"] = sc_
        bb_ = np.array(res[f"{fold}:base"])
        print(f"  -> {'base':5s} {bb_.mean():8.2f} ± {bb_.std(ddof=1) if len(bb_)>1 else 0:.2f}",
              flush=True)
        for v in vs:
            gg = np.array(res[f"{fold}:{v}"])
            d = gg - bb_                      # **시드별로 짝지어** 뺀다
            print(f"  -> {v:5s} {gg.mean():8.2f} ± "
                  f"{gg.std(ddof=1) if len(gg)>1 else 0:5.2f}   대조 대비 "
                  f"{d.mean():+7.2f} ± {d.std(ddof=1) if len(d)>1 else 0:5.2f}  "
                  f"({(gg.mean()/bb_.mean()-1)*100:+.3f}%)  "
                  f"시드별 {' '.join(f'{x:+.1f}' for x in d)}", flush=True)
        json.dump(res, open(a.out, "w"), indent=1)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
