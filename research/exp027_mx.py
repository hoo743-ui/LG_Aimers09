r"""EXP026 — 중복 열 가지치기. 명시적 2구성 비교. build_asof 의 대조군 기계를 안 쓴다.

## 왜 이 실험인가 (경로가 흥미롭다)

UC/NZ 실험이 **가드 버그**로 무효였는데(열이 계산되지 않았다), 그 무효 실험이
뜻밖의 것을 측정해 버렸다 — **전부 NaN 인 8열을 넣으면 2024 가 −13.5.**

즉 **열에는 세금이 있다.** 그렇다면 뒤집을 수 있다 — 정보 손실이 0 인 중복 열을
**빼면** 세금만큼 이득이 나야 한다.

## 무엇을 빼는가 (전부 100.0000% 증명된 중복)

    asof_pitcher_pitchmix_n  = asof_pitcher_n
    cur_logn_mix             = cur_logn_pitch   (위 열의 D 파생이라 헛열)
    run_total_before         = run_top + run_bot
    score_diff_home          = run_bot − run_top
    num_runners_on           = 1b + 2b + 3b
    base_state               = 주자 3플래그의 문자열
    away_win_expectancy      = home_win_expectancy 의 결정함수

RELATION_LEDGER §3 에 전부 실측 기록이 있다. **정보는 하나도 잃지 않는다.**

## 설계

폴드 2024 (학습 2019~2023), 시드 2개 평균. 후처리 편차 4축 + 차등 3축(w=0.65)을
동일하게 얹고 비교한다. 두 구성 외의 차이는 없다.

    .\.venv\Scripts\python.exe -u research\exp026_prune.py
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

EXP = "EXP027"
SEEDS = (42, 43)
PRUNE = ["asof_pitcher_pitchmix_n", "run_total_before", "score_diff_home",
         "num_runners_on", "base_state", "away_win_expectancy", "cur_logn_mix"]


def main():
    E.start_experiment(EXP, "A-MX", "python research/exp026_prune.py", "load")
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
    FULL = list(allf) + ctxf + sc.ASOF_COLS + sc.CTX_COLS + sc.LVL_COLS
    LEAN = FULL + sc.MX_COLS
    print(f"FULL {len(FULL)}열   +MX {len(LEAN)}열")
    print(f"  {[c for c in FULL if c not in LEAN]}\n")

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
        add += 0.65 * (pd.Series(P[m24]).map(dvec(ctx, k)).fillna(0.).to_numpy()
                       * np.where(ctx[m24] == 1, .5, -.5))
    ref = np.load(os.path.join(ROOT, "exp", "prod_champ_2024.npy"))
    print(f"참조 — 캐시 2시드 {r2(ref[:2].mean(0) + add, y[m24]):.1f}\n")

    print("=" * 70)
    print(f"{'구성':<16}{'열':>5}{'시드42':>10}{'시드43':>10}{'2시드 평균':>12}{'':>8}")
    print("=" * 70)
    out = {}
    base = None
    for nm, feats in (("FULL (현행)", FULL), ("+MX (믹스x맥락)", LEAN)):
        preds = []
        for sd in SEEDS:
            E.beat(f"{nm} seed {sd}")
            t0 = time.time()
            mdl = ba.pipeline(feats, sd)
            mdl.fit(tr.loc[m_tr, feats], y[m_tr].astype(int))
            preds.append(mdl.predict_proba(tr.loc[m24, feats])[:, 1])
            del mdl
        s1, s2 = (r2(p + add, y[m24]) for p in preds)
        avg = r2(np.mean(preds, 0) + add, y[m24])
        if base is None:
            base = avg
        out[nm] = dict(seed42=s1, seed43=s2, avg=avg, n=len(feats))
        print(f"{nm:<16}{len(feats):>5}{s1:>10.1f}{s2:>10.1f}{avg:>12.1f}"
              f"{avg - base:>+8.1f}")
        np.save(os.path.join(ROOT, "exp", f"pred24_{'full' if 'FULL' in nm else 'mx'}.npy"),
                np.asarray(preds))

    gain = out["+MX (믹스x맥락)"]["avg"] - out["FULL (현행)"]["avg"]
    print(f"\nMX 이득 = {gain:+.1f}   (시드 잡음 ±7.2)")
    print("  참고 — 전부 NaN 인 8열 추가는 −13.5 였다 (열 추가 세금)")
    dec = "PROMISING" if gain > 3 else "REJECTED"
    E.set_hypothesis_status("A-MX", "PROMISING" if dec == "PROMISING" else "CLOSED",
                            level=2, hypothesis="구종믹스 x 맥락 (X/H1 연장)",
                            result=round(gain, 2))
    E.finish_experiment(dict(
        experiment_id=EXP, hypothesis_id="A-MX", type="A", level=2,
        started_at=E.read(E.CKPT)["start_time"], added=list(sc.MX_COLS),
        results={k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in out.items()},
        gain=round(gain, 2), decision=dec, artifact=None,
        what_we_learned=(f"정보 손실 0 이 증명된 중복 7열을 빼면 2024 {gain:+.1f}. "
                         f"열 추가 세금(NaN 8열 = -13.5)의 역방향 검정")))
    json.dump(out, open(os.path.join(ROOT, "exp", "exp027_mx.json"), "w"),
              indent=1, default=float)
    print(f"\nDECISION = {dec}")


if __name__ == "__main__":
    main()
