r"""현행 TrackMan 8열 **제거** 실험 — 죽은 무게인지 잰다.

## 왜 이 실험인가

TMX(2024 −0.9)와 TMR(−3.9)이 같은 서명을 보였고 원인이 규명됐다 —
TrackMan 결측(F 40% vs R 18%)이 `game_type` F 를 대리한다. 2022 에서는
공짜 신호를 실어 나르고 2024 에서는 죽는다. 조인 구조가 원인이라 **어떤
TrackMan 파생도 피할 수 없다.**

그런데 Champion 은 이미 TrackMan 8열을 지고 있다 (`tmc_*_dev` 4 + `tmh_*_dev` 4).
그것은 7회차(LB 825, 구체제)에 채택된 것이고 D/X/H1 이 얹히기 전에 검증됐다.
같은 오염을 지고 있다면 **지금은 죽은 무게**일 수 있다.

새 피처가 아니라 **제거** 실험이라는 점이 다르다. 통과하면 82피처가 74피처가
되고 추론도 빨라진다.

## 무엇을 재는가

    champ           = 기본47 + TrackMan ctx 8 + D 13 + X 8 + H1 6   (82피처)
    champ_notm      = − TrackMan 8열 전부                            (74피처)
    champ_nocnt     = − 카운트 조건부 4열 (tmc_*_dev)                (78피처)
    champ_nohand    = − 타자손 조건부 4열 (tmh_*_dev)                (78피처)

분해 두 개를 같이 넣는 이유 — 8열 전부가 죽었는지, 한쪽만 죽었는지는 다른
결론이다. 학습 비용이 폴드당 200초라 한 번에 잰다.

## 경로

생산 경로 3폴드 x 2시드, 후처리 편차 4축 유지, 아핀 이전 `rho^2`.
`tmx_probe.py` 와 **완전히 같은 틀**이라 champ 값이 재현되는지로 검증된다
(tmx.json: 2022 2475.8 / 2023 211.8 / 2024 943.8).

판정은 CLAUDE.md 5절 — 2024 부호가 유일한 신뢰 지표다. 제거가 2024 에서
양수면 후보, 음수면 현행 유지.
"""
import io
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "exp"))
import build_asof as ba                                    # noqa: E402
from path_alloc import build_df                            # noqa: E402

ft, sc = ba.ft, ba.sc
REF = {2022: 2475.8, 2023: 211.8, 2024: 943.8}             # tmx.json 의 champ


def main():
    tr = build_df()
    tc = pd.read_csv(os.path.join(ft.DATA_DIR, "test.csv"),
                     encoding="utf-8-sig", nrows=0).columns
    allf = [c for c in tc if c != ft.ID]
    ctxf = [c for c in ft.COUNT_FEATS + ft.HAND_FEATS
            if c not in ("tmc_n", "tmh_n")]
    CHAMP = list(allf) + ctxf + sc.ASOF_COLS + sc.CTX_COLS + sc.LVL_COLS
    assert len(ctxf) == 8, f"TrackMan ctx 열 수가 8 이 아니다: {ctxf}"
    assert len(CHAMP) == 82, f"Champion 피처 수가 82 가 아니다: {len(CHAMP)}"
    cnt4 = [c for c in ctxf if c.startswith("tmc_")]
    hnd4 = [c for c in ctxf if c.startswith("tmh_")]
    print(f"  제거 대상 8열: {ctxf}", flush=True)

    cfgs = [("champ", CHAMP),
            ("champ_notm", [c for c in CHAMP if c not in ctxf]),
            ("champ_nocnt", [c for c in CHAMP if c not in cnt4]),
            ("champ_nohand", [c for c in CHAMP if c not in hnd4])]

    y = tr[ft.TARGET].to_numpy(np.float64)
    season = tr["season"].to_numpy()
    P = tr["pitcher_id"].to_numpy(np.int64)
    BH = tr["batter_hand"].to_numpy(np.int64)
    BB = tr["balls_before"].to_numpy(np.int64)
    SS = tr["strikes_before"].to_numpy(np.int64)
    OB = (tr["num_runners_on"].to_numpy(np.int64) > 0).astype(np.int64)
    PH = P * 10 + BH
    PHA = PH * 10 + (SS > BB).astype(np.int64)
    AX = [(P, PH), (PH, PHA), (PHA, PHA * 100 + (BB * 4 + SS)), (PH, PH * 10 + OB)]

    seeds = [int(v) for v in
             (sys.argv[sys.argv.index("--seeds") + 1] if "--seeds" in sys.argv
              else "42,43").split(",")]
    folds = [2022, 2023, 2024]
    R = {n: {} for n, _ in cfgs}
    for f in folds:
        m_tr, m_va = season < f, season == f
        yv = y[m_va]
        post = np.column_stack([
            ba.look(*ba.nested_dev(p[m_tr], c[m_tr], y[m_tr], k), c[m_va])
            for (p, c), k in zip(AX, ba.KSH)]) @ ba.WPOST
        print(f"\n=== 폴드 {f}  학습 {int(m_tr.sum()):,}행 ===", flush=True)
        for n, fs in cfgs:
            t = time.time()
            acc = np.zeros(int(m_va.sum()))
            per = []
            for sd in seeds:                                # 시드는 순차
                mm = ba.pipeline(fs, sd)
                mm.fit(tr.loc[m_tr, fs], y[m_tr].astype(int))
                pv = mm.predict_proba(tr.loc[m_va, fs])[:, 1]
                per.append(1e5 * np.corrcoef(pv + post, yv)[0, 1] ** 2)
                acc += pv
                del mm
            r = 1e5 * np.corrcoef(acc / len(seeds) + post, yv)[0, 1] ** 2
            R[n][f] = dict(rho2=r, per=per)
            print(f"  {n:<14}{len(fs):>4}p  {r:>9.1f}   시드별 "
                  + " ".join(f"{v:.1f}" for v in per)
                  + f"   {time.time()-t:.0f}s", flush=True)
        if seeds == [42, 43]:                   # 틀을 재현하는지 즉시 확인한다
            got, exp = R["champ"][f]["rho2"], REF[f]
            print(f"  [재현] champ {got:.1f} vs tmx.json {exp:.1f} "
                  f"({got-exp:+.1f})", flush=True)

    print(f"\n=== champ 대비 증분 (생산 경로, 후처리 4축 유지) ===")
    print(f"{'후보':<16}" + "".join(f"{f:>10}" for f in folds)
          + f"{'평균':>9}{'최악':>9}{'시드폭':>9}")
    for n, _ in cfgs:
        d = [R[n][f]["rho2"] - R["champ"][f]["rho2"] for f in folds]
        sv = max(max(R[n][f]["per"]) - min(R[n][f]["per"]) for f in folds)
        print(f"{n:<16}" + "".join(f"{x:>+10.1f}" for x in d)
              + f"{np.mean(d):>+9.1f}{min(d):>+9.1f}{sv:>9.1f}")
    json.dump({n: {str(f): R[n][f] for f in folds} for n, _ in cfgs},
              io.open(os.path.join(ROOT, "exp", "drop_tm.json"), "w",
                      encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
