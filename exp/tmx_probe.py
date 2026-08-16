r"""TMX 프로브 — 상황 조건부 TrackMan 편차 6열 (spin / ivb / hb).

## 무엇이 새로운가

TrackMan 30컬럼 중 Champion 이 실제로 쓰는 **원천은 두 개뿐**이다 —
`pitch_type_group`(breaking/fastball/offspeed 편차 6열)과 `rel_speed`(speed
편차 2열). `spin_rate` · `induced_vert_break` · `horz_break` 는 미사용이다.

그 6열이 `cols.json` 의 `extra` 에 **이미 만들어져 있는데** 원장에도
`run_exp.py` 에도 시험 이력이 없다.

## 기각된 TrackMan 실험과 무엇이 다른가

- **6-o 행 단위 조인** — 불가(키 없음) + TrackMan 이 2024 에서 끝남. TMX 는
  행 조인이 아니라 **투수 x 맥락 집계 조회**라 그 벽에 걸리지 않는다.
- **25-a TR-VAR** — `sd(rel_height)` 등 **산포**를 투수 상수로 준 것이고
  타깃 상관이 0.01 대라 기각됐다. TMX 는 산포가 아니라 **그 투수가 이 카운트/
  타자손에서 자기 평균 대비 얼마나 벗어나는가**이고, 행마다 카운트·손으로
  조회되므로 **행 단위로 값이 변한다**.
- **11-c 구종 상호작용** — 해당 투구의 구종이 train/test 에 없어 불가.
  TMX 는 그 투구의 구종을 쓰지 않는다.

즉 TMX 는 **채택된 8열(LB +10.98)과 완전히 같은 구성**이고 측정 물리량만 다르다.

## 경로

생산 경로 3폴드 x 2시드, 후처리 편차 4축 유지, 아핀 이전 `rho^2`.
캐시 경로는 채택 근거로 쓰지 않는다 — K2 에서 캐시 +11.5 / 생산 -8.2 /
실제 LB -4.72 로 생산 쪽이 맞았다.
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
CACHE = os.path.join(ROOT, "exp", "cache")


def attach_tmx(tr):
    """`extra` 6열을 캐시 X 에서 가져온다. **행 정렬을 반드시 검증한다.**"""
    meta = json.load(io.open(os.path.join(CACHE, "cols.json"), encoding="utf-8"))
    ixc = {c: i for i, c in enumerate(meta["cols"])}
    X = np.load(os.path.join(CACHE, "X.npy"), mmap_mode="r")
    assert len(X) == len(tr), f"행수 불일치 {len(X)} vs {len(tr)}"
    a = np.asarray(X[:, ixc["asof_pitcher_n"]], dtype=np.float64)
    b = tr["asof_pitcher_n"].to_numpy(np.float64)
    bad = int((a != b).sum())
    assert bad == 0, f"행 정렬 불일치 {bad:,}행 — 캐시와 생산 df 순서가 다르다"
    print(f"  정렬 검증 OK ({len(tr):,}행)", flush=True)
    for c in meta["extra"]:
        tr[c] = np.asarray(X[:, ixc[c]], dtype=np.float32)
    return tr, list(meta["extra"])


def main():
    tr = build_df()
    tr, TMX = attach_tmx(tr)
    tc = pd.read_csv(os.path.join(ft.DATA_DIR, "test.csv"),
                     encoding="utf-8-sig", nrows=0).columns
    allf = [c for c in tc if c != ft.ID]
    ctxf = [c for c in ft.COUNT_FEATS + ft.HAND_FEATS
            if c not in ("tmc_n", "tmh_n")]
    CHAMP = list(allf) + ctxf + sc.ASOF_COLS + sc.CTX_COLS + sc.LVL_COLS
    GROUPS = {"champ_tmx": TMX}
    if "--split" in sys.argv:                       # 2단계: 내부 분해
        GROUPS = {"champ_tmx": TMX,
                  "tmx_spin": [c for c in TMX if "spin" in c],
                  "tmx_ivb": [c for c in TMX if "ivb" in c],
                  "tmx_hb": [c for c in TMX if "hb" in c],
                  "tmx_no_ivbh": [c for c in TMX if "tmh_ivb" not in c]}
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
    cfgs = [("champ", CHAMP)] + [(n, CHAMP + g) for n, g in GROUPS.items()]
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
    print(f"\n=== champ 대비 증분 (생산 경로, 후처리 4축 유지) ===")
    print(f"{'후보':<16}" + "".join(f"{f:>10}" for f in folds)
          + f"{'평균':>9}{'최악':>9}{'시드폭':>9}")
    for n, _ in cfgs:
        d = [R[n][f]["rho2"] - R["champ"][f]["rho2"] for f in folds]
        sv = max(max(R[n][f]["per"]) - min(R[n][f]["per"]) for f in folds)
        print(f"{n:<16}" + "".join(f"{x:>+10.1f}" for x in d)
              + f"{np.mean(d):>+9.1f}{min(d):>+9.1f}{sv:>9.1f}")
    json.dump({n: {str(f): R[n][f] for f in folds} for n, _ in cfgs},
              io.open(os.path.join(ROOT, "exp", "tmx.json"), "w",
                      encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
