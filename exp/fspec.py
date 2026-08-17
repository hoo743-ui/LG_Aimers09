r"""F-specific 관계 분리 — **생산 경로** 재측정 (제출 후보 선정용).

## 왜 다시 재는가

`champ_xt` / `champ_ht` / `champ_st` 는 **캐시 경로**(`run_exp.py`)로만 쟀다.
K2 에서 캐시 +11.5 / 생산 −8.2 / 실제 LB −4.72 로 갈린 전례가 있으므로,
제출 후보 판정은 반드시 생산 경로로 한다.

    champ_xf   = Champion + X x is_F      (dx_* 8열의 F 전용 사본)
    champ_hf   = Champion + H1 x is_F     (lx_* 6열의 F 전용 사본)
    champ_xhf  = Champion + 둘 다 (14열)

## 가설

F 와 R 은 라벨 정의가 다른 두 체제다 (2023 년에 F 정의가 바뀌었고 F−R 격차가
2022 +0.205 -> 2024 −0.03). 같은 상태값이 두 체제에서 다른 의미를 가진다면
**타입 전용 사본**이 그 차이를 담을 수 있다. 트리는 `is_F` 로 분기한 뒤 같은
상호작용을 양쪽에서 다시 만들어야 하므로 원리적으로 비싸다.

## 판정 (사용자 규칙)

    A 2024 > 0            B 3폴드 중 2개 이상 양수
    D 시드 산포가 과대하지 않을 것

기준선은 캐시된 생산 경로 예측(`prod_champ_{f}.npy`)의 앞 2시드를 쓴다 —
`drop_tm.py` 에서 champ 재현이 소수점까지 확인됐다.

    .\.venv\Scripts\python.exe -u exp\fspec.py
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
from resid_table import post_for                           # noqa: E402

ft, sc = ba.ft, ba.sc
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

FOLDS = (2022, 2023, 2024)
SEEDS = (42, 43)


def main():
    tr = build_df()
    season = tr["season"].to_numpy()
    y = tr["control_success"].to_numpy(np.float64)
    tc = pd.read_csv(os.path.join(ft.DATA_DIR, "test.csv"),
                     encoding="utf-8-sig", nrows=0).columns
    allf = [c for c in tc if c != ft.ID]
    ctxf = [c for c in ft.COUNT_FEATS + ft.HAND_FEATS
            if c not in ("tmc_n", "tmh_n")]
    CHAMP = list(allf) + ctxf + sc.ASOF_COLS + sc.CTX_COLS + sc.LVL_COLS
    assert len(CHAMP) == 82

    isF = (tr["game_type"].to_numpy() == "F").astype(np.float64)
    XF, HF = [], []
    for c in sc.CTX_COLS:                       # X x is_F
        n = "xf_" + c[3:]
        tr[n] = tr[c].to_numpy(np.float64) * isF
        XF.append(n)
    for c in sc.LVL_COLS:                       # H1 x is_F
        n = "hf_" + c[3:]
        tr[n] = tr[c].to_numpy(np.float64) * isF
        HF.append(n)
    print(f"  F 전용 열 생성: X {len(XF)}개 {XF[:2]}... / H1 {len(HF)}개 {HF[:2]}...")
    print(f"  F 행 비율 (전체) {isF.mean():.1%}", flush=True)

    cfgs = [("champ_xf", CHAMP + XF),
            ("champ_hf", CHAMP + HF),
            ("champ_xhf", CHAMP + XF + HF)]
    R = {n: {} for n, _ in cfgs}
    B = {}
    for f in FOLDS:
        m_tr, m_va = season < f, season == f
        yv = y[m_va]
        post = post_for(tr, y, m_tr, m_va)
        P = np.load(os.path.join(ROOT, "exp", f"prod_champ_{f}.npy"))
        base = 1e5 * np.corrcoef(P[:2].mean(0) + post, yv)[0, 1] ** 2
        per_b = [1e5 * np.corrcoef(P[i] + post, yv)[0, 1] ** 2 for i in range(2)]
        B[f] = dict(rho2=base, per=per_b)
        print(f"\n=== 폴드 {f}  학습 {int(m_tr.sum()):,}행   champ 기준선 "
              f"{base:.1f} (시드별 {per_b[0]:.1f} {per_b[1]:.1f}) ===", flush=True)
        for n, fs in cfgs:
            t = time.time()
            acc = np.zeros(int(m_va.sum()))
            per = []
            for sd in SEEDS:                                # 시드는 순차
                mm = ba.pipeline(fs, sd)
                mm.fit(tr.loc[m_tr, fs], y[m_tr].astype(int))
                pv = mm.predict_proba(tr.loc[m_va, fs])[:, 1]
                per.append(1e5 * np.corrcoef(pv + post, yv)[0, 1] ** 2)
                acc += pv
                del mm
            r = 1e5 * np.corrcoef(acc / len(SEEDS) + post, yv)[0, 1] ** 2
            R[n][f] = dict(rho2=r, per=per)
            print(f"  {n:<12}{len(fs):>4}p  {r:>9.1f}  ({r-base:+.1f})   시드별 "
                  + " ".join(f"{v:.1f}" for v in per)
                  + f"   {time.time()-t:.0f}s", flush=True)

    print(f"\n=== champ 대비 증분 (생산 경로 3폴드 x 2시드) ===")
    print(f"{'후보':<12}" + "".join(f"{f:>10}" for f in FOLDS)
          + f"{'평균':>9}{'최악':>9}{'시드폭(2024)':>13}")
    for n, _ in cfgs:
        d = [R[n][f]["rho2"] - B[f]["rho2"] for f in FOLDS]
        sv = max(R[n][2024]["per"]) - min(R[n][2024]["per"])
        print(f"{n:<12}" + "".join(f"{x:>+10.1f}" for x in d)
              + f"{np.mean(d):>+9.1f}{min(d):>+9.1f}{sv:>13.1f}")
    print(f"\n  champ 자체 2024 시드폭 "
          f"{max(B[2024]['per'])-min(B[2024]['per']):.1f} (노이즈 참조)")
    json.dump({"cand": {n: {str(f): R[n][f] for f in FOLDS} for n, _ in cfgs},
               "base": {str(f): B[f] for f in FOLDS}},
              io.open(os.path.join(ROOT, "exp", "fspec.json"), "w",
                      encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
