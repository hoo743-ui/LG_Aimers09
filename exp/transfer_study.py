r"""LOCAL -> FUTURE 전이 연구 — pseudo-future 가 핵심 증거다.

실제 LB 6건으로 곡선을 맞추지 않는다. 과거 시즌을 가상의 미래로 만들어
**같은 후보를 여러 번 전이시켜** 본다.

## 설계 — 창마다 학습하고 이후 **모든** 시즌에 예측한다

    창 w:  학습 = 시즌 < w,   평가 = w, w+1, ..., 2024

예측은 몇 초라 시간 거리(1·2·3시즌 앞)가 **추가 학습 없이** 나온다.

    LOCAL  (시점 t 에서 손에 쥔 증거)  = 창 t 의 목표 t   증분
    FUTURE (그 다음에 실제로 일어난 일) = 창 t+1 의 목표 t+1 증분

두 개를 이어 붙이면 (LOCAL_t -> FUTURE_{t+1}) 쌍이 4개 나온다.

## 후보와 그 **올바른 기준선**

실제 LB 앵커가 있는 후보를 그 앵커와 같은 중첩 구조로 만든다.

    BASE 55 = 기본47 + TrackMan ctx 8
    D   68 = BASE + cur_* 13          gain = D - BASE       (LB +85.65)
    DX  76 = D + dx_* 8               gain = DX - D         (LB +3.90)
    CH  82 = DX + lx_* 6  (Champion)  gain = CH - DX        (LB +5.16)
    DF  74 = D + form_* 6             gain = DF - D         (LB -8.55)
    K2  86 = CH + k2_* 4              gain = K2 - CH        (LB -4.72)

`EB`(LB -13.83)는 시즌별 적률 추정이 필요해 이 판에서 제외하고 실제 LB
앵커로만 쓴다. `form_*` 는 `prev{k} - cur` 로 그 자리에서 만든다 (새 피처가
아니라 23회차에 실제 제출된 구성의 재현이다).

## 규약

후처리 편차 4축은 **각 창의 학습 행**으로 만들어 목표 시즌에 조회한다.
시드는 42/43 두 개. 모든 값은 아핀 이전 `rho^2 = 1e5 * corr^2`.

    .\.venv\Scripts\python.exe -u exp\transfer_study.py
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

WINDOWS = (2020, 2021, 2022, 2023, 2024)
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
    BASE = list(allf) + ctxf

    # form_* — 23회차(F) 구성의 재현. prev{k} - cur, 같은 창 안의 뺄셈.
    FORM = []
    for lb, pat in sc.FORM_SPEC:
        for k in sc.FORM_WIN:
            n = f"form_{lb}{k}"
            tr[n] = (tr[pat.format(k)].to_numpy(np.float64)
                     - tr[f"cur_{lb}"].to_numpy(np.float64))
            FORM.append(n)

    CFG = [("BASE", BASE),
           ("D", BASE + sc.ASOF_COLS),
           ("DX", BASE + sc.ASOF_COLS + sc.CTX_COLS),
           ("CH", BASE + sc.ASOF_COLS + sc.CTX_COLS + sc.LVL_COLS),
           ("DF", BASE + sc.ASOF_COLS + FORM),
           ("K2", BASE + sc.ASOF_COLS + sc.CTX_COLS + sc.LVL_COLS + sc.K2_COLS)]
    for n, f in CFG:
        print(f"  {n:<5}{len(f):>4}p")
    print(f"  창 {WINDOWS}, 시드 {SEEDS}", flush=True)

    R = {}
    for w in WINDOWS:
        m_tr = season < w
        targets = [s for s in sorted(set(season)) if s >= w]
        post = {s: post_for(tr, y, m_tr, season == s) for s in targets}
        print(f"\n=== 창 {w}  학습 {int(m_tr.sum()):,}행  ->  목표 {targets} ===",
              flush=True)
        for n, fs in CFG:
            t0 = time.time()
            acc = {s: np.zeros(int((season == s).sum())) for s in targets}
            per = {s: [] for s in targets}
            for sd in SEEDS:                               # 시드는 순차
                mm = ba.pipeline(fs, sd)
                mm.fit(tr.loc[m_tr, fs], y[m_tr].astype(int))
                for s in targets:
                    ms = season == s
                    p = mm.predict_proba(tr.loc[ms, fs])[:, 1]
                    acc[s] += p
                    per[s].append(1e5 * np.corrcoef(p + post[s],
                                                    y[ms])[0, 1] ** 2)
                del mm
            for s in targets:
                ms = season == s
                r = 1e5 * np.corrcoef(acc[s] / len(SEEDS) + post[s],
                                      y[ms])[0, 1] ** 2
                R[f"{n}|{w}|{s}"] = dict(rho2=r, per=per[s])
            print(f"  {n:<5}{len(fs):>4}p  "
                  + "  ".join(f"{s}:{R[f'{n}|{w}|{s}']['rho2']:.1f}"
                              for s in targets)
                  + f"   {time.time()-t0:.0f}s", flush=True)
        json.dump(R, io.open(os.path.join(ROOT, "exp", "transfer_study.json"),
                             "w", encoding="utf-8"), indent=1)

    print("\n" + "=" * 84)
    print("증분 행렬 (후보 - 그 후보의 기준선), 창 x 목표")
    print("=" * 84)
    PAIR = [("D", "BASE"), ("X", "D"), ("H1", "DX"), ("F", "D"), ("K2", "CH")]
    NUM = {"D": "D", "X": "DX", "H1": "CH", "F": "DF", "K2": "K2"}
    for lbl, base in PAIR:
        print(f"\n[{lbl}]  = {NUM[lbl]} - {base}")
        print(f"{'창':<7}" + "".join(f"{s:>10}" for s in (2020, 2021, 2022, 2023, 2024)))
        for w in WINDOWS:
            row = f"{w:<7}"
            for s in (2020, 2021, 2022, 2023, 2024):
                k1, k2 = f"{NUM[lbl]}|{w}|{s}", f"{base}|{w}|{s}"
                row += (f"{R[k1]['rho2'] - R[k2]['rho2']:>+10.1f}"
                        if k1 in R and k2 in R else f"{'':>10}")
            print(row)


if __name__ == "__main__":
    main()
