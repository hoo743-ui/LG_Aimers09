r"""잔차 표의 **시즌 간 전이** 측정 — 오라클이 가리킨 자리가 합법으로 닿는가.

## 어디서 왔나

`oracle_probe.py` 가 남은 정보의 소재를 특정했다 (2024, 기준선 944.0).

    pitcher x 타자손      cv2 1020.8   +76.8
    pitcher x 월          cv2 1005.2   +61.2
    pitcher x 시즌내십분위  cv2  996.2   +52.2
    batter / count / 국면  전부 기준선 이하

전부 **투수 단위**이고, 후처리 `dev_platoon`(+11.9)이 이미 얹힌 위의 잔여분이다.
그런데 cv2 는 2024 자신의 라벨로 교차적합한 값이라 **시즌 내** 정보다.

## 이 파일이 재는 것

평가셋(2025)에 쓰려면 **과거 시즌의 잔차로 만든 표가 다음 시즌에 통해야** 한다.
그것을 그대로 재현한다 — 한 시즌 앞으로.

    fold 2023 모델(학습 <2023)의 2023 잔차로 셀 표를 만든다
        -> 그 표를 2024 행에 조회해 더한다
        -> 2024 rho^2 가 기준선 944.0 보다 오르는가

기존 후처리 편차 4축은 **라벨 편차**(y - 그룹평균)로 만든다. 여기서는 **잔차**
(y - 예측)로 만든다 — 모델이 이미 회수한 몫을 빼고 남은 것만 싣는다는 점이
다르다. 축 자체(투수 x 타자손)는 편차 B 축과 같은 자리이므로, 오르면 그것은
**추정 대상을 바꾼 이득**이다.

규정 4) 안이다 — 표는 학습 구간에서만 만들고 조회 키는 그 행 자신의
`pitcher_id`/`batter_hand`/`game_month` 뿐이다.

    .\.venv\Scripts\python.exe -u exp\resid_table.py
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
from oracle_probe import apply_table, group_mean, r2       # noqa: E402

ft, sc = ba.ft, ba.sc
KS = [500, 2000, 5000, 10000, 20000, 50000, 100000, 200000]
PRED = os.path.join(ROOT, "exp", "prod_champ_%d.npy")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def post_for(tr, y, m_tr, m_va):
    """후처리 편차 4축을 학습 구간에서 만들어 검증 구간에 조회한다."""
    P = tr["pitcher_id"].to_numpy(np.int64)
    BH = tr["batter_hand"].to_numpy(np.int64)
    BB = tr["balls_before"].to_numpy(np.int64)
    SS = tr["strikes_before"].to_numpy(np.int64)
    OB = (tr["num_runners_on"].to_numpy(np.int64) > 0).astype(np.int64)
    PH = P * 10 + BH
    PHA = PH * 10 + (SS > BB).astype(np.int64)
    AX = [(P, PH), (PH, PHA), (PHA, PHA * 100 + (BB * 4 + SS)), (PH, PH * 10 + OB)]
    return np.column_stack([
        ba.look(*ba.nested_dev(p[m_tr], c[m_tr], y[m_tr], k), c[m_va])
        for (p, c), k in zip(AX, ba.KSH)]) @ ba.WPOST


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

    def pv_for(fold):
        """폴드 `fold` 의 생산 경로 예측(후처리 포함)을 만들거나 캐시에서 읽는다."""
        m = season == fold
        f = PRED % fold
        if os.path.exists(f):
            p = np.load(f)
            print(f"  {fold} 예측 캐시 사용 {p.shape}", flush=True)
        else:
            acc = []
            for sd in (42, 43):                             # 시드는 순차
                t = time.time()
                mm = ba.pipeline(CHAMP, sd)
                mm.fit(tr.loc[season < fold, CHAMP],
                       y[season < fold].astype(int))
                acc.append(mm.predict_proba(tr.loc[m, CHAMP])[:, 1])
                print(f"  fold{fold} seed {sd} {time.time()-t:.0f}s", flush=True)
                del mm
            p = np.asarray(acc)
            np.save(f, p)
        return p[:3].mean(0) + post_for(tr, y, season < fold, m)

    pv = {}
    for fold in (2022, 2023, 2024):
        m = season == fold
        pv[fold] = pv_for(fold)
        print(f"  {fold} 기준선 rho^2 = {r2(pv[fold], y[m]):.1f}", flush=True)
    print("  (원장 champ 2022 2475.8 / 2023 211.8 / 2024 943.8)\n", flush=True)

    PID = tr["pitcher_id"].to_numpy(np.int64)
    BH = tr["batter_hand"].to_numpy(np.int64)
    MON = tr["game_month"].to_numpy(np.int64)
    GROUPS = {"pitcher x 타자손": PID * 10 + BH,
              "pitcher": PID,
              "pitcher x 월": PID * 100 + MON}
    #  (원천 시즌들) -> 목표 시즌. 마지막 판이 실제 제출 구조(전 학습구간 -> 다음 시즌)와 같다
    RUNS = [((2022,), 2023), ((2023,), 2024), ((2022, 2023), 2024)]

    out = {}
    print(f"{'원천->목표':<16}{'그룹':<18}{'k':>8}{'rho^2':>10}{'vs 기준':>10}"
          f"{'셀 겹침':>10}")
    for src, tgt in RUNS:
        m_src = np.isin(season, src)
        m_tgt = season == tgt
        res = np.concatenate([y[season == s] - pv[s] for s in src])
        base = r2(pv[tgt], y[m_tgt])
        tag = f"{'+'.join(str(s) for s in src)}->{tgt}"
        for name, keyall in GROUPS.items():
            ks, kt = keyall[m_src], keyall[m_tgt]
            cover = float(np.isin(kt, np.unique(ks)).mean())
            row, bestk, bestv = {}, None, -1e9
            for k in KS:
                u, tab, _ = group_mean(ks, res, k)
                v = r2(pv[tgt] + apply_table(u, tab, kt), y[m_tgt])
                row[k] = v
                if v > bestv:
                    bestk, bestv = k, v
                print(f"{tag:<16}{name:<18}{k:>8}{v:>10.1f}{v-base:>+10.1f}"
                      f"{cover:>10.1%}")
            out[f"{tag}|{name}"] = dict(cover=cover, base=base, best_k=bestk,
                                        best=bestv, gain=bestv - base,
                                        by_k={str(k): row[k] for k in KS})
        print("", flush=True)
    out["_base"] = {str(f): float(r2(pv[f], y[season == f]))
                    for f in (2022, 2023, 2024)}
    json.dump(out, io.open(os.path.join(ROOT, "exp", "resid_table.json"), "w",
                           encoding="utf-8"), indent=1, ensure_ascii=False)
    print("\n판정 — 어느 k 에서도 2024 가 기준선을 못 넘으면 그 정보는 "
          "시즌을 건너 전이되지 않는다. 오라클이 큰데 여기서 0 이면 "
          "'합법으로 닿지 않는 자리'로 확정된다.")


if __name__ == "__main__":
    main()
