r"""경로 배분 재최적화 — 후처리 편차 4축과 in-model 피처의 중복을 실측한다.

## 왜 이 실험인가

같은 정보(카운트)가 경로에 따라 정반대 결과를 냈다.

    카운트 in-model  (K2, 26회차)          실제 LB  -4.72
    카운트 후처리 편차 (17/19회차)          실제 LB  +2.41, +1.48

26회차 원장이 원인을 적어놨다 — K2 의 2스트라이크 국면이 편차 2번축
(플래툰x투수유리x카운트)과 겹치고, 투수유리 = `strikes>balls` 라 후처리가
이미 그 국면을 다룬다. **편차 4축은 D 이전 모델 위에서 튜닝된 뒤 고정됐고,
X/H1 은 그것을 고정한 채 얹혔다. 둘의 상호작용은 측정된 적이 없다.**

## 왜 학습 1회로 되는가

편차 축을 빼는 것은 **모델을 바꾸지 않는다.** 예측에 더하는 보정항의 가중을
0 으로 두는 것뿐이다. 그래서 축별 보정을 따로 구해 두면 학습 한 번으로
Champion + 단일제거 4가지를 전부 잰다. K2 를 넣는 판만 학습이 더 필요하다.

## 판정

생산 경로 2024 홀드아웃의 `rho^2` (아핀 이전, 원장 `local_2024` 와 같은 정의).
캐시 경로 결과는 채택 근거로 쓰지 않는다 — K2 에서 캐시 +11.5 / 생산 -8.2 /
실제 LB -4.72 로 생산 쪽이 맞았다.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "exp"))
import build_asof as ba                                    # noqa: E402

ft, sc = ba.ft, ba.sc
CACHE_DF = os.path.join(ROOT, "exp", "cache", "prod_df.pkl")
AXNAME = ["B dev_platoon   (투수x타자손)",
          "C dev_plat_cnt  (플래툰x투수유리)",
          "A dev_count     (플래툰x투수유리x카운트)",
          "D dev_runner    (플래툰x주자유무)"]


def build_df():
    """build_asof.main() 의 구축부와 **같은 순서**로 만든다. K2 까지 포함한 상위집합."""
    if os.path.exists(CACHE_DF):
        print("  생산 데이터프레임 캐시 사용", flush=True)
        return pd.read_pickle(CACHE_DF)
    t0 = time.time()
    tc = pd.read_csv(os.path.join(ft.DATA_DIR, "test.csv"),
                     encoding="utf-8-sig", nrows=0).columns
    allf = [c for c in tc if c != ft.ID]
    tr = pd.read_csv(os.path.join(ft.DATA_DIR, "train.csv"),
                     encoding="utf-8-sig", usecols=allf + [ft.TARGET])
    tm = ft.load_trackman()
    tr = ft.attach_ctx_train(tr, tm)
    acols = sc.ASOF_COLS + sc.CTX_COLS + sc.LVL_COLS + sc.K2_COLS
    season = tr["season"].to_numpy()
    for c in acols:
        tr[c] = np.nan
    for g in sorted(np.unique(season)):
        m = season == g
        part = ba.add_state(tr.loc[m].copy(), ba.prior_tables(tr, season < g))
        for c in acols:
            tr.loc[m, c] = part[c].to_numpy()
        print(f"    {g} 상태 분해 완료 {time.time()-t0:.0f}s", flush=True)
    for c in tr.columns:                       # 수치 컬럼만 내린다 (범주형은 문자열)
        if tr[c].dtype == np.float64:
            tr[c] = tr[c].astype(np.float32)
    tr.to_pickle(CACHE_DF)
    print(f"  구축 {time.time()-t0:.0f}s -> 캐시 저장", flush=True)
    return tr


def main():
    tr = build_df()
    tc = pd.read_csv(os.path.join(ft.DATA_DIR, "test.csv"),
                     encoding="utf-8-sig", nrows=0).columns
    allf = [c for c in tc if c != ft.ID]
    ctxf = [c for c in ft.COUNT_FEATS + ft.HAND_FEATS if c not in ("tmc_n", "tmh_n")]
    CHAMP = list(allf) + ctxf + sc.ASOF_COLS + sc.CTX_COLS + sc.LVL_COLS
    WITHK2 = CHAMP + sc.K2_COLS
    y = tr[ft.TARGET].to_numpy(np.float64)
    season = tr["season"].to_numpy()
    m_tr, m_va = season < 2024, season == 2024
    yv = y[m_va]

    P = tr["pitcher_id"].to_numpy(np.int64)
    BH = tr["batter_hand"].to_numpy(np.int64)
    BB = tr["balls_before"].to_numpy(np.int64)
    SS = tr["strikes_before"].to_numpy(np.int64)
    OB = (tr["num_runners_on"].to_numpy(np.int64) > 0).astype(np.int64)
    PH, PHA = P * 10 + BH, (P * 10 + BH) * 10 + (SS > BB).astype(np.int64)
    AX = [(P, PH), (PH, PHA), (PHA, PHA * 100 + (BB * 4 + SS)), (PH, PH * 10 + OB)]
    posts = np.column_stack([                       # 축별 보정 (n_va, 4)
        ba.look(*ba.nested_dev(p[m_tr], c[m_tr], y[m_tr], k), c[m_va])
        for (p, c), k in zip(AX, ba.KSH)])
    W = ba.WPOST
    r2 = lambda p: 1e5 * np.corrcoef(p, yv)[0, 1] ** 2

    preds = {}
    for lbl, fs in (("champ", CHAMP), ("k2", WITHK2)):
        t = time.time()
        mm = ba.pipeline(fs, 42)
        mm.fit(tr.loc[m_tr, fs], y[m_tr].astype(int))
        preds[lbl] = mm.predict_proba(tr.loc[m_va, fs])[:, 1]
        print(f"  {lbl} {len(fs)}p 학습 {time.time()-t:.0f}s", flush=True)
        del mm

    base = r2(preds["champ"] + posts @ W)
    print(f"\n=== 1단계: 편차 축 단일 제거 (모델 동일, 후처리만 변경) ===")
    print(f"{'구성':<40}{'2024':>10}{'vs champ':>11}")
    print(f"{'Champion (편차 4축 전부)':<40}{base:>10.1f}{0.0:>+11.1f}")
    order = [(0, "B"), (1, "C"), (2, "A"), (3, "D")]
    single = {}
    for i, tag in order:
        w = W.copy(); w[i] = 0.0
        v = r2(preds["champ"] + posts @ w)
        single[tag] = v - base
        print(f"{'  - ' + AXNAME[i]:<40}{v:>10.1f}{v-base:>+11.1f}")
    v0 = r2(preds["champ"])
    print(f"{'  - 편차 전부 제거':<40}{v0:>10.1f}{v0-base:>+11.1f}")

    print(f"\n=== 2단계: K2 in-model 복귀 x 편차 축 제거 ===")
    print(f"{'구성':<40}{'2024':>10}{'vs champ':>11}")
    kk = r2(preds["k2"] + posts @ W)
    print(f"{'champ + K2 (편차 그대로)':<40}{kk:>10.1f}{kk-base:>+11.1f}")
    for i, tag in order:
        w = W.copy(); w[i] = 0.0
        v = r2(preds["k2"] + posts @ w)
        print(f"{'  K2, - ' + AXNAME[i]:<40}{v:>10.1f}{v-base:>+11.1f}")
    vk = r2(preds["k2"])
    print(f"{'  K2, 편차 전부 제거':<40}{vk:>10.1f}{vk-base:>+11.1f}")
    print(f"\n참고 — 원장 기준선: champ 944.0 / champ+K2 935.8 (실제 LB -4.72)")


if __name__ == "__main__":
    main()
