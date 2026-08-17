r"""정보 한계 vs 적합 한계 — 새 피처 0개, Champion 82p 고정, 제출 없음.

## 무엇을 가르려는가

    A  train 높음 / OOS 낮음        -> 과적합
    B  train·OOS 둘 다 낮음          -> 현 표현에서의 **정보 한계**
    C  train·OOS 둘 다 높고 OOS 가   -> **적합이 병목** (이때만 새 모델 탐색)
       현재보다 훨씬 좋음

## 설계 — 학습을 늘리지 않고 세 곡선을 얻는다

CatBoost 는 `ntree_end` 로 **같은 모델의 부분 앙상블**을 평가할 수 있다. 그래서
적합 1회에서 반복수별 train/OOS 곡선이 나온다. 과소적합이면 둘 다 아직 오르고,
과적합이면 train 만 오르고 OOS 가 꺾인다. **하이퍼파라미터 재탐색이 아니다** —
현행 설정(iterations 1200, depth 6, lr 0.02, l2 100)의 궤적을 읽는 것뿐이다.

표본 크기 곡선은 학습 행을 비율로 줄여 잰다. OOS 가 표본에 따라 계속 오르면
정보가 아직 남은 것이고, 평평하면 표본이 아니라 **정보**가 한계다.

## 규약

`rho^2 = 1e5 * corr(pred, y)^2`. train 쪽은 **모델 단독**으로만 잰다 (후처리
편차는 학습 행으로 만들어 학습 행에 다시 얹으면 in-sample 이라 의미가 없다).
OOS 는 모델 단독과 후처리 포함을 둘 다 낸다 — 후자가 원장 기준선(944.0)이다.

    .\.venv\Scripts\python.exe -u exp\fit_limit.py
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

NTREES = [100, 200, 400, 600, 800, 1000, 1200]
FRACS = [0.125, 0.25, 0.5, 1.0]
TRAIN_EVAL_N = 200_000          # train 쪽 평가 표본 (예측 비용 절약)
SEED = 42


def r2(p, y):
    return 1e5 * np.corrcoef(p, y)[0, 1] ** 2


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
    rng = np.random.default_rng(0)
    out = {}

    print("=" * 88)
    print("1. train / OOS 격차 (3폴드, 시드 42, 모델 단독)")
    print("=" * 88)
    print(f"{'폴드':<7}{'학습행':>11}{'train rho^2':>13}{'OOS rho^2':>11}"
          f"{'격차':>9}{'OOS+후처리':>11}{'예측 sd(tr/oos)':>18}")
    models = {}
    for f in (2022, 2023, 2024):
        m_tr, m_va = season < f, season == f
        ix = np.flatnonzero(m_tr)
        ev = rng.choice(ix, min(TRAIN_EVAL_N, len(ix)), replace=False)
        t0 = time.time()
        mm = ba.pipeline(CHAMP, SEED)
        mm.fit(tr.loc[m_tr, CHAMP], y[m_tr].astype(int))
        p_tr = mm.predict_proba(tr.iloc[ev][CHAMP])[:, 1]
        p_va = mm.predict_proba(tr.loc[m_va, CHAMP])[:, 1]
        post = post_for(tr, y, m_tr, m_va)
        a, b = r2(p_tr, y[ev]), r2(p_va, y[m_va])
        out[f"fold{f}"] = dict(train=a, oos=b, gap=a - b,
                               oos_post=r2(p_va + post, y[m_va]),
                               sd_train=float(p_tr.std()),
                               sd_oos=float(p_va.std()),
                               n_train=int(m_tr.sum()),
                               resid_corr=float(np.corrcoef(
                                   p_va, y[m_va] - p_va)[0, 1]))
        print(f"{f:<7}{int(m_tr.sum()):>11,}{a:>13.1f}{b:>11.1f}"
              f"{a-b:>+9.1f}{out[f'fold{f}']['oos_post']:>11.1f}"
              f"{p_tr.std():>10.4f}/{p_va.std():.4f}"
              f"   ({time.time()-t0:.0f}s)", flush=True)
        if f == 2024:
            models[f] = (mm, ev, m_tr, m_va, post)
        else:
            del mm

    print()
    print("=" * 88)
    print("2. 반복수 곡선 (폴드 2024, 같은 적합의 부분 앙상블)")
    print("=" * 88)
    mm, ev, m_tr, m_va, post = models[2024]
    Xtr, Xva = tr.iloc[ev][CHAMP], tr.loc[m_va, CHAMP]
    clf = mm.named_steps["clf"]
    pre = mm.named_steps["pre"]
    Ztr, Zva = pre.transform(Xtr), pre.transform(Xva)
    print(f"{'나무 수':>8}{'train rho^2':>13}{'OOS rho^2':>11}{'격차':>9}"
          f"{'OOS+후처리':>11}")
    curve = {}
    for k in NTREES:
        a = r2(clf.predict_proba(Ztr, ntree_end=k)[:, 1], y[ev])
        pv = clf.predict_proba(Zva, ntree_end=k)[:, 1]
        b = r2(pv, y[m_va])
        curve[k] = dict(train=a, oos=b, oos_post=r2(pv + post, y[m_va]))
        print(f"{k:>8}{a:>13.1f}{b:>11.1f}{a-b:>+9.1f}"
              f"{curve[k]['oos_post']:>11.1f}", flush=True)
    out["iter_curve"] = curve
    del mm, models

    print()
    print("=" * 88)
    print("3. 표본 크기 곡선 (폴드 2024, 학습 행을 비율로 줄임)")
    print("=" * 88)
    print(f"{'비율':>7}{'학습행':>11}{'train rho^2':>13}{'OOS rho^2':>11}"
          f"{'격차':>9}{'OOS+후처리':>11}")
    ixall = np.flatnonzero(m_tr)
    lc = {}
    for fr in FRACS:
        sub = rng.choice(ixall, int(len(ixall) * fr), replace=False)
        sm = np.zeros(len(tr), bool)
        sm[sub] = True
        t0 = time.time()
        mm = ba.pipeline(CHAMP, SEED)
        mm.fit(tr.loc[sm, CHAMP], y[sm].astype(int))
        ev2 = rng.choice(sub, min(TRAIN_EVAL_N, len(sub)), replace=False)
        a = r2(mm.predict_proba(tr.iloc[ev2][CHAMP])[:, 1], y[ev2])
        pv = mm.predict_proba(tr.loc[m_va, CHAMP])[:, 1]
        b = r2(pv, y[m_va])
        lc[fr] = dict(n=int(sm.sum()), train=a, oos=b,
                      oos_post=r2(pv + post, y[m_va]))
        print(f"{fr:>7.3f}{int(sm.sum()):>11,}{a:>13.1f}{b:>11.1f}"
              f"{a-b:>+9.1f}{lc[fr]['oos_post']:>11.1f}"
              f"   ({time.time()-t0:.0f}s)", flush=True)
        del mm
    out["size_curve"] = {str(k): v for k, v in lc.items()}

    json.dump(out, io.open(os.path.join(ROOT, "exp", "fit_limit.json"), "w",
                           encoding="utf-8"), indent=1, ensure_ascii=False)
    print("\n판정 — train 이 OOS 보다 크게 높으면 A(과적합), 둘 다 낮고 표본 곡선이 "
          "평평하면 B(정보 한계), 반복수 곡선에서 OOS 가 아직 오르는 중이면 C(적합 한계).")


if __name__ == "__main__":
    main()
