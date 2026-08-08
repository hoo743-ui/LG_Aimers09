r"""제출 전 실제 규모 점검 — 6-2 점검 목록 3단계.

로컬 test.csv 는 5행뿐이라 이것만으로는 부족하다. 10분 추론 제한에 걸리는지,
확률이 범위를 벗어나는지, 결측이 생기는지 알 수 없다. 2024 시즌(25만 행)을
평가셋 대역으로 써서 실제 규모로 확인한다.

**script.py 의 함수를 그대로 불러 쓴다.** 여기서 따로 구현하면 제출 경로와
어긋나도 모른다 — 그게 이 점검의 존재 이유다.

정답을 알고 있으므로 점수도 같이 낸다. 다만 이 값은 참고용이다. 실제 pkl 은
2024 를 포함한 전체로 재학습됐으므로 여기서는 학습 데이터를 채점하는 셈이라
부풀려진다. **점수가 아니라 동작을 보는 도구다.**

    .\.venv\Scripts\python.exe check_submit.py
"""
import os
import time

import joblib
import numpy as np
import pandas as pd

import script as sub          # 제출 경로 그대로 쓴다

DATA_DIR = "./data"
MODEL_PATH = "./model/rf.pkl"
BAND_SEASON = 2024
EVAL_ROWS = 245_789           # 실제 평가셋 행 수 (6-5)
TIME_LIMIT = 600              # 초


def main():
    print("모델 로드 ...")
    bundle = joblib.load(MODEL_PATH)
    feats = bundle.get("features", []) if isinstance(bundle, dict) else []
    print(f"  앙상블 {len(bundle['models'])}개 | 피처 {len(feats)}개 | "
          f"alpha={bundle.get('alpha')} center={bundle.get('center'):.4f}")
    print(f"  spec={bundle.get('spec')}")
    print(f"  shift={bundle.get('shift')} detrend={bundle.get('detrend')}")
    if bundle.get("shift") is not None:
        raise SystemExit("🚫 shift 가 켜져 있다 — 규칙 위반 (4-3). 제출 불가")

    # ---- 평가셋 대역 만들기 ----
    # test.csv 의 컬럼 구조를 그대로 쓴다. 정답 컬럼은 채점용으로 따로 뺀다.
    test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig",
                            nrows=0).columns.tolist()
    df = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                     usecols=test_cols + ["control_success"])
    band = df[df["season"] == BAND_SEASON].reset_index(drop=True)
    y = band["control_success"].to_numpy()
    band = band[test_cols]
    print(f"\n평가셋 대역: {BAND_SEASON} 시즌 {len(band):,} 행 "
          f"(실제 평가 {EVAL_ROWS:,} 행)")

    # ---- 제출 경로 그대로 추론 ----
    t = time.time()
    X = sub.build_features(sub.add_derived(band), bundle)
    p = sub.predict_proba(bundle, X)
    elapsed = time.time() - t
    scaled = elapsed * EVAL_ROWS / len(band)

    # ---- 점검 ----
    n_nan = int(np.isnan(p).sum())
    n_bad = int(((p < 0) | (p > 1)).sum())
    r = y.mean()
    score = max(0.0, 100000 * (1 - ((p - y) ** 2).mean() / (r * (1 - r))))

    print(f"\n{'항목':22s}{'측정':>22s}   {'제한':>10s}")
    print("-" * 58)
    print(f"{'추론 시간':22s}{f'{elapsed:.1f}초':>22s}   {'':>10s}")
    print(f"{'  실제 규모 환산':22s}{f'{scaled:.1f}초':>22s}   {f'{TIME_LIMIT}초':>10s}"
          f"   {'OK' if scaled < TIME_LIMIT else '★초과'}")
    print(f"{'입력 피처 수':22s}{X.shape[1]:>22d}   {len(feats):>10d}"
          f"   {'OK' if X.shape[1] == len(feats) else '★불일치'}")
    print(f"{'확률 범위':22s}{f'{p.min():.4f}~{p.max():.4f}':>22s}   {'0~1':>10s}"
          f"   {'OK' if n_bad == 0 else '★이탈'}")
    print(f"{'결측':22s}{n_nan:>22d}   {0:>10d}"
          f"   {'OK' if n_nan == 0 else '★있음'}")
    print(f"{'예측 평균':22s}{p.mean():>22.4f}   {'':>10s}")
    print(f"{'중심 편차':22s}{p.mean() - r:>+22.4f}   {'':>10s}"
          f"   (실제 {r:.4f})")
    print(f"\n참고 점수 {score:.2f} — 이 대역은 pkl 학습에 포함돼 있어 부풀려진다. "
          f"동작 확인용이지 성능 지표가 아니다.")

    # ---- 파생 컬럼이 실제로 쓰이는지 ----
    derived = [c for c in feats if c not in test_cols]
    print(f"\n파생 컬럼 {derived}")
    for c in derived:
        v = X[c]
        print(f"  {c}: 고유값 {sorted(v.unique().tolist())[:6]} | "
              f"결측 {int(v.isna().sum())}")


if __name__ == "__main__":
    main()
