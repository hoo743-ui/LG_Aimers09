r"""피처 행렬을 한 번만 만들어 float32 npy 로 캐시한다.

왜 필요한가 — 기존 실험 스크립트(`ctx_feat.py` 등)는 매 실행마다 train.csv
368MB 와 trackman_history.csv 354MB 를 다시 파싱한다. 실험 한 번에 1~2분이
파싱으로 날아가고, 이 PC 는 가용 메모리가 7GB 뿐이라 병렬 실행도 막힌다.

한 번 만들어 두면
  - 파싱 0초 (mmap 으로 읽는다)
  - float32 라 메모리 절반 (1.47M x 55 = 323MB)
  - 워커 여러 개가 같은 파일을 mmap 해 공유

피처 정의는 `final_train.py` 를 그대로 import 해서 쓴다. 실험과 제출물이
서로 다른 피처를 쓰면 측정이 의미를 잃기 때문에 절대 복붙하지 않는다.

    .\.venv\Scripts\python.exe exp\prep.py
"""
import importlib.util
import json
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "exp", "cache")
DATA = os.path.join(ROOT, "data")
TARGET = "control_success"


def load_final_train():
    """final_train.py 를 모듈로 읽는다 (era_screen.py 와 같은 방식)."""
    spec = importlib.util.spec_from_file_location(
        "ft", os.path.join(ROOT, "final_train.py"))
    ft = importlib.util.module_from_spec(spec)
    cwd = os.getcwd()
    os.chdir(ROOT)          # final_train 의 상대경로("./data") 를 맞춘다
    try:
        spec.loader.exec_module(ft)
    finally:
        os.chdir(cwd)
    return ft


def main():
    os.makedirs(CACHE, exist_ok=True)
    ft = load_final_train()
    os.chdir(ROOT)

    test_cols = pd.read_csv(f"{DATA}/test.csv", encoding="utf-8-sig",
                            nrows=0).columns
    all_features = [c for c in test_cols if c != "row_id"]
    print(f"기본 피처 {len(all_features)}개")

    train = pd.read_csv(f"{DATA}/train.csv", encoding="utf-8-sig",
                        usecols=all_features + [TARGET])
    print(f"train {train.shape}")

    # 상황 조건부 Trackman 피처 (4-9). 시점 규칙은 final_train 이 정의한다.
    train = ft.attach_ctx_train(train, ft.load_trackman())
    ctx_feats = ft.COUNT_FEATS + ft.HAND_FEATS          # tmc_n/tmh_n 포함
    print(f"상황 피처 {len(ctx_feats)}개 부착 (시계 tmc_n/tmh_n 포함 — "
          f"실험에서 켜고 끌 수 있게 캐시에는 담는다)")

    # 범주형 3개는 여기서 정수로 굳힌다. 라벨 대응일 뿐 타깃을 보지 않으므로
    # 전체에 대해 fit 해도 누설이 아니다. 제출 파이프라인의
    # ColumnTransformer(OrdinalEncoder) 와 같은 결과를 낸다.
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    cat_vals = enc.fit_transform(train[ft.CAT_COLS].astype(object))

    num_cols = [c for c in all_features if c not in ft.CAT_COLS] + ctx_feats
    cols = list(ft.CAT_COLS) + num_cols

    n = len(train)
    X = np.empty((n, len(cols)), dtype=np.float32)
    X[:, :len(ft.CAT_COLS)] = cat_vals.astype(np.float32)
    for j, c in enumerate(num_cols, start=len(ft.CAT_COLS)):
        X[:, j] = pd.to_numeric(train[c], errors="coerce").to_numpy(np.float32)

    y = train[TARGET].to_numpy(np.int8)
    season = train["season"].to_numpy(np.int16)

    np.save(f"{CACHE}/X.npy", X)
    np.save(f"{CACHE}/y.npy", y)
    np.save(f"{CACHE}/season.npy", season)
    with open(f"{CACHE}/cols.json", "w") as f:
        json.dump({"cols": cols, "cat": list(ft.CAT_COLS),
                   "ctx": ctx_feats,
                   "n_base": len(all_features)}, f, indent=1)

    print(f"\n저장: X {X.shape} {X.nbytes/1e6:.0f}MB  →  {CACHE}")
    for S in sorted(set(season.tolist())):
        m = season == S
        print(f"  {S}  n={m.sum():>9,}  성공률 {y[m].mean():.4f}")


if __name__ == "__main__":
    main()
