r"""관문 2 — 제출 zip 의 **생산 추론 경로 그대로** 단독 행 vs 배치를 비교한다.

데이콘 공지(2026-08-18)의 판정 기준을 그대로 재현한다.

> test.csv 에 해당 행 1개만 있는 경우와 전체 평가 데이터가 함께 있는 경우의
> 예측값이 **같아야** 한다.

저장소의 `script.py` 가 아니라 **zip 안의 script.py 와 rf.pkl** 을 쓴다.
제출되는 것이 그 둘이기 때문이다.

    .\.venv\Scripts\python.exe -u exp\rowindep_zip.py --zip submissions\cand_kbf.zip
"""
import argparse
import importlib.util
import io
import os
import sys
import tempfile
import zipfile

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_COLS = None

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def load_zip(path):
    z = zipfile.ZipFile(path)
    d = tempfile.mkdtemp()
    sp = os.path.join(d, "script.py")
    with open(sp, "wb") as f:
        f.write(z.read("script.py"))
    spec = importlib.util.spec_from_file_location("zsc", sp)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    b = joblib.load(io.BytesIO(z.read("model/rf.pkl")))
    return m, b


def infer(sc, b, df):
    X = df.copy()
    X = sc.attach_asof_state(X, b)
    if hasattr(sc, "attach_ctx"):
        X = sc.attach_ctx(X, b)
    if hasattr(sc, "attach_aux"):
        X = sc.attach_aux(X, b)
    F = sc.build_features(X, b)
    return np.asarray(sc.predict_proba(b, F), float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--rows", type=int, default=200)
    a = ap.parse_args()

    cols = list(pd.read_csv(os.path.join(ROOT, "data", "test.csv"),
                            encoding="utf-8-sig").columns)
    # 2024 행만 쓴다. 표 키가 2023+24 라 옛 시즌 행으로 재면 바뀐 항목을
    # 하나도 건드리지 못해 검사가 무의미해진다.
    n_all = sum(1 for _ in open(os.path.join(ROOT, "data", "train.csv"),
                                encoding="utf-8-sig")) - 1
    tr = pd.read_csv(os.path.join(ROOT, "data", "train.csv"),
                     skiprows=range(1, n_all - 253507 + 1), encoding="utf-8-sig")
    assert (tr.season == 2024).all(), tr.season.value_counts().to_dict()
    step = max(1, len(tr) // a.rows)
    df = tr.iloc[::step][:a.rows][[c for c in cols if c in tr.columns]]            .reset_index(drop=True)
    print(f"{os.path.basename(a.zip)}  검사행 {len(df)}  컬럼 {len(df.columns)}")

    sc, b = load_zip(a.zip)
    batch = infer(sc, b, df)
    solo = np.array([infer(sc, b, df.iloc[[i]])[0] for i in range(len(df))])
    mx = float(np.max(np.abs(batch - solo)))
    print(f"  배치 예측 범위 {batch.min():.6f} ~ {batch.max():.6f}")
    print(f"  단독 vs 배치 최대 절대오차  {mx:.3e}")
    print("  ✅ 통과 — 행 하나만 있어도 같은 값이다 (규정 4)" if mx == 0.0
          else f"  ❌ 실패 — 다른 행이 추론에 영향을 준다")
    return 0 if mx == 0.0 else 1


if __name__ == "__main__":
    sys.exit(main())
