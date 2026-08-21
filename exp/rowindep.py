r"""규정 4 수치 검사 — **zip 안의 script.py 와 pkl 을 그대로** 써서
`test.csv` 에 그 행 하나만 있을 때와 전체가 있을 때의 예측이 같은지 실측한다.

공지의 판정 기준 그 자체다. 제출 전 예외 없이 통과시킨다.

    .\.venv\Scripts\python.exe -u exp\rowindep.py --zip submissions\cand_rob1.zip
"""
import argparse
import importlib.util
import os
import shutil
import sys
import tempfile
import zipfile

import joblib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "exp"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="rowindep_")
    try:
        with zipfile.ZipFile(a.zip) as z:
            z.extract("script.py", tmp)
            z.extract("model/rf.pkl", tmp)
        spec = importlib.util.spec_from_file_location(
            "sc_zip", os.path.join(tmp, "script.py"))
        sc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sc)
        b = joblib.load(os.path.join(tmp, "model", "rf.pkl"))

        from path_alloc import build_df
        tr = build_df()
        X = tr.loc[tr["season"].to_numpy() == 2024].iloc[:a.n].copy()

        batch = np.asarray(sc.platoon_adjust(b, X), float)
        solo = np.array([float(sc.platoon_adjust(b, X.iloc[[i]])[0])
                         for i in range(len(X))])
        err = float(np.max(np.abs(batch - solo)))
        print(f"{os.path.basename(a.zip)}  {a.n}행 배치 vs 1행 단독 "
              f"— 최대 절대차 {err:.3e}")
        print("  ✅ 통과 (행 하나만 있어도 같은 값)" if err == 0.0
              else "  ❌ 실패 — 다른 행이 추론에 영향을 준다. 제출 금지")
        sys.exit(0 if err == 0.0 else 1)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
