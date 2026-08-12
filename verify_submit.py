r"""6-2 제출 전 점검 목록을 자동으로 돌린다.

1·2회차 제출 실패가 각각 이 목록의 첫 항목과 마지막 항목이었다. 손으로 하면
빠뜨리므로 스크립트로 고정한다.

  1. zip 을 **빈 폴더**에 풀어 model/ 이 폴더로 생기는지 (6-1)
  2. 그 폴더에서 script.py 실행 → output/submission.csv, 확률 0~1
  3. 실제 규모(약 24.6만 행)로 추론 시간 측정 — 제한 600초
  4. 낮은 numpy 환경에서 pkl 로드 (6-4) — 이건 별도 venv 가 필요해 여기서는
     존재하면 돌리고 없으면 안내만 한다

3번은 평가셋 대역을 흉내내야 한다. 로컬 test.csv 는 5행뿐이라 시간도 형식도
확인이 안 된다. train.csv 의 2024 시즌을 잘라 test 스키마로 만들어 쓴다.
**정답 컬럼은 반드시 뺀다** — 남으면 script.py 가 그걸 피처로 먹어서 실제 평가와
다른 상황을 검증하게 된다.

사용법:
    .\.venv\Scripts\python.exe verify_submit.py
    .\.venv\Scripts\python.exe verify_submit.py --zip submissions\후보.zip
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

import pandas as pd

ZIP = os.environ.get("VERIFY_ZIP", "submit.zip")   # 후보 검증용 (기본은 제출본)
DATA_DIR = "./data"
TARGET = "control_success"
ID = "row_id"
TIME_LIMIT = 600
OLDNP = "./oldnp/Scripts/python.exe"


def fail(msg):
    print(f"\n[실패] {msg}")
    sys.exit(1)


def step1_extract(work):
    print("=== 1. zip 구조 ===")
    with zipfile.ZipFile(ZIP) as zf:
        names = zf.namelist()
        bad = [n for n in names if "\\" in n]
        if bad:
            fail(f"역슬래시 경로: {bad}  (6-1)")
        zf.extractall(work)
    print(f"  엔트리: {names}")

    mdir = os.path.join(work, "model")
    if not os.path.isdir(mdir):
        fail("model/ 이 폴더로 풀리지 않았다 — 6-1 의 그 사고다")
    pkl = os.path.join(mdir, "rf.pkl")
    if not os.path.isfile(pkl):
        fail("model/rf.pkl 없음")
    print(f"  model/ 폴더 OK, rf.pkl {os.path.getsize(pkl)/1e6:.1f} MB")

    extra = [n for n in names if n.startswith("model/") and n != "model/rf.pkl"]
    if extra:
        print(f"  경고: model/ 에 rf.pkl 말고 다른 게 들어 있다 → {extra}")
        print("        백업 pkl 이 딸려 들어가면 용량만 늘고 득이 없다 (6-4)")

    req = open(os.path.join(work, "requirements.txt"), "rb").read()
    try:
        req.decode("ascii")
    except UnicodeDecodeError:
        fail("requirements.txt 에 비ASCII 문자 — 환경에 따라 실행 실패 (6-4)")
    print(f"  requirements.txt:\n    " + req.decode().strip().replace("\n", "\n    "))
    if b"numpy" not in req:
        fail("requirements.txt 에 numpy 핀이 없다 — 2회차가 이걸로 죽었다 (6-4)")
    return work


def step2_smoke(work):
    print("\n=== 2. 격리 실행 (로컬 test.csv) ===")
    shutil.copytree(DATA_DIR, os.path.join(work, "data"),
                    ignore=shutil.ignore_patterns("train.csv",
                                                  "trackman_history.csv"))
    run(work, "로컬 5행")
    out = os.path.join(work, "output", "submission.csv")
    sub = pd.read_csv(out)
    check_probs(sub)
    print(f"  submission.csv {len(sub)}행 OK")


def step3_scale(work):
    print("\n=== 3. 실제 규모 추론 ===")
    test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig",
                            nrows=0).columns.tolist()
    need = [c for c in test_cols if c != ID]
    big = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                      usecols=need + ["season"])
    big = big[big["season"] == 2024].copy()
    big.insert(0, ID, range(1, len(big) + 1))
    big = big[test_cols]
    if TARGET in big.columns:
        fail("정답 컬럼이 test 스키마에 남아 있다")

    dst = os.path.join(work, "data")
    big.to_csv(os.path.join(dst, "test.csv"), index=False, encoding="utf-8")
    pd.DataFrame({ID: big[ID], TARGET: 0.5}).to_csv(
        os.path.join(dst, "sample_submission.csv"), index=False, encoding="utf-8")
    print(f"  평가셋 대역 {len(big):,}행 생성 (2024 시즌, 정답 제거)")

    t = time.time()
    run(work, "실제 규모")
    el = time.time() - t
    sub = pd.read_csv(os.path.join(work, "output", "submission.csv"))
    check_probs(sub)
    p = sub[TARGET]
    print(f"  추론 {el:.1f}초 / 제한 {TIME_LIMIT}초  ({el/TIME_LIMIT:.1%})")
    print(f"  확률 범위 {p.min():.4f}~{p.max():.4f}  평균 {p.mean():.4f}")
    if el > TIME_LIMIT:
        fail(f"시간 초과: {el:.0f}초")
    if p.std() < 1e-6:
        fail("예측이 상수다 — 모델이 제대로 안 붙었다")


def step4_oldnp(work):
    print("\n=== 4. 낮은 numpy 환경에서 pkl 로드 (6-4) ===")
    if not os.path.exists(OLDNP):
        print(f"  건너뜀 — {OLDNP} 없음. 만들려면:")
        print("    python -m venv oldnp")
        print("    .\\oldnp\\Scripts\\python.exe -m pip install "
              "scikit-learn==1.7.2 joblib==1.5.3 pandas==2.3.3 numpy==1.26.4")
        return False
    pkl = os.path.join(work, "model", "rf.pkl").replace("\\", "/")
    r = subprocess.run([OLDNP, "-c",
                        f"import joblib; b=joblib.load('{pkl}'); "
                        f"print('loaded', type(b).__name__)"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        fail(f"낮은 numpy 에서 로드 실패 — 2회차와 같은 사고:\n{r.stderr[-1200:]}")
    print(f"  {r.stdout.strip()}  → 통과")
    return True


def run(work, tag):
    r = subprocess.run([os.path.abspath(sys.executable), "script.py"],
                       cwd=work, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        fail(f"{tag} script.py 실행 실패:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    for line in r.stdout.strip().splitlines():
        print(f"    | {line}")


def check_probs(sub):
    if list(sub.columns[:2]) != [ID, TARGET]:
        fail(f"컬럼이 다르다: {list(sub.columns)}")
    p = sub[TARGET]
    if p.isna().any():
        fail(f"결측 {int(p.isna().sum())}건")
    if not ((p >= 0) & (p <= 1)).all():
        fail("확률이 0~1 범위를 벗어났다")


def main():
    # 12회차에 --zip 을 줬는데 argparse 가 없어 조용히 무시됐다. 기본값 submit.zip
    # (6회차 RF 판)이 검증되고 "통과"가 찍혔다 — 엉뚱한 파일에 도장을 찍는 사고다.
    # 이제 플래그를 받고, 모르는 인자는 오류로 낸다.
    global ZIP
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=ZIP, help="검증할 zip (기본: VERIFY_ZIP 또는 submit.zip)")
    ZIP = ap.parse_args().zip

    if not os.path.exists(ZIP):
        fail(f"{ZIP} 없음 — make_submit.py 를 먼저 돌릴 것")
    work = tempfile.mkdtemp(prefix="submit_check_")
    print(f"작업 폴더: {work}\n")
    try:
        step1_extract(work)
        step2_smoke(work)
        step3_scale(work)
        did4 = step4_oldnp(work)
        print("\n" + "=" * 46)
        print("점검 통과 — 1,2,3" + (",4" if did4 else " (4번 미실행)"))
        if not did4:
            print("4번은 2회차를 죽인 항목이다. 반드시 따로 돌릴 것.")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
