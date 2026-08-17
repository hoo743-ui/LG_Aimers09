r"""후처리 7축의 가중만 바꿔 제출 후보를 즉시 만든다. 학습 0회 · Champion 불변.

## 왜 이 도구인가

대회가 **최고점 채점 · private 분할 없음**이다. 그러면 일반화 격차가 존재하지
않고, LB 가 곧 목적함수다. 같은 제출은 같은 점수를 내므로 표본 잡음도 0 이다.
즉 이건 통계 추정 문제가 아니라 **결정론적 함수의 최적화 문제**이고, 예산은
남은 제출 횟수다.

그래서 필요한 것은 "한 번 잘 고르는 판단"이 아니라 **결과를 보고 몇 초 만에
다음 점을 만드는 회전 속도**다.

## 좌표

Champion 번들 `cat_submit_3.pkl` 의 `platoon` 7축.

    [0..3] 후처리 편차 4축   w = 0.20 · 0.825 · 0.280 · 0.45
    [4..6] 잔차 차등 3축     w = 1.0 · 1.0 · 1.0   (손 · 2S · 주자)

배수는 **곱셈**으로 준다 (`--dev 0.75` 는 네 축 전부 x0.75).

## 사용법

    # 전역 배수
    python exp\build_w.py --name cand_wd060 --diff 0.6
    python exp\build_w.py --name cand_dev075 --dev 0.75

    # 축별 세밀 조정 (7개 절대 가중을 직접)
    python exp\build_w.py --name cand_fine1 --abs 0.20,0.825,0.280,0.45,0.7,0.6,0.5

행 독립성은 구조적으로 보존된다 — 표와 조회 키는 손대지 않고 스칼라만 곱한다.
"""
import argparse
import hashlib
import os
import subprocess
import sys

import joblib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_DEV = 4
BASE_PKL = os.path.join(ROOT, "model_cand", "cat_submit_3.pkl")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def build(name, dev=1.0, diff=1.0, abs_w=None, force=False):
    base = joblib.load(BASE_PKL)
    pl = base["platoon"]
    assert len(pl) == 7, f"축 수가 7이 아니다: {len(pl)}"
    specs = []
    for i, s in enumerate(pl):
        s2 = dict(s)
        if abs_w is not None:
            s2["w"] = float(abs_w[i])
        else:
            s2["w"] = float(s["w"]) * (dev if i < N_DEV else diff)
        specs.append(s2)
    b = dict(base)
    b["platoon"] = specs
    tag = (f"abs={abs_w}" if abs_w is not None else f"dev x{dev:g} diff x{diff:g}")
    b["note"] = str(base.get("note", "")) + f" | W: {tag}"

    zp = os.path.join(ROOT, "submissions", f"{name}.zip")
    if os.path.exists(zp) and not force:
        raise SystemExit(f"이미 있다 — 덮어쓰지 않는다: {zp}")
    pkl = os.path.join(ROOT, "model_cand", f"{name}.pkl")
    joblib.dump(b, pkl, compress=3)
    r = subprocess.run(
        [sys.executable, os.path.join(ROOT, "make_submit.py"),
         "--model", os.path.relpath(pkl, ROOT),
         "--requirements", "requirements_cat.txt",
         "--out", os.path.relpath(zp, ROOT)],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if not os.path.exists(zp):
        raise SystemExit("빌드 실패\n" + r.stdout[-400:] + r.stderr[-800:])
    h = hashlib.sha256(open(zp, "rb").read()).hexdigest()
    print(f"  {name}.zip  {os.path.getsize(zp)/1e6:.2f} MB")
    print(f"  가중 " + " ".join(f"{s['w']:.4g}" for s in specs))
    print(f"  sha256 {h}")
    return zp, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--dev", type=float, default=1.0, help="편차 4축 배수")
    ap.add_argument("--diff", type=float, default=1.0, help="차등 3축 배수")
    ap.add_argument("--abs", type=str, default=None,
                    help="7개 절대 가중을 콤마로 (배수 무시)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    abs_w = [float(x) for x in a.abs.split(",")] if a.abs else None
    if abs_w is not None and len(abs_w) != 7:
        raise SystemExit("--abs 는 7개여야 한다")
    build(a.name, a.dev, a.diff, abs_w, a.force)


if __name__ == "__main__":
    main()
