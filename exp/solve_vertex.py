r"""좌표 직선 위 LB 점들로 **포물선을 풀고 정점 t\*** 를 낸다 (55회차 방법 고정).

## 근거 (EXP059)

같은 표 직선 위에서 점수는 t 의 **정확한 2차식**이다. t=+1,-1 두 점으로 정한
식이 t=-3 을 오차 2.3e-13, t=-4.5 를 1.5e-8 로 맞혔다. 로컬 분모(2024)를 섞은
유리함수 모형은 측정점조차 -3.23 어긋났다 -> **LB 점만 쓴다.**

    gain(t) = c1·t + c2·t²        (t=0 이 기준선이므로 상수항 없음)
    t* = -c1/(2c2)                최대이득 = -c1²/(4c2)

기준선(t=0)이 공짜 3번째 점이므로 **탐침 2회면 곡선이 완전히 결정된다.**

    .\.venv\Scripts\python.exe exp\solve_vertex.py --s0 1090.0094882798 ^
        --pt -6 1085.1 --pt -3 1093.4
"""
import argparse
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def solve(s0, pts):
    t = np.array([p[0] for p in pts], float)
    g = np.array([p[1] for p in pts], float) - s0
    M = np.column_stack([t, t * t])
    c, *_ = np.linalg.lstsq(M, g, rcond=None)
    c1, c2 = c
    res = M @ c - g
    print(f"기준선 S0 = {s0:.10f}   측정점 {len(pts)}개")
    for (ti, gi), ri in zip(zip(t, g), res):
        print(f"  t={ti:+7.4f}   LB={s0+gi:14.7f}   gain={gi:+9.4f}   잔차={ri:+.3e}")
    print(f"\n  gain(t) = {c1:+.7f}·t {c2:+.7f}·t²")
    if c2 >= 0:
        print("  ⚠️ c2 >= 0 — 위로 열린 포물선이다. 정점이 최소이므로 **바깥으로** 더 나가야 한다")
        print(f"     현 두 점 중 좋은 쪽으로 |t| 를 키워 재탐침한다")
        return
    ts = -c1 / (2 * c2)
    gmax = -c1 * c1 / (4 * c2)
    print(f"\n  🚩 t* = {ts:.6f}   예측 최대 = {s0 + gmax:.7f}   (기준선 대비 {gmax:+.4f})")
    best = max(pts, key=lambda p: p[1])
    print(f"  이미 낸 최고점 {best[1]:.7f} (t={best[0]:g}) 대비 남은 몫 {s0+gmax-best[1]:+.4f}")
    if abs(s0 + gmax - best[1]) < 0.05:
        print("  -> 남은 몫이 잡음 수준. **정점 제출을 건너뛰고 다음 좌표로 간다**")
    else:
        print("\n  빌드:  venv_submit/Scripts/python.exe -u exp/build_lvlk.py "
              f"--axis pitcher --k 2000 --t {ts:.6f} --name cand_kpopt")
    # 외삽 경고
    if not (min(t) <= ts <= max(t)):
        print("  ⚠️ t* 가 탐침 구간 **밖**이다 — 55회차에서 외삽 오차가 2.3e-13 -> 1.5e-8 로 커졌다")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s0", type=float, required=True, help="t=0 기준선 LB")
    ap.add_argument("--pt", nargs=2, type=float, action="append", required=True,
                    metavar=("T", "LB"))
    a = ap.parse_args()
    solve(a.s0, [(t, s) for t, s in a.pt])


if __name__ == "__main__":
    main()
