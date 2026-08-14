r"""1120 에 닿으려면 2번째 모델이 **얼마나 강하고 얼마나 달라야** 하는가.

## 왜 이 계산인가

점수 = `1e5 * rho^2` 이고 `rho` 는 아핀 불변이다. 즉 13회차가 6자리까지 맞춘
아핀 보정으로는 `rho` 가 1 도 안 움직인다. 955 -> 1120 은 **오직 예측 자체**의
문제이고, LB 역산으로 번 40점과는 직교하는 축이다.

우리가 안 해본 것이 정확히 하나 있다 — **강하면서 상관이 낮은 2번째 모델.**

    LightGBM   상관 0.994   이득 +0.5      (같은 모델을 두 번 돌린 셈)
    HGB        3폴드 모두 순수 cat 이 최고
    FM         상관 0.50~0.86 인데 rho^2 229 vs 798 — 너무 약하다

두 예측 `p1`(챔피언), `p2`(도전자)를 섞으면 표준화 단위에서

    rho_blend(w) = (w*r1 + (1-w)*r2) / sqrt(w^2 + (1-w)^2 + 2w(1-w)c)

`w` 로 최적화한다. 같은 세기(r2=r1)에 상관 `c` 면 배수가 `2/(1+c)` 이므로

    955.22 -> 1120 은 배수 1.1726  ->  c = 0.7057

즉 **챔피언과 같은 세기이고 상관이 0.71 인 모델 하나**면 정확히 1120 이다.
이 스크립트는 세기를 낮췄을 때 필요한 상관까지 표로 낸다.

    .\.venv\Scripts\python.exe exp\blend_spec.py
"""
import numpy as np

S_CUR = 955.2193198652
TARGETS = (1000, 1050, 1120, 1200)
# 세기 비 r2/r1. 우리 실측 대조군: FM 은 sqrt(229/798)=0.536 이었다
RATIOS = (1.00, 0.90, 0.80, 0.70, 0.60, 0.536, 0.40)
CORRS = (0.99, 0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50)


def mult(ratio, c):
    """혼합 최적 w 에서의 점수 배수 (rho^2 비)."""
    r1, r2 = 1.0, ratio
    w = np.linspace(0, 1, 20001)
    num = (w * r1 + (1 - w) * r2) ** 2
    den = w ** 2 + (1 - w) ** 2 + 2 * w * (1 - w) * c
    return float(np.max(num / den)) / (r1 * r1)


def need_corr(ratio, target):
    """목표 점수에 닿는 최대 허용 상관. 없으면 nan."""
    lo, hi = -0.99, 0.999
    if mult(ratio, lo) * S_CUR < target:
        return float("nan")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if mult(ratio, mid) * S_CUR >= target:
            lo = mid
        else:
            hi = mid
    return lo


def main():
    print(f"현 챔피언 {S_CUR:.4f}   (rho = {np.sqrt(S_CUR / 1e5):.6f})")
    print("점수 = 1e5*rho^2 이고 rho 는 아핀 불변 — 보정으로는 못 움직인다.\n")

    print("=== 1) 2번째 모델을 섞었을 때 도달 점수 ===")
    print(f"  {'상관 c':>7}" + "".join(f"{f'r2/r1={x:.2f}':>12}" for x in RATIOS))
    for c in CORRS:
        print(f"  {c:>7.2f}" + "".join(
            f"{mult(x, c) * S_CUR:>12.1f}" for x in RATIOS))

    print("\n=== 2) 목표 점수별 — 필요한 최대 상관 ===")
    print(f"  {'목표':>6}" + "".join(f"{f'r2/r1={x:.2f}':>12}" for x in RATIOS))
    for t in TARGETS:
        row = ""
        for x in RATIOS:
            v = need_corr(x, t)
            row += f"{'불가':>12}" if np.isnan(v) else f"{v:>12.3f}"
        print(f"  {t:>6}" + row)

    print("\n=== 3) 우리가 실제로 만들었던 도전자들 ===")
    for nm, ratio, c in [("LightGBM", 1.00, 0.994),
                         ("FM (2024 폴드)", 0.536, 0.499),
                         ("FM (2022 폴드)", 0.778, 0.807)]:
        print(f"  {nm:<16} r2/r1={ratio:.3f}  c={c:.3f}  "
              f"-> {mult(ratio, c) * S_CUR:.1f}  ({mult(ratio, c) * S_CUR - S_CUR:+.1f})")

    print("\n=== 4) 세기를 포기하고 다양성만으로 갈 수 있나 ===")
    print("  (약한 모델은 상관이 0 이어도 배수 상한이 있다)")
    print(f"  {'r2/r1':>7}{'c=0 배수':>12}{'점수':>10}")
    for x in (0.8, 0.6, 0.4, 0.2):
        m = mult(x, 0.0)
        print(f"  {x:>7.2f}{m:>12.4f}{m * S_CUR:>10.1f}")


if __name__ == "__main__":
    main()
