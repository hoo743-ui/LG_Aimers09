r"""전이 연구 분석 — `transfer_study.json` 을 읽어 전이 모델을 세운다. 학습 0회.

## 규약

시즌마다 `rho^2` 규모가 30~2400 으로 흔들리므로 **상대 이득(%)** 만 쓴다.

    gain%(창 w, 목표 s) = (후보 - 기준선) / 기준선 x 100

창 `w` 는 시즌 `< w` 로 학습하므로 **목표 `w` 가 이미 1시즌 앞**이다.
따라서 시간 거리는 `거리 = s - w + 1`.

## LOCAL -> FUTURE 쌍

의사결정 시점 `t` 에서 손에 쥔 증거는 "가장 최근 시즌을 홀드아웃한 값"이고,
실제로 일어나는 일은 "한 시즌 뒤에 같은 후보를 그대로 배치한 값"이다.

    LOCAL_t   = gain%(창 t,   목표 t)      <- 거리 1
    FUTURE_t  = gain%(창 t+1, 목표 t+1)    <- 한 시즌 뒤의 거리 1

우리 실제 상황과 정확히 같다 (로컬은 학습 <=2023 / 검증 2024,
배치는 학습 <=2024 / 예측 2025).

    .\.venv\Scripts\python.exe -u exp\transfer_fit.py
"""
import io
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

R = json.load(io.open(os.path.join(ROOT, "exp", "transfer_study.json"),
                      encoding="utf-8"))
WINDOWS = (2020, 2021, 2022, 2023, 2024)
CAND = {"D": ("D", "BASE", "current-state"),
        "X": ("DX", "D", "context-interaction"),
        "H1": ("CH", "DX", "context-interaction"),
        "F": ("DF", "D", "recent-history"),
        "K2": ("K2", "CH", "regime")}
# 실제 LB 앵커 (생산 경로 로컬% -> 실제 LB%)
LB = {"D": (12.87, 8.97), "X": (2.78, 0.375), "H1": (2.12, 0.494),
      "F": (1.36, -0.82), "K2": (-0.87, -0.45)}


def gp(cand, base, w, s):
    """상대 이득(%)과 시드별 이득."""
    a, b = f"{cand}|{w}|{s}", f"{base}|{w}|{s}"
    if a not in R or b not in R:
        return None
    g = (R[a]["rho2"] - R[b]["rho2"]) / R[b]["rho2"] * 100
    per = [(x - y) / y * 100 for x, y in zip(R[a]["per"], R[b]["per"])]
    return g, float(np.std(per)), R[b]["rho2"]


print("=" * 96)
print("1. 상대 이득 행렬 (%) — 창 x 목표.  거리 = 목표 - 창 + 1")
print("=" * 96)
M = {}
for lbl, (c, b, fam) in CAND.items():
    print(f"\n[{lbl}] {c} - {b}   ({fam})")
    print(f"{'창':<7}" + "".join(f"{s:>9}" for s in WINDOWS) + "     (시드 산포)")
    for w in WINDOWS:
        row, sds = f"{w:<7}", []
        for s in WINDOWS:
            r = gp(c, b, w, s)
            if r is None:
                row += f"{'':>9}"
            else:
                M[(lbl, w, s)] = r
                row += f"{r[0]:>+9.2f}"
                if s == w:
                    sds.append(f"거리1 sd {r[1]:.2f}")
        print(row + "   " + " ".join(sds))

print()
print("=" * 96)
print("2. LOCAL_t -> FUTURE_{t+1} 쌍 (둘 다 거리 1)")
print("=" * 96)
print(f"{'후보':<5}{'t':>6}{'LOCAL%':>9}{'FUTURE%':>9}{'비율':>8}{'델타%p':>9}"
      f"{'LOCAL sd':>10}{'SNR':>7}")
P = []
for lbl in CAND:
    for t in (2020, 2021, 2022, 2023):
        a, b = M.get((lbl, t, t)), M.get((lbl, t + 1, t + 1))
        if a is None or b is None:
            continue
        ratio = b[0] / a[0] if abs(a[0]) > 1e-9 else np.nan
        snr = abs(a[0]) / a[1] if a[1] > 1e-9 else np.nan
        P.append(dict(cand=lbl, t=t, local=a[0], future=b[0], ratio=ratio,
                      delta=b[0] - a[0], sd=a[1], snr=snr,
                      fam=CAND[lbl][2]))
        print(f"{lbl:<5}{t:>6}{a[0]:>+9.2f}{b[0]:>+9.2f}{ratio:>8.2f}"
              f"{b[0]-a[0]:>+9.2f}{a[1]:>10.2f}{snr:>7.1f}")

L = np.array([p["local"] for p in P])
Fu = np.array([p["future"] for p in P])
Ra = np.array([p["ratio"] for p in P])
print(f"\n  쌍 {len(P)}개")
print(f"  비율   평균 {np.nanmean(Ra):+.3f}  중앙값 {np.nanmedian(Ra):+.3f}"
      f"  표준편차 {np.nanstd(Ra):.3f}  최악 {np.nanmin(Ra):+.3f}")
print(f"  델타   평균 {np.mean(Fu-L):+.3f}%p  표준편차 {np.std(Fu-L):.3f}%p")
print(f"  corr(LOCAL, FUTURE) = {np.corrcoef(L, Fu)[0,1]:+.3f}")
pos, neg = L > 0, L < 0
print(f"  부호 보존  LOCAL+ {pos.sum()}건 중 FUTURE+ {(Fu[pos] > 0).sum()}건"
      f"   LOCAL- {neg.sum()}건 중 FUTURE- {(Fu[neg] < 0).sum()}건")

print()
print("  --- 가설 검증 (쌍 20개, 계수 2개 이하만) ---")
b_only = np.sum(L * Fu) / np.sum(L * L)
res_a = Fu - b_only * L
A = np.column_stack([np.ones(len(L)), L])
ab = np.linalg.lstsq(A, Fu, rcond=None)[0]
res_b = Fu - A @ ab
print(f"  A  Future = b x Local            b={b_only:+.3f}"
      f"   잔차 RMS {np.sqrt((res_a**2).mean()):.3f}%p")
print(f"  B  Future = a + b x Local        a={ab[0]:+.3f} b={ab[1]:+.3f}"
      f"   잔차 RMS {np.sqrt((res_b**2).mean()):.3f}%p")
for t0 in (0.5, 1.0, 2.0):
    hi, lo = L >= t0, L < t0
    print(f"  C  임계 {t0:.1f}%  이상 {hi.sum()}건 평균 전이 "
          f"{np.nanmean(Ra[hi]) if hi.sum() else float('nan'):+.2f}"
          f"   미만 {lo.sum()}건 평균 전이 "
          f"{np.nanmean(Ra[lo]) if lo.sum() else float('nan'):+.2f}")

print()
print("=" * 96)
print("3. 구간별 · family별 전이")
print("=" * 96)
for name, sel in (("small  0<L<1%", (L > 0) & (L < 1)),
                  ("medium 1~3%", (L >= 1) & (L < 3)),
                  ("large  >=3%", L >= 3),
                  ("negative L<0", L < 0)):
    if sel.sum():
        print(f"  {name:<14}{sel.sum():>3}건  전이 평균 {np.nanmean(Ra[sel]):+.3f}"
              f"  중앙값 {np.nanmedian(Ra[sel]):+.3f}"
              f"  FUTURE 평균 {Fu[sel].mean():+.2f}%")
print()
fams = sorted(set(p["fam"] for p in P))
print(f"  {'family':<22}{'건수':>5}{'전이 평균':>10}{'표준편차':>9}{'부호보존':>9}"
      f"{'FUTURE 평균':>12}")
for f in fams:
    ix = [i for i, p in enumerate(P) if p["fam"] == f]
    r = Ra[ix]
    sgn = np.mean([(L[i] > 0) == (Fu[i] > 0) for i in ix])
    print(f"  {f:<22}{len(ix):>5}{np.nanmean(r):>10.3f}{np.nanstd(r):>9.3f}"
          f"{sgn:>9.0%}{Fu[ix].mean():>+12.2f}")

print()
print("=" * 96)
print("4. 시간 거리 감쇠 (거리별 상대 이득 평균 %)")
print("=" * 96)
print(f"{'후보':<6}" + "".join(f"{'거리'+str(d):>10}" for d in (1, 2, 3, 4, 5)))
for lbl in CAND:
    row = f"{lbl:<6}"
    for d in (1, 2, 3, 4, 5):
        v = [M[(lbl, w, s)][0] for (l2, w, s) in M
             if l2 == lbl and s - w + 1 == d]
        row += (f"{np.mean(v):>+10.2f}" if v else f"{'':>10}")
    print(row)

print()
print("=" * 96)
print("5. SNR 과 전이의 관계")
print("=" * 96)
sn = np.array([p["snr"] for p in P])
ok = np.isfinite(sn) & np.isfinite(Ra)
print(f"  corr(SNR, 전이비율) = {np.corrcoef(sn[ok], Ra[ok])[0,1]:+.3f}")
print(f"  corr(LOCAL sd, |델타|) = "
      f"{np.corrcoef([p['sd'] for p in P], np.abs(Fu-L))[0,1]:+.3f}")
lo, hi = sn < np.median(sn[ok]), sn >= np.median(sn[ok])
print(f"  SNR 낮은 절반 전이 평균 {np.nanmean(Ra[lo]):+.3f}"
      f"   높은 절반 {np.nanmean(Ra[hi]):+.3f}")

print()
print("=" * 96)
print("6. 실제 LB 앵커와의 정합 (pseudo-future 가 LB 를 맞히는가)")
print("=" * 96)
print(f"{'후보':<5}{'LB 로컬%':>10}{'LB 실측%':>10}{'LB 비율':>9}"
      f"{'의사미래 비율 평균':>18}{'차이':>9}")
for lbl in CAND:
    if lbl not in LB:
        continue
    l0, f0 = LB[lbl]
    pr = [p["ratio"] for p in P if p["cand"] == lbl]
    print(f"{lbl:<5}{l0:>+10.2f}{f0:>+10.2f}{f0/l0:>9.3f}"
          f"{np.nanmean(pr):>18.3f}{f0/l0-np.nanmean(pr):>+9.3f}")

json.dump({"pairs": P}, io.open(os.path.join(ROOT, "exp", "transfer_fit.json"),
                                "w", encoding="utf-8"), indent=1,
          ensure_ascii=False, default=float)
