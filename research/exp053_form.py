r"""EXP053 — 수준 축의 **형태**를 훑는다. 재학습 0회.

8개월간 후처리는 언제나 `p = m + Σ w_i c_i` 의 **선형 가산**이었다. 수준 축이
전이율 +1.49 / +2.34 로 가장 좋은데, 그 축을 **어떤 형태로** 싣는지는 한 번도
바꿔본 적이 없다. 부류 단위 사각지대일 수 있다.

    L_p x L_b     두 수준의 상호작용 (매치업 정체성이 아니라 수준의 곱)
    L x cur_n     시즌 진행도에 따라 수준의 무게를 바꾼다
    L^2, |L|      수준의 비선형 (큰 편차일수록 덜/더 믿는다)
    clip(L)       극단 절단
    rank(L)       분포 형태 제거

전부 그 행 자신의 컬럼과 학습 상수만 쓴다. 규정 4 안전.

    .\.venv\Scripts\python.exe -u research\exp053_form.py
"""
import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "exp"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

W9 = np.array([0.099765, 0.411532, 0.139671, 0.224472,
               0.703699, 0.775302, 0.775315, 1.984998, 2.105])
NAMES = ["dev0", "dev1", "dev2", "dev3", "c_hand", "c_2S", "c_run", "L_p", "L_b"]

z = np.load(os.path.join(ROOT, "exp", "cache", "exp043_comp.npz"))
mm, Cm, y = z["mm"], z["Cm"], z["y"]
from path_alloc import build_df                                    # noqa: E402
tr = build_df()
season = tr["season"].to_numpy(); m = season == 2024
cur = mm + Cm @ W9
base = 1e5 * np.corrcoef(cur, y)[0, 1] ** 2
print(f"현행 9축 폴드 2024 = {base:.2f}\n")

Lp, Lb = Cm[:, 7], Cm[:, 8]                  # 축소된 수준 (가중 전)
CH, C2, CR = Cm[:, 4], Cm[:, 5], Cm[:, 6]
D0, D1 = Cm[:, 0], Cm[:, 1]
logn = tr["cur_logn_pitch"].to_numpy(float)[m]
lognb = tr["cur_logn_bat"].to_numpy(float)[m]
zs = lambda v: (v - v.mean()) / (v.std() + 1e-12)


def rank(v):
    o = np.argsort(np.argsort(v))
    return (o / (len(v) - 1) - 0.5)


CAND = {
    "L_p x L_b (수준 상호작용)": Lp * Lb,
    "L_p x logn_pitch": Lp * zs(logn),
    "L_b x logn_bat": Lb * zs(lognb),
    "L_p^2 * sign": np.sign(Lp) * Lp ** 2,
    "L_b^2 * sign": np.sign(Lb) * Lb ** 2,
    "rank(L_p)": rank(Lp) * Lp.std(),
    "rank(L_b)": rank(Lb) * Lb.std(),
    "clip(L_p, ±1sd)": np.clip(Lp, -Lp.std(), Lp.std()),
    "clip(L_b, ±1sd)": np.clip(Lb, -Lb.std(), Lb.std()),
    "L_p x c_hand": Lp * zs(CH),
    "L_b x c_hand": Lb * zs(CH),
    "L_p x dev0": Lp * zs(D0),
    "L_p + L_b (합의 비선형)": np.sign(Lp + Lb) * (Lp + Lb) ** 2,
    "c_hand x c_2S": CH * C2 / (CH.std() + 1e-12),
    "c_hand x c_run": CH * CR / (CH.std() + 1e-12),
    "L_p x 모델예측편차": Lp * zs(mm),
    "L_b x 모델예측편차": Lb * zs(mm),
}
WS = (-3, -2, -1.5, -1, -0.5, -0.25, 0.25, 0.5, 1, 1.5, 2, 3, 5, 8)
print(f"{'후보 형태':26s} {'최적 이득':>9s} {'w':>6s}   {'w 곡선 (이득)':s}")
rows = []
for nm, v in CAND.items():
    v = np.nan_to_num(v)
    if v.std() == 0:
        continue
    v = v / v.std() * Lb.std()                       # 크기를 L_b 에 맞춘다
    g = [(1e5 * np.corrcoef(cur + w * v, y)[0, 1] ** 2 - base, w) for w in WS]
    s, w = max(g)
    rows.append((s, w, nm, g))
for s, w, nm, g in sorted(rows, reverse=True):
    top = "  ".join(f"{ww:+g}:{gg:+.2f}" for gg, ww in g if abs(ww) in (0.5, 1, 2))
    print(f"{nm:26s} {s:+9.2f} {w:+6g}   {top}")
print(f"\n[대조] 이미 실린 축을 더 넣으면 (0 근처여야 정상)")
for nm, v in (("L_p 재투입", Lp), ("L_b 재투입", Lb), ("c_hand 재투입", CH)):
    v = v / v.std() * Lb.std()
    g = [(1e5 * np.corrcoef(cur + w * v, y)[0, 1] ** 2 - base, w) for w in WS]
    s, w = max(g)
    print(f"  {nm:22s} {s:+8.2f} @ w={w:+g}")
