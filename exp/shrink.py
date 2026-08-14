r"""PARTIAL POOLING — 세밀한 셀을 안정적인 상위 구조로 빌려온다.

## 출발점 (§12-b)

| 분할 | 셀수 | in-fold 오라클 | out-of-fold | 전이효율 |
|---|---:|---:|---:|---:|
| 손조합4 | 4 | 114.5 | 93.1 | **81.3%** |
| 투수×타자손 | 772 | 1425.5 | 109.2 | 7.7% |
| 투수×손×카운트 | 8,492 | 4593.5 | 128.6 | 2.8% |
| 투수×손×카운트×주자 | 15,680 | 7481.4 | 130.7 | 1.7% |

**세밀할수록 정보는 폭발하고 전이는 붕괴한다.** 따라서 셀 암기를 강화할 것이
아니라 **안정적인 상위 구조에서 정보를 빌려와야** 한다.

## 평가 지표 — 왜 `corr(d,y)^2` 인가

편차항을 `p += w*d` 로 더할 때, 적률 모형에서 최적 가중의 이득은

    gain ∝ (k*cov(d,y))^2 / (lam*var(d)) ∝ corr(d,y)^2

이므로 `1e5*corr(d, y_eval)^2` 이 정확한 비교 지표다. CatBoost 학습 없이
수십 가지를 훑고, 승자만 실제 파이프라인으로 검증한다.

**주의** — 이 지표는 `d` 와 기저 모델 예측의 상관(`cm` 항)을 무시한다. 같은
`corr(d,y)` 라도 모델과 더 겹치는 쪽이 나쁘다. 그래서 승자는 반드시 CatBoost 로
재검증한다.

## 비교하는 추정량 (전부 시즌 `<f` 로만 구축)

    G          전체 평균
    H[h]       손조합4 평균          (전이효율 81.3% — 가장 안정적인 상위 구조)
    P[p]       투수 평균
    PH[p,h]    투수x타자손 평균

    A raw          d = PH - P                       축소 없음
    B count-k      d = n/(n+k) * (PH - P)           <- 현행 d1 (k=300)
    C 분산인지 EB   d = n/(n+s2/tau2) * (PH - P)     관측이 많아도 변동 크면 강하게 축소
    D 계층분해      d = hE[h] + shrink(PH - P - hE[h])
                   손조합 효과를 먼저 떼고 **잔차만** 축소한다. H1/H3 의 핵심.
    E D + EB       D 의 잔차를 분산인지 축소로
    F 부모교체      d = n/(n+k) * (PH - P - hE) + hE

`hE[h] = H[h] - G` 는 4셀이라 사실상 축소가 필요 없다 (전이 81.3%).

    .\.venv\Scripts\python.exe -u exp\shrink.py
"""
import io
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "exp", "cache")
FOLDS = (2022, 2023, 2024)


def gstat(key, y):
    """키별 합/개수/평균."""
    o = np.argsort(key, kind="stable")
    k, v = key[o], y[o]
    u, s = np.unique(k, return_index=True)
    cnt = np.diff(np.append(s, len(k))).astype(np.float64)
    tot = np.add.reduceat(v, s)
    return u, tot, cnt


def take(u, val, keys, fill=0.0):
    ix = np.clip(np.searchsorted(u, keys), 0, max(len(u) - 1, 0))
    ok = (u[ix] == keys) if len(u) else np.zeros(len(keys), bool)
    out = np.full(len(keys), fill, dtype=np.float64)
    out[ok] = val[ix[ok]]
    return out


def rho2(d, y):
    if np.std(d) < 1e-12:
        return 0.0
    return 1e5 * np.corrcoef(d, y)[0, 1] ** 2


def build(P, H4, PH, y, tr, ev_keys, scheme, k=300.0):
    """시즌 <f 로 추정량을 만들고 평가 행에 붙인다."""
    Pt, H4t, PHt, yt = P[tr], H4[tr], PH[tr], y[tr]
    G = float(yt.mean())
    uH, tH, cH = gstat(H4t, yt)
    uP, tP, cP = gstat(Pt, yt)
    uPH, tPH, cPH = gstat(PHt, yt)
    mH, mP, mPH = tH / cH, tP / cP, tPH / cPH
    hE = mH - G                                   # 손조합 효과 (4셀)

    kp, kh, kph = ev_keys                          # 평가행의 P / H4 / PH 키
    p_par = take(uP, mP, kp, G)
    ph_cell = take(uPH, mPH, kph, np.nan)
    n = take(uPH, cPH, kph, 0.0)
    h_eff = take(uH, hE, kh, 0.0)
    seen = ~np.isnan(ph_cell)
    ph_cell = np.where(seen, ph_cell, p_par)

    raw = ph_cell - p_par                          # 투수 평균 대비 편차
    if scheme == "A":
        return np.where(seen, raw, 0.0)
    if scheme == "B":
        return np.where(seen, n / (n + k) * raw, 0.0)
    if scheme in ("C", "E"):
        # 경험적 베이즈: w = n/(n + s2/tau2)
        # s2 = 셀 내 이항분산, tau2 = 셀 간 참효과 분산 (적률법)
        base = raw if scheme == "C" else raw - h_eff
        s2 = np.clip(ph_cell * (1 - ph_cell), 1e-6, None)
        obs = base[seen & (n >= 5)]
        nn = n[seen & (n >= 5)]
        tau2 = max(float(np.var(obs) - np.mean(s2[seen & (n >= 5)] / nn)), 1e-8)
        w = n / (n + s2 / tau2)
        out = np.where(seen, w * base, 0.0)
        return out + (h_eff if scheme == "E" else 0.0)
    if scheme == "D":
        resid = raw - h_eff                        # 손조합 효과를 먼저 뗀다
        return h_eff + np.where(seen, n / (n + k) * resid, 0.0)
    if scheme == "F":
        return h_eff + np.where(seen, n / (n + k) * (raw - h_eff), 0.0)
    raise ValueError(scheme)


def main():
    meta = json.load(io.open(f"{CACHE}/cols.json", encoding="utf-8"))
    ixc = {c: i for i, c in enumerate(meta["cols"])}
    X = np.load(f"{CACHE}/X.npy", mmap_mode="r")
    y = np.load(f"{CACHE}/y.npy").astype(np.float64)
    season = np.load(f"{CACHE}/season.npy")
    C = lambda n: np.asarray(X[:, ixc[n]], dtype=np.float64)

    P = C("pitcher_id").astype(np.int64)
    BH = C("batter_hand").astype(np.int64)
    PHD = C("pitcher_hand").astype(np.int64)
    H4 = PHD * 2 + BH
    PH = P * 10 + BH

    print("=== 1) 축소 상수 k 훑기 (Model B = 현행 d1 형태) ===")
    print(f"  {'k':>7}" + "".join(f"{f:>10}" for f in FOLDS) + f"{'기하평균':>10}")
    for k in (50, 100, 200, 300, 500, 1000, 2000, 5000):
        rs = []
        for f in FOLDS:
            tr, va = season < f, season == f
            d = build(P, H4, PH, y, tr, (P[va], H4[va], PH[va]), "B", k)
            rs.append(max(rho2(d, y[va]), 1e-9))
        print(f"  {k:>7}" + "".join(f"{r:>10.1f}" for r in rs)
              + f"{np.exp(np.mean(np.log(rs))):>10.1f}")

    print("\n=== 2) 방식 비교 (k 는 각 방식의 최적 근처 300 고정) ===")
    NAMES = {"A": "raw (축소 없음)", "B": "count-k (현행 d1)",
             "C": "분산인지 EB", "D": "계층분해+count",
             "E": "계층분해+EB", "F": "부모교체"}
    print(f"  {'방식':<18}" + "".join(f"{f:>10}" for f in FOLDS)
          + f"{'기하평균':>10}{'in-fold':>10}{'전이효율':>9}")
    res = {}
    for s in "ABCDEF":
        rs, ins = [], []
        for f in FOLDS:
            tr, va = season < f, season == f
            d = build(P, H4, PH, y, tr, (P[va], H4[va], PH[va]), s)
            rs.append(max(rho2(d, y[va]), 1e-9))
            di = build(P, H4, PH, y, va, (P[va], H4[va], PH[va]), s)
            ins.append(max(rho2(di, y[va]), 1e-9))
        g = float(np.exp(np.mean(np.log(rs))))
        gi = float(np.exp(np.mean(np.log(ins))))
        res[s] = (rs, g, gi)
        print(f"  {NAMES[s]:<18}" + "".join(f"{r:>10.1f}" for r in rs)
              + f"{g:>10.1f}{gi:>10.1f}{g / gi:>9.3f}")

    print("\n=== 3) 계층분해(D)의 k 훑기 — 잔차에만 축소를 건다 ===")
    print(f"  {'k':>7}" + "".join(f"{f:>10}" for f in FOLDS) + f"{'기하평균':>10}")
    best = None
    for k in (100, 300, 1000, 3000, 10000, 1e9):
        rs = []
        for f in FOLDS:
            tr, va = season < f, season == f
            d = build(P, H4, PH, y, tr, (P[va], H4[va], PH[va]), "D", k)
            rs.append(max(rho2(d, y[va]), 1e-9))
        g = float(np.exp(np.mean(np.log(rs))))
        lbl = "inf(=손조합만)" if k > 1e8 else f"{int(k)}"
        print(f"  {lbl:>7}" + "".join(f"{r:>10.1f}" for r in rs)
              + f"{g:>10.1f}")
        if best is None or g > best[0]:
            best = (g, k)

    print(f"\n  현행 d1 (B, k=300) 기하평균 {res['B'][1]:.1f}")
    print(f"  최고 방식 기하평균 "
          f"{max(v[1] for v in res.values()):.1f} "
          f"({[k for k, v in res.items() if v[1] == max(x[1] for x in res.values())][0]})")
    print(f"  D 최적 k={best[1]:.0f} 에서 {best[0]:.1f}")
    print(f"\n  주의 — 이 지표는 기저 모델과의 상관(cm)을 무시한다. "
          f"승자는 CatBoost 로 재검증할 것.")


if __name__ == "__main__":
    main()
