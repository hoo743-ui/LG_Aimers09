r"""계층분해 편차를 **실제 파이프라인**으로 검증한다.

## `exp/shrink.py` 가 낸 것 (`corr(d,y)^2 * 1e5`)

    현행 d1 (count-k, k=300)      65.4      전이효율 0.189
    계층분해 D (k=1000)            91.3      +39.6%
    계층분해+EB                    84.4      전이효율 0.302

3폴드 모두 개선이다. 그러나 그 지표는 **기저 모델과의 상관을 무시한다**.
D 가 품은 손조합 효과 `hE` 는 CatBoost 가 이미 `pitcher_hand`/`batter_hand` 로,
CAAFE 판은 `cf_same_hand` 로 잡고 있을 수 있다. 중복이면 실전 이득이 사라진다.

## 그래서 세 형태를 분리해 잰다

    d1B = n/(n+300)  * (PH - P)                 현행
    d1D = hE + n/(n+1000) * (PH - P - hE)       계층분해 (손조합 효과 포함)
    d1R =      n/(n+1000) * (PH - P - hE)       잔차만 (손조합 효과 제외)

`d1D` 만 좋고 `d1R` 이 나쁘면 이득의 정체는 `hE` 이고, 그것은 모델이 이미 아는
정보를 후처리로 다시 넣은 것에 불과하다 (CAAFE 판에서는 특히). 반대로 `d1R` 이
`d1B` 를 이기면 **잔차 축소 자체가 개선된 것**이고 그것이 진짜 새 정보다.

나머지 3축(dC/dN/d3)은 19회차 가중 그대로 두고 `w1` 만 훑는다.

    .\.venv\Scripts\python.exe -u exp\shrink_cat.py
"""
import io
import json
import os
import time

import numpy as np
from catboost import CatBoostClassifier

from caafe import build

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "exp", "cache")
HP = dict(iterations=1200, learning_rate=0.02, depth=6, l2_leaf_reg=100.0,
          border_count=32, loss_function="Logloss", verbose=0,
          thread_count=16, allow_writing_files=False)
FOLDS = (2022, 2023, 2024)
W_REST = np.array([0.825, 0.280, 0.45])
KEEP = ["cf_same_hand", "cf_form5", "cf_share_reverse", "cf_share_ball",
        "cf_mix_entropy", "cf_log_pn", "cf_trend13", "cf_trend35",
        "cf_midform1", "cf_midtrend13", "cf_ball_minus_strike"]


def nested_dev(parent, child, y, k):
    o = np.argsort(child, kind="stable")
    Ys, Ps, Cs = y[o], parent[o], child[o]
    u, s = np.unique(Cs, return_index=True)
    cnt = np.diff(np.append(s, len(Cs)))
    cell = np.add.reduceat(Ys, s) / cnt
    par = Ps[s]
    op = np.argsort(parent, kind="stable")
    Yp, Pp = y[op], parent[op]
    pu, ps = np.unique(Pp, return_index=True)
    pc = np.diff(np.append(ps, len(Pp)))
    pmean = np.add.reduceat(Yp, ps) / pc
    return u, cnt * (cell - pmean[np.searchsorted(pu, par)]) / (cnt + k)


def look(u, d, keys):
    ix = np.clip(np.searchsorted(u, keys), 0, len(u) - 1)
    ok = u[ix] == keys
    out = np.zeros(len(keys))
    out[ok] = d[ix[ok]]
    return out


def gstat(key, y):
    o = np.argsort(key, kind="stable")
    k, v = key[o], y[o]
    u, s = np.unique(k, return_index=True)
    c = np.diff(np.append(s, len(k))).astype(np.float64)
    return u, np.add.reduceat(v, s) / c, c


def take(u, v, keys, fill=np.nan):
    ix = np.clip(np.searchsorted(u, keys), 0, max(len(u) - 1, 0))
    ok = (u[ix] == keys) if len(u) else np.zeros(len(keys), bool)
    out = np.full(len(keys), fill)
    out[ok] = v[ix[ok]]
    return out


def main():
    meta = json.load(io.open(f"{CACHE}/cols.json", encoding="utf-8"))
    ixc = {c: i for i, c in enumerate(meta["cols"])}
    prod = meta["prod"]
    X = np.load(f"{CACHE}/X.npy", mmap_mode="r")
    y = np.load(f"{CACHE}/y.npy").astype(np.float64)
    season = np.load(f"{CACHE}/season.npy")
    C = lambda n: np.asarray(X[:, ixc[n]], dtype=np.float64)

    P = C("pitcher_id").astype(np.int64)
    BH = C("batter_hand").astype(np.int64)
    PHD = C("pitcher_hand").astype(np.int64)
    BB, SS = C("balls_before").astype(np.int64), C("strikes_before").astype(np.int64)
    OB = (C("num_runners_on") > 0).astype(np.int64)
    PH, CNT = P * 10 + BH, BB * 4 + SS
    PHA = PH * 10 + (SS > BB).astype(np.int64)
    H4 = PHD * 2 + BH
    REST = [(PH, PHA, 2000), (PHA, PHA * 100 + CNT, 800), (PH, PH * 10 + OB, 2000)]

    base = np.column_stack([np.asarray(X[:, ixc[c]], dtype=np.float32)
                            for c in prod])
    F = build(C)
    base_c = np.hstack([base, np.column_stack([F[k] for k in KEEP])])

    print(f"{'폴드':>6}{'모델':<12}{'구성':<22}{'최적w1':>8}{'rho^2':>10}"
          f"{'현행대비':>10}")
    ACC = {}
    for f in FOLDS:
        tr, va = season < f, season == f
        yv = y[va]
        rest = np.column_stack([look(*nested_dev(a[tr], b[tr], y[tr], k), b[va])
                                for a, b, k in REST]) @ W_REST
        # --- d1 세 형태 ---
        uP, mP, cP = gstat(P[tr], y[tr])
        uH, mH, cH = gstat(H4[tr], y[tr])
        uPH, mPH, cPH = gstat(PH[tr], y[tr])
        G = float(y[tr].mean())
        p_par = take(uP, mP, P[va], G)
        cell = take(uPH, mPH, PH[va])
        n = take(uPH, cPH, PH[va], 0.0)
        hE = take(uH, mH - G, H4[va], 0.0)
        seen = ~np.isnan(cell)
        cell = np.where(seen, cell, p_par)
        raw = cell - p_par
        D1 = {
            "d1B 현행(k=300)": np.where(seen, n / (n + 300) * raw, 0.0),
            "d1D 계층(k=1000)": hE + np.where(seen, n / (n + 1000) * (raw - hE), 0.0),
            "d1R 잔차만(k=1000)": np.where(seen, n / (n + 1000) * (raw - hE), 0.0),
        }
        for mlbl, M in (("대조55p", base), ("CAAFE66p", base_c)):
            t = time.time()
            m = CatBoostClassifier(random_seed=42, **HP)
            m.fit(M[tr], y[tr].astype(int))
            p0 = m.predict_proba(M[va])[:, 1].astype(np.float64) + rest
            ref = None
            for dl, d in D1.items():
                ws = np.linspace(0, 1.2, 121)
                rr = np.array([1e5 * np.corrcoef(p0 + w * d, yv)[0, 1] ** 2
                               for w in ws])
                j = int(np.argmax(rr))
                # 현행은 w=0.20 고정이 실제 제출 조건이다
                fixed = 1e5 * np.corrcoef(p0 + 0.20 * d, yv)[0, 1] ** 2
                if ref is None:
                    ref = fixed
                ACC.setdefault((mlbl, dl), []).append(rr[j] / ref)
                print(f"{f:>6}{mlbl:<12}{dl:<22}{ws[j]:>8.2f}{rr[j]:>10.1f}"
                      f"{rr[j] / ref:>10.4f}", flush=True)
            print(f"{'':>6}{'':<12}{'(w=0.20 고정 현행)':<22}{0.20:>8.2f}"
                  f"{ref:>10.1f}{1.0:>10.4f}   {time.time() - t:.0f}s")

    print(f"\n=== 3폴드 기하평균 (현행 d1 w=0.20 대비) ===")
    print(f"  {'모델':<12}{'구성':<22}" + "".join(f"{f:>10}" for f in FOLDS)
          + f"{'기하평균':>10}{'3/3':>6}")
    for (mlbl, dl), v in ACC.items():
        g = float(np.exp(np.mean(np.log(v))))
        print(f"  {mlbl:<12}{dl:<22}" + "".join(f"{x:>10.4f}" for x in v)
              + f"{g:>10.4f}{str(sum(x > 1 for x in v)) + '/3':>6}")
    print("\n  * 최적 w1 은 그 폴드에서 고른 낙관값이다. 채택 시에는 "
          "폴드 밖에서 정한 고정 w 로 다시 재야 한다.")


if __name__ == "__main__":
    main()
