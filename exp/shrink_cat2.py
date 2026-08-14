r"""계층분해 판정 — **가중을 폴드 밖에서 정한다**.

## 1차 검증이 무효였던 이유

`exp/shrink_cat.py` 는 `w1` 을 **그 폴드에서 최적화**해 비교했다. 그러면 현행
`d1` 조차 1.0955 가 나온다 — 14회차에서 −23.26 을 맞은 바로 그 함정이다
(로컬 최적 `w`=0.3904 를 골랐는데 평가셋 최적은 0.1990 이었다).

낙관 상한끼리 비교한 것이라 판정에 쓸 수 없다.

## 정직한 프로토콜

  (1) **고정 `w`** — 실제 제출 조건이다. 19회차가 쓴 0.20 을 포함해 격자로 훑고
      **같은 `w` 에서** 세 형태를 비교한다.
  (2) **폴드 밖 `w`** — 폴드 `f` 의 가중을 **다른 두 폴드의 최적값 평균**으로
      정해 `f` 에 적용한다. 하이퍼파라미터를 대상 폴드에서 안 고른다.

## 판정 기준

`d1D` 가 `d1B` 를 (1)(2) **양쪽에서 3/3** 으로 이겨야 채택이다. 특히
**CAAFE66p 판**에서 이겨야 한다 — 실제 제출본이 그것이고, 그 모델은
`cf_same_hand` 로 `hE` 를 이미 갖고 있어 중복 위험이 가장 크다.

예측을 저장해 재학습 없이 여러 분석을 돌린다.

    .\.venv\Scripts\python.exe -u exp\shrink_cat2.py
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
PRED = os.path.join(ROOT, "exp", "preds", "shrink2.npz")
HP = dict(iterations=1200, learning_rate=0.02, depth=6, l2_leaf_reg=100.0,
          border_count=32, loss_function="Logloss", verbose=0,
          thread_count=16, allow_writing_files=False)
FOLDS = (2022, 2023, 2024)
W_REST = np.array([0.825, 0.280, 0.45])
KEEP = ["cf_same_hand", "cf_form5", "cf_share_reverse", "cf_share_ball",
        "cf_mix_entropy", "cf_log_pn", "cf_trend13", "cf_trend35",
        "cf_midform1", "cf_midtrend13", "cf_ball_minus_strike"]
FORMS = ["d1B", "d1D", "d1R"]


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


def collect():
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
    out = {}
    for f in FOLDS:
        tr, va = season < f, season == f
        out[f"y_{f}"] = y[va]
        out[f"rest_{f}"] = np.column_stack(
            [look(*nested_dev(a[tr], b[tr], y[tr], k), b[va])
             for a, b, k in REST]) @ W_REST
        uP, mP, cP = gstat(P[tr], y[tr])
        uH, mH, _ = gstat(H4[tr], y[tr])
        uPH, mPH, cPH = gstat(PH[tr], y[tr])
        G = float(y[tr].mean())
        p_par = take(uP, mP, P[va], G)
        cell = take(uPH, mPH, PH[va])
        n = take(uPH, cPH, PH[va], 0.0)
        hE = take(uH, mH - G, H4[va], 0.0)
        seen = ~np.isnan(cell)
        raw = np.where(seen, cell, p_par) - p_par
        out[f"d1B_{f}"] = np.where(seen, n / (n + 300) * raw, 0.0)
        out[f"d1R_{f}"] = np.where(seen, n / (n + 1000) * (raw - hE), 0.0)
        out[f"d1D_{f}"] = hE + out[f"d1R_{f}"]
        for mlbl, M in (("c55", base), ("c66", base_c)):
            t = time.time()
            m = CatBoostClassifier(random_seed=42, **HP)
            m.fit(M[tr], y[tr].astype(int))
            out[f"p_{mlbl}_{f}"] = m.predict_proba(M[va])[:, 1].astype(np.float64)
            print(f"  {f} {mlbl} 학습 {time.time() - t:.0f}s", flush=True)
    np.savez_compressed(PRED, **out)
    return out


def main():
    if os.path.exists(PRED):
        print(f"저장된 예측 사용: {PRED}\n")
        D = dict(np.load(PRED))
    else:
        print("학습 (6회)\n")
        D = collect()

    r2 = lambda p, y: 1e5 * np.corrcoef(p, y)[0, 1] ** 2
    WS = np.linspace(0.05, 1.2, 116)

    print("=== (1) 고정 w — 같은 w 에서 세 형태 비교 (실제 제출 조건) ===")
    for mlbl in ("c55", "c66"):
        print(f"\n  [{mlbl}]  {'w':>6}" + "".join(f"{x:>22}" for x in FORMS))
        for w in (0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
            cells = []
            for fm in FORMS:
                ms = [r2(D[f"p_{mlbl}_{f}"] + D[f"rest_{f}"] + w * D[f"{fm}_{f}"],
                         D[f"y_{f}"])
                      / r2(D[f"p_{mlbl}_{f}"] + D[f"rest_{f}"]
                           + 0.20 * D[f"d1B_{f}"], D[f"y_{f}"])
                      for f in FOLDS]
                g = float(np.exp(np.mean(np.log(ms))))
                cells.append(f"{g:.4f} ({sum(x > 1 for x in ms)}/3)")
            print(f"  {'':>7}{w:>6.2f}" + "".join(f"{c:>22}" for c in cells))

    print("\n=== (2) 폴드 밖 w — 다른 두 폴드의 최적값 평균을 적용 ===")
    print(f"  {'모델':<6}{'형태':<6}" + "".join(f"{f:>10}" for f in FOLDS)
          + f"{'기하평균':>10}{'3/3':>6}{'적용w':>18}")
    RES = {}
    for mlbl in ("c55", "c66"):
        for fm in FORMS:
            wopt = {}
            for f in FOLDS:
                rr = [r2(D[f"p_{mlbl}_{f}"] + D[f"rest_{f}"] + w * D[f"{fm}_{f}"],
                         D[f"y_{f}"]) for w in WS]
                wopt[f] = float(WS[int(np.argmax(rr))])
            ms, used = [], []
            for f in FOLDS:
                w = float(np.mean([wopt[g] for g in FOLDS if g != f]))
                used.append(w)
                num = r2(D[f"p_{mlbl}_{f}"] + D[f"rest_{f}"] + w * D[f"{fm}_{f}"],
                         D[f"y_{f}"])
                den = r2(D[f"p_{mlbl}_{f}"] + D[f"rest_{f}"]
                         + 0.20 * D[f"d1B_{f}"], D[f"y_{f}"])
                ms.append(num / den)
            g = float(np.exp(np.mean(np.log(ms))))
            RES[(mlbl, fm)] = (ms, g)
            print(f"  {mlbl:<6}{fm:<6}" + "".join(f"{m:>10.4f}" for m in ms)
                  + f"{g:>10.4f}{str(sum(x > 1 for x in ms)) + '/3':>6}"
                  + f"   {','.join(f'{w:.2f}' for w in used):>15}")

    print("\n=== 판정 — d1D 가 d1B 를 이기는가 (폴드 밖 w 기준) ===")
    for mlbl in ("c55", "c66"):
        b, d = RES[(mlbl, "d1B")], RES[(mlbl, "d1D")]
        rel = [x / z for x, z in zip(d[0], b[0])]
        g = float(np.exp(np.mean(np.log(rel))))
        tag = "채택" if sum(x > 1 for x in rel) == 3 else "게이트 미달"
        print(f"  {mlbl}  d1D/d1B = " + " ".join(f"{x:.4f}" for x in rel)
              + f"   기하평균 {g:.4f}   {sum(x > 1 for x in rel)}/3   -> {tag}")


if __name__ == "__main__":
    main()
