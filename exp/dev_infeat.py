r"""편차 신호를 **모델 바깥의 스칼라**에서 **모델 안의 피처**로 옮긴다.

## 왜

편차 신호는 LB 로 진짜임이 증명됐다 (13회차 942.46 -> 19회차 955.22, +12.76).
그런데 지금 그것이 들어가는 방식은 이렇다.

    p = model(x) + 0.20*d1 + 0.825*dC + 0.28*dN + 0.45*d3      <- w 가 상수 4개

**모델에게 피처로 준 적이 한 번도 없다.** 카운트·상황·투수유형에 따라 편차를
얼마나 믿을지가 다를 텐데 지금은 전부 하나의 상수다. 피처로 넣으면 `w(x)` 를
학습할 수 있다.

`BEST_CONFIG.closed_axes` 에 "상황별 성공률 편차 (신호/잡음 <1, **학습 없이**
기각)" 로 닫혀 있으나, 그것은 이 신호가 진짜인 줄 모르던 시점에 학습도 없이
내린 판정이다. 이제 신호 크기를 LB 로 알고 있으므로 다시 잰다.

## 🚩 누수 차단 — 이 실험의 성패가 여기 달렸다

편차는 타깃 인코딩이다. 학습 행의 편차에 **그 행 자신의 정답**이 들어가면 모델은
편차항을 과신하도록 학습되고, 검증에서 무너진다.

그래서 **시즌 `s` 행의 편차는 시즌 `< s` 로만 만든다** (as-of). 이는 배포 구조
(<=2024 표로 2025 예측)와 정확히 같은 형태다. 2019 행은 이전이 없으므로 0.

    2019 -> 0 (선행 시즌 없음)
    2020 -> 2019 로 만든 표
    2021 -> 2019~2020
    ...
    2024 -> 2019~2023        <- 후처리판이 쓰는 것과 동일

## 셀 관측수도 같이 넣는다

후처리판은 축소 상수 `k`(300/800/2000)로 신뢰도를 고정한다. 모델에게 셀
관측수 `n` 을 주면 **신뢰도 곡선 자체를 학습**할 수 있다. 이것이 상수 가중으로는
불가능한 부분이고, 이 실험이 노리는 지점이다.

## 판정 기준

    대조군 (prod 55, 같은 시드/설정)     <- 이걸 이겨야 의미가 있다
    Champion 후처리판  rho^2 798.1      <- 이걸 이겨야 채택 가치가 있다

    .\.venv\Scripts\python.exe -u exp\dev_infeat.py
"""
import argparse
import io
import json
import os
import time

import numpy as np
from catboost import CatBoostClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "exp", "cache")
HP = dict(iterations=1200, learning_rate=0.02, depth=6, l2_leaf_reg=100.0,
          border_count=32, loss_function="Logloss", verbose=0,
          thread_count=16, allow_writing_files=False)


def nested_dev(parent, child, y, k):
    """부모 평균 대비 자식 셀 편차 + 셀 관측수. 축소 상수 k."""
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
    dev = cnt * (cell - pmean[np.searchsorted(pu, par)]) / (cnt + k)
    return u, dev, cnt


def lookup2(u, dev, cnt, keys):
    ix = np.clip(np.searchsorted(u, keys), 0, max(len(u) - 1, 0))
    ok = (len(u) > 0) & (u[ix] == keys) if len(u) else np.zeros(len(keys), bool)
    d = np.zeros(len(keys), dtype=np.float32)
    n = np.zeros(len(keys), dtype=np.float32)
    d[ok], n[ok] = dev[ix[ok]], cnt[ix[ok]]
    return d, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", type=int, default=2024)
    ap.add_argument("--seeds", type=int, default=1)
    a = ap.parse_args()
    VAL = a.val
    meta = json.load(io.open(f"{CACHE}/cols.json", encoding="utf-8"))
    ixc = {c: i for i, c in enumerate(meta["cols"])}
    prod = meta["prod"]
    X = np.load(f"{CACHE}/X.npy", mmap_mode="r")
    y = np.load(f"{CACHE}/y.npy").astype(np.float64)
    season = np.load(f"{CACHE}/season.npy")
    col = lambda n: np.asarray(X[:, ixc[n]], dtype=np.float64)

    P = col("pitcher_id").astype(np.int64)
    BH = col("batter_hand").astype(np.int64)
    BB, SS = col("balls_before").astype(np.int64), col("strikes_before").astype(np.int64)
    OB = (col("num_runners_on") > 0).astype(np.int64)
    PH, CNT = P * 10 + BH, BB * 4 + SS
    ADV = (SS > BB).astype(np.int64)
    PHA = PH * 10 + ADV

    # 챔피언 후처리판이 쓰는 네 축을 그대로 옮긴다
    AXES = [("d1", P, PH, 300),
            ("dC", PH, PHA, 2000),
            ("dN", PHA, PHA * 100 + CNT, 800),
            ("d3", PH, PH * 10 + OB, 2000)]

    n = len(y)
    D = {nm: np.zeros(n, dtype=np.float32) for nm, _, _, _ in AXES}
    N = {nm: np.zeros(n, dtype=np.float32) for nm, _, _, _ in AXES}
    print("as-of 편차 생성 (시즌 s 행 <- 시즌 <s 로만)")
    for s in sorted(np.unique(season)):
        tr, va = season < s, season == s
        if tr.sum() == 0:
            print(f"  {s}  선행 시즌 없음 -> 0")
            continue
        for nm, par, ch, k in AXES:
            u, dv, ct = nested_dev(par[tr], ch[tr], y[tr], k)
            d, c = lookup2(u, dv, ct, ch[va])
            D[nm][va], N[nm][va] = d, c
        cov = float(np.mean(N["d1"][va] > 0))
        print(f"  {s}  학습 {int(tr.sum()):>9,}행  d1 커버리지 {cov:6.1%}")

    base = np.column_stack([np.asarray(X[:, ixc[c]], dtype=np.float32)
                            for c in prod])
    dev4 = np.column_stack([D[nm] for nm, _, _, _ in AXES])
    cnt4 = np.column_stack([np.log1p(N[nm]) for nm, _, _, _ in AXES])

    tr, va = season < VAL, season == VAL
    yv = y[va]
    cp = os.path.join(ROOT, "exp", f"champ_oof_{VAL}.npy")
    r_champ = (1e5 * np.corrcoef(np.load(cp), yv)[0, 1] ** 2
               if os.path.exists(cp) else float("nan"))
    print(f"\n=== 검증 폴드 {VAL}   Champion OOF {r_champ:.1f} ===")
    print(f"학습 {int(tr.sum()):,}행 / 검증 {int(va.sum()):,}행  "
          f"시드 {a.seeds}개\n")

    CONFIGS = [("대조 prod55", base),
               ("+편차4", np.hstack([base, dev4])),
               ("+편차4+logn4", np.hstack([base, dev4, cnt4]))]

    print(f"  {'설정':<16}{'피처':>5}{'rho^2':>10}{'vs 대조':>10}"
          f"{'vs Champ':>10}{'초':>6}")
    ctrl = None
    out = {}
    for nm, F in CONFIGS:
        t = time.time()
        acc = np.zeros(int(va.sum()), dtype=np.float64)
        for sd in range(42, 42 + a.seeds):
            m = CatBoostClassifier(random_seed=sd, **HP)
            m.fit(F[tr], y[tr].astype(int))
            acc += m.predict_proba(F[va])[:, 1]
        p = acc / a.seeds
        r = 1e5 * np.corrcoef(p, yv)[0, 1] ** 2
        if ctrl is None:
            ctrl = r
        out[nm] = p
        print(f"  {nm:<16}{F.shape[1]:>5}{r:>10.1f}{r - ctrl:>+10.1f}"
              f"{r - r_champ:>+10.1f}{time.time() - t:>6.0f}")

    np.savez_compressed(os.path.join(ROOT, "exp", "preds",
                                     f"dev_infeat_{VAL}.npz"),
                        **{k: v.astype(np.float32) for k, v in out.items()})

    # 후처리를 추가로 얹으면? (in-model 이 이미 먹었는지 확인)
    print(f"\n=== in-model 위에 후처리를 또 얹으면 (이미 먹었는지 확인) ===")
    dv = dev4[va]
    W = [0.20, 0.825, 0.280, 0.45]
    for nm, p in out.items():
        ph = p + dv @ np.array(W)
        r0 = 1e5 * np.corrcoef(p, yv)[0, 1] ** 2
        r1 = 1e5 * np.corrcoef(ph, yv)[0, 1] ** 2
        print(f"  {nm:<16} 후처리 전 {r0:>8.1f}  후 {r1:>8.1f}  {r1 - r0:>+8.1f}")


if __name__ == "__main__":
    main()
