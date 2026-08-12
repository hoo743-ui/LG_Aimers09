r"""캐시된 예측들의 **혼합**을 rho^2 로 잰다. 재학습 0회.

## 왜 이걸 다시 보는가

앙상블(HGB+Cat)과 LightGBM 은 **"예측이 중심으로 오므라든다"** 는 이유로 기각됐다
(4-5, 4-10). 그런데 **그게 정확히 `alpha` 가 고치는 것이다** (4-23). 즉 기각 근거가
지금은 무효일 수 있다.

`rho_rank.py` 는 이 의심을 **개별 설정에만** 적용했고 혼합은 재지 않았다. 혼합은
`rho` 를 올리는 고전적 수단이다 — 오차가 완전히 같지만 않으면 평균이 더 높은 `rho`
를 갖는다. 그리고 **고정 가중이면 계수를 폴드에서 맞추지 않으므로 4-26/4-28 이 죽은
전이 문제에 걸리지 않는다.**

## 판정

`w` 를 폴드마다 고르면 그건 폴드 안에서 맞춘 계수다 (함정 ③). 그래서 **고정 `w`**
에서 3폴드 부호일치를 본다. 폴드별 최적 `w` 는 낙관 상한으로 참고만 찍는다.

    .\.venv\Scripts\python.exe exp\blend_rank.py
    .\.venv\Scripts\python.exe exp\blend_rank.py --top 12
"""
import argparse
import itertools
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "exp", "cache")
PREDS = os.path.join(ROOT, "exp", "preds")
LOG = os.path.join(ROOT, "experiment_log.jsonl")

FOLDS = [2021, 2022, 2024]
REF = "cat_tuned"
SEEDS = 3
WGRID = np.round(np.arange(0.05, 0.75, 0.05), 2)


def rho2(y, p):
    if p.std() < 1e-12:
        return 0.0
    return float(1e5 * np.corrcoef(p, y) [0, 1] ** 2)


def load_all(seeds):
    rows = {}
    with open(LOG, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if os.path.exists(f"{PREDS}/{r['key']}.npz"):
                rows[r["key"]] = r
    y = np.load(f"{CACHE}/y.npy")
    season = np.load(f"{CACHE}/season.npy")
    tags = sorted({v["tag"] for v in rows.values()})
    P, Y = {}, {}
    for fold in FOLDS:
        Y[fold] = y[season == fold].astype(np.float64)
    for tag in tags:
        got = {}
        for fold in FOLDS:
            keys = [k for k, v in rows.items()
                    if v["tag"] == tag and v["fold"] == fold]
            if not keys:
                break
            keys = sorted(keys, key=lambda k: rows[k]["seed"])[:seeds]
            ps = [np.load(f"{PREDS}/{k}.npz")["p"] for k in keys]
            got[fold] = np.mean(ps, axis=0).astype(np.float64)
        if len(got) == len(FOLDS):
            P[tag] = got
    return P, Y


def z(p):
    """혼합 전 표준화. rho^2 는 아핀 불변이므로 각 판의 척도 차이를 없애도
    점수가 안 바뀐다 — 그래야 w 가 '어느 판을 얼마나' 의 의미가 된다."""
    return (p - p.mean()) / p.std()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--top", type=int, default=14)
    a = ap.parse_args()

    P, Y = load_all(a.seeds)
    if REF not in P:
        raise SystemExit(f"{REF} 캐시 없음")
    base = {f: rho2(Y[f], P[REF][f]) for f in FOLDS}
    print(f"기준 {REF}  rho^2 " + "  ".join(f"{f}:{base[f]:.2f}" for f in FOLDS)
          + f"   평균 {np.mean(list(base.values())):.2f}\n")

    rows = []
    for tag in P:
        if tag == REF:
            continue
        corr = np.mean([np.corrcoef(P[REF][f], P[tag][f])[0, 1] for f in FOLDS])
        # 고정 w 에서의 폴드별 델타
        fixed = {}
        for w in WGRID:
            d = [rho2(Y[f], (1 - w) * z(P[REF][f]) + w * z(P[tag][f])) - base[f]
                 for f in FOLDS]
            fixed[w] = d
        # 부호일치를 먼저, 그 다음 평균으로 고른다
        best_w = max(WGRID, key=lambda w: (sum(v > 0 for v in fixed[w]),
                                           np.mean(fixed[w])))
        d = fixed[best_w]
        rows.append({"tag": tag, "corr": corr, "w": best_w, "d": d,
                     "mean": float(np.mean(d)), "pos": sum(v > 0 for v in d)})

    rows.sort(key=lambda r: (-r["pos"], -r["mean"]))
    print("2모델 혼합 — 고정 w (폴드마다 다시 고르지 않는다)")
    hdr = (f"{'tag':16s} {'상관':>6s} {'w':>5s} {'평균':>8s} {'부호':>5s}   "
           + "  ".join(str(f).rjust(8) for f in FOLDS))
    print(hdr)
    print("-" * len(hdr))
    for r in rows[:a.top]:
        print(f"{r['tag']:16s} {r['corr']:6.3f} {r['w']:5.2f} {r['mean']:8.2f} "
              f"{r['pos']}/3   " + "  ".join(f"{v:+8.2f}" for v in r["d"]))

    # 3모델 균등 혼합 — 가중을 아예 고르지 않는 형태
    print("\n3모델 균등 혼합 (w 를 고르지 않는다: 기준 + 둘을 1/3씩)")
    cands = [r["tag"] for r in rows if r["pos"] >= 2][:8]
    tri = []
    for t1, t2 in itertools.combinations(cands, 2):
        d = [rho2(Y[f], (z(P[REF][f]) + z(P[t1][f]) + z(P[t2][f])) / 3) - base[f]
             for f in FOLDS]
        tri.append((float(np.mean(d)), sum(v > 0 for v in d), t1, t2, d))
    tri.sort(key=lambda r: (-r[1], -r[0]))
    for m, pos, t1, t2, d in tri[:8]:
        print(f"  {t1:14s} + {t2:14s} {m:8.2f} {pos}/3   "
              + "  ".join(f"{v:+8.2f}" for v in d))


if __name__ == "__main__":
    main()
