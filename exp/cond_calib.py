r"""조건부(그룹별) 아핀 보정. 재학습 0회.

## 왜 이건 다른 축인가

지금까지 보정은 항상 **전역** 아핀이었다 — `p' = A p + B`. 그건 `rho` 를 바꾸지
못한다 (4-25: 아핀 최적 점수 = `1e5 * rho^2`, A/B 무관). 즉 지금까지의 보정 작업은
전부 "이미 있는 `rho` 를 다 쓰는" 일이었고 `rho` 자체는 건드린 적이 없다.

**그룹별로 다른 아핀**은 다르다.

    p' = A_g * p + B_g        g = 카운트 / 이닝 / 손 조합 / 경력 구간 ...

이건 `p` 와 `g` 의 상호작용을 넣는 것이라 **`rho` 를 올릴 수 있다.** 트리가 이미
`g` 를 피처로 갖고 있지만, 규제를 세게 건 상태(l2=100, border=32, depth6)에서는
"이 그룹에서는 예측을 더 벌려라" 같은 곱셈 형태를 표현하기 어렵다 — 4-1 과
`prep_shrink.py` 가 같은 이유로 축소값을 **직접** 만들어 준 것과 같은 논리다.

상수 `A_g`,`B_g` 는 학습 시점에 정해져 pkl 에 담기고 추론은 행 독립이다. 5) 안전.

## 판정은 워크포워드로만

폴드 안에서 맞추면 파라미터가 늘어난 만큼 **반드시** 오른다. 4-26 이 정확히 거기서
죽었다 (폴드 내 +33.76, 3/3 -> 워크포워드 -88.89). 계수는 이전 폴드에서만 맞춘다.

    .\.venv\Scripts\python.exe exp\cond_calib.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import level_probe as L                                   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "exp", "cache")
FOLDS = [2021, 2022, 2024]
MIN_CELL = 2000          # 이보다 작은 칸은 전역 계수로 되돌린다


def bucket(v, edges):
    return np.digitize(v, edges)


def build_groups(X, ix, mask):
    def c(name):
        return np.asarray(X[mask, ix[name]], dtype=np.float64)
    b, s = c("balls_before"), c("strikes_before")
    n = np.where(np.isnan(c("asof_pitcher_n")), 0.0, c("asof_pitcher_n"))
    g = {
        "balls": b,
        "strikes": s,
        "count(b,s)": b * 4 + s,
        "outs": c("outs_before"),
        "inning(1-3/4-6/7+)": bucket(c("inning"), [3.5, 6.5]),
        "month": c("game_month"),
        "pit_hand": c("pitcher_hand"),
        "hand 조합": c("pitcher_hand") * 8 + c("batter_hand"),
        "주자상황": c("num_runners_on"),
        "투수경력 5분위": bucket(n, np.nanquantile(n, [.2, .4, .6, .8])),
    }
    return {k: v.astype(np.int64) for k, v in g.items()}


def fit_cond(p, y, g):
    """그룹별 (A_g, B_g). 표본이 적은 칸은 전역 계수로 되돌린다."""
    A0, B0 = np.polyfit(p, y, 1)
    out = {}
    for lev in np.unique(g):
        m = g == lev
        if m.sum() < MIN_CELL or p[m].std() < 1e-9:
            out[lev] = (A0, B0)
        else:
            a, b = np.polyfit(p[m], y[m], 1)
            out[lev] = (a, b)
    return out, (A0, B0)


def apply_cond(p, g, coef, glob):
    q = np.empty_like(p)
    for lev in np.unique(g):
        m = g == lev
        a, b = coef.get(lev, glob)
        q[m] = a * p[m] + b
    return q


def rho2(y, p):
    if p.std() < 1e-12:
        return 0.0
    return float(1e5 * np.corrcoef(p, y)[0, 1] ** 2)


def main():
    meta = json.load(open(f"{CACHE}/cols.json"))
    ix = {c: i for i, c in enumerate(meta["cols"])}
    X = np.load(f"{CACHE}/X.npy", mmap_mode="r")
    y_all = np.load(f"{CACHE}/y.npy")
    season = np.load(f"{CACHE}/season.npy")

    D = {}
    for fold in FOLDS:
        mask = season == fold
        D[fold] = (L.model_preds(fold), y_all[mask].astype(np.float64),
                   build_groups(X, ix, mask))

    names = list(D[FOLDS[0]][2].keys())
    base = {f: rho2(D[f][1], D[f][0]) for f in FOLDS}
    print("조건부 아핀 — 폴드 내 적합은 파라미터가 늘어난 만큼 반드시 오른다.")
    print("판정은 워크포워드다 (4-26 이 여기서 죽었다).\n")
    print(f"기준 rho^2: " + "  ".join(f"{f}:{base[f]:.2f}" for f in FOLDS))
    print()
    hdr = (f"{'그룹':22s} {'칸':>4s} | " +
           "  ".join(f"{f} 내적합".rjust(11) for f in FOLDS) + " | " +
           "  ".join(f"{f} WF".rjust(10) for f in FOLDS[1:]) +
           f" {'WF평균':>9s} {'부호':>5s}")
    print(hdr)
    print("-" * len(hdr))

    for name in names:
        fits, ins, wf = {}, {}, {}
        for fold in FOLDS:
            p, y, g = D[fold]
            coef, glob = fit_cond(p, y, g[name])
            fits[fold] = (coef, glob)
            ins[fold] = rho2(y, apply_cond(p, g[name], coef, glob)) - base[fold]
        for i, fold in enumerate(FOLDS[1:], start=1):
            p, y, g = D[fold]
            # 이전 폴드들의 계수를 평균해서 쓴다 (postcal_rank 와 같은 규약)
            levs = set()
            for f in FOLDS[:i]:
                levs |= set(fits[f][0])
            coef = {}
            for lev in levs:
                vals = [fits[f][0][lev] for f in FOLDS[:i] if lev in fits[f][0]]
                coef[lev] = tuple(np.mean(vals, axis=0))
            glob = tuple(np.mean([fits[f][1] for f in FOLDS[:i]], axis=0))
            wf[fold] = rho2(y, apply_cond(p, g[name], coef, glob)) - base[fold]
        ncell = len(np.unique(D[FOLDS[0]][2][name]))
        wfm = float(np.mean(list(wf.values())))
        pos = sum(1 for v in wf.values() if v > 0)
        print(f"{name:22s} {ncell:4d} | "
              + "  ".join(f"{ins[f]:+11.2f}" for f in FOLDS) + " | "
              + "  ".join(f"{wf[f]:+10.2f}" for f in FOLDS[1:])
              + f" {wfm:+9.2f} {pos}/{len(wf)}")


if __name__ == "__main__":
    main()
