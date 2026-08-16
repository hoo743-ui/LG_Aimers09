r"""후보를 Full / R-only / F-only 로 나눠 평가하고 2025 혼합을 추정한다.

## rho 는 그룹별로 분해되지 않는다

`rho` 를 그룹별로 재서 가중평균하면 틀린다. 전체 상관에는 **그룹 간(between)
성분**이 들어가기 때문이다. 2022 폴드 점수의 74% 가 바로 그 성분이었다.

올바른 재조립은 공분산 수준에서 한다.

    E[p]    = sum_g w_g * mp_g
    Var(p)  = sum_g w_g * (vp_g + (mp_g - E[p])^2)
    Cov     = sum_g w_g * (cov_g + (mp_g - E[p]) * (my_g - E[y]))

within 성분과 between 성분이 이렇게 갈리므로, F 비중 `w` 를 바꿔가며
2025 예상 점수를 낼 수 있다. **단 그룹 내부 통계와 그룹 평균이 그대로
전이된다는 가정**이 붙는다 — 2023/2024 가 같은 체제이므로 그 구간에서만 쓴다.
"""
import glob
import io
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "exp", "cache")


def stats(p, y):
    """한 그룹의 (비중용) 크기·평균·분산·공분산."""
    return dict(n=len(y), mp=p.mean(), my=y.mean(),
                vp=p.var(), vy=y.var(), cov=np.mean((p - p.mean()) * (y - y.mean())))


def mix(groups, weights):
    """그룹 통계와 비중으로 전체 rho^2 를 재조립한다."""
    w = np.array(weights, float); w = w / w.sum()
    mp = sum(wi * g["mp"] for wi, g in zip(w, groups))
    my = sum(wi * g["my"] for wi, g in zip(w, groups))
    vp = sum(wi * (g["vp"] + (g["mp"] - mp) ** 2) for wi, g in zip(w, groups))
    vy = sum(wi * (g["vy"] + (g["my"] - my) ** 2) for wi, g in zip(w, groups))
    cv = sum(wi * (g["cov"] + (g["mp"] - mp) * (g["my"] - my))
             for wi, g in zip(w, groups))
    return 1e5 * cv ** 2 / (vp * vy)


def main():
    y = np.load(os.path.join(CACHE, "y.npy")).astype(np.float64)
    season = np.load(os.path.join(CACHE, "season.npy"))
    meta = json.load(io.open(os.path.join(CACHE, "cols.json"), encoding="utf-8"))
    ixc = {c: i for i, c in enumerate(meta["cols"])}
    X = np.load(os.path.join(CACHE, "X.npy"), mmap_mode="r")
    isF_all = np.asarray(X[:, ixc["game_type"]]) == 0

    FOLDS = [int(v) for v in (sys.argv[1] if len(sys.argv) > 1
                              else "2021,2022,2023,2024").split(",")]
    names, res = [], {}
    for f in FOLDS:
        va = season == f
        yv, fv = y[va], isF_all[va]
        for path in sorted(glob.glob(os.path.join(ROOT, "exp", f"pred_*_{f}.npy"))):
            n = os.path.basename(path)[5:-(len(str(f)) + 5)]
            p = np.load(path)
            if len(p) != va.sum():
                continue
            if n not in names:
                names.append(n)
            r2 = lambda a, b: 1e5 * np.corrcoef(a, b)[0, 1] ** 2
            res[(n, f)] = dict(
                A=r2(p, yv), B=r2(p[~fv], yv[~fv]), C=r2(p[fv], yv[fv]),
                wF=fv.mean(), gR=stats(p[~fv], yv[~fv]), gF=stats(p[fv], yv[fv]))
    return FOLDS, names, res


if __name__ == "__main__":
    FOLDS, names, res = main()
    for lab, key in (("A  Full", "A"), ("B  R-only", "B"), ("C  F-only", "C")):
        print(f"\n=== {lab} ===")
        print(f"{'후보':<20}" + "".join(f"{f:>10}" for f in FOLDS))
        for n in names:
            print(f"{n:<20}" + "".join(
                (f"{res[(n, f)][key]:>10.1f}" if (n, f) in res else f"{'-':>10}")
                for f in FOLDS))
        base = names[0]
        print(f"  -- {base} 대비 증분 --")
        for n in names[1:]:
            print(f"{n:<20}" + "".join(
                (f"{res[(n, f)][key] - res[(base, f)][key]:>+10.1f}"
                 if (n, f) in res and (base, f) in res else f"{'-':>10}")
                for f in FOLDS))
    print(f"\n=== F 비중 ===")
    print(f"{'':<20}" + "".join(f"{res[(names[0], f)]['wF']:>10.1%}" for f in FOLDS
                                if (names[0], f) in res))
    print("\n=== 2025 혼합 추정 (2024 그룹 통계 고정, F 비중만 변화) ===")
    print(f"{'후보':<20}" + "".join(f"{w:>10.1%}" for w in (0.00, 0.095, 0.109, 0.123, 0.15)))
    for n in names:
        if (n, 2024) not in res:
            continue
        r = res[(n, 2024)]
        print(f"{n:<20}" + "".join(
            f"{mix([r['gR'], r['gF']], [1 - w, w]):>10.1f}"
            for w in (0.00, 0.095, 0.109, 0.123, 0.15)))
