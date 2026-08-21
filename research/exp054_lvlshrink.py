r"""EXP054 — 🚩 수준 축의 가중이 **시즌 진행도**에 따라 변해야 하는가. 3폴드 검증.

## 발견 (EXP053, 폴드 2024)

    L_p x zs(cur_logn_pitch)   +3.18 @ w=-1.5   대조 바닥의 5배, 곡선이 매끄럽다

## 기제

`L_p` 는 **직전 2시즌** 잔차로 만든 투수 수준이다. `cur_n` 은 **이번 시즌** 투구
수다. 음수 가중은 "이번 시즌 표본이 쌓일수록 과거 수준을 덜 싣는다"이고, 이는
베이즈 축소 그 자체다. 그런데 8개월간 수준 축은 **고정 가중**이었다 — 그 투수의
평가 시즌 표본이 5구든 2,000구든 같은 무게였다.

모델은 이 상호작용을 만들 수 없다. `L_p` 는 후처리 표라 모델이 보지 못한다.

## 판정

EXP053 은 폴드 2024 에서 골랐다 (후보 17 x 가중 14). §5-a 조건 3 위반이므로
**2022·2023 에서 독립 확인**한다. 3폴드 부호가 같아야 후보다.

    .\.venv\Scripts\python.exe -u research\exp054_lvlshrink.py
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "exp"))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import build_asof as ba                                          # noqa: E402

W9 = np.array([0.099765, 0.411532, 0.139671, 0.224472,
               0.703699, 0.775302, 0.775315, 1.984998, 2.105])
CACHE = os.path.join(ROOT, "exp", "cache", "exp054_comp_{f}.npz")


def components(fold):
    """폴드 `fold` 의 모델예측 + 9개 보정열. exp042 와 같은 절차."""
    p = CACHE.format(f=fold)
    if os.path.exists(p):
        z = np.load(p)
        return z["mm"], z["Cm"], z["y"], z["logn"]
    from path_alloc import build_df
    tr = build_df()
    season = tr["season"].to_numpy()
    y = tr["control_success"].to_numpy(np.float64)
    P = tr["pitcher_id"].to_numpy(np.int64)
    B = tr["batter_id"].to_numpy(np.int64)
    BH = tr["batter_hand"].to_numpy(np.int64)
    BB = tr["balls_before"].to_numpy(np.int64)
    SS = tr["strikes_before"].to_numpy(np.int64)
    OB = (tr["num_runners_on"].to_numpy(np.int64) > 0).astype(np.int64)
    PH = P * 10 + BH
    PHA = PH * 10 + (SS > BB).astype(np.int64)
    AXD = [(P, PH), (PH, PHA), (PHA, PHA * 100 + (BB * 4 + SS)), (PH, PH * 10 + OB)]
    SAME = (tr["pitcher_hand"].to_numpy() == tr["batter_hand"].to_numpy()).astype(int)
    AXC = [(SAME, 1000.), ((SS == 2).astype(int), 1000.), (OB, 2000.)]
    src_seasons = [fold - 2, fold - 1]
    MODEL = {f: np.load(os.path.join(ROOT, "exp", f"prod_champ_{f}.npy")).mean(0)
             for f in src_seasons + [fold]}
    D, res = {}, {}
    for f in src_seasons + [fold]:
        mtr = season < f
        D[f] = np.column_stack([
            ba.look(*ba.nested_dev(pp[mtr], cc[mtr], y[mtr], k), cc[season == f])
            for (pp, cc), k in zip(AXD, ba.KSH)])
        res[f] = y[season == f] - (MODEL[f] + D[f] @ ba.WPOST)
    m = season == fold
    cols = [D[fold][:, i] for i in range(4)]

    def frames(key, ctx=None):
        out = []
        for s in src_seasons:
            ms = season == s
            d = {"k": key[ms], "sr": res[s], "n": np.ones(int(ms.sum()))}
            if ctx is not None:
                d["c"] = ctx[ms]
            out.append(pd.DataFrame(d))
        return pd.concat(out)

    for ctx, k in AXC:
        q = frames(P, ctx).groupby(["k", "c"])[["sr", "n"]].sum().unstack()
        n0, n1 = q[("n", 0)].fillna(0), q[("n", 1)].fillna(0)
        m0, m1 = q[("sr", 0)] / n0.replace(0, np.nan), q[("sr", 1)] / n1.replace(0, np.nan)
        ne = (n0 * n1) / (n0 + n1).replace(0, np.nan)
        t = ((m1 - m0) * ne / (ne + k)).dropna()
        v = pd.Series(P[m]).map(t).fillna(0.).to_numpy()
        cols.append(v * np.where(ctx[m] == 1, .5, -.5))
    for key, k in ((P, 50000.), (B, 20000.)):
        q = frames(key).groupby("k")[["sr", "n"]].sum()
        t = (q["sr"] / q["n"]) * q["n"] / (q["n"] + k)
        cols.append(pd.Series(key[m]).map(t).fillna(0.).to_numpy())
    out = (MODEL[fold], np.column_stack(cols), y[m],
           tr["cur_logn_pitch"].to_numpy(float)[m])
    np.savez_compressed(p, mm=out[0], Cm=out[1], y=out[2], logn=out[3])
    return out


WS = (-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1)
print(f"{'폴드':>6s} {'기준':>9s}  " + "  ".join(f"{w:>+6g}" for w in WS) + "   최적")
res = {}
for fold in (2022, 2023, 2024):
    mm, Cm, y, logn = components(fold)
    cur = mm + Cm @ W9
    base = 1e5 * np.corrcoef(cur, y)[0, 1] ** 2
    Lp = Cm[:, 7]
    v = Lp * ((logn - logn.mean()) / (logn.std() + 1e-12))
    v = v / v.std() * Cm[:, 8].std()
    g = [1e5 * np.corrcoef(cur + w * v, y)[0, 1] ** 2 - base for w in WS]
    res[fold] = g
    b = max(range(len(WS)), key=lambda i: g[i])
    print(f"{fold:6d} {base:9.2f}  " + "  ".join(f"{x:+6.2f}" for x in g) +
          f"   {g[b]:+.2f}@{WS[b]:+g}")
print("\n3폴드 평균")
avg = np.mean([res[f] for f in (2022, 2023, 2024)], 0)
print("       " + " " * 10 + "  ".join(f"{x:+6.2f}" for x in avg))
b = int(np.argmax(avg))
print(f"\n공통 최적 w = {WS[b]:+g}   평균 이득 {avg[b]:+.2f}   "
      f"폴드별 {[round(res[f][b],2) for f in (2022,2023,2024)]}")
print(f"3폴드 부호: {sum(1 for f in (2022,2023,2024) if res[f][b] > 0)}/3")
