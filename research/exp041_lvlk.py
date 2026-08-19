r"""EXP041 — 수준 축의 **축소 상수 k** 가 모양을 바꾸는가. 학습 0회.

## 왜 여기인가

38회차로 표 제작(원천 창·감쇠)은 두 번 다 LB 음수임이 확정됐고, 남은 것은
"이미 LB 로 검증된 축의 가중 조정"뿐이다. 그런데 가중 곡선 하나가 이상한 값을
가리킨다.

    타자 수준   gain(w) = 2.414(2*2.105*w - w^2)      b = 2.105

`b` 는 보정 벡터를 얼마로 곱해야 최적인가다. **b = 2.1 은 우리 보정이 참값의
절반도 안 된다는 뜻**이고, 그 원인 후보가 k 다.

    타자 축소   n_eff / (n_eff + 20000)
    타자 한 명의 시즌 투구수는 대략 1,000~3,000 이므로 그 인자는 0.05~0.13 이다

즉 k >> n 인 선형 영역이라 보정이 사실상 **평균이 아니라 잔차 합(= n x 평균)**에
비례한다. 우리는 "실력이 좋은 타자"가 아니라 **"많이 나온 타자"**를 크게 보정하고
있다. k 를 낮추면 벡터의 **모양 자체**가 바뀐다.

EXP033 의 "k 와 w 는 중복(RMS 맞추면 상관 >= 0.995)"은 좁은 k 구간의 관찰이었을
수 있다. 그것을 먼저 확인한다.

## 재는 것

k 마다 보정 벡터 c_k 를 만들고 **각자의 최적 w** 에서의 이득을 본다. 최적 w 에서의
이득은 `corr(c_k, r)^2 x var(r)` 에 비례하므로 스케일이 소거되고 **모양의 질**만
남는다. 선택은 하지 않는다 — 로컬 선택은 38회차까지 네 번 틀렸다. 여기서는
"LB 를 쓸 가치가 있는가"만 판정한다.

    .\.venv\Scripts\python.exe -u research\exp041_lvlk.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "exp"))
import build_asof as ba                                     # noqa: E402
from path_alloc import build_df                             # noqa: E402
from traj_probe import r2                                   # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

FOLDS = (2022, 2023, 2024)
KGRID = [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]
WGRID = np.arange(0.0, 8.01, 0.05)


def main():
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
    SAME = ((tr["pitcher_hand"].to_numpy() == tr["batter_hand"].to_numpy())).astype(int)
    TWO = (SS == 2).astype(int)
    AXC = [(SAME, 1000.0), (TWO, 1000.0), (OB, 2000.0)]
    MODEL = {f: np.load(os.path.join(ROOT, "exp", f"prod_champ_{f}.npy"))[:2].mean(0)
             for f in (2020, 2021, 2022, 2023, 2024)}

    DEV = {f: np.column_stack([
        ba.look(*ba.nested_dev(p[season < f], c[season < f], y[season < f], k),
                c[season == f])
        for (p, c), k in zip(AXD, ba.KSH)]) for f in (2020, 2021, 2022, 2023, 2024)}
    res = {f: y[season == f] - (MODEL[f] + DEV[f] @ ba.WPOST)
           for f in (2020, 2021, 2022, 2023, 2024)}

    def level(key, f, k):
        """직전 2시즌 잔차의 개체별 축소 평균을 f 시즌 행에 조회."""
        parts = [pd.DataFrame({"k": key[season == s], "sr": res[s],
                               "n": np.ones(int((season == s).sum()))})
                 for s in (f - 2, f - 1)]
        q = pd.concat(parts).groupby("k")[["sr", "n"]].sum()
        t = (q["sr"] / q["n"]) * q["n"] / (q["n"] + k)
        return pd.Series(key[season == f]).map(t).fillna(0.).to_numpy()

    def contrast(f, ctx, k, w=0.65):
        parts = []
        for s in (f - 2, f - 1):
            m = season == s
            parts.append(pd.DataFrame({"k": P[m], "c": ctx[m], "sr": res[s],
                                       "n": np.ones(int(m.sum()))}))
        D = pd.concat(parts)
        q = D.groupby(["k", "c"])[["sr", "n"]].sum().unstack()
        n0, n1 = q[("n", 0)].fillna(0), q[("n", 1)].fillna(0)
        m0 = q[("sr", 0)] / n0.replace(0, np.nan)
        m1 = q[("sr", 1)] / n1.replace(0, np.nan)
        ne = (n0 * n1) / (n0 + n1).replace(0, np.nan)
        t = ((m1 - m0) * ne / (ne + k)).dropna()
        m = season == f
        return w * (pd.Series(P[m]).map(t).fillna(0.).to_numpy()
                    * np.where(ctx[m] == 1, .5, -.5))

    out = {}
    for f in FOLDS:
        m = season == f
        base = MODEL[f] + DEV[f] @ ba.WPOST
        for ctx, k0 in AXC:
            base = base + contrast(f, ctx, k0)
        base = base + 1.0 * level(P, f, 50000.0)          # 투수 수준 (현행)
        b0 = r2(base, y[m])
        row = {}
        ref = None
        for k in KGRID:
            c = level(B, f, float(k))
            g = np.array([r2(base + w * c, y[m]) for w in WGRID]) - b0
            i = int(np.argmax(g))
            sd = c.std()
            rr = float(np.corrcoef(c, ref)[0, 1]) if ref is not None else 1.0
            if k == 20000:
                ref20 = c
            row[k] = dict(best_w=float(WGRID[i]), best_gain=float(g[i]),
                          gain_at_25=float(g[int(round(2.5 / 0.05))]), sd=float(sd))
            if ref is None:
                ref = c
        # k=20000 기준 상관
        c20 = level(B, f, 20000.0)
        for k in KGRID:
            c = level(B, f, float(k))
            row[k]["corr_vs_20000"] = float(np.corrcoef(c, c20)[0, 1])
        out[f] = dict(base=float(b0), rows=row)
        print(f"\n=== 폴드 {f}   기준선 {b0:.1f}")
        print("   k        최적w   최적이득   w=2.5 이득   sd        corr(c,c_20000)")
        for k in KGRID:
            r = row[k]
            print(f"  {k:7d}   {r['best_w']:5.2f}   {r['best_gain']:+7.2f}   "
                  f"{r['gain_at_25']:+7.2f}      {r['sd']:.5f}   {r['corr_vs_20000']:.4f}")

    json.dump(out, open(os.path.join(ROOT, "exp", "exp041_lvlk.json"), "w"), indent=1)
    print("\n-> exp/exp041_lvlk.json")


if __name__ == "__main__":
    main()
