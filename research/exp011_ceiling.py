r"""EXP011 — LEVEL 4 정보 해상도 감사. 천장을 정보원별로 분해한다. 학습 0회.

## 왜 이 실험인가

지금까지 열 번의 실험이 전부 "없다"로 끝났다. 그러면 물어야 할 것이 바뀐다.

    "무엇을 더 넣을까"  ->  "애초에 얼마가 있고, 우리는 그 중 얼마를 가졌는가"

`rho` 는 아핀 불변이므로 최적 예측은 참 조건부확률 `p*` 이고

    rho_max = sd(p*) / sd(y)

다. `p*` 는 관측할 수 없지만 **정보원별 하한**은 관측할 수 있다. 어떤 변수
묶음으로 행을 그룹핑하고 그룹 평균의 분산에서 이항 잡음을 빼면, 그 묶음이
가진 신호 분산의 불편추정이 된다.

    sd_signal^2 = Var(그룹평균) - E[p(1-p)/n]

이 값을 정보원마다 계산하면 **천장의 분해도**가 나온다.

## 함께 확인하는 것

    1  현행 예측의 분산이 참 신호 분산에 비해 부족한가 (under-dispersion)
    2  ID 가 의미를 가지는가 (데뷔 순서 등 — 그렇다면 미사용 정보다)
    3  이산 국면만으로 얻을 수 있는 상한 vs 투수 정체성을 더했을 때

## 무엇을 반증하려는가

가설 H: "우리는 이미 가용 정보의 대부분을 가졌다."
반증되면(=천장이 현재보다 훨씬 높으면) 남은 것은 표현이 아니라 **미사용 정보**다.

    .\.venv\Scripts\python.exe -u research\exp011_ceiling.py
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
import engine as E                                          # noqa: E402
from path_alloc import build_df                             # noqa: E402
from resid_table import post_for                            # noqa: E402
from traj_probe import r2                                   # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

EXP = "EXP011"


def signal_sd(key, y, minn=2):
    """그룹 평균 분산에서 이항 잡음을 뺀 진짜 신호 sd (행 가중)."""
    d = pd.DataFrame({"k": key, "y": y}).groupby("k")["y"].agg(["mean", "size"])
    d = d[d["size"] >= minn]
    w = d["size"] / d["size"].sum()
    mu = float((w * d["mean"]).sum())
    vo = float((w * (d["mean"] - mu) ** 2).sum())
    vn = float((w * (d["mean"] * (1 - d["mean"]) / d["size"])).sum())
    return max(vo - vn, 0.0) ** .5, len(d), float(d["size"].mean())


def main():
    E.start_experiment(EXP, "L4-A", "python research/exp011_ceiling.py", "load")
    tr = build_df()
    season = tr["season"].to_numpy()
    y = tr["control_success"].to_numpy(np.float64)
    P = tr["pitcher_id"].to_numpy(np.int64)
    B = tr["batter_id"].to_numpy(np.int64)

    def g(c):
        return tr[c].to_numpy(np.float64)

    m24 = season == 2024
    AX = {"hand": ((g("pitcher_hand") == g("batter_hand")).astype(int), 1000),
          "2S": ((g("strikes_before").astype(int) == 2).astype(int), 1000),
          "runner": ((g("num_runners_on") > 0).astype(int), 2000)}
    res0 = {}
    for f in (2022, 2023):
        m = season == f
        res0[f] = y[m] - (np.load(os.path.join(ROOT, "exp", f"prod_champ_{f}.npy"))[:2].mean(0)
                          + post_for(tr, y, season < f, m))

    def dvec(ctx, k):
        p = np.concatenate([P[season == f] for f in (2022, 2023)])
        c = np.concatenate([ctx[season == f] for f in (2022, 2023)])
        r = np.concatenate([res0[f] for f in (2022, 2023)])
        gg = pd.DataFrame({"p": p, "c": c, "r": r}).groupby(["p", "c"])["r"].agg(
            ["mean", "size"]).unstack()
        n0, n1 = gg[("size", 0)].fillna(0), gg[("size", 1)].fillna(0)
        ne = (n0 * n1) / (n0 + n1).replace(0, np.nan)
        return ((gg[("mean", 1)] - gg[("mean", 0)]) * ne / (ne + k)).dropna()

    C3 = (np.load(os.path.join(ROOT, "exp", "prod_champ_2024.npy"))[:2].mean(0)
          + post_for(tr, y, season < 2024, m24))
    for a, (ctx, k) in AX.items():
        C3 += (pd.Series(P[m24]).map(dvec(ctx, k)).fillna(0.).to_numpy()
               * np.where(ctx[m24] == 1, .5, -.5))
    y24 = y[m24]
    sdy = y24.std()
    rho = float(np.corrcoef(C3, y24)[0, 1])
    print(f"2024  n={len(y24):,}  기저율 {y24.mean():.4f}  sd(y)={sdy:.4f}")
    print(f"C3    rho={rho:.5f}  점수 {1e5*rho**2:.1f}  sd(예측)={C3.std():.5f}\n")

    print("=" * 92)
    print("1. under-dispersion — 참 신호는 우리 예측보다 얼마나 넓게 퍼져 있는가")
    print("=" * 92)
    lb_sd = rho * sdy
    print(f"  sd(p*) >= rho * sd(y) = {lb_sd:.5f}      (예측이 완벽하면 등호)")
    print(f"  sd(C3)                 = {C3.std():.5f}")
    print(f"  비 sd(C3)/하한         = {C3.std() / lb_sd:.3f}"
          f"   <1 이면 예측이 과소산포")
    for tag, sc in (("현 Champion", 1057.34), ("10위", 1157.96), ("1위권", 1290.0)):
        r_ = (sc / 1e5) ** .5
        print(f"  {tag:<12} LB {sc:>7.1f} -> rho {r_:.5f} -> 필요한 sd(p*) "
              f">= {r_ * sdy:.5f}   (현재의 {r_ * sdy / C3.std():.2f}배)")

    print("\n" + "=" * 92)
    print("2. 정보원별 신호 분산 — 이항 잡음을 뺀 진짜 성분 (2024)")
    print("=" * 92)
    cnt = g("balls_before").astype(int) * 3 + g("strikes_before").astype(int)
    disc = (cnt[m24] * 1000 + g("outs_before").astype(int)[m24] * 100
            + g("num_runners_on").astype(int)[m24] * 10
            + (g("pitcher_hand") == g("batter_hand")).astype(int)[m24])
    SRC = {
        "카운트만 (12셀)": cnt[m24],
        "이산 국면 전체 (카운트x아웃x주자x손)": disc,
        "투수 정체성": P[m24],
        "투수 x 이산 국면": P[m24] * 10000 + disc,
        "투수 x 타자": P[m24] * 100000 + B[m24],
        "C3 예측값 (100분위)": pd.qcut(C3, 100, labels=False, duplicates="drop"),
    }
    print(f"  {'정보원':<38}{'셀':>9}{'셀당':>8}{'신호 sd':>10}{'rho 상한':>10}"
          f"{'점수 상한':>11}")
    out = {}
    for nm, key in SRC.items():
        E.beat(nm)
        s, nc, per = signal_sd(np.asarray(key), y24)
        rm = s / sdy
        out[nm] = dict(sd=s, rho=rm, score=1e5 * rm ** 2, cells=nc)
        print(f"  {nm:<38}{nc:>9,}{per:>8.1f}{s:>10.5f}{rm:>10.5f}"
              f"{1e5*rm**2:>11.1f}")
    print("\n  주의 — '투수 x 타자' 같은 희소 셀의 상한은 그 시즌 안에서만 성립한다."
          "\n  시즌을 건너가는 몫은 별도이고 실측 +3.9 였다.")

    print("\n" + "=" * 92)
    print("3. ID 가 의미를 가지는가 — 미사용 정보인지")
    print("=" * 92)
    first = pd.DataFrame({"p": P, "s": season}).groupby("p")["s"].min()
    n_at = pd.DataFrame({"p": P, "n": g("asof_pitcher_n")}).groupby("p")["n"].min()
    idx = first.index.to_numpy(np.float64)
    print(f"  corr(pitcher_id, 첫 관측 시즌)      {np.corrcoef(idx, first.to_numpy())[0,1]:+.4f}")
    print(f"  corr(pitcher_id, 최초 asof_n)      "
          f"{np.corrcoef(idx, np.log1p(n_at.to_numpy()))[0,1]:+.4f}")
    fb = pd.DataFrame({"b": B, "s": season}).groupby("b")["s"].min()
    print(f"  corr(batter_id, 첫 관측 시즌)       "
          f"{np.corrcoef(fb.index.to_numpy(np.float64), fb.to_numpy())[0,1]:+.4f}")
    r24 = y24 - C3
    for nm, v in (("pitcher_id", P[m24].astype(float)),
                  ("batter_id", B[m24].astype(float))):
        print(f"  corr({nm}, C3 잔차)              "
              f"{np.corrcoef(v, r24)[0,1]:+.4f}   (잡음 1SE {1/np.sqrt(len(r24)):.4f})")

    ceil_disc = out["이산 국면 전체 (카운트x아웃x주자x손)"]["score"]
    ceil_pxd = out["투수 x 이산 국면"]["score"]
    print("\n" + "=" * 92)
    print("해석")
    print("=" * 92)
    print(f"  이산 국면만의 상한        {ceil_disc:>8.1f}")
    print(f"  투수 x 이산 국면 상한     {ceil_pxd:>8.1f}   (시즌 내 기준)")
    print(f"  현재 C3                {1e5*rho**2:>8.1f}")
    dec = "REJECTED"
    E.set_hypothesis_status("L4-A", "MEASURED", level=4,
                            hypothesis="정보 천장의 정보원별 분해", result=round(ceil_pxd, 1))
    E.finish_experiment(dict(
        experiment_id=EXP, hypothesis_id="L4-A", type="AUDIT", level=4,
        started_at=E.read(E.CKPT)["start_time"],
        rho=round(rho, 5), sd_pred=round(float(C3.std()), 5),
        sd_star_lower=round(float(lb_sd), 5),
        sources={k: {kk: round(vv, 5) for kk, vv in v.items()} for k, v in out.items()},
        decision=dec, artifact=None,
        what_we_learned=(f"C3 예측 sd {C3.std():.5f} vs 참 신호 하한 {lb_sd:.5f} "
                         f"(비 {C3.std()/lb_sd:.3f}). 이산 국면 상한 {ceil_disc:.1f}, "
                         f"투수x이산 국면 시즌내 상한 {ceil_pxd:.1f}")))
    json.dump(out, open(os.path.join(ROOT, "exp", "exp011_ceiling.json"), "w"),
              indent=1, default=float)


if __name__ == "__main__":
    main()
