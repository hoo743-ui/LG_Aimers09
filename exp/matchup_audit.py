r"""(투수 x 타자) 매치업 감사 — 합법·과적합 없이 추정 가능한가. 학습 0회.

원시 매치업 비율을 바로 피처로 만들지 않는다. 순서는

    1 표본 구조   pair 당 관측수·시즌수·경험 비율 분산 vs 이항 잡음
    2 계층 분해   pair 효과 = 관측 - 투수 주효과 - 타자 주효과
    3 부분 풀링   축소 강도 k 를 **학습 이력에서 적률법으로 추정**한다 (고정값 금지)
    4 전이        2022->2023, 2023->2024 에서 pair 효과가 유지되는가

## 누수 규칙

표는 **학습 시즌 잔차로만** 만들고 목표 시즌 행에는 그 행의 (투수,타자) 키로
조회만 한다. 평가셋 다른 행·미래 매치업·전체 분포를 쓰지 않는다.

## 적률법 축소 (고정 k 를 안 쓰는 이유)

    Var_w(관측 그룹평균) = sigma_b^2 + E_w[v/n]      v = 잔차 분산 (약 0.25)
    -> sigma_b^2 = Var_w(평균) - E_w[v/n],   k = v / sigma_b^2

`k` 가 데이터에서 정해지므로 2024 를 보고 고르지 않는다.

    .\.venv\Scripts\python.exe -u exp\matchup_audit.py
"""
import io
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "exp"))
from path_alloc import build_df                            # noqa: E402
from resid_table import post_for                           # noqa: E402
from traj_probe import cells, look, r2                     # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def eb_k(keys, vals):
    """적률법으로 축소 상수 k 와 성분 분산을 추정한다."""
    u, inv = np.unique(keys, return_inverse=True)
    n = np.bincount(inv).astype(np.float64)
    m = np.bincount(inv, weights=vals) / n
    v = float(np.var(vals))                       # 그룹 내 잔차 분산
    w = n / n.sum()
    var_m = float(np.sum(w * (m - np.sum(w * m)) ** 2))
    noise = float(np.sum(w * (v / n)))
    sb = max(var_m - noise, 1e-9)
    return v / sb, dict(var_obs=var_m, noise=noise, sigma_b2=sb,
                        cells=len(u), k=v / sb)


def shrunk(keys, vals, k):
    """n/(n+k) 축소 평균 표."""
    u, inv = np.unique(keys, return_inverse=True)
    n = np.bincount(inv).astype(np.float64)
    m = np.bincount(inv, weights=vals) / n
    return u, m * (n / (n + k))


def main():
    tr = build_df()
    season = tr["season"].to_numpy()
    y = tr["control_success"].to_numpy(np.float64)
    PID = tr["pitcher_id"].to_numpy(np.int64)
    BID = tr["batter_id"].to_numpy(np.int64)
    PAIR = PID * 100000 + BID
    g = lambda c: tr[c].to_numpy(np.float64)
    pv, res = {}, {}
    for f in (2022, 2023, 2024):
        m = season == f
        P = np.load(os.path.join(ROOT, "exp", f"prod_champ_{f}.npy"))
        pv[f] = P[:3].mean(0) + post_for(tr, y, season < f, m)
        res[f] = y[m] - pv[f]
    m24 = season == 2024
    y24, p24 = y[m24], pv[2024]
    base = r2(p24, y24)
    out = {"base": base}

    print("=" * 88)
    print("1. 매치업 표본 구조")
    print("=" * 88)
    for f in (2022, 2023, 2024):
        m = season == f
        u, c = np.unique(PAIR[m], return_counts=True)
        print(f"  {f}  pair {len(u):>7,}  투구/pair 중앙 {np.median(c):>2.0f}"
              f"  평균 {c.mean():>4.1f}  90분위 {np.percentile(c,90):>3.0f}"
              f"  최대 {c.max():>3}   1~3투구 pair 비율 {np.mean(c<=3):.1%}")
    msrc = np.isin(season, (2022, 2023))
    src_pair = PAIR[msrc]
    rs = np.concatenate([res[2022], res[2023]])
    u_s, c_s = np.unique(src_pair, return_counts=True)
    ns = pd.Series(c_s, index=u_s)
    seasons_per_pair = pd.DataFrame({"p": src_pair, "s": season[msrc]}
                                    ).groupby("p")["s"].nunique()
    print(f"\n  학습 2시즌 pair {len(u_s):,}개, 두 시즌 모두 등장 "
          f"{(seasons_per_pair == 2).mean():.1%}")
    cov = float(np.isin(PAIR[m24], u_s).mean())
    print(f"  2024 행 커버리지 {cov:.1%}"
          f"   (그중 학습 표본 10투구 이상인 pair 비율 "
          f"{float(np.isin(PAIR[m24], ns[ns>=10].index).mean()):.1%})")
    # 잡음 대비 관측 분산
    kk, info = eb_k(src_pair, rs)
    print(f"\n  적률법: 관측 pair 평균 분산 {info['var_obs']:.5f}"
          f"  = 잡음 {info['noise']:.5f} + 신호 {info['sigma_b2']:.5f}"
          f"   -> 신호비 {info['sigma_b2']/info['var_obs']:.3f},  k = {kk:.0f}")
    out["eb_pair_raw"] = info

    print("\n" + "=" * 88)
    print("2. 계층 분해 — 투수/타자 주효과를 걷어낸 pair 잔차")
    print("=" * 88)
    kp, ip = eb_k(PID[msrc], rs)
    up, tp = shrunk(PID[msrc], rs, kp)
    r1 = rs - look(up, tp, PID[msrc])
    kb, ib = eb_k(BID[msrc], r1)
    ub, tb = shrunk(BID[msrc], r1, kb)
    r2_ = r1 - look(ub, tb, BID[msrc])
    kx, ix = eb_k(src_pair, r2_)
    print(f"  투수 주효과   k={kp:>7.0f}  신호비 {ip['sigma_b2']/ip['var_obs']:.3f}"
          f"  셀 {ip['cells']:,}")
    print(f"  타자 주효과   k={kb:>7.0f}  신호비 {ib['sigma_b2']/ib['var_obs']:.3f}"
          f"  셀 {ib['cells']:,}")
    print(f"  pair 잔차     k={kx:>7.0f}  신호비 {ix['sigma_b2']/ix['var_obs']:.3f}"
          f"  셀 {ix['cells']:,}")
    out.update(eb_pitcher=ip, eb_batter=ib, eb_pair_resid=ix)

    print("\n" + "=" * 88)
    print("3. pair 효과가 시즌을 건너 유지되는가 (직접 상관)")
    print("=" * 88)
    T = {}
    for f in (2022, 2023, 2024):
        m = season == f
        T[f] = pd.DataFrame({"p": PAIR[m], "r": res[f]}).groupby("p")["r"].agg(
            ["mean", "size"])
    print(f"  {'시즌쌍':<14}{'nmin':>6}{'공통 pair':>10}{'상관':>9}")
    for a, b in ((2022, 2023), (2023, 2024), (2022, 2024)):
        for nmin in (5, 10):
            J = T[a].join(T[b], lsuffix="_a", rsuffix="_b", how="inner")
            J = J[(J["size_a"] >= nmin) & (J["size_b"] >= nmin)]
            if len(J) < 30:
                continue
            c = float(np.corrcoef(J["mean_a"], J["mean_b"])[0, 1])
            print(f"  {a}~{b:<9}{nmin:>6}{len(J):>10,}{c:>+9.3f}")
            out[f"pair_cross_{a}_{b}_n{nmin}"] = dict(n=len(J), corr=c)

    print("\n" + "=" * 88)
    print("4. A/B/C 비교 — 2024 이득 (표는 학습 시즌으로만 제작)")
    print("=" * 88)
    pair24 = PAIR[m24]
    print(f"{'구성':<44}{'2024 이득':>11}")
    # A raw pair, 적률법 k
    u, t = shrunk(src_pair, rs, kk)
    a_gain = r2(p24 + look(u, t, pair24), y24) - base
    print(f"  A  원시 pair (적률 k={kk:.0f})".ljust(44) + f"{a_gain:>+11.1f}")
    # B 계층 분해 후 pair 잔차만
    ux, tx = shrunk(src_pair, r2_, kx)
    b_gain = r2(p24 + look(ux, tx, pair24), y24) - base
    print(f"  B  주효과 제거 후 pair 잔차 (적률 k={kx:.0f})".ljust(44)
          + f"{b_gain:>+11.1f}")
    # C 전체 계층 (투수 + 타자 + pair 잔차)
    full = (look(up, tp, PID[m24]) + look(ub, tb, BID[m24])
            + look(ux, tx, pair24))
    c_gain = r2(p24 + full, y24) - base
    print(f"  C  투수 + 타자 + pair 잔차 (전부 적률 k)".ljust(44)
          + f"{c_gain:>+11.1f}")
    only_main = look(up, tp, PID[m24]) + look(ub, tb, BID[m24])
    print(f"     참고: 주효과만 (투수+타자)".ljust(44)
          + f"{r2(p24 + only_main, y24) - base:>+11.1f}")
    out.update(A=a_gain, B=b_gain, C=c_gain)

    print("\n  --- 축소를 세게/약하게 흔들었을 때 (민감도, 2024 를 보고 고르지 않음) ---")
    for mult in (0.5, 1.0, 2.0, 5.0):
        ux2, tx2 = shrunk(src_pair, r2_, kx * mult)
        v = r2(p24 + look(ux2, tx2, pair24), y24) - base
        print(f"    k x {mult:<4} = {kx*mult:>8.0f}   {v:>+7.1f}")

    json.dump(out, io.open(os.path.join(ROOT, "exp", "matchup_audit.json"), "w",
                           encoding="utf-8"), indent=1, ensure_ascii=False,
              default=float)
    print("\n판정 — 게이트는 +3.8% (약 +36점). +1~+3 은 기록만, 0 근처면 종료.")


if __name__ == "__main__":
    main()
