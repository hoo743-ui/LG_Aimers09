r"""투수 x 타자손 축 최종 심층 — 남은 잔여분이 전이되는가. 학습 0회.

## 전제

이 축은 **이미 Champion 에 있다** — 후처리 편차 1번축 `dev_platoon`
(투수 x 타자손, 부모=투수, k=300, w=0.20), 2024 기여 +11.9.
따라서 아래 모든 수치는 **그 위에 남은 잔여분**이다.

## 왜 이 축만 남았나

경기 단위 분할로 다시 잰 오라클에서 유일하게 전이와 큰 격차가 남았다.

    투수            경기분할 +1.0    전이 +1.0     격차 없음
    투수 x 타자      경기분할 +4.0    전이 +3.9     격차 없음
    투수 x 타자손    경기분할 +20.7   전이 +6.9     **격차 +13.8**

## 이번에 보는 것

`2025 의 투수 x 타자손 현재상태`는 카운터에 손 분할이 없어 복원 불가다.
그렇다면 **과거 이력으로 추정 가능한 안정적 감도**가 있는가.

    d_p = (같은손 잔차평균) - (반대손 잔차평균)      투수별 손 차등

    1 표본 구조   투수-시즌-손별 표본, 커버리지, 신규 투수 cold start
    2 지속성      d_p 의 시즌 간 상관 (4개 전이)
    3 추정량 비교 RAW / 표준 EB / 전이인식(ρ x EB) 를 다음 시즌에 적용
    4 오라클      **경기 단위 분할**로만 (투구 단위 금지)

    .\.venv\Scripts\python.exe -u exp\hand_axis.py
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
from game_decomp import games                              # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

FOLDS = (2020, 2021, 2022, 2023, 2024)
TRANS = [(2020, 2021), (2021, 2022), (2022, 2023), (2023, 2024)]
KS = [0, 50, 150, 500, 2000, 10000, 50000]


def main():
    tr = build_df()
    season = tr["season"].to_numpy()
    y = tr["control_success"].to_numpy(np.float64)
    PID = tr["pitcher_id"].to_numpy(np.int64)
    g = lambda c: tr[c].to_numpy(np.float64)
    SAME = (g("pitcher_hand") == g("batter_hand")).astype(np.int64)
    pv, res, GID = {}, {}, {}
    for f in FOLDS:
        m = season == f
        P = np.load(os.path.join(ROOT, "exp", f"prod_champ_{f}.npy"))
        pv[f] = P[:2].mean(0) + post_for(tr, y, season < f, m)
        res[f] = y[m] - pv[f]
        GID[f] = games(PID[m], g("asof_pitcher_n")[m],
                       g("asof_pitcher_prev1_game_success_rate")[m],
                       g("asof_pitcher_prev1_game_middle_rate")[m])

    print("=" * 92)
    print("1. 표본 구조 — 투수 x 손")
    print("=" * 92)
    for f in FOLDS:
        m = season == f
        d = pd.DataFrame({"p": PID[m], "s": SAME[m]})
        c = d.groupby(["p", "s"]).size().unstack(fill_value=0)
        both = ((c.get(0, 0) >= 50) & (c.get(1, 0) >= 50)).sum()
        print(f"  {f}  투수 {len(c):>4}  양쪽 50투구 이상 {both:>4}"
              f" ({both/len(c):.0%})   같은손 비율 {SAME[m].mean():.1%}"
              f"   손별 중앙 투구 {c.replace(0, np.nan).median().mean():.0f}")
    msrc = np.isin(season, (2022, 2023))
    m24 = season == 2024
    seen = set(PID[msrc])
    cold = float(~pd.Series(PID[m24]).isin(seen).mean() if False
                 else np.mean(~np.isin(PID[m24], list(seen))))
    print(f"\n  cold start — 2024 행 중 학습 2시즌에 없던 투수 {cold:.1%}")
    d24 = pd.DataFrame({"p": PID[msrc], "s": SAME[msrc]}).groupby(["p", "s"]
                                                                  ).size().unstack(fill_value=0)
    ok_both = set(d24[(d24.get(0, 0) >= 50) & (d24.get(1, 0) >= 50)].index)
    print(f"  2024 행 중 학습 2시즌에 **양손 50투구 이상**인 투수 "
          f"{np.mean(np.isin(PID[m24], list(ok_both))):.1%}")

    print("\n" + "=" * 92)
    print("2. 손 차등 d_p = (같은손 잔차평균) − (반대손 잔차평균) 의 지속성")
    print("=" * 92)
    D = {}
    for f in FOLDS:
        m = season == f
        t = pd.DataFrame({"p": PID[m], "s": SAME[m], "r": res[f]})
        gg = t.groupby(["p", "s"])["r"].agg(["mean", "size"]).unstack()
        okm = (gg[("size", 0)] >= 50) & (gg[("size", 1)] >= 50)
        D[f] = pd.DataFrame({"d": gg[("mean", 1)] - gg[("mean", 0)],
                             "n": gg[("size", 0)] + gg[("size", 1)]})[okm]
    print(f"  {'전이':<14}{'공통 투수':>10}{'상관':>9}{'가중 회귀 기울기':>16}")
    rows = {}
    for a, b in TRANS:
        J = D[a].join(D[b], lsuffix="_a", rsuffix="_b", how="inner").dropna()
        if len(J) < 30:
            continue
        c = float(np.corrcoef(J["d_a"], J["d_b"])[0, 1])
        w = J["n_b"].to_numpy()
        beta = float(np.sum(w * J["d_a"] * J["d_b"]) / np.sum(w * J["d_a"] ** 2))
        rows[f"{a}->{b}"] = dict(n=len(J), corr=c, beta=beta)
        print(f"  {a}->{b:<9}{len(J):>10}{c:>+9.3f}{beta:>+16.3f}")
    print(f"\n  참고 — 투수 주효과 잔차의 시즌 간 상관은 +0.19(2023~2024), "
          f"pair 는 +0.02~+0.04 였다")

    print("\n" + "=" * 92)
    print("3. 경기 단위 분할 오라클 (폴드 2024) — 투구 단위 금지")
    print("=" * 92)
    y24, p24 = y[m24], pv[2024]
    base = r2(p24, y24)
    r24 = y24 - p24
    key = PID[m24] * 10 + SAME[m24]
    rng = np.random.default_rng(0)
    u, inv = np.unique(GID[2024], return_inverse=True)
    h = (rng.random(len(u)) < 0.5)[inv]
    best = -1e9
    for k in KS:
        add = np.zeros(len(y24))
        for m in (h, ~h):
            uu, tab, _ = cells(key[m], r24[m], k)
            add[~m] = look(uu, tab, key[~m])
        best = max(best, r2(p24 + add, y24) - base)
    print(f"  투수 x 손 (경기 분할)  {best:+.1f}     [기준선 {base:.1f}]")

    print("\n" + "=" * 92)
    print("4. 추정량 비교 — 다음 시즌에 실제 적용")
    print("=" * 92)
    print(f"{'전이':<12}{'RAW':>10}{'표준 EB':>10}{'전이인식':>10}{'과거 ρ̂':>9}")
    out = {"structure": {}, "persist": rows, "oracle_game": best}
    for a, b in TRANS:
        mb = season == b
        bs = r2(pv[b], y[mb])
        t = pd.DataFrame({"p": PID[season == a], "s": SAME[season == a],
                          "r": res[a]})
        gg = t.groupby(["p", "s"])["r"].agg(["mean", "size"])
        v = float(np.var(res[a]))
        m_ = gg["mean"].to_numpy()
        n_ = gg["size"].to_numpy()
        w = n_ / n_.sum()
        sb = max(float(np.sum(w * (m_ - np.sum(w * m_)) ** 2))
                 - float(np.sum(w * (v / n_))), 1e-9)
        k_eb = v / sb
        eb = gg["mean"] * gg["size"] / (gg["size"] + k_eb)
        # 과거 전이들로만 rho 적합
        num = den = 0.0
        for a2, b2 in TRANS:
            if b2 > a:
                continue
            t2 = pd.DataFrame({"p": PID[season == a2], "s": SAME[season == a2],
                               "r": res[a2]}).groupby(["p", "s"])["r"].agg(
                ["mean", "size"])
            v2 = float(np.var(res[a2]))
            m2, n2 = t2["mean"].to_numpy(), t2["size"].to_numpy()
            w2 = n2 / n2.sum()
            sb2 = max(float(np.sum(w2 * (m2 - np.sum(w2 * m2)) ** 2))
                      - float(np.sum(w2 * (v2 / n2))), 1e-9)
            e2 = t2["mean"] * t2["size"] / (t2["size"] + v2 / sb2)
            nx = pd.DataFrame({"p": PID[season == b2], "s": SAME[season == b2],
                               "r": res[b2]}).groupby(["p", "s"])["r"].agg(
                ["mean", "size"])
            J = pd.DataFrame({"e": e2}).join(nx, how="inner").dropna()
            num += float(np.sum(J["size"] * J["e"] * J["mean"]))
            den += float(np.sum(J["size"] * J["e"] ** 2))
        rho = num / den if den > 0 else None
        keyb = pd.MultiIndex.from_arrays([PID[mb], SAME[mb]])
        vals = {}
        for nm, s in (("RAW", gg["mean"]), ("EB", eb),
                      ("TA", eb * rho if rho is not None else None)):
            if s is None:
                vals[nm] = np.nan
                continue
            add = pd.Series(s).reindex(keyb).fillna(0.0).to_numpy()
            vals[nm] = r2(pv[b] + add, y[mb]) - bs
        out[f"{a}->{b}"] = dict(k_eb=k_eb, rho=rho, **vals)
        rt = f"{rho:+.3f}" if rho is not None else "-"
        print(f"  {a}->{b:<7}{vals['RAW']:>+10.1f}{vals['EB']:>+10.1f}"
              f"{vals['TA'] if not np.isnan(vals['TA']) else float('nan'):>+10.1f}"
              f"{rt:>9}")

    json.dump(out, io.open(os.path.join(ROOT, "exp", "hand_axis.json"), "w",
                           encoding="utf-8"), indent=1, ensure_ascii=False,
              default=float)
    print("\n  승격 조건: 경기분할 오라클 > +10 · 전이 양수 · 2024 OOS 양수 · "
          "기대 이득 >= +3.8%(약 +36점)")


if __name__ == "__main__":
    main()
