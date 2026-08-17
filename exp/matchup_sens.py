r"""매치업 감도(sensitivity) — pair 는 안 되지만 **기울기**는 되는가. 학습 0회.

## 왜 이 각도인가

pair 효과는 시즌 간 상관이 0 이었다(+0.017~+0.037). 이유는 표본이다 —
pair 당 중앙 7투구라 애초에 추정이 불가능하다.

그런데 **"그 투수가 어떤 타자 유형에 민감한가"** 는 다르다. 그 투수의 모든
타자를 모아 기울기 하나를 추정하므로 표본이 650투구다. 그리고 투수 수준 효과는
지속성이 확인돼 있다 (인접 시즌 감쇠보정 +0.774).

    pair 효과      θ(투수, 타자)        표본 7      지속 0
    감도(기울기)    b(투수) x trait(타자)  표본 650    ?  <- 이번에 잰다

`batter_id` 를 쓰지 않고 타자를 **저차원 형질**로만 표현한다. 형질은 전부
그 행의 공식 컬럼이다.

## 이미 닫힌 것과 무엇이 다른가

`투수상태 x 타자상태`(2024 −12.7)는 **전역** 곱 하나였다. 여기서는 투수마다
**서로 다른 기울기** 391개를 본다. 트리는 투수를 격리한 뒤 각자 기울기를
만들어야 하므로 원리적으로 비싸다.

## 절차

    1 조건 분해   pair 잔차가 어떤 조건에서 커지는가
    2 감도 추정   투수별 [잔차 ~ 타자형질] 기울기, 시즌마다
    3 지속성      corr(b_{p,s}, b_{p,s+1})  <- pair 상관(0)과 직접 비교
    4 오라클      그 시즌 안에서 교차적합했을 때의 이득
    5 전이        2022+2023 기울기를 2024 에 적용

    .\.venv\Scripts\python.exe -u exp\matchup_sens.py
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
from traj_probe import r2                                  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

NMIN = 200


def slopes(pid, x, r, nmin=NMIN):
    """투수별 단순회귀 기울기와 절편. x 는 중심화해서 넣는다."""
    d = pd.DataFrame({"p": pid, "x": x, "r": r}).dropna()
    g = d.groupby("p")
    n = g.size()
    mx = g["x"].mean()
    my = g["r"].mean()
    cov = g.apply(lambda t: np.mean((t["x"] - t["x"].mean())
                                    * (t["r"] - t["r"].mean())),
                  include_groups=False)
    var = g["x"].var(ddof=0)
    b = (cov / var.replace(0, np.nan))
    ok = (n >= nmin) & var.gt(1e-12)
    return pd.DataFrame({"b": b, "a": my, "n": n, "mx": mx})[ok]


def main():
    tr = build_df()
    season = tr["season"].to_numpy()
    y = tr["control_success"].to_numpy(np.float64)
    PID = tr["pitcher_id"].to_numpy(np.int64)
    BID = tr["batter_id"].to_numpy(np.int64)
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

    SAME = (g("pitcher_hand") == g("batter_hand")).astype(np.float64)
    TRAITS = {
        "타자 통산 성공률": g("asof_batter_success_rate"),
        "타자 통산 가운데율": g("asof_batter_middle_rate"),
        "타자 시즌 성공률(cur_bsucc)": g("cur_bsucc"),
        "같은손": SAME,
        "타자 경험(log n)": np.log1p(g("asof_batter_n")),
    }

    print("=" * 88)
    print("1. pair 잔차는 어떤 조건에서 커지는가 (폴드 2024)")
    print("=" * 88)
    pair24 = (PID * 100000 + BID)[m24]
    d = pd.DataFrame({"k": pair24, "r": res[2024]})
    pm = d.groupby("k")["r"].agg(["mean", "size"])
    big = pm[pm["size"] >= 10]
    print(f"  10투구 이상 pair {len(big):,}개, |잔차평균| 중앙 "
          f"{big['mean'].abs().median():.4f}")
    cond = {"같은손": SAME[m24],
            "F 경기": (tr["game_type"].to_numpy()[m24] == "F").astype(float),
            "2스트라이크": (g("strikes_before")[m24] == 2).astype(float),
            "주자 있음": (g("num_runners_on")[m24] > 0).astype(float)}
    print(f"  {'조건':<12}{'조건=1 pair 잔차 sd':>22}{'조건=0':>12}")
    for n, c in cond.items():
        s1 = pd.DataFrame({"k": pair24[c > 0.5], "r": res[2024][c > 0.5]}
                          ).groupby("k")["r"].mean().std()
        s0 = pd.DataFrame({"k": pair24[c <= 0.5], "r": res[2024][c <= 0.5]}
                          ).groupby("k")["r"].mean().std()
        print(f"  {n:<12}{s1:>22.4f}{s0:>12.4f}")

    print("\n" + "=" * 88)
    print("2~3. 투수별 감도 기울기와 그 지속성")
    print("=" * 88)
    print(f"  (투수당 {NMIN}투구 이상, 시즌별로 따로 적합)")
    print(f"{'타자 형질':<28}{'투수수':>7}{'기울기 sd':>11}"
          f"{'22~23':>9}{'23~24':>9}{'22~24':>9}")
    S = {}
    for name, x in TRAITS.items():
        row = {}
        for f in (2022, 2023, 2024):
            m = season == f
            row[f] = slopes(PID[m], x[m], res[f])
        S[name] = row
        cs = []
        for a, b in ((2022, 2023), (2023, 2024), (2022, 2024)):
            J = row[a].join(row[b], lsuffix="_a", rsuffix="_b", how="inner")
            J = J.replace([np.inf, -np.inf], np.nan).dropna(subset=["b_a", "b_b"])
            cs.append(float(np.corrcoef(J["b_a"], J["b_b"])[0, 1])
                      if len(J) > 30 else np.nan)
        print(f"  {name:<26}{len(row[2024]):>7}{row[2024]['b'].std():>11.4f}"
              + "".join(f"{c:>+9.3f}" for c in cs))
        out[f"persist|{name}"] = cs

    print("\n  참고 — pair 효과의 시즌 간 상관은 +0.017~+0.037 이었다")

    print("\n" + "=" * 88)
    print("4~5. 오라클(시즌 내 교차적합)과 전이(2022+2023 -> 2024)")
    print("=" * 88)
    rng = np.random.default_rng(0)
    half = rng.random(len(y24)) < 0.5
    print(f"{'타자 형질':<28}{'오라클':>9}{'전이 이득':>11}{'축소 배율':>10}")
    for name, x in TRAITS.items():
        x24 = x[m24]
        # 오라클 — 절반으로 기울기 적합, 반대쪽에 적용
        add = np.zeros(len(y24))
        for msk in (half, ~half):
            sl = slopes(PID[m24][msk], x24[msk], res[2024][msk], nmin=100)
            cen = sl["mx"]
            bb = pd.Series(PID[m24][~msk]).map(sl["b"]).to_numpy()
            cc = pd.Series(PID[m24][~msk]).map(cen).to_numpy()
            v = np.nan_to_num(bb) * (np.nan_to_num(x24[~msk],
                                                   nan=np.nanmean(x24))
                                     - np.nan_to_num(cc))
            add[~msk] = np.nan_to_num(v)
        orc = r2(p24 + add, y24) - base
        # 전이 — 학습 2시즌 기울기, 축소 배율은 과거에서 고른다
        msrc = np.isin(season, (2022, 2023))
        sl2 = slopes(PID[msrc], x[msrc], np.concatenate([res[2022], res[2023]]))
        bb = pd.Series(PID[m24]).map(sl2["b"]).to_numpy()
        cc = pd.Series(PID[m24]).map(sl2["mx"]).to_numpy()
        addt = np.nan_to_num(bb) * (np.nan_to_num(x24, nan=np.nanmean(x24))
                                    - np.nan_to_num(cc))
        best, bw = -1e9, None
        for w in (0.05, 0.1, 0.2, 0.4, 0.7, 1.0):
            v = r2(p24 + w * np.nan_to_num(addt), y24) - base
            if v > best:
                best, bw = v, w
        out[f"gain|{name}"] = dict(oracle=orc, transfer=best, w=bw)
        print(f"  {name:<26}{orc:>+9.1f}{best:>+11.1f}{bw:>10.2f}")
    print("\n  (전이 열의 축소 배율은 2024 에서 고른 값이라 **낙관적 상한**이다)")

    json.dump(out, io.open(os.path.join(ROOT, "exp", "matchup_sens.json"), "w",
                           encoding="utf-8"), indent=1, ensure_ascii=False,
              default=float)


if __name__ == "__main__":
    main()
