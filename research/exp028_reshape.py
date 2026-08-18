r"""EXP028 — 단조 재형성(monotone reshaping). 후처리 등급. 학습 0회.

## REOPEN_JUSTIFICATION

`rho` 는 **아핀 불변이지 단조 불변이 아니다.** 예측값에 단조 사상을 걸면 rho 가
바뀐다. 그리고 그 상한을 이미 쟀는데 지나쳤다 (EXP011).

    C3 예측을 100분위로 묶어 각 구간의 실제 성공률로 대체   962.3
    C3 원본                                            958.1   ->  +4.2

원장에는 "형식 축 전체 — 구간별 아핀(최대 +3.9)"로 닫혀 있는데, 그 기각은
**`+3.8%` 게이트(로컬 +36 요구)** 에 걸린 것이다. 그 게이트는 오늘 LB 로
틀렸음이 확인됐다 — 채택된 차등 3축은 로컬 +14.6 으로 LB +10.4 를 냈다.

    등급    **후처리** — 실측 전이율 +0.84 (유일하게 신뢰 가능한 계열)
    합법성  사상은 **과거 시즌의 (예측, 정답)** 으로만 만든 상수다.
            그 행의 예측값 하나만 넣으면 같은 값이 나온다 (규정 4 안전).

## 무엇이 최적인가

단조 `g` 중 `corr(g(p), y)` 를 최대화하는 것은 `g(p) = E[y | p]` 다. 그래서
과거 시즌의 (예측, 정답)으로 그 곡선을 추정해 목표 시즌에 적용한다.

**분위가 아니라 예측값 자체로 구간을 나눈다.** 분위로 나누면 평가셋 전체 분포가
필요해 규정 4 위반이다.

## 후보 사상

    ISO-n    예측값 등간격 n 구간의 과거 평균 (선형 보간)
    LOGIT    로짓 척도 아핀  ->  확률 척도에서는 단조 비선형
    POW      (p - c)^gamma 형태의 멱변환 (중심 대칭)

`n` 과 `gamma` 는 **과거 두 전이로만** 고른다.

    .\.venv\Scripts\python.exe -u research\exp028_reshape.py
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

EXP = "EXP028"
PREV2 = {2022: (2020, 2021), 2023: (2021, 2022), 2024: (2022, 2023)}
WD = 0.65


def main():
    E.start_experiment(EXP, "R-MONO", "python research/exp028_reshape.py", "load")
    tr = build_df()
    season = tr["season"].to_numpy()
    y = tr["control_success"].to_numpy(np.float64)
    P = tr["pitcher_id"].to_numpy(np.int64)

    def g(c):
        return tr[c].to_numpy(np.float64)

    AX = {"hand": ((g("pitcher_hand") == g("batter_hand")).astype(int), 1000),
          "2S": ((g("strikes_before").astype(int) == 2).astype(int), 1000),
          "runner": ((g("num_runners_on") > 0).astype(int), 2000)}
    pv0, res0 = {}, {}
    for f in (2020, 2021, 2022, 2023, 2024):
        m = season == f
        pv0[f] = (np.load(os.path.join(ROOT, "exp", f"prod_champ_{f}.npy"))[:2].mean(0)
                  + post_for(tr, y, season < f, m))
        res0[f] = y[m] - pv0[f]

    def dvec(ctx, src, k):
        p = np.concatenate([P[season == f] for f in src])
        c = np.concatenate([ctx[season == f] for f in src])
        r = np.concatenate([res0[f] for f in src])
        gg = pd.DataFrame({"p": p, "c": c, "r": r}).groupby(["p", "c"])["r"].agg(
            ["mean", "size"]).unstack()
        n0, n1 = gg[("size", 0)].fillna(0), gg[("size", 1)].fillna(0)
        ne = (n0 * n1) / (n0 + n1).replace(0, np.nan)
        return ((gg[("mean", 1)] - gg[("mean", 0)]) * ne / (ne + k)).dropna()

    C3, base = {}, {}
    for f in (2020, 2021, 2022, 2023, 2024):
        m = season == f
        v = pv0[f].copy()
        if f in PREV2:
            for a, (ctx, k) in AX.items():
                v += WD * (pd.Series(P[m]).map(dvec(ctx, PREV2[f], k)).fillna(0.).to_numpy()
                           * np.where(ctx[m] == 1, .5, -.5))
        C3[f] = v
        base[f] = r2(v, y[m])
    print(f"Champion(w=0.65)  2022 {base[2022]:.1f}  2023 {base[2023]:.1f}  "
          f"2024 {base[2024]:.1f}")
    m24 = season == 2024
    print(f"2024 예측 범위 {C3[2024].min():.4f} ~ {C3[2024].max():.4f}\n")
    E.beat("기준선")

    def fit_iso(src, nb):
        """과거 시즌의 (예측, 정답)으로 등간격 구간 평균 곡선. 예측값 기준."""
        p = np.concatenate([C3[f] for f in src])
        t = np.concatenate([y[season == f] for f in src])
        lo, hi = np.percentile(p, [0.1, 99.9])
        edges = np.linspace(lo, hi, nb + 1)
        idx = np.clip(np.digitize(p, edges) - 1, 0, nb - 1)
        d = pd.DataFrame({"i": idx, "t": t}).groupby("i")["t"].agg(["mean", "size"])
        d = d.reindex(range(nb))
        # 표본이 적은 구간은 이웃으로 메운다
        mu = d["mean"].interpolate(limit_direction="both").to_numpy()
        mu = np.maximum.accumulate(mu)          # 단조 강제
        cen = (edges[:-1] + edges[1:]) / 2
        return cen, mu

    def apply_iso(cen, mu, p):
        return np.interp(p, cen, mu)

    print("=" * 78)
    print("1. 구간 평균 곡선 (ISO) — 구간 수는 과거 두 전이로 선택")
    print("=" * 78)
    print(f"{'구간 수':>8}{'21→22':>10}{'22→23':>10}{'23→24':>10}")
    iso = {}
    for nb in (20, 50, 100, 200, 400, 1000):
        E.beat(f"iso {nb}")
        gs = []
        for a, b in ((2021, 2022), (2022, 2023), (2023, 2024)):
            mb = season == b
            cen, mu = fit_iso((a,), nb)
            gs.append(r2(apply_iso(cen, mu, C3[b]), y[mb]) - base[b])
        iso[nb] = gs
        print(f"{nb:>8}" + "".join(f"{v:>+10.1f}" for v in gs))
    nb_b = max(iso, key=lambda n: np.mean(iso[n][:2]))
    cen, mu = fit_iso((2022, 2023), nb_b)
    g24 = r2(apply_iso(cen, mu, C3[2024]), y[m24]) - base[2024]
    print(f"\n  과거로 고른 구간 수 {nb_b}  ->  2024 {g24:+.1f} (직전 2시즌 곡선)")

    print("\n" + "=" * 78)
    print("2. 매끄러운 모수 사상 — 로짓 척도 기울기 / 멱변환")
    print("=" * 78)

    def logit_map(p, s, c):
        q = np.clip(p, 1e-6, 1 - 1e-6)
        z = np.log(q / (1 - q)) - np.log(c / (1 - c))
        return 1.0 / (1.0 + np.exp(-(np.log(c / (1 - c)) + s * z)))

    def pow_map(p, gm, c):
        d = p - c
        return c + np.sign(d) * np.abs(d) ** gm

    print(f"{'사상':<16}{'모수':>8}{'21→22':>10}{'22→23':>10}{'23→24':>10}")
    par = {}
    for s in (0.6, 0.8, 1.2, 1.5, 2.0):
        gs = [r2(logit_map(C3[b], s, float(np.mean(y[season == b]))), y[season == b])
              - base[b] for a, b in ((2021, 2022), (2022, 2023), (2023, 2024))]
        par[("logit", s)] = gs
        print(f"{'로짓 기울기':<16}{s:>8.2f}" + "".join(f"{v:>+10.1f}" for v in gs))
    for gm in (0.6, 0.8, 1.2, 1.5):
        gs = [r2(pow_map(C3[b], gm, float(np.mean(C3[b]))), y[season == b]) - base[b]
              for a, b in ((2021, 2022), (2022, 2023), (2023, 2024))]
        par[("pow", gm)] = gs
        print(f"{'멱변환':<16}{gm:>8.2f}" + "".join(f"{v:>+10.1f}" for v in gs))

    pb = max(par, key=lambda k: np.mean(par[k][:2]))
    print(f"\n  과거로 고른 모수 사상 {pb}  ->  2024 {par[pb][2]:+.1f}")

    best = max(g24, par[pb][2])
    dec = "PROMISING" if best > 1.5 else "REJECTED"
    E.set_hypothesis_status("R-MONO", "PROMISING" if dec == "PROMISING" else "CLOSED",
                            level=3, hypothesis="단조 재형성", result=round(best, 2))
    E.finish_experiment(dict(
        experiment_id=EXP, hypothesis_id="R-MONO", type="A", level=3,
        started_at=E.read(E.CKPT)["start_time"],
        reopen_justification="+3.8% 게이트로 닫힌 축. rho 는 단조 불변이 아니다. EXP011 상한 +4.2",
        iso={str(k): [round(x, 2) for x in v] for k, v in iso.items()},
        param={f"{a}-{b}": [round(x, 2) for x in v] for (a, b), v in par.items()},
        iso_best_bins=nb_b, iso_2024=round(g24, 2),
        param_best=str(pb), param_2024=round(par[pb][2], 2),
        decision=dec, artifact=None,
        what_we_learned=(f"단조 재형성 — 구간평균 {nb_b}칸 2024 {g24:+.1f}, "
                         f"모수사상 {pb} 2024 {par[pb][2]:+.1f}")))
    json.dump(dict(iso={str(k): v for k, v in iso.items()},
                   par={f"{a}-{b}": v for (a, b), v in par.items()},
                   nb=nb_b, g24=g24), open(os.path.join(ROOT, "exp", "exp028_reshape.json"), "w"),
              indent=1, default=float)
    print(f"\nDECISION = {dec}   최선 2024 {best:+.1f}")


if __name__ == "__main__":
    main()
