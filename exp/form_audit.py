r"""형식(formulation) 감사 — D 를 타깃 확률로 잇는 **방식** 자체에 손실이 있는가.

D 의 구성은 고정한다. 피처를 만들지 않고 하이퍼파라미터도 건드리지 않는다.
이미 닫힌 것(§30 objective · 비선형 재보정 · 다양성 · 행 게이팅 오라클 +0.7)은
반복하지 않고, **아직 시험된 적 없는 두 가지**의 상한만 잰다.

## 시험 1 — 조건부 아핀 (Q3·Q4 의 상한)

`rho` 는 전역 아핀에 불변이다. 그러나 **구간마다 다른 아핀**은 불변이 아니다.

    y ~ a_g + b_g * pred     구간 g 마다 따로 적합

`b_g` 가 구간마다 다르다면 "같은 예측값이 구간에 따라 다른 의미"라는 뜻이고,
그것이 곧 **상태 의존 확률 모형**(Q3)과 **조건부 예측함수**(Q4)가 회수할 수 있는
전부의 상한이다. 라벨로 적합하므로 진단 전용이고, 2024 를 반으로 갈라
교차적합해 과적합을 제거한다 (§30-b 에서 in-sample 재보정에 한 번 속았다).

## 시험 2 — 잠재 상태 회전 (Q2 의 상한)

10개 `cur_*` 를 독립 축으로 주는 것이 손실인가. 트리는 축평행이라 **대각 방향**을
못 만든다. 그래서 학습 시즌 잔차로 방향 `w` 를 적합하고 (라벨 사용은 학습
구간에서만), 그 사영이 **다음 시즌** 잔차와 상관되는지 본다.

    w = argmin || resid_{2022,2023} - cur_{2022,2023} w ||     -> 2024 에서 평가

이것은 §22 의 "상태 프로파일/기하" 와 다르다 — 새 피처를 만드는 것이 아니라
**기존 10축의 선형결합 한 개**가 잔차를 설명하는지만 본다.

    .\.venv\Scripts\python.exe -u exp\form_audit.py
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

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

CUR = ["cur_succ", "cur_mid", "cur_ball", "cur_rev", "cur_str",
       "cur_fb", "cur_bb", "cur_os", "cur_bsucc", "cur_bmid"]


def r2(p, y):
    return 1e5 * np.corrcoef(p, y)[0, 1] ** 2


def cond_affine(keys, pred, y, half, slope_only=False):
    """구간별 아핀을 교차적합으로 적용했을 때의 rho^2 증분과 기울기 산포."""
    base = r2(pred, y)
    out = np.zeros(len(y))
    slopes = {}
    for m in (half, ~half):
        for gval in np.unique(keys):
            tr_m = m & (keys == gval)
            te_m = (~m) & (keys == gval)
            if tr_m.sum() < 200 or te_m.sum() == 0:
                out[te_m] = pred[te_m]
                continue
            x, t = pred[tr_m], y[tr_m]
            b = np.cov(x, t)[0, 1] / max(np.var(x), 1e-12)
            a = t.mean() - b * x.mean()
            if slope_only:
                a = 0.0
            out[te_m] = a + b * pred[te_m]
            slopes.setdefault(gval, []).append(b)
    sl = {k: float(np.mean(v)) for k, v in slopes.items()}
    return r2(out, y) - base, sl


def main():
    tr = build_df()
    season = tr["season"].to_numpy()
    y = tr["control_success"].to_numpy(np.float64)
    pv, res = {}, {}
    for f in (2022, 2023, 2024):
        m = season == f
        P = np.load(os.path.join(ROOT, "exp", f"prod_champ_{f}.npy"))
        pv[f] = P[:3].mean(0) + post_for(tr, y, season < f, m)
        res[f] = y[m] - pv[f]
    m24 = season == 2024
    y24, pred = y[m24], pv[2024]
    base = r2(pred, y24)
    rng = np.random.default_rng(0)
    half = rng.random(int(m24.sum())) < 0.5
    g = lambda c: tr[c].to_numpy(np.float64)
    curn = np.expm1(g("cur_logn_pitch"))
    out = {}

    print(f"기준선 rho^2 = {base:.1f}   예측 범위 {pred.min():.3f}~{pred.max():.3f}")
    print()
    print("=" * 84)
    print("시험 1. 구간별 아핀의 상한 — 같은 예측값이 구간마다 다른 의미인가")
    print("=" * 84)

    def dec(x, q=10):
        v = x[m24] if len(x) == len(season) else x      # 이미 폴드로 자른 배열도 받는다
        qs = np.nanquantile(v, np.linspace(0, 1, q + 1)[1:-1])
        return np.searchsorted(qs, np.nan_to_num(v, nan=np.nanmedian(v)))

    GR = {
        "game_type (F/R)": (tr["game_type"].to_numpy()[m24] == "F").astype(int),
        "cur_n 십분위": dec(curn),
        "cur_n 4분위": dec(curn, 4),
        "같은손": (g("pitcher_hand") == g("batter_hand")).astype(int)[m24],
        "카운트 우위": (g("strikes_before") > g("balls_before")).astype(int)[m24],
        "2스트라이크": (g("strikes_before") == 2).astype(int)[m24],
        "cur_succ 십분위": dec(g("cur_succ")),
        "예측값 십분위": dec(pred),
        "투수 (391)": tr["pitcher_id"].to_numpy()[m24],
    }
    print(f"{'구간 축':<20}{'아핀 상한':>10}{'기울기만':>10}{'기울기 범위':>22}")
    for n, k in GR.items():
        v_af, sl = cond_affine(k, pred, y24, half)
        v_sl, _ = cond_affine(k, pred, y24, half, slope_only=True)
        vals = np.array(list(sl.values()))
        rng_s = (f"{vals.min():.2f} ~ {vals.max():.2f}" if len(vals) <= 20
                 else f"{np.percentile(vals,10):.2f} ~ {np.percentile(vals,90):.2f} (10-90%)")
        out[f"affine|{n}"] = dict(
            gain=float(v_af), gain_slope=float(v_sl),
            slopes={str(k): v for k, v in sl.items()} if len(sl) <= 20 else None)
        print(f"{n:<20}{v_af:>+10.1f}{v_sl:>+10.1f}{rng_s:>22}")

    print()
    print("  참고 — 전역 아핀은 rho 를 바꾸지 않는다. 위 이득은 전부 **이질성**의 몫이다.")

    print()
    print("=" * 84)
    print("시험 2. 잠재 상태 — 10개 cur_* 를 독립 축으로 주는 것이 손실인가")
    print("=" * 84)
    C = np.column_stack([g(c) for c in CUR])
    ok = np.isfinite(C).all(1)
    Ztr = C[ok & (season < 2024)]
    Z = (Ztr - Ztr.mean(0)) / np.maximum(Ztr.std(0), 1e-9)
    ev = np.linalg.svd(Z / np.sqrt(len(Z)), compute_uv=False) ** 2
    ev = ev / ev.sum()
    print(f"  10축 상관구조의 주성분 설명력: "
          + " ".join(f"{v:.2f}" for v in ev))
    print(f"  누적 1~3성분 {ev[:3].sum():.2f}  -> 잠재 차원이 1개면 0.9 이상이어야 한다")

    m_fit = ok & np.isin(season, (2022, 2023))
    Xf = C[m_fit]
    rf = np.concatenate([res[2022][ok[season == 2022]],
                         res[2023][ok[season == 2023]]])
    A = np.column_stack([Xf, np.ones(len(Xf))])
    w = np.linalg.lstsq(A, rf, rcond=None)[0]
    proj_all = np.column_stack([C, np.ones(len(C))]) @ w
    p24 = proj_all[m24]
    o24 = ok[m24]
    c_proj = float(np.corrcoef(p24[o24], res[2024][o24])[0, 1])
    se = 1 / np.sqrt(o24.sum())
    print(f"\n  학습시즌(2022+2023) 잔차로 적합한 방향의 2024 잔차상관 "
          f"{c_proj:+.4f} ({c_proj/se:+.1f}SE)")
    print(f"  개별 cur_* 의 2024 잔차상관 최대 |0.0060| (앞선 asof 감사)")
    print(f"  선형 상한 1e5*c^2 = {1e5*c_proj**2:+.1f}점")
    out["latent_proj"] = dict(corr=c_proj, se_mult=c_proj / se,
                              ceiling=1e5 * c_proj ** 2,
                              pca=[float(v) for v in ev])

    add = np.zeros(int(m24.sum()))
    q = np.nanquantile(p24[o24], np.linspace(0, 1, 11)[1:-1])
    keys = np.searchsorted(q, np.nan_to_num(p24, nan=np.nanmedian(p24[o24])))
    for m in (half, ~half):
        u, inv = np.unique(keys[m], return_inverse=True)
        n = np.bincount(inv, minlength=len(u))
        mu = np.bincount(inv, weights=res[2024][m], minlength=len(u)) / np.maximum(n, 1)
        ix = np.clip(np.searchsorted(u, keys[~m]), 0, len(u) - 1)
        add[~m] = np.where(u[ix] == keys[~m], mu[ix], 0.0)
    print(f"  사영 십분위 오라클(2024 라벨 교차적합) {r2(pred + add, y24) - base:+.1f}")

    json.dump(out, io.open(os.path.join(ROOT, "exp", "form_audit.json"), "w",
                           encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
