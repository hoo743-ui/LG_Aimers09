r"""ASOF 잔차 정보 감사 — **D 가 asof 의 정보를 얼마나 회수했는가.**

원시 `asof_*` 의 단순 상관이 낮다고 정보가 없다고 결론내지 않는다. 원시에서
D 가 설명하는 부분을 걷어낸 뒤 **남은 것**이 타깃과 관계있는지 본다.

## 분해가 대수적으로 정확하다

`asof_rate` 는 통산이고 D 는 그것을 시즌내 상태로 갈랐다. 정의상

    asof_n * asof_rate = prior_events + cur_n * cur_rate
    => asof_rate = w * prior_rate + (1 - w) * cur_rate ,  w = prior_n / asof_n

즉 원시 = **경력 성분(prior)** 과 **현재 시즌 성분(cur)** 의 볼록결합이고,
D 가 꺼낸 것은 뒤쪽이다. 그래서 "D 를 제거한 나머지"는 근사가 아니라
`prior_rate` 와 혼합비 `w` 로 **정확히** 특정된다.

    prior_rate = (asof_n * asof_rate - cur_n * cur_rate) / prior_n

두 경로로 잰다.

    대수 분해   prior_rate (경력 성분) 자체의 신호
    사영 잔차   원시를 D 열 전체로 회귀한 잔차 (학습 구간에서만 적합)

## 검증

폴드 2022 / 2023 / 2024 전부. **계수는 학습 구간(season < f)에서만** 적합하고
폴드 행에 적용한다. 모델 잔차는 그 폴드의 생산 경로 예측에서 만든다
(`prod_champ_{f}.npy`, 후처리 편차 포함 — 세 폴드 모두 원장 기준선 재현 확인).

## 분류

    A  raw 약한데 D 복원 후 강함        -> D 가 이미 취한 이득 (설명)
    B  raw 와 D 가 같은 정보            -> 새 후보 아님
    C  D 를 걷어내도 residual 이 남음    -> **승격 후보**
    D  residual 도 없음                 -> 종료

    .\.venv\Scripts\python.exe -u exp\asof_resid.py
"""
import io
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "exp"))
from path_alloc import build_df                            # noqa: E402
from resid_table import post_for                           # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# (원시 비율, 그 비율의 분모, 대응하는 D 열)
SPEC = [("asof_pitcher_success_rate", "asof_pitcher_n", "cur_succ"),
        ("asof_pitcher_middle_rate", "asof_pitcher_n", "cur_mid"),
        ("asof_pitcher_ball_rate", "asof_pitcher_n", "cur_ball"),
        ("asof_pitcher_reverse_rate", "asof_pitcher_n", "cur_rev"),
        ("asof_pitcher_strike_rate", "asof_pitcher_n", "cur_str"),
        ("asof_pitcher_fastball_rate", "asof_pitcher_pitchmix_n", "cur_fb"),
        ("asof_pitcher_breaking_rate", "asof_pitcher_pitchmix_n", "cur_bb"),
        ("asof_pitcher_offspeed_rate", "asof_pitcher_pitchmix_n", "cur_os"),
        ("asof_batter_success_rate", "asof_batter_n", "cur_bsucc"),   # 보조
        ("asof_batter_middle_rate", "asof_batter_n", "cur_bmid")]     # 보조
DCOLS = ["cur_succ", "cur_mid", "cur_ball", "cur_rev", "cur_str",
         "cur_fb", "cur_bb", "cur_os", "cur_bsucc", "cur_bmid",
         "cur_logn_pitch", "cur_logn_bat"]        # cur_logn_mix 는 pitch 와 동일
FOLDS = (2022, 2023, 2024)


def cc(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 100 or a[m].std() < 1e-12 or b[m].std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a[m], b[m])[0, 1])


def oracle_decile(x, pred, y, half):
    """x 십분위의 잔차 평균을 교차적합해 더했을 때의 rho^2 증분."""
    r2 = lambda p: 1e5 * np.corrcoef(p, y)[0, 1] ** 2
    base = r2(pred)
    xf = np.where(np.isfinite(x), x, np.nanmedian(x[np.isfinite(x)]))
    q = np.quantile(xf, np.linspace(0, 1, 11)[1:-1])
    keys = np.searchsorted(q, xf)
    res = y - pred
    add = np.zeros(len(y))
    for msk in (half, ~half):
        u, inv = np.unique(keys[msk], return_inverse=True)
        n = np.bincount(inv, minlength=len(u))
        mu = np.bincount(inv, weights=res[msk], minlength=len(u)) / np.maximum(n, 1)
        ix = np.clip(np.searchsorted(u, keys[~msk]), 0, len(u) - 1)
        add[~msk] = np.where(u[ix] == keys[~msk], mu[ix], 0.0)
    return r2(pred + add) - base


def main():
    tr = build_df()
    season = tr["season"].to_numpy()
    y = tr["control_success"].to_numpy(np.float64)
    g = lambda c: tr[c].to_numpy(np.float64)
    cur_n = {"asof_pitcher_n": np.expm1(g("cur_logn_pitch")),
             "asof_pitcher_pitchmix_n": np.expm1(g("cur_logn_mix")),
             "asof_batter_n": np.expm1(g("cur_logn_bat"))}
    D = np.column_stack([g(c) for c in DCOLS])
    Dok = np.isfinite(D).all(1)

    out = {}
    for f in FOLDS:
        m_va, m_tr = season == f, season < f
        yv = y[m_va]
        P = np.load(os.path.join(ROOT, "exp", f"prod_champ_{f}.npy"))
        pred = P[:3].mean(0) + post_for(tr, y, m_tr, m_va)
        mres = yv - pred
        base = 1e5 * np.corrcoef(pred, yv)[0, 1] ** 2
        rng = np.random.default_rng(0)
        half = rng.random(int(m_va.sum())) < 0.5
        print("=" * 100)
        print(f"폴드 {f}   기준선 rho^2 = {base:.1f}   검증 {int(m_va.sum()):,}행"
              f"   (학습 구간에서만 계수 적합)")
        print("=" * 100)
        print(f"{'원시 컬럼':<34}{'raw~y':>9}{'cur~y':>9}{'prior~y':>9}"
              f"{'사영잔차~y':>11}{'사영잔차~모델잔차':>17}{'오라클':>9}")
        for rc, nc, dc in SPEC:
            n_now, cn = g(nc), cur_n[nc]
            pn = np.maximum(n_now - cn, 0.0)
            # 대수 분해 — 경력 성분을 정확히 되살린다
            prior = np.where(pn > 0,
                             (n_now * g(rc) - cn * g(dc)) / np.maximum(pn, 1e-9),
                             np.nan)
            # 사영 잔차 — 원시를 D 열 전체로 회귀 (계수는 학습 구간에서만)
            mtr = m_tr & Dok & np.isfinite(g(rc))
            ix = np.flatnonzero(mtr)
            if len(ix) > 400000:                  # 계수 적합은 표본으로 충분하다
                ix = rng.choice(ix, 400000, replace=False)
            A = np.column_stack([D[ix], np.ones(len(ix))])
            beta = np.linalg.lstsq(A, g(rc)[ix], rcond=None)[0]
            fit = np.column_stack([D, np.ones(len(D))]) @ beta
            pres = g(rc) - fit
            r2fit = 1.0 - np.nanvar(pres[m_tr]) / np.nanvar(g(rc)[m_tr])
            o = oracle_decile(pres[m_va], pred, yv, half)
            row = dict(raw_y=cc(g(rc)[m_va], yv), cur_y=cc(g(dc)[m_va], yv),
                       prior_y=cc(prior[m_va], yv),
                       pres_y=cc(pres[m_va], yv),
                       pres_mres=cc(pres[m_va], mres),
                       prior_mres=cc(prior[m_va], mres),
                       r2_D_explains=float(r2fit), oracle=float(o))
            out[f"{f}|{rc}"] = row
            print(f"  {rc:<32}{row['raw_y']:>+9.4f}{row['cur_y']:>+9.4f}"
                  f"{row['prior_y']:>+9.4f}{row['pres_y']:>+11.4f}"
                  f"{row['pres_mres']:>+17.4f}{o:>+9.1f}"
                  f"   D설명 R2={r2fit:.3f}")
        print()

    json.dump(out, io.open(os.path.join(ROOT, "exp", "asof_resid.json"), "w",
                           encoding="utf-8"), indent=1, ensure_ascii=False)
    print("분류 — pres_mres 가 세 폴드에서 일관되게 크고 오라클이 손익분기(2024 "
          "+19.8점)에 근접하면 C(승격). 그렇지 않으면 B/D.")


if __name__ == "__main__":
    main()
