r"""`asof_*` 전수 감사 — D 가 아직 꺼내지 못한 합법 상태 정보가 남아 있는가.

## 감사 대상

공식 `asof_*` 는 19열이다 (`data_description.md`).

    카운터 3   asof_pitcher_n · asof_pitcher_pitchmix_n · asof_batter_n
    투수비율 5  success · reverse · middle · ball · strike
    prev 6     prev{1,3,5}_game_{success,middle}_rate
    구종믹스 3  fastball · breaking · offspeed
    타자 2     asof_batter_{success,middle}_rate

D 가 꺼낸 것은 `cur_*` 10비율 + `cur_logn_*` 3개다. **prev 6열은 원시 그대로**
모델에 있고, F(prev − cur)는 LB 전이 −60% 로 기각됐다.

## 이 파일이 재는 것

각 후보 수량마다 다섯 가지를 한 번에 낸다.

    target      corr(x, y)                     그 자체로 신호가 있는가
    residual    corr(x, Champion 잔차)          모델이 아직 못 쓴 몫이 있는가
    inpred      corr(x, Champion 예측)          이미 반영된 정도
    within      선수내 분산 비율                 pitcher_id 상수인가 (상수면 4-4 로 기각된 족보)
    persist     시즌 s -> s+1 선수평균 상관       내년에도 같은 값인가

`residual` 은 CLAUDE.md 4절의 사전 선별이다 — 이미 모델에 있는 피처와 **같은
대역**이면 회수된 것이다. 기준선으로 in-model 피처를 함께 찍는다.

## 카운터에서 만드는 후보 (규정 안)

전부 **그 행의 공식 컬럼 + 학습 구간 상수(prior 표)** 로만 만든다. 라벨·미래
행·다른 test 행·test 분포를 안 쓴다.

    prior_logn      log1p(prior_n)         이 시즌 시작 시점의 통산 투구수 (경력)
    cur_share       cur_n / asof_n         통산 중 이번 시즌 몫
    mix_cover_cur   cur_n_mix / cur_n      이번 시즌 구종기록 커버리지
    mix_cover_car   asof_mix_n / asof_n    통산 커버리지
    cur_ev_succ     cur_n * cur_succ       이번 시즌 성공 **개수** (양 x 질)
    pace            cur_n / 경과월          시즌내 투구 강도
    d_succ          cur_succ - prior_succ  경력 기준선 대비 현재 시즌 수준차
    bat_share       cur_n_bat / asof_bat_n 타자쪽 같은 몫

`cur_n` 은 `expm1(cur_logn_pitch)` 로 되살린다 (모델에 든 것은 로그뿐이다).
`prior_n = asof_n - cur_n` 은 **그 시즌 시작 시점**의 통산이다 — 생산 df 가
시즌마다 그 이전 시즌들로 prior 표를 만들기 때문이다 (`path_alloc.build_df`).

    .\.venv\Scripts\python.exe -u exp\asof_audit.py
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

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

RAW_CNT = ["asof_pitcher_n", "asof_pitcher_pitchmix_n", "asof_batter_n"]
RAW_RATE = ["asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
            "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
            "asof_pitcher_strike_rate"]
RAW_PREV = [f"asof_pitcher_prev{k}_game_{w}_rate"
            for w in ("success", "middle") for k in (1, 3, 5)]
RAW_MIX = ["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
           "asof_pitcher_offspeed_rate"]
RAW_BAT = ["asof_batter_success_rate", "asof_batter_middle_rate"]
CUR = ["cur_succ", "cur_mid", "cur_ball", "cur_rev", "cur_str",
       "cur_fb", "cur_bb", "cur_os", "cur_bsucc", "cur_bmid",
       "cur_logn_pitch", "cur_logn_mix", "cur_logn_bat"]


def cc(a, b):
    """결측을 뺀 상관. 상수면 0."""
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 100 or np.nanstd(a[m]) < 1e-12 or np.nanstd(b[m]) < 1e-12:
        return 0.0
    return float(np.corrcoef(a[m], b[m])[0, 1])


def main():
    tr = build_df()
    season = tr["season"].to_numpy()
    y = tr["control_success"].to_numpy(np.float64)
    m24 = season == 2024
    P = np.load(os.path.join(ROOT, "exp", "prod_champ_2024.npy"))
    pred = P[:3].mean(0) + np.load(os.path.join(ROOT, "exp",
                                                "prod_post_2024.npy"))
    resid = y[m24] - pred

    g = lambda c: tr[c].to_numpy(np.float64)
    cur_n = np.expm1(g("cur_logn_pitch"))
    cur_nm = np.expm1(g("cur_logn_mix"))
    cur_nb = np.expm1(g("cur_logn_bat"))
    an, amn, abn = g("asof_pitcher_n"), g("asof_pitcher_pitchmix_n"), g("asof_batter_n")
    prior_n = np.maximum(an - cur_n, 0.0)
    mon = g("game_month")
    elapsed = np.maximum(mon - 2.0, 1.0)          # 시즌 시작(3월) 기준 경과 개월

    print("=" * 78)
    print("1. 대수적 종속성 — 원문 정의대로 서로 묶여 있는가")
    print("=" * 78)
    s5 = np.column_stack([g(c) for c in RAW_RATE])
    ok = np.isfinite(s5).all(1)
    print(f"  투수 5비율 합 (succ+rev+mid+ball+str)  평균 {s5[ok].sum(1).mean():.4f}"
          f"  표준편차 {s5[ok].sum(1).std():.4f}")
    print(f"  ball+strike            평균 {(s5[ok][:, 3]+s5[ok][:, 4]).mean():.4f}"
          f"  표준편차 {(s5[ok][:, 3]+s5[ok][:, 4]).std():.4f}")
    print(f"  succ+mid               평균 {(s5[ok][:, 0]+s5[ok][:, 2]).mean():.4f}"
          f"  표준편차 {(s5[ok][:, 0]+s5[ok][:, 2]).std():.4f}")
    A = np.column_stack([s5[ok], np.ones(ok.sum())])
    sv = np.linalg.svd(A, compute_uv=False)
    print(f"  5비율+절편 특이값 비 {sv[-1]/sv[0]:.2e}"
          f"   (1e-8 미만이면 정확한 선형 종속)")
    mx = np.column_stack([g(c) for c in RAW_MIX])
    okm = np.isfinite(mx).all(1)
    print(f"  구종 3비율 합          평균 {mx[okm].sum(1).mean():.4f}"
          f"  표준편차 {mx[okm].sum(1).std():.4f}")
    print(f"  카운터 3종 상관: n~mix {cc(an, amn):+.4f}   n~bat {cc(an, abn):+.4f}"
          f"   mix~bat {cc(amn, abn):+.4f}")
    print(f"  mix_n <= n 위반 행 {(amn > an + 1e-6).mean():.2%}"
          f"   커버리지 중앙값 {np.nanmedian(amn/np.maximum(an,1)):.3f}")

    CAND = {
        "prior_logn": np.log1p(prior_n),
        "cur_share": cur_n / np.maximum(an, 1.0),
        "mix_cover_cur": cur_nm / np.maximum(cur_n, 1.0),
        "mix_cover_car": amn / np.maximum(an, 1.0),
        "cur_ev_succ": cur_n * g("cur_succ"),
        "cur_ev_ball": cur_n * g("cur_ball"),
        "pace": cur_n / elapsed,
        "d_succ": g("cur_succ") - np.where(prior_n > 0,
                                           (an * g("asof_pitcher_success_rate")
                                            - cur_n * g("cur_succ"))
                                           / np.maximum(prior_n, 1.0), np.nan),
        "bat_share": cur_nb / np.maximum(abn, 1.0),
        "prev1_gap_prior": g("asof_pitcher_prev1_game_success_rate")
        - g("asof_pitcher_success_rate"),
    }

    print()
    print("=" * 78)
    print("2. 전수 측정 (폴드 2024, Champion 잔차 기준)")
    print("=" * 78)
    print(f"{'컬럼':<34}{'target':>9}{'residual':>10}{'inpred':>9}"
          f"{'within':>8}{'persist':>9}")

    def within_share(x):
        """선수내 분산 / 전체 분산. 0 이면 pitcher_id 상수다."""
        v = x[m24]
        pid = tr["pitcher_id"].to_numpy()[m24]
        ok = np.isfinite(v)
        if ok.sum() < 1000:
            return np.nan
        v, pid = v[ok], pid[ok]
        u, inv = np.unique(pid, return_inverse=True)
        n = np.bincount(inv)
        mu = np.bincount(inv, weights=v) / n
        tot = v.var()
        return float(((v - mu[inv]) ** 2).mean() / tot) if tot > 0 else np.nan

    def persist(x):
        """시즌 s 와 s+1 의 선수평균 상관. 중앙값으로 요약한다."""
        pid = tr["pitcher_id"].to_numpy()
        out = []
        for s in range(2019, 2024):
            r = []
            for ss in (s, s + 1):
                m = (season == ss) & np.isfinite(x)
                d = pd.Series(x[m]).groupby(pid[m]).mean()
                r.append(d)
            j = r[0].to_frame("a").join(r[1].to_frame("b"), how="inner")
            if len(j) > 30:
                out.append(cc(j["a"].to_numpy(), j["b"].to_numpy()))
        return float(np.median(out)) if out else np.nan

    rows = {}
    BLOCKS = [("원시 카운터", RAW_CNT), ("원시 투수비율", RAW_RATE),
              ("원시 prev", RAW_PREV), ("원시 구종믹스", RAW_MIX),
              ("원시 타자", RAW_BAT), ("D 의 cur_*", CUR),
              ("카운터 후보", list(CAND))]
    for title, cols in BLOCKS:
        print(f"-- {title}")
        for c in cols:
            x = CAND[c] if c in CAND else g(c)
            r = dict(target=cc(x[m24], y[m24]), residual=cc(x[m24], resid),
                     inpred=cc(x[m24], pred), within=within_share(x),
                     persist=persist(x))
            rows[c] = r
            print(f"  {c:<32}{r['target']:>+9.4f}{r['residual']:>+10.4f}"
                  f"{r['inpred']:>+9.3f}{r['within']:>8.2f}{r['persist']:>+9.3f}")

    print()
    print("=" * 78)
    print("3. 카운터 3종의 정확한 관계 — 중복이면 D 의 13열 중 하나가 헛열이다")
    print("=" * 78)
    d = np.abs(an - amn)
    print(f"  |asof_pitcher_n - asof_pitchmix_n|  최대 {np.nanmax(d):.6f}"
          f"  평균 {np.nanmean(d):.6f}  일치율 {(d < 1e-6).mean():.4%}")
    dl = np.abs(g("cur_logn_pitch") - g("cur_logn_mix"))
    print(f"  |cur_logn_pitch - cur_logn_mix|     최대 {np.nanmax(dl):.6f}"
          f"  일치율 {(dl < 1e-6).mean():.4%}")

    print()
    print("=" * 78)
    print("4. 규모 상한 — 잡음 대역과 오라클(십분위 교차적합)")
    print("=" * 78)
    n24 = int(m24.sum())
    se = 1.0 / np.sqrt(n24)
    print(f"  폴드 2024 {n24:,}행 -> 상관 표준오차 약 {se:.4f}."
          f"  in-model 대역(asof_pitcher_n) +0.0033 은 {0.0033/se:.1f}SE 다.")
    base = 1e5 * np.corrcoef(pred, y[m24])[0, 1] ** 2
    print(f"  기준선 {base:.1f}\n")
    rng = np.random.default_rng(0)
    half = rng.random(n24) < 0.5
    print(f"{'수량':<34}{'residual':>10}{'SE배수':>8}{'선형상한':>10}"
          f"{'오라클십분위':>12}")
    scale = {}
    TESTED = (RAW_CNT + RAW_RATE + RAW_PREV + RAW_MIX + RAW_BAT
              + CUR + list(CAND))
    for c in TESTED:
        x = (CAND[c] if c in CAND else g(c))[m24]
        r = cc(x, resid)
        xf = np.where(np.isfinite(x), x, np.nanmedian(x))
        lin = 1e5 * (np.corrcoef(pred + resid.std() * r * (xf - xf.mean())
                                 / max(xf.std(), 1e-12), y[m24])[0, 1] ** 2)
        q = np.quantile(xf, np.linspace(0, 1, 11)[1:-1])
        keys = np.searchsorted(q, xf)
        add = np.zeros(n24)
        for msk in (half, ~half):
            u, inv = np.unique(keys[msk], return_inverse=True)
            n = np.bincount(inv, minlength=len(u))
            mu = np.bincount(inv, weights=resid[msk], minlength=len(u)) / np.maximum(n, 1)
            ix = np.clip(np.searchsorted(u, keys[~msk]), 0, len(u) - 1)
            add[~msk] = np.where(u[ix] == keys[~msk], mu[ix], 0.0)
        orc = 1e5 * np.corrcoef(pred + add, y[m24])[0, 1] ** 2
        scale[c] = dict(resid=r, se=r / se, lin=lin - base, oracle=orc - base)
        rows.setdefault(c, {}).update(scale[c])
        print(f"  {c:<32}{r:>+10.4f}{r/se:>+8.1f}{lin-base:>+10.1f}"
              f"{orc-base:>+12.1f}")

    json.dump(rows, io.open(os.path.join(ROOT, "exp", "asof_audit.json"), "w",
                            encoding="utf-8"), indent=1, ensure_ascii=False)
    print("\n판정 기준 — residual 이 in-model 피처 대역(asof_pitcher_n 부근)이면 "
          "회수된 것이다. within 이 0 에 가까우면 pitcher_id 상수라 4-4 족보다.")
    print("손익분기는 2024 에서 +2.1% = 약 +19.8 점이다. 오라클 십분위가 그 아래면 "
          "그 수량 하나로는 원리적으로 못 넘는다.")


if __name__ == "__main__":
    main()
