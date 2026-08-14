r"""CAAFE — 의미 기반 피처 생성. 마지막 미착수 축.

## 왜 이것만 남았는가

모델 계열은 전부 소진됐다 (§9-a, §10). 그리고 그 결과가 이 방향을 가리킨다 —
**모델을 아무리 바꿔도 `rho` 가 안 움직였고 두 계열이 같은 곳으로 수렴했다.**
주어진 피처가 담은 정보를 이미 다 짜냈다는 뜻이고, 남은 것은 정보를 늘리는 것뿐이다.

그런데 55피처 중 **47개가 주최측 제공 그대로**다. 나머지 8개는 TrackMan 집계고,
**야구 도메인 지식으로 만든 파생변수는 하나도 없다.** 편차 표도 통계적 타깃
인코딩이지 야구 지식이 아니다.

`05` 덱의 CAAFE 패턴 — LLM 이 의미 기반 피처를 코드로 제안하고 표준 ML 이
평가해 살아남은 것만 채택 — 을 그대로 따른다. 추론 시점에 API 를 쓰는 것이
아니라 **개발 시점에 피처 코드를 짜는 것**이라 규정 2)에 걸리지 않는다.
모든 피처가 **행 안의 값만으로 계산**되므로 규정 4)에도 걸리지 않는다.

## 설계 — 네 갈래

### 1. 카운트의 의미 (타깃 정의에서 직접 온다)

제구 실패 = ① 존 한가운데 ② 존에서 크게 벗어남 ③ 포수 요구 반대방향.
그래서 **2스트라이크(유인구를 던져도 됨)** 와 **3볼(스트라이크를 던져야 함)** 은
투수의 *의도*가 정반대인데, 지금 모델에는 `balls`/`strikes` 숫자로만 들어간다.
0-2 에서 일부러 뺀 공은 ②로 실패 기록되고, 3-0 에서 가운데 몰면 ①로 실패다.

### 2. 차이·비율 — 트리가 원리적으로 못 만드는 형태

축평행 분할은 `a - b` 를 한 번에 못 만든다. 최근 폼 편차(`prev1 - 시즌`),
폼 추세(`prev1 - prev3`), 투수-타자 격차가 여기 해당한다.

### 3. 실패의 구성 — 투수의 지문

실패가 가운데/볼/반대로 쪼개지는데 그 **비율**이 투수마다 다르다.
같은 성공률이라도 "가운데로 몰려 실패하는 투수"와 "빠져서 실패하는 투수"는
카운트·주자 상황에 다르게 반응할 것이다.

### 4. 구종 다양성 — 분포의 엔트로피

한 구종만 던지는 투수와 셋을 고루 섞는 투수는 제구 난이도가 다르다.
`fastball/breaking/offspeed` 세 비율이 있으나 엔트로피는 트리가 못 만든다.

## 평가

대조군과 동일한 시드·설정으로 3폴드(2022/2023/2024) 전부에서 잰다. §9-b 에서
`log(n)` 이 2024 에서만 +1.43% 였다가 폴드 검증에서 죽은 전례가 있으므로
**한 폴드 결과는 채택 근거가 못 된다.**

    .\.venv\Scripts\python.exe -u exp\caafe.py
"""
import argparse
import io
import json
import os
import time

import numpy as np
from catboost import CatBoostClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "exp", "cache")
HP = dict(iterations=1200, learning_rate=0.02, depth=6, l2_leaf_reg=100.0,
          border_count=32, loss_function="Logloss", verbose=0,
          thread_count=16, allow_writing_files=False)
EPS = 1e-6


def build(C):
    """의미 기반 파생변수. 전부 행 안의 값만 쓴다 (규정 4) 준수)."""
    B, S = C("balls_before"), C("strikes_before")
    ph, bh = C("pitcher_hand"), C("batter_hand")
    inn, outs = C("inning"), C("outs_before")
    sd = C("score_diff_pitcher_team")
    r1, r2, r3 = C("runner_on_1b"), C("runner_on_2b"), C("runner_on_3b")
    li = C("li")
    ps = C("asof_pitcher_success_rate")
    pm, pb_, pr = (C("asof_pitcher_middle_rate"), C("asof_pitcher_ball_rate"),
                   C("asof_pitcher_reverse_rate"))
    pst = C("asof_pitcher_strike_rate")
    p1, p3, p5 = (C("asof_pitcher_prev1_game_success_rate"),
                  C("asof_pitcher_prev3_game_success_rate"),
                  C("asof_pitcher_prev5_game_success_rate"))
    m1, m3, m5 = (C("asof_pitcher_prev1_game_middle_rate"),
                  C("asof_pitcher_prev3_game_middle_rate"),
                  C("asof_pitcher_prev5_game_middle_rate"))
    pn, bn = C("asof_pitcher_n"), C("asof_batter_n")
    bs, bm = C("asof_batter_success_rate"), C("asof_batter_middle_rate")
    ff, fb, fo = (C("asof_pitcher_fastball_rate"),
                  C("asof_pitcher_breaking_rate"),
                  C("asof_pitcher_offspeed_rate"))

    fail = np.clip(1.0 - ps, EPS, None)
    mix = np.stack([np.clip(ff, EPS, 1), np.clip(fb, EPS, 1),
                    np.clip(fo, EPS, 1)])
    mix = mix / mix.sum(0, keepdims=True)
    F = {
        # --- 1. 카운트의 의미 ---
        "cf_must_strike": (B >= 3).astype(np.float32),
        "cf_can_waste": (S >= 2).astype(np.float32),
        "cf_first_pitch": ((B == 0) & (S == 0)).astype(np.float32),
        "cf_full_count": ((B == 3) & (S == 2)).astype(np.float32),
        "cf_count_diff": (B - S).astype(np.float32),
        "cf_pitcher_ahead": (S > B).astype(np.float32),
        "cf_hitter_ahead": (B > S).astype(np.float32),
        "cf_total_pitches": (B + S).astype(np.float32),
        # 3볼인데 유인구도 못 던지는 상황 vs 2스트라이크 여유
        "cf_waste_room": (S - B).astype(np.float32),
        # --- 2. 차이 · 추세 (트리가 못 만드는 형태) ---
        "cf_form1": (p1 - ps).astype(np.float32),
        "cf_form3": (p3 - ps).astype(np.float32),
        "cf_form5": (p5 - ps).astype(np.float32),
        "cf_trend13": (p1 - p3).astype(np.float32),
        "cf_trend35": (p3 - p5).astype(np.float32),
        "cf_midform1": (m1 - pm).astype(np.float32),
        "cf_midtrend13": (m1 - m3).astype(np.float32),
        "cf_pb_gap": (ps - bs).astype(np.float32),
        "cf_pb_midgap": (pm - bm).astype(np.float32),
        "cf_ball_minus_strike": (pb_ - pst).astype(np.float32),
        # --- 3. 실패의 구성 — 투수의 지문 ---
        "cf_share_middle": (pm / fail).astype(np.float32),
        "cf_share_ball": (pb_ / fail).astype(np.float32),
        "cf_share_reverse": (pr / fail).astype(np.float32),
        "cf_mid_vs_ball": (pm - pb_).astype(np.float32),
        # --- 4. 구종 다양성 ---
        "cf_mix_entropy": (-(mix * np.log(mix)).sum(0)).astype(np.float32),
        "cf_mix_dominant": mix.max(0).astype(np.float32),
        "cf_non_fastball": (fb + fo).astype(np.float32),
        # --- 5. 표본 신뢰도 ---
        "cf_log_pn": np.log1p(pn).astype(np.float32),
        "cf_log_bn": np.log1p(bn).astype(np.float32),
        "cf_p_cold": (pn == 0).astype(np.float32),
        "cf_b_cold": (bn == 0).astype(np.float32),
        # --- 6. 상황 압력 ---
        "cf_same_hand": (ph == bh).astype(np.float32),
        "cf_abs_sd": np.abs(sd).astype(np.float32),
        "cf_close": (np.abs(sd) <= 1).astype(np.float32),
        "cf_blowout": (np.abs(sd) >= 5).astype(np.float32),
        "cf_late_close": ((inn >= 7) & (np.abs(sd) <= 3)).astype(np.float32),
        "cf_risp": ((r2 + r3) > 0).astype(np.float32),
        "cf_loaded": ((r1 > 0) & (r2 > 0) & (r3 > 0)).astype(np.float32),
        "cf_outs_risp": (outs * ((r2 + r3) > 0)).astype(np.float32),
        "cf_log_li": np.log1p(np.clip(li, 0, None)).astype(np.float32),
        "cf_deep": (inn >= 7).astype(np.float32),
    }
    return F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="2022,2023,2024")
    a = ap.parse_args()
    meta = json.load(io.open(f"{CACHE}/cols.json", encoding="utf-8"))
    ixc = {c: i for i, c in enumerate(meta["cols"])}
    prod = meta["prod"]
    X = np.load(f"{CACHE}/X.npy", mmap_mode="r")
    y = np.load(f"{CACHE}/y.npy").astype(np.float64)
    season = np.load(f"{CACHE}/season.npy")
    C = lambda n: np.asarray(X[:, ixc[n]], dtype=np.float64)

    base = np.column_stack([np.asarray(X[:, ixc[c]], dtype=np.float32)
                            for c in prod])
    F = build(C)
    names = list(F)
    new = np.column_stack([F[n] for n in names])
    print(f"파생변수 {len(names)}개 생성   기본 {base.shape[1]} -> "
          f"{base.shape[1] + len(names)}\n")

    res = {}
    for f in [int(v) for v in a.folds.split(",")]:
        tr, va = season < f, season == f
        yv = y[va]
        cp = os.path.join(ROOT, "exp", f"champ_oof_{f}.npy")
        rch = (1e5 * np.corrcoef(np.load(cp), yv)[0, 1] ** 2
               if os.path.exists(cp) else float("nan"))
        print(f"=== 폴드 {f}   학습 {int(tr.sum()):,}행   "
              f"Champion OOF {rch:.1f} ===")
        row = {}
        KEEP = ["cf_same_hand", "cf_form5", "cf_share_reverse",
                "cf_share_ball", "cf_mix_entropy", "cf_log_pn",
                "cf_trend13", "cf_trend35", "cf_midform1",
                "cf_midtrend13", "cf_ball_minus_strike"]
        ki = [names.index(k) for k in KEEP]
        for nm, M in (("대조 prod55", base),
                      ("+같은손 1개", np.hstack([base, new[:, [names.index("cf_same_hand")]]])),
                      ("+생존 11개", np.hstack([base, new[:, ki]])),
                      ("+CAAFE 40개", np.hstack([base, new]))):
            t = time.time()
            m = CatBoostClassifier(random_seed=42, **HP)
            m.fit(M[tr], y[tr].astype(int))
            p = m.predict_proba(M[va])[:, 1].astype(np.float64)
            r = 1e5 * np.corrcoef(p, yv)[0, 1] ** 2
            row[nm] = r
            print(f"  {nm:<14}{M.shape[1]:>5}피처{r:>10.1f}"
                  f"{r - row['대조 prod55']:>+9.1f}{time.time() - t:>6.0f}s",
                  flush=True)
            if nm == "+CAAFE 40개":
                imp = m.get_feature_importance()
                top = np.argsort(imp[len(prod):])[::-1][:10]
                print("   상위 파생변수: " + ", ".join(
                    f"{names[i]}({imp[len(prod) + i]:.2f})" for i in top))
        res[f] = row

    print(f"\n=== 3폴드 요약 (대조 대비 배수) ===")
    variants = ["+같은손 1개", "+생존 11개", "+CAAFE 40개"]
    print(f"  {'폴드':>7}{'대조':>10}" + "".join(f"{v:>14}" for v in variants))
    R = {v: [] for v in variants}
    for f, row in res.items():
        print(f"  {f:>7}{row['대조 prod55']:>10.1f}", end="")
        for v in variants:
            m = row[v] / row["대조 prod55"]
            R[v].append(m)
            print(f"{m:>14.4f}", end="")
        print()
    print(f"  {'기하평균':>7}{'':>10}" + "".join(
        f"{np.exp(np.mean(np.log(R[v]))):>14.4f}" for v in variants))
    print(f"  {'개선폴드':>7}{'':>10}" + "".join(
        f"{str(sum(x > 1 for x in R[v])) + '/3':>14}" for v in variants))


if __name__ == "__main__":
    main()
