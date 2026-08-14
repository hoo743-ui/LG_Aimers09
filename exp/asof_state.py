r"""AS-OF 분해 — 통산 누적에서 **현재 시즌 상태**를 대수적으로 복원한다.

## 발견

`asof_*` 는 시즌 리셋이 아니라 **통산 누적**이다 (`data_description.md` L86:
"해당 행의 투구 직전까지"). 실측으로 확인했다 —

    투수 23633  2019:2,757  2020:5,427  2021:8,378  2022:10,893  2023:13,636  2024:15,449

따라서 그 투수의 **직전 시즌까지 통산**을 학습 데이터에서 빼면 시즌 내 상태가 나온다.

    n_cur    = asof_pitcher_n(행) - N_prior[투수]
    succ_cur = (asof_n * asof_rate)(행) - S_prior[투수]
    rate_cur = succ_cur / n_cur

**검증 (2024 폴드, `N_prior`/`S_prior` 는 <=2023 에서)**

    n 분해: 실제 시즌내 순번과 100.0000% 일치, 음수 0%
    rate 복원: 평균절대오차 3.1e-6 (잔차는 저장된 rate 의 float32 반올림)

## 왜 새 정보인가

모델이 보는 `asof_pitcher_success_rate` 는 통산이라 **2019~2024 이력과 현재 폼이
섞여 있고**, 모델은 그것을 못 푼다 (그 투수의 직전 시즌말 통산을 모른다).
우리는 안다. 분해하면

    통산 rate = [안정적 이력]  +  [현재 폼]

두 성분이 분리되고, `n_cur` 이 **현재 폼 추정의 신뢰도**까지 준다.
CAAFE 의 `prev1 - season`(경기 단위 창)과 질적으로 다르다 — 시즌 전체를 정확히
가르는 분해다.

## 규정 (4)

행 자신의 공식 `asof_*` 컬럼(`data_description.md` L182 에서 사용 허가 명시)과
**학습 데이터만으로 만든 투수별 상수**만 쓴다. 다른 test 행을 참조하지 않으므로
test.csv 에 그 행 하나만 있어도 값이 같다.

## 프로토콜 — 지시된 walk-forward

폴드 `f` 에서 `N_prior`/`S_prior` 는 시즌 `<f` 로 만든다. 그런데 **학습 행에도
같은 규율을 적용해야 한다** — 시즌 `g` 의 행은 `<g` 의 상수로 분해한다.
그래야 추론 시점(2025 는 `<=2024` 상수)과 같은 형태가 된다.

하이퍼파라미터를 폴드에서 고르지 않는다. 피처군은 사전에 정하고 그대로 잰다.

## 판정 게이트 (지시 15)

    A 3개 temporal split 에서 방향 일관
    B **min gain** 이 음수/미미하지 않을 것   <- CAAFE 가 여기서 걸렸어야 했다
    C walk-forward 에서 유지
    D 하이퍼파라미터가 미래 정보 미사용
    E 시드 안정성

    .\.venv\Scripts\python.exe -u exp\asof_state.py
"""
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
FOLDS = (2022, 2023, 2024)
SEEDS = (42, 43)

# 분해 대상 — (as-of 비율 컬럼, 그 비율의 분모가 되는 n 컬럼, 라벨)
RATE_COLS = [("asof_pitcher_success_rate", "asof_pitcher_n", "succ"),
             ("asof_pitcher_middle_rate", "asof_pitcher_n", "mid"),
             ("asof_pitcher_ball_rate", "asof_pitcher_n", "ball"),
             ("asof_pitcher_reverse_rate", "asof_pitcher_n", "rev"),
             ("asof_pitcher_strike_rate", "asof_pitcher_n", "str"),
             ("asof_pitcher_fastball_rate", "asof_pitcher_pitchmix_n", "fb"),
             ("asof_pitcher_breaking_rate", "asof_pitcher_pitchmix_n", "bb"),
             ("asof_pitcher_offspeed_rate", "asof_pitcher_pitchmix_n", "os"),
             ("asof_batter_success_rate", "asof_batter_n", "bsucc"),
             ("asof_batter_middle_rate", "asof_batter_n", "bmid")]


def build_state(C, ID_P, ID_B, season):
    """시즌 `g` 의 행을 시즌 `<g` 상수로 분해. 추론 시점과 같은 형태."""
    out = {}
    n_p, n_b = C("asof_pitcher_n"), C("asof_batter_n")
    n_mix = C("asof_pitcher_pitchmix_n")
    NCOL = {"asof_pitcher_n": n_p, "asof_batter_n": n_b,
            "asof_pitcher_pitchmix_n": n_mix}
    IDOF = {"asof_pitcher_n": ID_P, "asof_batter_n": ID_B,
            "asof_pitcher_pitchmix_n": ID_P}
    # 누적 상태 (prior) 와 현재 상태 (cur)
    for nc in NCOL:
        out[f"prior_n_{nc}"] = np.zeros(len(season))
        out[f"cur_n_{nc}"] = np.zeros(len(season))
    for rc, nc, lb in RATE_COLS:
        out[f"prior_{lb}"] = np.full(len(season), np.nan)
        out[f"cur_{lb}"] = np.full(len(season), np.nan)

    for g in sorted(np.unique(season)):
        m = season == g
        pr = season < g
        for nc in NCOL:
            ids, cnt = IDOF[nc], NCOL[nc]
            # prior 표본수 = 그 선수의 <g 행 수 (해당 n 정의에 맞게 근사)
            u, c = np.unique(ids[pr], return_counts=True)
            look = np.zeros(len(season))
            ix = np.clip(np.searchsorted(u, ids[m]), 0, max(len(u) - 1, 0))
            ok = (u[ix] == ids[m]) if len(u) else np.zeros(m.sum(), bool)
            look[np.where(m)[0][ok]] = c[ix[ok]]
            out[f"prior_n_{nc}"][m] = look[m]
            out[f"cur_n_{nc}"][m] = np.maximum(cnt[m] - look[m], 0)
        for rc, nc, lb in RATE_COLS:
            ids, cnt, rate = IDOF[nc], NCOL[nc], C(rc)
            tot = cnt * np.nan_to_num(rate)          # 통산 성공(해당 사건) 수
            pn = out[f"prior_n_{nc}"][m]
            # prior 사건수: 그 선수의 <g 마지막 행의 통산값을 쓸 수 없으므로
            # <g 구간의 (n*rate) 최대값으로 근사한다 — 통산이 단조증가라 정확하다
            u = np.unique(ids[pr])
            mx = np.zeros(len(u))
            o = np.argsort(ids[pr], kind="stable")
            k, v = ids[pr][o], tot[pr][o]
            uu, s0 = np.unique(k, return_index=True)
            mx = np.maximum.reduceat(v, s0)
            ix = np.clip(np.searchsorted(uu, ids[m]), 0, max(len(uu) - 1, 0))
            ok = (uu[ix] == ids[m]) if len(uu) else np.zeros(m.sum(), bool)
            ps = np.zeros(m.sum())
            ps[ok] = mx[ix[ok]]
            cn = out[f"cur_n_{nc}"][m]
            out[f"prior_{lb}"][m] = np.where(pn > 0, ps / np.maximum(pn, 1), np.nan)
            out[f"cur_{lb}"][m] = np.where(cn > 0,
                                           (tot[m] - ps) / np.maximum(cn, 1), np.nan)
    return out


def main():
    meta = json.load(io.open(f"{CACHE}/cols.json", encoding="utf-8"))
    ixc = {c: i for i, c in enumerate(meta["cols"])}
    prod = meta["prod"]
    X = np.load(f"{CACHE}/X.npy", mmap_mode="r")
    y = np.load(f"{CACHE}/y.npy").astype(np.float64)
    season = np.load(f"{CACHE}/season.npy")
    C = lambda n: np.asarray(X[:, ixc[n]], dtype=np.float64)
    ID_P = C("pitcher_id").astype(np.int64)
    ID_B = C("batter_id").astype(np.int64)

    t = time.time()
    S = build_state(C, ID_P, ID_B, season)
    print(f"as-of 분해 생성 {time.time() - t:.0f}s")
    cov = float(np.mean(S["cur_n_asof_pitcher_n"] > 0))
    print(f"  cur_n > 0 비율 {cov:.1%}   "
          f"중앙 {np.median(S['cur_n_asof_pitcher_n']):.0f}\n")

    base = np.column_stack([np.asarray(X[:, ixc[c]], dtype=np.float32)
                            for c in prod])
    L = lambda a: np.log1p(np.clip(a, 0, None)).astype(np.float32)
    lbls = [lb for _, _, lb in RATE_COLS]
    GROUPS = {
        # C — 누적(이력) 상태: 직전 시즌까지의 안정적 능력치
        "C 이력(prior)": [S[f"prior_{lb}"] for lb in lbls]
        + [L(S[f"prior_n_{n}"]) for n in
           ("asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n")],
        # D — 현재 시즌 상태
        "D 현재(cur)": [S[f"cur_{lb}"] for lb in lbls]
        + [L(S[f"cur_n_{n}"]) for n in
           ("asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n")],
        # E — 현재 vs 이력 (핵심)
        "E 현재-이력": [S[f"cur_{lb}"] - S[f"prior_{lb}"] for lb in lbls]
        + [L(S["cur_n_asof_pitcher_n"])],
    }
    GROUPS["J 전부"] = sum(GROUPS.values(), [])
    CONFIGS = [("A 기준 55p", base)] + [
        (k, np.hstack([base, np.column_stack(v).astype(np.float32)]))
        for k, v in GROUPS.items()]

    R = {}
    for f in FOLDS:
        tr, va = season < f, season == f
        yv = y[va]
        print(f"=== 폴드 {f}  학습 {int(tr.sum()):,}행 ===")
        for nm, M in CONFIGS:
            t = time.time()
            acc = np.zeros(int(va.sum()))
            for sd in SEEDS:
                m = CatBoostClassifier(random_seed=sd, **HP)
                m.fit(M[tr], y[tr].astype(int))
                acc += m.predict_proba(M[va])[:, 1]
            r = 1e5 * np.corrcoef(acc / len(SEEDS), yv)[0, 1] ** 2
            R.setdefault(nm, {})[f] = r
            print(f"  {nm:<14}{M.shape[1]:>5}p{r:>10.1f}"
                  f"{r - R['A 기준 55p'][f]:>+9.1f}{time.time() - t:>6.0f}s",
                  flush=True)

    print(f"\n=== 기준 55p 대비 배수 (시드 {len(SEEDS)}개 평균) ===")
    print(f"  {'구성':<14}" + "".join(f"{f:>10}" for f in FOLDS)
          + f"{'기하평균':>10}{'min gain':>10}{'3/3':>6}")
    for nm in R:
        ms = [R[nm][f] / R["A 기준 55p"][f] for f in FOLDS]
        g = float(np.exp(np.mean(np.log(ms))))
        print(f"  {nm:<14}" + "".join(f"{m:>10.4f}" for m in ms)
              + f"{g:>10.4f}{min(ms):>10.4f}"
              + f"{str(sum(x > 1 for x in ms)) + '/3':>6}")
    print(f"\n  게이트 B — min gain 이 1.00 근처면 CAAFE 와 같은 실패다 "
          f"(2022 1.0001 이었고 평가셋 이득 0).")
    json.dump({k: {str(a): b for a, b in v.items()} for k, v in R.items()},
              io.open(os.path.join(ROOT, "exp", "asof_state.json"), "w",
                      encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
