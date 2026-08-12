r"""투수 수준 추정치가 모델 예측에 직교하는 신호를 남기고 있는지 잰다. 학습 0회.

## 왜 여기인가

4-21 의 구조 진단: 그 시즌 자신의 `pitcher_id` 평균은 **874.8** 로 모델 전체
(748.77)보다 높다. 신호는 투수 단위로 크게 있고 '지금 시즌 기준'으로 알아야 한다.

12회차 이후 지표의 정체가 드러났다 — 아핀 최적에서 **점수 = 1e5 * rho^2** 다.
그래서 "이 추정치가 쓸모 있는가"는 학습 없이 답할 수 있다:

    R^2(y ~ p_model)              현재 도달점
    R^2(y ~ p_model + level)      선형으로 섞기만 해도 되는 지점

증분이 크면 **모델이 이미 쓰고 있는 정보가 아니다.** 그리고 2변수 선형결합은
행 독립이고 `asof_*` 공식 컬럼만 쓰므로 5) 원칙에 안전하다 — 트리를 다시 학습하지
않고 추론 경로에 그대로 실을 수 있는 형태다.

## 후보

`asof_pitcher_success_rate` 는 **통산 누적**이라 하락을 0.025 늦게 따라간다 (4-2).
`prev{1,3,5}_game` 은 현재 국면을 거의 정확히 반영하지만 표본이 적어 시끄럽다.
둘을 섞는 것이 경험 베이즈이고, 5-d 가 남겨 둔 축이다.

오라클 두 개는 **규칙상 쓸 수 없다** (평가셋 내부 누적 = 5) 금지). 천장을 보기
위한 참조값이다.

    .\.venv\Scripts\python.exe exp\level_probe.py
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "exp", "cache")
PREDS = os.path.join(ROOT, "exp", "preds")
LOG = os.path.join(ROOT, "experiment_log.jsonl")

FOLDS = [2021, 2022, 2024]
REF_TAG = "cat_tuned"
SEEDS = 3


def r2_of(y, X):
    """R^2 of OLS y ~ [1, X].  아핀 최적 점수 = 1e5 * R^2 이므로 그대로 점수다."""
    X = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return 1.0 - resid.var() / y.var()


def model_preds(fold):
    rows = []
    with open(LOG, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (r.get("tag") == REF_TAG and r.get("fold") == fold
                    and os.path.exists(f"{PREDS}/{r['key']}.npz")):
                rows.append(r)
    rows = sorted(rows, key=lambda r: r["seed"])[:SEEDS]
    ps = [np.load(f"{PREDS}/{r['key']}.npz")["p"] for r in rows]
    return np.mean(ps, axis=0).astype(np.float64)


def main():
    meta = json.load(open(f"{CACHE}/cols.json"))
    ix = {c: i for i, c in enumerate(meta["cols"])}
    X = np.load(f"{CACHE}/X.npy", mmap_mode="r")
    y_all = np.load(f"{CACHE}/y.npy")
    season = np.load(f"{CACHE}/season.npy")
    MU = meta["shrink_prior"]["mu"]["p_succ"]

    def col(name, mask):
        return np.asarray(X[mask, ix[name]], dtype=np.float64)

    print(f"지표 = 1e5 * R^2 (아핀 최적). 사전값 mu={MU:.4f}\n")
    summary = {}
    for fold in FOLDS:
        mask = season == fold
        y = y_all[mask].astype(np.float64)
        pid = col("pitcher_id", mask)
        career = col("asof_pitcher_success_rate", mask)
        n = col("asof_pitcher_n", mask)
        prev = {k: col(f"asof_pitcher_prev{k}_game_success_rate", mask)
                for k in (1, 3, 5)}
        p_mod = model_preds(fold)

        career = np.where(np.isnan(career), MU, career)
        n = np.where(np.isnan(n), 0.0, n)
        for k in prev:
            prev[k] = np.where(np.isnan(prev[k]), career, prev[k])

        # 오라클 — 규칙 위반. 천장 참조용
        order = np.argsort(pid, kind="stable")
        uniq, start = np.unique(pid[order], return_index=True)
        oracle = np.empty(len(y))
        for i, s in enumerate(start):
            e = start[i + 1] if i + 1 < len(start) else len(order)
            idx = order[s:e]
            oracle[idx] = y[idx].mean()

        # 경험 베이즈: 통산을 리그 사전값으로 당긴다 (5-d 의 sk*)
        cands = {
            "career (asof 통산)": career,
            "prev1": prev[1], "prev3": prev[3], "prev5": prev[5],
        }
        for k in (200, 1000):
            cands[f"sk{k} (통산->리그)"] = (n * career + k * MU) / (n + k)
        # 통산 + 최근 선형혼합. w 는 폴드마다 다시 고르지 않고 격자를 다 찍는다
        for w in (0.3, 0.5, 0.7):
            cands[f"mix w={w} (prev5)"] = w * prev[5] + (1 - w) * career

        print(f"--- fold {fold}  n={len(y):,}  r={y.mean():.4f} ---")
        print(f"{'단독 예측자':26s} {'단독':>9s} {'모델+이것':>10s} {'증분':>8s}")
        solo_m = 1e5 * r2_of(y, [p_mod])
        print(f"{'모델 (cat_tuned)':26s} {solo_m:9.2f} {'':>10s} {'':>8s}")
        for name, v in cands.items():
            solo = 1e5 * r2_of(y, [v])
            both = 1e5 * r2_of(y, [p_mod, v])
            print(f"{name:26s} {solo:9.2f} {both:10.2f} {both-solo_m:+8.2f}")
            summary.setdefault(name, []).append(both - solo_m)
        # 최근 3개를 한꺼번에
        allp = [prev[1], prev[3], prev[5], career]
        both = 1e5 * r2_of(y, [p_mod] + allp)
        print(f"{'career+prev1/3/5 전부':26s} {1e5*r2_of(y, allp):9.2f} "
              f"{both:10.2f} {both-solo_m:+8.2f}")
        summary.setdefault("career+prev1/3/5 전부", []).append(both - solo_m)
        o_solo = 1e5 * r2_of(y, [oracle])
        o_both = 1e5 * r2_of(y, [p_mod, oracle])
        print(f"{'[오라클] 시즌내 투수평균':26s} {o_solo:9.2f} {o_both:10.2f} "
              f"{o_both-solo_m:+8.2f}   <- 규칙 위반. 천장")
        print()

    print("=== 증분 요약 (모델 대비, 3폴드) ===")
    print(f"{'예측자':26s} {'평균':>8s} {'부호':>5s}   " +
          "  ".join(str(f).rjust(8) for f in FOLDS))
    for name, ds in sorted(summary.items(), key=lambda kv: -np.mean(kv[1])):
        print(f"{name:26s} {np.mean(ds):8.2f} {sum(d>0 for d in ds)}/3   "
              + "  ".join(f"{d:+8.2f}" for d in ds))

    print("\n" + "=" * 60)
    walk_forward(X, ix, y_all, season, MU)


def blend_cols(X, ix, mask, MU):
    """추론 경로에 실을 수 있는 형태 그대로. 결측은 통산->사전값 순으로 메운다."""
    def c(name):
        return np.asarray(X[mask, ix[name]], dtype=np.float64)
    career = c("asof_pitcher_success_rate")
    career = np.where(np.isnan(career), MU, career)
    out = [career]
    for k in (1, 3, 5):
        v = c(f"asof_pitcher_prev{k}_game_success_rate")
        out.append(np.where(np.isnan(v), career, v))
    return out


def walk_forward(X, ix, y_all, season, MU):
    """계수를 이전 시즌에서만 맞춰 다음 폴드에 쓴다.

    폴드 안에서 맞춘 계수는 낙관 상한이다 (4-3 이 중심 보정에서 배운 함정과 같은
    형태). 실제로 쓸 수 있는 값은 이것이다 — 시즌을 건너 계수가 전이되는가.
    """
    print("워크포워드 — 계수는 이전 폴드에서만 맞춘다\n")
    print(f"{'폴드':>6s} {'모델':>9s} {'폴드내적합':>10s} {'워크포워드':>10s} "
          f"{'WF증분':>8s}  계수출처")
    print("-" * 66)
    fits = {}
    for fold in FOLDS:
        mask = season == fold
        y = y_all[mask].astype(np.float64)
        p = model_preds(fold)
        Z = np.column_stack([p] + blend_cols(X, ix, mask, MU))
        A = np.column_stack([np.ones(len(y)), Z])
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        fits[fold] = beta
        solo = 1e5 * r2_of(y, [p])
        insample = 1e5 * r2_of(y, list(Z.T))

        prev = [f for f in FOLDS if f < fold]
        if not prev:
            print(f"{fold:6d} {solo:9.2f} {insample:10.2f} {'—':>10s} "
                  f"{'—':>8s}  (이전 폴드 없음)")
            continue
        b = np.mean([fits[f] for f in prev], axis=0)
        q = A @ b
        wf = 1e5 * (1 - (y - q).var() / y.var()) if q.std() > 1e-12 else 0.0
        # 아핀은 공짜이므로 워크포워드 예측도 아핀 최적으로 읽는다
        wf = 1e5 * np.corrcoef(q, y)[0, 1] ** 2
        print(f"{fold:6d} {solo:9.2f} {insample:10.2f} {wf:10.2f} "
              f"{wf-solo:+8.2f}  {'+'.join(str(f) for f in prev)}")

    print("\n계수 (상수, 모델, 통산, prev1, prev3, prev5)")
    for fold in FOLDS:
        print(f"  {fold}  " + " ".join(f"{v:+7.4f}" for v in fits[fold]))


if __name__ == "__main__":
    main()
