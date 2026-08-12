r"""외부 데이터의 천장을 학습 0회로 잰다. 인터넷도 필요 없다.

## 무엇을 검산하는가

외부 데이터(KBO 공개 기록 등)가 결국 주는 것은 **"그 투수의 2025 현재 시즌 수준"**
이다. 4-21 이 가리킨 바로 그 정보이고, 4-26 이 죽은 이유(시즌 간 관계 전이 실패)를
정면으로 우회한다 — 학습 구간에서 옮겨 오는 게 아니라 평가 시즌 자체의 값이니까.

그런데 **`test.csv` 에는 `game_date` 가 없다.** `game_month` 뿐이다. 규칙 6) 이
"현재 투구 이후에 확정되는 모든 정보"를 금지하므로 시즌 집계를 그대로 쓸 수 없고,
쓸 수 있는 최선의 형태는 **전월까지의 시즌 성적**이다.

그건 train 으로 **그대로 흉내낼 수 있다.** 그것도 외부 지표보다 유리한 조건으로 —
외부 기록은 `control_success` 의 비공개 운영 정의를 모르는 대리지표지만, 여기서는
진짜 타깃으로 만든다. **이 상한이 안 나오면 외부 데이터는 더 안 나온다.**

    lvl[row] = mean(control_success)  over  같은 투수 & 같은 시즌 & game_month < 이 행의 월

## 이건 제출용이 아니다

평가셋 안에서 이렇게 누적하면 5) 위반이다 (평가 데이터 내부 행을 이용한 선수별
누적). 여기서는 **같은 정보를 외부 출처에서 합법적으로 얻었다고 가정했을 때의
천장**을 재는 것이고, 그 천장이 비용을 정당화하는지만 본다.

    .\.venv\Scripts\python.exe exp\extern_probe.py
"""
import json
import os

import numpy as np

import level_probe as L

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "exp", "cache")
FOLDS = [2021, 2022, 2024]


def asof_month_level(pid, month, y, min_n):
    """같은 시즌 안에서 '전월까지' 의 투수 성공률. 없으면 NaN.

    월 경계로만 자르므로 같은 달 안의 행끼리는 서로를 보지 않는다 — 외부 기록을
    월 단위로 받아 쓰는 상황과 정확히 같다.
    """
    out = np.full(len(y), np.nan)
    months = np.unique(month)
    order = np.argsort(pid, kind="stable")
    uniq, start = np.unique(pid[order], return_index=True)
    for i, s in enumerate(start):
        e = start[i + 1] if i + 1 < len(start) else len(order)
        idx = order[s:e]
        mm, yy = month[idx], y[idx]
        # 월별 누계를 한 번에
        tot = {}
        for m in months:
            sel = mm == m
            tot[m] = (yy[sel].sum(), sel.sum())
        run_s = run_n = 0.0
        for m in sorted(months):
            if run_n >= min_n:
                out[idx[mm == m]] = run_s / run_n
            run_s += tot[m][0]
            run_n += tot[m][1]
    return out


def main():
    meta = json.load(open(f"{CACHE}/cols.json"))
    ix = {c: i for i, c in enumerate(meta["cols"])}
    X = np.load(f"{CACHE}/X.npy", mmap_mode="r")
    y_all = np.load(f"{CACHE}/y.npy")
    season = np.load(f"{CACHE}/season.npy")
    MU = meta["shrink_prior"]["mu"]["p_succ"]

    print("외부 데이터 천장 — '전월까지의 시즌 성적' (진짜 타깃으로 만든 상한)\n")
    fits, data = {}, {}
    for fold in FOLDS:
        mask = season == fold
        y = y_all[mask].astype(np.float64)
        pid = np.asarray(X[mask, ix["pitcher_id"]])
        month = np.asarray(X[mask, ix["game_month"]])
        p = L.model_preds(fold)
        career = np.asarray(X[mask, ix["asof_pitcher_success_rate"]], float)
        career = np.where(np.isnan(career), MU, career)

        lvl = asof_month_level(pid, month, y, min_n=30)
        cov = np.isfinite(lvl).mean()
        lvl_f = np.where(np.isfinite(lvl), lvl, career)

        solo = 1e5 * L.r2_of(y, [p])
        both = 1e5 * L.r2_of(y, [p, lvl_f])
        print(f"--- fold {fold}  n={len(y):,}  커버리지 {cov:.1%} "
              f"(나머지는 통산으로 대체) ---")
        print(f"  모델                {solo:9.2f}")
        print(f"  전월까지 수준 단독      {1e5*L.r2_of(y, [lvl_f]):9.2f}")
        print(f"  모델 + 전월까지 수준    {both:9.2f}   ({both-solo:+.2f})")
        # 커버되는 행만 따로 — 대체값이 희석하는 몫을 가른다
        c = np.isfinite(lvl)
        s_c = 1e5 * L.r2_of(y[c], [p[c]])
        b_c = 1e5 * L.r2_of(y[c], [p[c], lvl[c]])
        print(f"  (커버된 행만)         {s_c:9.2f} -> {b_c:9.2f}   ({b_c-s_c:+.2f})")

        A = np.column_stack([np.ones(len(y)), p, lvl_f])
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        fits[fold] = beta
        data[fold] = (A, y, p)
        print()

    print("워크포워드 — 계수는 이전 폴드에서만 (4-26 이 여기서 죽었다)")
    print(f"{'폴드':>6s} {'모델':>9s} {'폴드내적합':>10s} {'워크포워드':>10s} {'WF증분':>8s}")
    print("-" * 50)
    for i, fold in enumerate(FOLDS):
        A, y, p = data[fold]
        solo = 1e5 * L.r2_of(y, [p])
        ins = 1e5 * L.r2_of(y, list(A[:, 1:].T))
        if i == 0:
            print(f"{fold:6d} {solo:9.2f} {ins:10.2f} {'—':>10s} {'—':>8s}")
            continue
        b = np.mean([fits[f] for f in FOLDS[:i]], axis=0)
        q = A @ b
        wf = 1e5 * np.corrcoef(q, y)[0, 1] ** 2
        print(f"{fold:6d} {solo:9.2f} {ins:10.2f} {wf:10.2f} {wf-solo:+8.2f}")

    print("\n계수 (상수, 모델, 전월까지수준)")
    for fold in FOLDS:
        print(f"  {fold}  " + " ".join(f"{v:+7.4f}" for v in fits[fold]))


if __name__ == "__main__":
    main()
