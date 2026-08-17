r"""D 의 한계 감사 — **무엇을 잘 복원하고 무엇을 잃는가.** 학습 0회.

D 를 "성공한 피처"로 고정하지 않고 잃은 것을 찾는다. 한계 후보 7개를 전부
정보량으로 먼저 잰다 (오라클 = 그 그룹의 실제 잔차 평균을 2024 를 반으로 갈라
교차적합해 더했을 때의 `rho^2` 증분. 라벨을 쓰므로 **진단 전용**이다).

## 최우선 질문 — F/R 을 섞지 않고 복원할 수 있는가

**대수적으로 불가능하다.** D 가 푼 식은 하나다.

    asof_n * asof_rate = prior_events + cur_n * cur_rate      (미지수 2, 식 1)

F/R 로 가르려면 미지수가 넷(`cur_n_F, cur_rate_F, cur_n_R, cur_rate_R`)이 되는데
식은 그대로 하나다. 공식 카운터는 **F/R 이 섞인 단일 계열**이고, 평가셋의 다른
행을 못 보므로(규정 4) 2025 행에서 그 행의 시즌내 F 투구수를 알 방법이 없다.

그래서 합법으로 가능한 것은 **학습 이력으로 만든 정적 표** 하나뿐이다.

    D_F[투수] , D_R[투수]  <- 학습 구간 라벨로 만든 상수, 그 행의 game_type 으로 조회

이 파일은 그것의 값을 두 각도에서 잰다.

    (1) 상한  — 그 시즌 자신의 F/R 별 상태를 알면 얼마나 오르는가 (오라클)
    (2) 전이  — 투수의 F-R 격차가 시즌을 건너 유지되는가 (정적 표의 실현 가능치)

(1)이 작으면 (2)를 볼 것도 없이 끝이고, (1)이 커도 (2)가 0 이면 정적 표로는
못 가져온다. 2023 은 F 라벨 정의가 바뀐 해이므로 시즌쌍별로 따로 본다.

## 나머지 한계 후보

    1 궤적 상실     pitcher x 시즌내십분위 (D 는 시즌 평균만 준다)
    2 작은 cur_n    cur_n 십분위별 잔차 구조 (축소/게이팅은 이미 실패했다)
    3 F/R 혼합      pitcher x game_type
    4 타입별 의미차  game_type x cur_succ 십분위
    5 타자손 분할    pitcher x 타자손 (카운터에 분할이 없다)
    6 실패유형 겹침  실패 구성비 (rev/ball/mid 가 실패 안에서 차지하는 몫)
    7 생성 메커니즘  구종 구성 x 상태

    .\.venv\Scripts\python.exe -u exp\d_limits.py
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


KS = [0, 20, 50, 150, 500, 2000, 10000]


def cv2_gain(keys, pred, y, half):
    """그룹 잔차 평균을 교차적합해 더했을 때의 rho^2 증분 (정직한 상한).

    축소 `n/(n+k)` 를 훑어 **최댓값**을 쓴다 — 축소 없이는 셀 잡음이 이겨서
    큰 그룹이 전부 음수로 깔린다 (`oracle_probe.py` 와 같은 규약).
    """
    r2 = lambda p: 1e5 * np.corrcoef(p, y)[0, 1] ** 2
    res = y - pred
    b = r2(pred)
    best, bk = -1e9, None
    for k in KS:
        add = np.zeros(len(y))
        for msk in (half, ~half):
            u, inv = np.unique(keys[msk], return_inverse=True)
            n = np.bincount(inv, minlength=len(u)).astype(np.float64)
            mu = np.bincount(inv, weights=res[msk], minlength=len(u)) / np.maximum(n, 1)
            mu = mu * (n / (n + k))
            ix = np.clip(np.searchsorted(u, keys[~msk]), 0, len(u) - 1)
            add[~msk] = np.where(u[ix] == keys[~msk], mu[ix], 0.0)
        v = r2(pred + add) - b
        if v > best:
            best, bk = v, k
    return best, len(np.unique(keys)), bk


def dec(x, m):
    """구간 `m` 안에서의 십분위."""
    v = np.where(np.isfinite(x), x, np.nan)
    q = np.nanquantile(v[m], np.linspace(0, 1, 11)[1:-1])
    return np.searchsorted(q, np.nan_to_num(v, nan=np.nanmedian(v[m])))


def main():
    tr = build_df()
    season = tr["season"].to_numpy()
    y = tr["control_success"].to_numpy(np.float64)
    m24 = season == 2024
    P = np.load(os.path.join(ROOT, "exp", "prod_champ_2024.npy"))
    pred = P[:3].mean(0) + np.load(os.path.join(ROOT, "exp",
                                                "prod_post_2024.npy"))
    y24 = y[m24]
    base = 1e5 * np.corrcoef(pred, y24)[0, 1] ** 2
    rng = np.random.default_rng(0)
    half = rng.random(int(m24.sum())) < 0.5
    g = lambda c: tr[c].to_numpy(np.float64)

    PID = tr["pitcher_id"].to_numpy(np.int64)
    BH = tr["batter_hand"].to_numpy(np.int64)
    GT = pd.factorize(tr["game_type"])[0].astype(np.int64)
    gtname = list(pd.factorize(tr["game_type"])[1])
    isF = (tr["game_type"].to_numpy() == "F").astype(np.int64)
    curn = np.expm1(g("cur_logn_pitch"))
    fail = np.maximum(1.0 - g("cur_succ"), 1e-6)

    print(f"기준선 rho^2 = {base:.1f}   game_type 값 {gtname}"
          f"   F 비율 2024 {isF[m24].mean():.1%}")
    print()
    print("=" * 82)
    print("A. 한계 후보별 정보량 상한 (오라클 cv2, 폴드 2024)")
    print("=" * 82)
    GROUPS = {
        "pitcher (기준)": PID,
        "3 F/R 혼합: pitcher x game_type": PID * 10 + GT,
        "5 타자손: pitcher x 타자손": PID * 10 + BH,
        "  pitcher x game_type x 타자손": (PID * 10 + GT) * 10 + BH,
        "1 궤적: pitcher x 시즌내십분위": PID * 100 + dec(curn, m24),
        "  pitcher x game_type x 십분위": (PID * 10 + GT) * 100 + dec(curn, m24),
        "4 타입별 의미차: gt x cur_succ 십분위": GT * 100 + dec(g("cur_succ"), m24),
        "  game_type 단독": GT,
        "2 작은 cur_n: cur_n 십분위": dec(curn, m24),
        "  cur_n 십분위 x game_type": dec(curn, m24) * 10 + GT,
        "6 실패구성: rev 몫 십분위": dec(g("cur_rev") / fail, m24),
        "  ball 몫 십분위": dec(g("cur_ball") / fail, m24),
        "  mid 몫 십분위": dec(g("cur_mid") / fail, m24),
        "7 생성: cur_bb 십분위 x cur_succ 십분위": dec(g("cur_bb"), m24) * 100
        + dec(g("cur_succ"), m24),
        "  pitcher x cur_bb 십분위": PID * 100 + dec(g("cur_bb"), m24),
        # 위약 대조 — 정보가 없는 분할도 같은 이득을 내는가 (셀 세분화 산물 검사)
        "위약: pitcher x 무작위2분할": PID * 10
        + (np.random.default_rng(7).random(len(PID)) < 0.5).astype(np.int64),
        "위약: pitcher x 무작위10분할": PID * 100
        + np.random.default_rng(8).integers(0, 10, len(PID)),
        "참고: pitcher x count(12)": PID * 100
        + (tr["balls_before"].to_numpy(np.int64) * 4
           + tr["strikes_before"].to_numpy(np.int64)),
    }
    out = {}
    print(f"{'그룹':<42}{'셀수':>8}{'cv2 증분':>10}{'최적 k':>8}")
    for name, k in GROUPS.items():
        v, nc, bk = cv2_gain(k[m24], pred, y24, half)
        out[name] = dict(cells=nc, gain=float(v), k=bk)
        print(f"{name:<42}{nc:>8,}{v:>+10.1f}{bk:>8}")

    print()
    print("=" * 82)
    print("B. 최우선 질문 — 투수의 F-R 격차가 시즌을 건너는가")
    print("=" * 82)
    print("  (정적 D_F/D_R 표로 가져올 수 있는 몫의 실현 가능치)")
    rows = []
    for s in range(2019, 2025):
        m = season == s
        d = pd.DataFrame({"p": PID[m], "f": isF[m], "y": y[m]})
        piv = d.groupby(["p", "f"])["y"].agg(["mean", "size"]).unstack()
        ok = (piv[("size", 0)] >= 50) & (piv[("size", 1)] >= 50)
        gap = (piv[("mean", 1)] - piv[("mean", 0)])[ok]
        rows.append(pd.Series(gap, name=s))
        print(f"  {s}  투수 {ok.sum():>4}명  F-R 격차 평균 {gap.mean():+.4f}"
              f"  표준편차 {gap.std():.4f}   F 비율 {isF[m].mean():.1%}")
    print()
    print(f"  {'시즌쌍':<14}{'공통 투수':>10}{'격차 상관':>12}{'격차 회귀계수':>14}")
    pers = {}
    for a, b in zip(rows[:-1], rows[1:]):
        j = pd.concat([a, b], axis=1, join="inner").dropna()
        if len(j) < 8:
            continue
        c = float(np.corrcoef(j.iloc[:, 0], j.iloc[:, 1])[0, 1])
        beta = float(np.polyfit(j.iloc[:, 0], j.iloc[:, 1], 1)[0])
        pers[f"{a.name}->{b.name}"] = dict(n=len(j), corr=c, beta=beta)
        print(f"  {a.name}->{b.name:<9}{len(j):>10}{c:>+12.3f}{beta:>+14.3f}")
    out["_gap_persistence"] = pers
    out["_base"] = float(base)

    json.dump(out, io.open(os.path.join(ROOT, "exp", "d_limits.json"), "w",
                           encoding="utf-8"), indent=1, ensure_ascii=False)
    print("\n판정 — A 에서 pitcher x game_type 이 pitcher 단독을 크게 넘지 못하면 "
          "F/R 혼합은 D 의 한계가 아니다. B 의 격차 상관이 0 이면 정적 D_F/D_R "
          "표로는 그 몫을 못 가져온다.")


if __name__ == "__main__":
    main()
