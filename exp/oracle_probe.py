r"""오라클 진단 — **남은 점수가 어느 그룹에 있는가.** 학습 0회.

## 왜 이걸 재는가

닫힌 축 목록이 소진됐고, 10위 고원까지 +108 점(rho^2 +10.3%)이 남았다.
그 크기는 +0.2~0.5% 짜리를 모아 닿지 않는다 — **D 급(+8.97%)이 한 번 더**
필요하다. 그러면 다음 축을 추측으로 고를 것이 아니라 **어디에 정보가 남아
있는지**를 먼저 재야 한다.

## 세 가지 상한을 구분한다

그룹 `G` 마다 Champion 잔차의 그룹 평균을 예측에 더해 `rho^2` 를 다시 잰다.
더하는 값을 **무엇으로 추정했는가**가 셋을 가른다.

    infold    2024 자신의 라벨로 추정        불법. 잡음까지 맞히므로 과대평가
    cv2       2024 를 반으로 갈라 교차적합    **정직한 상한.** 그 그룹이 실제로
                                             가진 정보량 (시즌내 정보를 완벽히
                                             주는 피처가 있다면 여기까지)
    transfer  학습 시즌(<2024)에서 추정       지금 피처로 이미 회수 가능한 몫

읽는 법은 이렇다.

    cv2 가 크고 transfer 가 작다  ->  시즌내 정보다. D 가 정확히 이 자리였다
    cv2 와 transfer 가 둘 다 작다 ->  그 그룹에 정보가 없다. 축을 열지 않는다
    transfer 가 이미 크다         ->  모델이 회수 중이거나 회수 가능. 신규 아님

`transfer` 는 규정 4)를 지키는 조작이지만(학습 구간 표 + 그 행 자신의 키),
`infold`/`cv2` 는 **평가셋 라벨을 쓰므로 제출에 못 쓴다.** 진단 전용이다.

## 축소

그룹 평균은 표본이 적을수록 잡음이다. `n/(n+k)` 로 축소하고 `k` 를 훑는다.
보고는 **k 별 곡선 전체**로 한다 — 최댓값만 보면 평가셋에 맞춘 값이 된다.

    .\.venv\Scripts\python.exe -u exp\oracle_probe.py
"""
import io
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "exp"))
from path_alloc import build_df                            # noqa: E402

KS = [0, 20, 50, 150, 500, 2000]

try:                                      # cp949 콘솔에서 em-dash 로 죽지 않게
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def r2(p, y):
    return 1e5 * np.corrcoef(p, y)[0, 1] ** 2


def group_mean(keys, vals, k, uniq=None):
    """`keys` 별 `vals` 평균을 n/(n+k) 로 축소해 각 행에 돌려준다."""
    if uniq is None:
        uniq, inv = np.unique(keys, return_inverse=True)
    else:
        inv = np.searchsorted(uniq, keys)
    n = np.bincount(inv, minlength=len(uniq)).astype(np.float64)
    s = np.bincount(inv, weights=vals, minlength=len(uniq))
    m = np.divide(s, np.maximum(n, 1))
    return uniq, m * (n / (n + k)), inv


def apply_table(uniq, tab, keys):
    """학습 구간 표를 다른 구간의 키로 조회한다. 없는 키는 0."""
    ix = np.clip(np.searchsorted(uniq, keys), 0, max(len(uniq) - 1, 0))
    ok = uniq[ix] == keys if len(uniq) else np.zeros(len(keys), bool)
    out = np.zeros(len(keys))
    out[ok] = tab[ix[ok]]
    return out


def main():
    tr = build_df()
    season = tr["season"].to_numpy()
    y = tr["control_success"].to_numpy(np.float64)
    m_va, m_tr = season == 2024, season < 2024
    yv = y[m_va]

    P = np.load(os.path.join(ROOT, "exp", "prod_champ_2024.npy"))
    post = np.load(os.path.join(ROOT, "exp", "prod_post_2024.npy"))
    assert P.shape[1] == m_va.sum(), f"행수 불일치 {P.shape} vs {m_va.sum()}"
    pv = P[:3].mean(0) + post              # 원장 기준선과 같은 3시드 구성
    base = r2(pv, yv)
    print(f"  기준선 (3시드 + 후처리 4축)  2024 rho^2 = {base:.1f}", flush=True)
    print(f"  참고 — 원장 champ 943.8 / 7시드도 함께 잰다", flush=True)
    print(f"  7시드: {r2(P.mean(0) + post, yv):.1f}\n", flush=True)

    resid = yv - pv                        # 잔차. 그룹 평균을 이 위에 얹는다
    rtr = None                             # 학습 구간 잔차는 예측이 없으므로
    #  transfer 는 라벨 편차로 대신한다 — 학습 구간에서 그 그룹이 전체 평균보다
    #  얼마나 높은가. 모델이 이미 회수했다면 2024 잔차와 상관이 없다.
    ytr = y[m_tr]
    ybar = ytr.mean()
    dev_tr = ytr - ybar

    def col(name):
        return tr[name].to_numpy()

    BB, SS = col("balls_before").astype(np.int64), col("strikes_before").astype(np.int64)
    CNT = BB * 4 + SS
    PID, BID = col("pitcher_id").astype(np.int64), col("batter_id").astype(np.int64)
    BH = col("batter_hand").astype(np.int64)
    MON = col("game_month").astype(np.int64)
    # 시즌내 진행도 — D 가 복원한 시즌내 투구수(로그)의 그 시즌 내 십분위
    curn = tr["cur_logn_pitch"].to_numpy(np.float64)
    dec = np.zeros(len(curn), np.int64)
    for s in np.unique(season):
        m = season == s
        q = np.quantile(curn[m], np.linspace(0, 1, 11)[1:-1])
        dec[m] = np.searchsorted(q, curn[m])

    GROUPS = {
        "pitcher": PID,
        "batter": BID,
        "count(12)": CNT,
        "pitcher x count": PID * 100 + CNT,
        "pitcher x 타자손": PID * 10 + BH,
        "pitcher x 월": PID * 100 + MON,
        "pitcher x 시즌내십분위": PID * 100 + dec,
        "시즌내십분위": dec,
        "월": MON,
        "이닝": col("inning").astype(np.int64),
        "base_state": pd.factorize(tr["base_state"])[0].astype(np.int64),
        "투수팀": col("pitcher_team_id").astype(np.int64),
        "타자팀": col("batter_team_id").astype(np.int64),
        "구종믹스 n 십분위": np.digitize(
            tr["asof_pitcher_pitchmix_n"].to_numpy(np.float64),
            np.quantile(tr["asof_pitcher_pitchmix_n"].to_numpy(np.float64),
                        np.linspace(0, 1, 11)[1:-1])).astype(np.int64),
    }

    rng = np.random.default_rng(0)
    half = rng.random(int(m_va.sum())) < 0.5

    print(f"{'그룹':<24}{'셀수':>8}{'infold':>10}{'cv2':>10}{'transfer':>10}"
          f"{'cv2-기준':>10}")
    out = {}
    for name, keyall in GROUPS.items():
        t = time.time()
        kv, kt = keyall[m_va], keyall[m_tr]
        ncell = len(np.unique(kv))
        best = {"infold": (-1e9, None), "cv2": (-1e9, None), "transfer": (-1e9, None)}
        for k in KS:
            u, tab, _ = group_mean(kv, resid, k)          # infold
            v_in = r2(pv + apply_table(u, tab, kv), yv)
            add = np.zeros(len(kv))                        # cv2 (교차적합)
            for msk in (half, ~half):
                u2, tab2, _ = group_mean(kv[msk], resid[msk], k)
                add[~msk] = apply_table(u2, tab2, kv[~msk])
            v_cv = r2(pv + add, yv)
            u3, tab3, _ = group_mean(kt, dev_tr, k)        # transfer
            v_tf = r2(pv + apply_table(u3, tab3, kv), yv)
            for nm, v in (("infold", v_in), ("cv2", v_cv), ("transfer", v_tf)):
                if v > best[nm][0]:
                    best[nm] = (v, k)
        out[name] = {nm: dict(rho2=best[nm][0], k=best[nm][1]) for nm in best}
        out[name]["cells"] = ncell
        print(f"{name:<24}{ncell:>8,}"
              f"{best['infold'][0]:>10.1f}{best['cv2'][0]:>10.1f}"
              f"{best['transfer'][0]:>10.1f}{best['cv2'][0]-base:>+10.1f}"
              f"   ({time.time()-t:.0f}s)", flush=True)

    out["_base"] = dict(rho2=base, seeds=3)
    json.dump(out, io.open(os.path.join(ROOT, "exp", "oracle_probe.json"), "w",
                           encoding="utf-8"), indent=1, ensure_ascii=False)
    print("\n읽는 법 — cv2 가 기준선보다 크게 높고 transfer 가 낮은 그룹이 "
          "'시즌내 정보가 남은 자리'다. D 가 그 자리였다.")


if __name__ == "__main__":
    main()
