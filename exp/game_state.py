r"""압축된 상태 탐색 — **경기 단위 상태**의 크기를 처음으로 잰다. 학습 0회.

## 착안 — 경기 경계는 복원된다

`asof_pitcher_prev1_game_success_rate` 는 **직전 경기**의 비율이므로 한 경기
안에서는 상수이고 **경기가 바뀔 때만 값이 변한다.** `asof_pitcher_n` 은 투구마다
정확히 1씩 증가한다(연속성 100% 검증). 따라서 투수별로 `asof_n` 순으로 정렬하고
prev 값이 바뀌는 지점을 끊으면 **경기 단위가 복원된다.**

이건 train 안에서만 가능한 진단이다 (test 는 다른 행을 못 본다 — 규정 4).
그러나 **"우리가 무엇을 잃고 있는가"의 크기를 처음으로 측정**할 수 있다.

## 무엇을 재는가

목표 `control_success` 는 투구 위치로 정해지고, 실패 세 유형 중 하나가
**"포수의 요구 방향과 반대"** 다. 즉 라벨은 포수라는 **제3의 행위자**에
의존하는데 데이터에 포수가 없다. 같은 경기 안에서 공유되는 상태는 그밖에도
심판·구장·날씨·휴식일·경기 내 피로가 있고 **전부 데이터에 없다.**

    (투수, 경기) 오라클  =  그 경기의 실제 잔차 평균을 알면 얼마나 오르는가

이 값이 크면 "압축돼 사라진 상태"의 크기가 그만큼이라는 뜻이고, 작으면
경기 단위 상태는 애초에 없다는 뜻이다. 위약(같은 크기의 무작위 그룹)과 함께 잰다.

## 함께 재는 것

    경기 내 투구 순번의 잔차 추세   경기 내 피로가 있는가
    경기 크기 분포                 복원이 맞는지 검증 (불펜 1~30 / 선발 90~119)

    .\.venv\Scripts\python.exe -u exp\game_state.py
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
from d_limits import cv2_gain                              # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def main():
    tr = build_df()
    season = tr["season"].to_numpy()
    y = tr["control_species"] if False else tr["control_success"].to_numpy(np.float64)
    m24 = season == 2024
    P = np.load(os.path.join(ROOT, "exp", "prod_champ_2024.npy"))
    pred = P[:3].mean(0) + np.load(os.path.join(ROOT, "exp",
                                                "prod_post_2024.npy"))
    y24 = y[m24]
    base = 1e5 * np.corrcoef(pred, y24)[0, 1] ** 2
    half = np.random.default_rng(0).random(int(m24.sum())) < 0.5
    print(f"기준선 {base:.1f}   폴드 2024 {int(m24.sum()):,}행")

    # --- 경기 경계 복원 ---
    pid = tr["pitcher_id"].to_numpy(np.int64)[m24]
    an = tr["asof_pitcher_n"].to_numpy(np.float64)[m24]
    p1 = tr["asof_pitcher_prev1_game_success_rate"].to_numpy(np.float64)[m24]
    p1m = tr["asof_pitcher_prev1_game_middle_rate"].to_numpy(np.float64)[m24]
    o = np.lexsort((an, pid))                      # 투수별 asof_n 오름차순
    key = np.zeros(len(o), np.int64)
    gid = 0
    prev_p, prev_v, prev_v2 = -1, np.nan, np.nan
    for i in o:
        newp = pid[i] != prev_p
        chg = not (np.isclose(p1[i], prev_v, equal_nan=True)
                   and np.isclose(p1m[i], prev_v2, equal_nan=True))
        if newp or chg:
            gid += 1
        key[i] = gid
        prev_p, prev_v, prev_v2 = pid[i], p1[i], p1m[i]
    u, cnt = np.unique(key, return_counts=True)
    print(f"\n=== 경기 경계 복원 ===")
    print(f"  복원된 (투수,경기) 단위 {len(u):,}개   투수 {len(set(pid)):,}명")
    print(f"  경기당 투구수  중앙값 {np.median(cnt):.0f}  평균 {cnt.mean():.1f}"
          f"  90분위 {np.percentile(cnt,90):.0f}  최대 {cnt.max()}")
    q = np.percentile(cnt, [10, 25, 50, 75, 90, 99])
    print(f"  분위 10/25/50/75/90/99 = " + " ".join(f"{v:.0f}" for v in q))
    print(f"  참고 — PN 감사의 경기당 투구수: 불펜 1~30, 선발 90~119, 최대 151")
    print(f"  1투구 단위 비율 {np.mean(cnt == 1):.1%}  (교체 등판/복원 오차)")

    # 경기 내 순번
    idx = np.zeros(len(o), np.int64)
    pos = {}
    for i in o:
        k = key[i]
        pos[k] = pos.get(k, 0) + 1
        idx[i] = pos[k]

    print(f"\n=== 오라클 (폴드 2024, 교차적합 + 축소 훑기) ===")
    rng = np.random.default_rng(7)
    sizes = pd.Series(cnt, index=u)
    # 위약 — 같은 투수 안에서 같은 크기 분포로 무작위 그룹
    plac = np.zeros(len(key), np.int64)
    for p in np.unique(pid):
        m = pid == p
        n = int(m.sum())
        ng = max(1, len(np.unique(key[m])))
        plac[m] = p * 10000 + rng.integers(0, ng, n)
    GR = {
        "(투수, 경기)": key,
        "위약: 투수 x 무작위(같은 그룹수)": plac,
        "투수 (참조)": pid,
        "경기 내 순번 십분위": np.digitize(idx, np.percentile(idx, np.arange(10, 100, 10))),
        "경기 크기 십분위": np.digitize(sizes.reindex(key).to_numpy(),
                                 np.percentile(cnt, np.arange(10, 100, 10))),
    }
    out = {}
    print(f"{'그룹':<34}{'셀':>8}{'오라클':>10}{'k':>8}")
    for n, k in GR.items():
        v, nc, bk = cv2_gain(k, pred, y24, half)
        out[n] = dict(cells=nc, gain=float(v), k=bk)
        print(f"  {n:<32}{nc:>8,}{v:>+10.1f}{bk:>8}")

    print(f"\n=== 경기 내 피로 — 순번별 잔차 ===")
    res = y24 - pred
    print(f"{'경기 내 순번':<14}{'행수':>10}{'잔차평균':>10}{'성공률':>9}")
    for lo, hi in ((1, 5), (6, 15), (16, 30), (31, 60), (61, 90), (91, 200)):
        m = (idx >= lo) & (idx <= hi)
        if m.sum() > 500:
            print(f"  {f'{lo}~{hi}':<12}{int(m.sum()):>10,}{res[m].mean():>+10.4f}"
                  f"{y24[m].mean():>9.4f}")
    c = np.corrcoef(idx, res)[0, 1]
    print(f"  corr(경기 내 순번, 잔차) = {c:+.4f}  ({c*np.sqrt(len(idx)):+.1f}SE)")

    json.dump(out, io.open(os.path.join(ROOT, "exp", "game_state.json"), "w",
                           encoding="utf-8"), indent=1, ensure_ascii=False)
    print("\n판정 — (투수,경기) 오라클이 위약보다 크게 높으면 '경기 단위로 압축된 "
          "상태'가 실재한다. 그 크기가 곧 우리가 잃고 있는 정보의 상한이다.")


if __name__ == "__main__":
    main()
