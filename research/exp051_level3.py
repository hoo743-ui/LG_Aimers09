r"""EXP051 — 세 번째 **수준(level)** 축을 찾는다. 재학습 0회.

## 왜 수준인가

실측된 전이율 중 1 을 넘는 것은 수준 축뿐이다 (투수 +1.49, 타자 +2.34).
그런데 수준 축은 두 개뿐이고, 원장 §9 는 이렇게 경고한다 —

> 어떤 축을 닫을 때 "대비로 닫았는가 수준으로 닫았는가"를 구분해 적어라.
> 타자 축도 전부 **대비**로만 재고 닫았는데 수준으로 보니 2024 +4.2 가 나왔다.

팀 축은 "오라클 +11.6, 전이 −9.7" 로 닫혀 있는데, §5-b 의 `k` 규칙
("전이 최적 k 는 적률법의 10~100배") 은 **그 뒤에 나왔다.** 팀은 셀당 15만 행이라
`k < 10^5` 면 `n/(n+k) ≈ 1` 로 축소가 사실상 안 걸린다. 큰 `k` 로는 재본 적이 없다.

## 절차

후처리 축은 모델을 바꾸지 않으므로 캐시된 보정열에 10번째 열을 더해 재면 된다.
`exp043_comp.npz` (모델 예측 + 현행 9개 보정열, 폴드 2024) 를 그대로 쓴다.

    .\.venv\Scripts\python.exe -u research\exp051_level3.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "exp"))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

W9 = np.array([0.099765, 0.411532, 0.139671, 0.224472,
               0.703699, 0.775302, 0.775315, 1.984998, 2.105])   # rob2


def main():
    z = np.load(os.path.join(ROOT, "exp", "cache", "exp043_comp.npz"))
    mm, Cm, y = z["mm"], z["Cm"], z["y"]
    from path_alloc import build_df
    tr = build_df()
    season = tr["season"].to_numpy()
    m = season == 2024
    assert int(m.sum()) == len(y), "폴드 정렬 불일치"
    ya = tr["control_success"].to_numpy(np.float64)

    cur = mm + Cm @ W9
    base = 1e5 * np.corrcoef(cur, y)[0, 1] ** 2
    print(f"현행 9축 (rob2 가중)  폴드 2024 = {base:.2f}")

    # 현행 예측 기준의 잔차로 표를 만든다 (직전 2시즌, strictly OOF)
    # 학습 시즌 행의 잔차는 워크포워드 모델이 없으므로, 원장의 표 제작 관행대로
    # 각 원천 시즌을 그 이전 시즌으로 학습한 예측의 잔차로 쓴다.
    RES = {}
    for f in (2022, 2023):
        mf = season == f
        p = np.load(os.path.join(ROOT, "exp", f"prod_champ_{f}.npy")).mean(0)
        RES[f] = ya[mf] - p

    KEYS = {
        "투수팀": "pitcher_team_id",
        "타자팀": "batter_team_id",
        "구장(홈팀)": None,          # top_bottom 으로 유도
        "투수x투수팀": None,
        "타자x타자팀": None,
    }
    TB = tr["top_bottom"].to_numpy()
    home = np.where(pd.Series(TB).astype(str).str.upper().str[0].to_numpy() == "T",
                    tr["pitcher_team_id"].to_numpy(np.int64),
                    tr["batter_team_id"].to_numpy(np.int64))
    P = tr["pitcher_id"].to_numpy(np.int64)
    B = tr["batter_id"].to_numpy(np.int64)
    COL = {"투수팀": tr["pitcher_team_id"].to_numpy(np.int64),
           "타자팀": tr["batter_team_id"].to_numpy(np.int64),
           "구장(홈팀)": home,
           "투수x투수팀": P * 100 + tr["pitcher_team_id"].to_numpy(np.int64) % 100,
           "타자x타자팀": B * 100 + tr["batter_team_id"].to_numpy(np.int64) % 100}

    print(f"\n{'축':12s} {'셀수':>6s} {'셀당행':>8s} " +
          "  ".join(f"k={k:>7g}" for k in (5e3, 5e4, 2e5, 1e6, 5e6)))
    out = {}
    for nm, key in COL.items():
        src = pd.concat([pd.DataFrame({"k": key[season == f], "r": RES[f]})
                         for f in (2022, 2023)])
        g = src.groupby("k")["r"].agg(["sum", "size"])
        row, best = [], (-9, None, None)
        for k in (5e3, 5e4, 2e5, 1e6, 5e6):
            t = (g["sum"] / g["size"]) * g["size"] / (g["size"] + k)
            v = pd.Series(key[m]).map(t).fillna(0.).to_numpy()
            gains = []
            for w in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
                s = 1e5 * np.corrcoef(cur + w * v, y)[0, 1] ** 2 - base
                gains.append((s, w))
            s, w = max(gains)
            row.append(f"{s:+6.2f}@{w:g}")
            if s > best[0]:
                best = (s, w, k)
        print(f"{nm:12s} {len(g):6d} {g['size'].mean():8.0f}  " + "  ".join(row))
        out[nm] = best
    print(f"\n{'축':12s} {'최적 이득':>9s} {'w':>5s} {'k':>10s}")
    for nm, (s, w, k) in sorted(out.items(), key=lambda x: -x[1][0]):
        print(f"{nm:12s} {s:+9.2f} {w:5g} {k:10g}")
    print("\n[대조] 현행 수준 2축을 같은 방법으로 재현하면")
    for nm, key, k0 in (("투수 수준", P, 50000.), ("타자 수준", B, 20000.)):
        src = pd.concat([pd.DataFrame({"k": key[season == f], "r": RES[f]})
                         for f in (2022, 2023)])
        g = src.groupby("k")["r"].agg(["sum", "size"])
        t = (g["sum"] / g["size"]) * g["size"] / (g["size"] + k0)
        v = pd.Series(key[m]).map(t).fillna(0.).to_numpy()
        gains = [(1e5 * np.corrcoef(cur + w * v, y)[0, 1] ** 2 - base, w)
                 for w in (0.5, 1., 1.5, 2., 2.5, 3.)]
        s, w = max(gains)
        print(f"  {nm:10s} 추가 이득 {s:+6.2f} @ w={w:g}  "
              f"(이미 실려 있으므로 0 근처여야 정상)")


if __name__ == "__main__":
    main()
