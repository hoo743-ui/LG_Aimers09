r"""TMR — 릴리스 기하(extension / rel_height / rel_side)의 상황 조건부 편차.

## 기각된 TR-VAR 과 무엇이 다른가

25-a 가 기각한 것은 **산포**다 — `sd(rel_height)` 등을 투수 상수로 줬고 타깃
상관이 0.01 대라 떨어졌다. 그 절의 교훈은 "투수 상수는 이미 `asof_*` 요약이
담고 있고, 타깃과 같은 **시간 해상도**를 가져야 한다"였다.

TMR 은 산포가 아니라 **위치의 상황별 이동**이다.

    tmc_ext_dev = (이 투수가 이 카운트에서의 평균 extension)
                - (이 투수의 전체 평균 extension)

행마다 카운트·타자손으로 조회되므로 **행 단위로 값이 변한다.** 채택된 8열
(LB +10.98)과 같은 구성이고 측정 물리량만 다르다.

## 야구 메커니즘

제구는 반복성이다. 투수가 **카운트에 따라 릴리스 포인트를 옮긴다면** 그것은
의도적 조정이거나 무너짐이고, 둘 다 그 상황의 제구 성공률과 연결된다.
`extension` 이 줄면 팔이 덜 나온 것이고, `rel_side` 가 움직이면 슬롯이
바뀐 것이다. 결과 비율(`asof_*`)에는 **왜** 그랬는지가 없다.

## 캐시를 건드리지 않는다

`X.npy` 는 406MB 다. 여기에 덧붙이면 통째로 다시 써야 하고 기존 실험의
`prod` 의미도 흔들린다. TMR 은 **별도 파일**로 낸다.

    exp/cache/tmr.npy   (n, 8) float32
    exp/cache/tmr.json  컬럼명

    .\.venv\Scripts\python.exe exp\prep_tmr.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prep as P                                          # noqa: E402

ROOT, CACHE, DATA = P.ROOT, P.CACHE, P.DATA
QUANT = {"extension": "ext", "rel_height": "relh", "rel_side": "rels"}
MIN_COUNT_CELL = 30
MIN_HAND_CELL = 50
COUNT_KEY = ["pitcher_id", "balls_before", "strikes_before"]
HAND_KEY = ["pitcher_id", "batter_hand"]


def load_tm(ft):
    id_map = pd.read_csv(os.path.join(ROOT, "pitcher_id_map.csv"))
    id_map = id_map[id_map["conf"] >= ft.MIN_CONF]
    tm = pd.read_csv(f"{DATA}/trackman_history.csv", encoding="utf-8-sig",
                     usecols=["season", "pitcher_trackman_id", "balls_before",
                              "strikes_before", "batter_hand"] + list(QUANT))
    tm["pitcher_id"] = tm["pitcher_trackman_id"].map(
        id_map.set_index("pitcher_trackman_id")["pitcher_id"])
    tm = tm.dropna(subset=["pitcher_id"])
    tm["pitcher_id"] = tm["pitcher_id"].astype(np.int64)
    return tm


def tables(tm, upto):
    """`upto` 미만 시즌만 쓴다 (시점 규율). 4-9 구현과 같은 구조."""
    past = tm[tm["season"] < upto]
    if not len(past):
        return None, None
    base = past.groupby("pitcher_id")[list(QUANT)].mean()
    out = []
    for keys, prefix, min_cell in ((COUNT_KEY, "tmc", MIN_COUNT_CELL),
                                   (HAND_KEY, "tmh", MIN_HAND_CELL)):
        g = past.groupby(keys)
        tab = g[list(QUANT)].mean()
        tab["_n"] = g.size()
        tab = tab[tab["_n"] >= min_cell]
        tab = tab.join(base, on="pitcher_id", rsuffix="_all")
        cols = {f"{prefix}_{s}_dev": tab[q] - tab[f"{q}_all"]
                for q, s in QUANT.items()}
        # 릴리스 **변위 크기** — 세 축의 부호를 지우고 "얼마나 옮겼나"만 남긴다.
        # 방향은 투수마다 다르지만 '옮겼다'는 사실은 공통이라는 가설.
        cols[f"{prefix}_reldisp"] = np.sqrt(sum(v ** 2 for v in cols.values()))
        out.append(pd.DataFrame(cols))
    return out[0], out[1]


def main():
    os.chdir(ROOT)
    ft = P.load_final_train()
    os.chdir(ROOT)
    meta = json.load(open(f"{CACHE}/cols.json"))
    ix = {c: i for i, c in enumerate(meta["cols"])}
    X = np.load(f"{CACHE}/X.npy", mmap_mode="r")           # mmap — 통째 적재 안 함
    season = np.load(f"{CACHE}/season.npy")
    tm = load_tm(ft)
    print(f"대응된 trackman {len(tm):,}건")

    key = pd.DataFrame({
        "pitcher_id": np.asarray(X[:, ix["pitcher_id"]]).astype(np.int64),
        "balls_before": np.asarray(X[:, ix["balls_before"]]).astype(np.int64),
        "strikes_before": np.asarray(X[:, ix["strikes_before"]]).astype(np.int64),
        "batter_hand": np.asarray(X[:, ix["batter_hand"]]).astype(np.int64),
    })
    key["_hand"] = key["batter_hand"].map(ft.HAND)
    cols = ([f"tmc_{s}_dev" for s in QUANT.values()] + ["tmc_reldisp"] +
            [f"tmh_{s}_dev" for s in QUANT.values()] + ["tmh_reldisp"])
    NEW = np.full((len(key), len(cols)), np.nan, dtype=np.float32)
    for S in sorted(set(season.tolist())):
        cnt, hnd = tables(tm, S)
        if cnt is None:
            continue
        m = season == S
        sub = key[m]
        c = sub.join(cnt, on=COUNT_KEY)
        h = sub.join(hnd, on=["pitcher_id", "_hand"])
        w = np.where(m)[0]
        for j, col in enumerate(cols):
            NEW[w, j] = (c if col.startswith("tmc_") else h)[col].to_numpy(np.float32)
        print(f"  {S} 커버리지 카운트 {c[cols[0]].notna().mean():6.1%} "
              f"좌우 {h[cols[4]].notna().mean():6.1%}", flush=True)
    np.save(f"{CACHE}/tmr.npy", NEW)
    json.dump({"cols": cols}, open(f"{CACHE}/tmr.json", "w"), indent=1)
    print(f"\n저장 exp/cache/tmr.npy {NEW.shape}  ({NEW.nbytes/1e6:.0f}MB)")
    print(f"  {cols}")


if __name__ == "__main__":
    main()
