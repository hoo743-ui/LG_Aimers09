r"""4-9 패턴을 trackman 물리량으로 확장한 피처를 캐시에 덧붙인다.

`tmx_probe.py` 가 통과시킨 3개만 쓴다 — spin_rate / horz_break /
induced_vert_break. 셋 다 이미 채택된 `rel_speed`(신호잡음 3.33/1.84)보다
신호잡음과 실무적 편차 크기가 **둘 다** 높다.

시점 규칙·편차 형태·최소 셀 크기는 `final_train.py` 의 4-9 구현을 그대로
따른다. 다른 규칙을 쓰면 채택돼도 제출물에 옮길 수 없다.

기존 캐시를 덮어쓰지 않고 컬럼을 **뒤에 덧붙인다.** cols.json 에 `prod` 를
명시 목록으로 박아, 이미 캐시된 실험의 피처 집합 의미가 바뀌지 않게 한다.

    .\.venv\Scripts\python.exe exp\prep_tmx.py
"""
import json
import os

import numpy as np
import pandas as pd

import prep as P                                  # exp/prep.py 재사용

ROOT = P.ROOT
CACHE = P.CACHE
DATA = P.DATA

# 확장 물리량. 이름은 짧게 — 컬럼명이 곧 캐시 키의 일부다.
QUANT = {"spin_rate": "spin", "horz_break": "hb", "induced_vert_break": "ivb"}
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
    """upto 미만 시즌으로 (카운트 편차표, 좌우 편차표). 4-9 와 같은 구조."""
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
        cols = {}
        for q, short in QUANT.items():
            cols[f"{prefix}_{short}_dev"] = tab[q] - tab[f"{q}_all"]
        out.append(pd.DataFrame(cols))
    return out[0], out[1]


def main():
    os.chdir(ROOT)
    ft = P.load_final_train()
    os.chdir(ROOT)

    meta = json.load(open(f"{CACHE}/cols.json"))
    X = np.load(f"{CACHE}/X.npy")
    season = np.load(f"{CACHE}/season.npy")
    print(f"기존 캐시 X {X.shape}")

    if any(c.endswith("_spin_dev") for c in meta["cols"]):
        print("이미 확장돼 있다. 중단.")
        return

    tm = load_tm(ft)
    print(f"대응된 trackman {len(tm):,}건")

    # 조인 키를 캐시에서 되살린다 (CSV 재파싱 회피)
    ix = {c: i for i, c in enumerate(meta["cols"])}
    key = pd.DataFrame({
        "pitcher_id": X[:, ix["pitcher_id"]].astype(np.int64),
        "balls_before": X[:, ix["balls_before"]].astype(np.int64),
        "strikes_before": X[:, ix["strikes_before"]].astype(np.int64),
        "batter_hand": X[:, ix["batter_hand"]].astype(np.int64),
        "season": season.astype(np.int64),
    })
    key["_hand"] = key["batter_hand"].map(ft.HAND)

    new_cols = ([f"tmc_{s}_dev" for s in QUANT.values()] +
                [f"tmh_{s}_dev" for s in QUANT.values()])
    NEW = np.full((len(X), len(new_cols)), np.nan, dtype=np.float32)

    for S in sorted(set(season.tolist())):
        cnt, hnd = tables(tm, S)
        if cnt is None:
            continue
        m = season == S
        sub = key[m]
        c = sub.join(cnt, on=COUNT_KEY)
        h = sub.join(hnd, on=["pitcher_id", "_hand"])
        for j, col in enumerate(new_cols):
            src = c if col.startswith("tmc_") else h
            NEW[np.where(m)[0], j] = src[col].to_numpy(np.float32)
        print(f"  {S} 커버리지 카운트 {c[new_cols[0]].notna().mean():6.1%} "
              f"좌우 {h[new_cols[3]].notna().mean():6.1%}")

    X2 = np.hstack([X, NEW])
    cols2 = meta["cols"] + new_cols
    # 'prod' 를 명시 목록으로 고정한다. 새 컬럼이 자동으로 딸려 들어가면
    # 이미 캐시된 실험들의 'prod' 의미가 달라져 비교가 깨진다.
    prod = [c for c in meta["cols"] if c not in ("tmc_n", "tmh_n")]
    meta2 = {**meta, "cols": cols2, "prod": prod, "extra": new_cols}

    np.save(f"{CACHE}/X.npy", X2)
    with open(f"{CACHE}/cols.json", "w") as f:
        json.dump(meta2, f, indent=1)
    print(f"\n확장 완료: X {X2.shape} ({X2.nbytes/1e6:.0f}MB), "
          f"새 컬럼 {len(new_cols)}개")
    print(f"  prod 고정 {len(prod)}개 (기존 실험 영향 없음)")
    print(f"  {new_cols}")


if __name__ == "__main__":
    main()
