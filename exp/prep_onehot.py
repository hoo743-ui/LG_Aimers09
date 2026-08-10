r"""저카디널리티 범주형의 one-hot 컬럼을 캐시에 덧붙인다.

`card_probe.py` 결과가 근거다.

    pitcher_team_id / batter_team_id : 13 범주, 범주당 137,000 행,
                                       순위상관 -0.12 / +0.15, 미지 0%
    base_state                       : 8 범주, 범주당 81,000 행, 순위상관 0.405

지금은 이것들이 **ordinal 정수**로 들어간다. 순위상관이 0 근처라는 건 그 정수
축에 순서 정보가 없다는 뜻이고, 트리는 `team_id < 7` 같은 무의미한 분할로
범주를 갈라야 한다. 13 범주면 원핫 13 컬럼이라 값이 싸다.

**CTR 이 아니라 one-hot 이다.** 규칙 5 가 기각한 Team ID CTR 은 타깃 기반이라
폴드마다 부호가 뒤집혔다. one-hot 은 타깃을 전혀 쓰지 않으므로 그 불안정성이
구조적으로 없다. 같은 실패를 반복하는 게 아니다.

`inning`(-0.945) 과 `game_month`(-0.905) 는 순서가 뚜렷하므로 건드리지 않는다.
`pitcher_id`/`batter_id` 는 792/830 범주에 미지 20% 라 원핫 대상이 아니다.

    .\.venv\Scripts\python.exe exp\prep_onehot.py
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "exp", "cache")

TARGETS = ["pitcher_team_id", "batter_team_id", "base_state"]


def main():
    meta = json.load(open(f"{CACHE}/cols.json"))
    if any(c.startswith("oh_") for c in meta["cols"]):
        print("이미 원핫이 붙어 있다. 중단.")
        return

    X = np.load(f"{CACHE}/X.npy")
    ix = {c: i for i, c in enumerate(meta["cols"])}
    print(f"기존 캐시 X {X.shape}")

    blocks, names = [], []
    for c in TARGETS:
        v = np.asarray(X[:, ix[c]])
        vals = np.unique(v[~np.isnan(v)])
        print(f"  {c}: {len(vals)} 범주 -> 원핫 {len(vals)} 컬럼")
        B = np.zeros((len(X), len(vals)), dtype=np.float32)
        for j, u in enumerate(vals):
            B[:, j] = (v == u).astype(np.float32)
        blocks.append(B)
        names += [f"oh_{c}_{int(u)}" for u in vals]

    NEW = np.hstack(blocks)
    X2 = np.hstack([X, NEW])
    meta2 = {**meta, "cols": meta["cols"] + names,
             "onehot": names,
             "onehot_src": TARGETS}
    np.save(f"{CACHE}/X.npy", X2)
    with open(f"{CACHE}/cols.json", "w") as f:
        json.dump(meta2, f, indent=1)
    print(f"\n확장 완료: X {X2.shape} ({X2.nbytes/1e6:.0f}MB), 새 컬럼 {len(names)}개")
    print(f"  prod 는 명시 목록이라 그대로 {len(meta2['prod'])}개 — 기존 실험 영향 없음")


if __name__ == "__main__":
    main()
