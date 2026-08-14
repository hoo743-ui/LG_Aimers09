r"""PHASE 3 — 상호작용. Phase 1-2 가 찾아낸 구조 위에 설계한다.

## Phase 1-2 가 남긴 것

    CAAFE10 (same_hand 없이 나머지 10개)   0.9912   해롭다
    same_hand 만                          1.0105   부족하다 (2/3)
    둘을 합치면                            1.0256   3/3

부분의 합보다 전체가 크고, **3폴드 모두에서 시너지가 양수**다
(2022 +0.35% / 2023 +6.2% / 2024 +0.7%).

> 폼 변화 · 실패 구성 · 구종 다양성은 그 자체로는 잡음이고,
> **손 조합을 조건으로 걸었을 때만 신호가 된다.**

야구적으로 — 투수의 난조나 실패 유형은 어느 손 타자를 상대하느냐에 따라 다르게
발현된다. 좌투수가 좌타 상대로 슬라이더 제구가 흔들리는 것과 우타 상대로
흔들리는 것은 다른 현상인데, 조건 없이 넣으면 상쇄돼 잡음이 된다.

## 🚩 데이터 제약 — 지시된 `pitch_type` 상호작용은 구성 불가

`train.csv` 원본 50컬럼에 **해당 투구의 구종이 없다.** 구종 관련은
`asof_pitcher_{fastball,breaking,offspeed}_rate` 셋뿐이고 이는 그 투수의 **과거
구사 비율**이다. TrackMan 에는 `tagged_pitch_type` 이 있으나 ① 2024 에서 끝나
2025 평가셋에 없고 ② 행 단위 조인이 다섯 겹으로 막힌다 (§6-o).

따라서 `count x pitch_type`, `pitch_type x same_hand` 는 만들 수 없다. 대신
**구종 성향(tendency)과의 상호작용**으로 대체한다 — 의미상 "이 투수가 이 상황에서
무엇을 던지려 하는가"의 근사다.

## 후보 — 각각이 표현하는 야구적 의사결정

  1. `hp_count`  손조합 x 카운트(12)        좌투-좌타 2S 의 유인구는 슬라이더고 존을
                                            크게 벗어난다. 우투-우타와 실패 양상이 다르다.
  2. `hp_adv`    손조합 x 투수유리           유리/불리에서 플래툰별 공격성 차이.
  3. `hp_risp`   손조합 x 득점권             주자 2·3루에서 플래툰 열세면 승부를 피한다.
  4. `sh_ent`    same_hand x 구종엔트로피    구종이 다양한 투수는 플래툰 열세를 보완한다.
  5. `sh_form5`  same_hand x 폼편차          난조가 손 조합별로 다르게 발현된다 (핵심 가설).
  6. `sh_srev`   same_hand x 반대방향비율     반대방향 실패는 좌우 대응에서 나온다.
  7. `sh_sball`  same_hand x 볼비율          유인구 성향의 플래툰 의존.
  8. `hp_rate`   손조합의 as-of 성공률        모집단 수준 플래툰 효과. `d1` 은 투수
                                            평균을 빼므로 이 수준 성분을 담지 못한다.
  9. `hp_inn`    손조합 x 이닝구간            불펜 좌완 스페셜리스트의 등판 국면.

`hp_rate` 만 학습 데이터에서 계산하며 **시즌 `<f` 로만** 만든다 (누수 차단).
나머지는 전부 행 안의 값이다.

## 평가

기준선은 **CAAFE11 + 후처리** (Phase 1-b 의 최고, 1.0256)다. 그 위에서 각
상호작용을 단독으로 얹어 3폴드 배수를 잰다. 채택 게이트는 3/3.

    .\.venv\Scripts\python.exe -u exp\phase3.py
"""
import io
import json
import os
import time

import numpy as np
from catboost import CatBoostClassifier

from caafe import build


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "exp", "cache")
HP = dict(iterations=1200, learning_rate=0.02, depth=6, l2_leaf_reg=100.0,
          border_count=32, loss_function="Logloss", verbose=0,
          thread_count=16, allow_writing_files=False)
WPOST = np.array([0.20, 0.825, 0.280, 0.45])
FOLDS = (2022, 2023, 2024)
KEEP = ["cf_same_hand", "cf_form5", "cf_share_reverse", "cf_share_ball",
        "cf_mix_entropy", "cf_log_pn", "cf_trend13", "cf_trend35",
        "cf_midform1", "cf_midtrend13", "cf_ball_minus_strike"]


def nested_dev(parent, child, y, k):
    o = np.argsort(child, kind="stable")
    Ys, Ps, Cs = y[o], parent[o], child[o]
    u, s = np.unique(Cs, return_index=True)
    cnt = np.diff(np.append(s, len(Cs)))
    cell = np.add.reduceat(Ys, s) / cnt
    par = Ps[s]
    op = np.argsort(parent, kind="stable")
    Yp, Pp = y[op], parent[op]
    pu, ps = np.unique(Pp, return_index=True)
    pc = np.diff(np.append(ps, len(Pp)))
    pmean = np.add.reduceat(Yp, ps) / pc
    return u, cnt * (cell - pmean[np.searchsorted(pu, par)]) / (cnt + k)


def lookup(u, dev, keys):
    ix = np.clip(np.searchsorted(u, keys), 0, len(u) - 1)
    ok = u[ix] == keys
    out = np.zeros(len(keys))
    out[ok] = dev[ix[ok]]
    return out


def main():
    meta = json.load(io.open(f"{CACHE}/cols.json", encoding="utf-8"))
    ixc = {c: i for i, c in enumerate(meta["cols"])}
    prod = meta["prod"]
    X = np.load(f"{CACHE}/X.npy", mmap_mode="r")
    y = np.load(f"{CACHE}/y.npy").astype(np.float64)
    season = np.load(f"{CACHE}/season.npy")
    C = lambda n: np.asarray(X[:, ixc[n]], dtype=np.float64)

    P = C("pitcher_id").astype(np.int64)
    BH = C("batter_hand").astype(np.int64)
    PHD = C("pitcher_hand").astype(np.int64)
    BB, SS = C("balls_before").astype(np.int64), C("strikes_before").astype(np.int64)
    OB = (C("num_runners_on") > 0).astype(np.int64)
    R2, R3 = C("runner_on_2b").astype(np.int64), C("runner_on_3b").astype(np.int64)
    INN = np.clip(C("inning"), 1, 9).astype(np.int64)
    PH, CNT = P * 10 + BH, BB * 4 + SS
    ADV = (SS > BB).astype(np.int64)
    PHA = PH * 10 + ADV
    AXES = [(P, PH, 300), (PH, PHA, 2000),
            (PHA, PHA * 100 + CNT, 800), (PH, PH * 10 + OB, 2000)]

    HP4 = (PHD * 2 + BH).astype(np.int64)          # 손 조합 4단
    SH = (PHD == BH).astype(np.float64)
    RISP = ((R2 + R3) > 0).astype(np.int64)
    INN3 = np.digitize(INN, [4, 7]).astype(np.int64)

    base = np.column_stack([np.asarray(X[:, ixc[c]], dtype=np.float32)
                            for c in prod])
    F = build(C)
    caafe = np.column_stack([F[k] for k in KEEP])
    base_c = np.hstack([base, caafe])

    ENT, F5 = F["cf_mix_entropy"].astype(np.float64), F["cf_form5"].astype(np.float64)
    SREV, SBAL = (F["cf_share_reverse"].astype(np.float64),
                  F["cf_share_ball"].astype(np.float64))

    INTER = {
        "hp_count": (HP4 * 12 + CNT).astype(np.float32),
        "hp_adv": (HP4 * 2 + ADV).astype(np.float32),
        "hp_risp": (HP4 * 2 + RISP).astype(np.float32),
        "sh_ent": (SH * ENT).astype(np.float32),
        "sh_form5": (SH * F5).astype(np.float32),
        "sh_srev": (SH * SREV).astype(np.float32),
        "sh_sball": (SH * SBAL).astype(np.float32),
        "hp_inn": (HP4 * 3 + INN3).astype(np.float32),
    }
    print(f"기준선 = CAAFE11 + 후처리 (Phase 1-b 최고, 1.0256)")
    print(f"상호작용 후보 {len(INTER) + 1}개 (hp_rate 는 폴드마다 재계산)\n")

    R = {}
    for f in FOLDS:
        tr, va = season < f, season == f
        yv = y[va]
        post = np.column_stack([lookup(*nested_dev(par[tr], ch[tr], y[tr], k),
                                       ch[va]) for par, ch, k in AXES]) @ WPOST
        # 손 조합의 as-of 성공률 — 시즌 <f 로만 만든다
        hr = np.zeros(len(y), dtype=np.float32)
        for v in np.unique(HP4):
            mm = HP4 == v
            hr[mm] = y[mm & tr].mean() if (mm & tr).sum() else 0.5
        cands = [("기준 CAAFE11", base_c)] + \
                [(k, np.hstack([base_c, v[:, None]])) for k, v in INTER.items()] + \
                [("hp_rate", np.hstack([base_c, hr[:, None]]))]
        print(f"=== 폴드 {f}  학습 {int(tr.sum()):,}행 ===")
        for nm, M in cands:
            t = time.time()
            m = CatBoostClassifier(random_seed=42, **HP)
            m.fit(M[tr], y[tr].astype(int))
            p = m.predict_proba(M[va])[:, 1].astype(np.float64)
            r1 = 1e5 * np.corrcoef(p + post, yv)[0, 1] ** 2
            R.setdefault(nm, {})[f] = r1
            print(f"  {nm:<16}{r1:>10.1f}"
                  f"{r1 - R['기준 CAAFE11'][f]:>+9.1f}{time.time() - t:>6.0f}s",
                  flush=True)

    print(f"\n=== 기준(CAAFE11+후처리) 대비 배수 ===")
    print(f"  {'상호작용':<16}" + "".join(f"{f:>10}" for f in FOLDS)
          + f"{'기하평균':>10}{'3/3':>6}")
    rows = []
    for nm in R:
        if nm == "기준 CAAFE11":
            continue
        ms = [R[nm][f] / R["기준 CAAFE11"][f] for f in FOLDS]
        g = float(np.exp(np.mean(np.log(ms))))
        rows.append((g, nm, ms, sum(x > 1 for x in ms)))
        print(f"  {nm:<16}" + "".join(f"{m:>10.4f}" for m in ms)
              + f"{g:>10.4f}{str(sum(x > 1 for x in ms)) + '/3':>6}")
    rows.sort(reverse=True)
    print(f"\n  최고: {rows[0][1]}  기하평균 {rows[0][0]:.4f}  "
          f"{rows[0][3]}/3   ->  955 환산 "
          f"{955.2193198652 * 1.0256 * rows[0][0]:.1f}")
    json.dump({k: {str(f): v for f, v in d.items()} for k, d in R.items()},
              io.open(os.path.join(ROOT, "exp", "phase3_result.json"), "w",
                      encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
