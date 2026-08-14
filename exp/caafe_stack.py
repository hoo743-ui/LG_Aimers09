r"""CAAFE 이득이 챔피언의 편차 후처리와 **더해지는가, 겹치는가**.

## 왜 이것이 채택의 결정적 검사인가

2라운드에서 생존 11개가 3/3 · 기하평균 1.0279 로 부호 게이트를 통과했다.
그러나 챔피언은 이미 편차 후처리(`p += 0.20 d1 + 0.825 dC + 0.28 dN + 0.45 d3`)를
쓰고 있고, 그것이 3/3 · 1.0591 이다.

**둘이 같은 정보면 합쳐도 안 는다.** 실제로 CAAFE 중요도 1위인 `cf_same_hand`는
투수×타자손이고, 편차항 `d1` 이 정확히 그 축이다 (`dev(투수x타자손 | 부모=투수)`).
겹칠 공산이 크다. 반면 `cf_share_reverse`(실패 구성비) · `cf_form5`(폼 편차) ·
`cf_mix_entropy`(구종 엔트로피)는 편차 표가 담지 않는 정보다.

§9-b 에서 in-model 편차가 후처리와 **총량이 같다**는 것을 후처리 재적용 증분이
0.0 인 것으로 확인했다. 같은 검사를 CAAFE 에 적용한다.

## 판정

    (대조+CAAFE)+후처리  >  대조+후처리      -> 더해진다. 채택 후보.
    두 값이 같다                            -> 겹친다. CAAFE 는 편차의 재표현일 뿐.

후처리 가중은 19회차 제출본 그대로 쓴다 (LB 로 맞춘 값이라 이 비교에서 유리하게
편향되지 않는다 — 오히려 후처리 쪽에 유리한 조건이다).

    .\.venv\Scripts\python.exe -u exp\caafe_stack.py
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
KEEP = ["cf_same_hand", "cf_form5", "cf_share_reverse", "cf_share_ball",
        "cf_mix_entropy", "cf_log_pn", "cf_trend13", "cf_trend35",
        "cf_midform1", "cf_midtrend13", "cf_ball_minus_strike"]
W = np.array([0.20, 0.825, 0.280, 0.45])      # 19회차 제출본
FOLDS = (2022, 2023, 2024)


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
    BB, SS = C("balls_before").astype(np.int64), C("strikes_before").astype(np.int64)
    OB = (C("num_runners_on") > 0).astype(np.int64)
    PH, CNT = P * 10 + BH, BB * 4 + SS
    PHA = PH * 10 + (SS > BB).astype(np.int64)
    AXES = [(P, PH, 300), (PH, PHA, 2000),
            (PHA, PHA * 100 + CNT, 800), (PH, PH * 10 + OB, 2000)]

    base = np.column_stack([np.asarray(X[:, ixc[c]], dtype=np.float32)
                            for c in prod])
    F = build(C)
    names = list(F)
    caafe = np.column_stack([F[k] for k in KEEP])
    print(f"CAAFE 생존 {len(KEEP)}개   후처리 가중 {W.tolist()}\n")

    res = {}
    for f in FOLDS:
        tr, va = season < f, season == f
        yv = y[va]
        dv = np.column_stack([lookup(*nested_dev(par[tr], ch[tr], y[tr], k),
                                     ch[va]) for par, ch, k in AXES])
        post = dv @ W
        print(f"=== 폴드 {f}   학습 {int(tr.sum()):,}행 ===")
        row = {}
        for nm, M in (("대조", base), ("대조+CAAFE", np.hstack([base, caafe]))):
            t = time.time()
            m = CatBoostClassifier(random_seed=42, **HP)
            m.fit(M[tr], y[tr].astype(int))
            p = m.predict_proba(M[va])[:, 1].astype(np.float64)
            r0 = 1e5 * np.corrcoef(p, yv)[0, 1] ** 2
            r1 = 1e5 * np.corrcoef(p + post, yv)[0, 1] ** 2
            row[nm], row[nm + "+후처리"] = r0, r1
            print(f"  {nm:<12}{r0:>9.1f}   +후처리 {r1:>9.1f}"
                  f"   (후처리 증분 {r1 - r0:>+7.1f}){time.time() - t:>6.0f}s",
                  flush=True)
        res[f] = row

    print(f"\n=== 채택 검사 — 대조+후처리 대비 배수 ===")
    print(f"  {'폴드':>7}{'대조+후처리':>13}{'대조+CAAFE+후처리':>19}{'배수':>9}")
    ms = []
    for f in FOLDS:
        a, b = res[f]["대조+후처리"], res[f]["대조+CAAFE+후처리"]
        ms.append(b / a)
        print(f"  {f:>7}{a:>13.1f}{b:>19.1f}{b / a:>9.4f}")
    g = np.exp(np.mean(np.log(ms)))
    print(f"  {'기하평균':>7}{'':>13}{'':>19}{g:>9.4f}   "
          f"개선폴드 {sum(x > 1 for x in ms)}/{len(ms)}")
    print(f"\n  955.22 환산  ->  {955.2193198652 * g:.1f} "
          f"({955.2193198652 * (g - 1):+.1f})")
    print(f"\n  참고 — CAAFE 단독 이득(§2라운드) 1.0279, 후처리 단독 1.0591.")
    print(f"  완전 가산이면 배수가 1.0279 근처, 완전 중복이면 1.0000 근처다.")


if __name__ == "__main__":
    main()
