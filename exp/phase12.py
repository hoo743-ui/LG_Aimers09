r"""PHASE 1-2 — CAAFE 와 기존 편차의 **가산성**, 그리고 정보축 분리.

## 답해야 할 질문

1. CAAFE 이득이 기존 편차 후처리와 **중복인가 / 부분 가산인가 / 완전 가산인가**
2. 이득이 어느 정보축에서 나오는가 (플래툰 / 폼 / 실패구성·전략)

## 왜 `cf_same_hand` 를 따로 떼는가

CAAFE 중요도 1위가 `cf_same_hand`(투수손==타자손)인데, 기존 편차항 `d1` 이
`dev(투수x타자손 | 부모=투수)` 로 **정확히 같은 축**이다. 중요도가 높다는 것이
새 정보라는 뜻이 아니다 — 같은 정보를 다른 형태로 다시 준 것일 수 있다.
그래서 `same_hand` 를 뺀 판과 그것만 넣은 판을 따로 잰다.

## 평가

한 폴드 결과는 채택 근거가 못 된다 (§9-b 에서 `log(n)` 이 2024 에서만 +1.43%
였다가 폴드 검증에서 죽었다). 폴드 규모가 31배 다르므로(2022 2371 vs 2023 76.5)
절대차가 아니라 **대조 대비 배수**로 본다. 채택 게이트는 **3/3 부호 일관성**이다.

후처리 가중은 19회차 제출본 그대로다 (LB 6장으로 맞춘 값이라 후처리 쪽에 유리한
조건이다 — 그런데도 CAAFE 가 위에 얹히면 진짜 새 정보다).

    .\.venv\Scripts\python.exe -u exp\phase12.py
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
W = np.array([0.20, 0.825, 0.280, 0.45])
FOLDS = (2022, 2023, 2024)

KEEP11 = ["cf_same_hand", "cf_form5", "cf_share_reverse", "cf_share_ball",
          "cf_mix_entropy", "cf_log_pn", "cf_trend13", "cf_trend35",
          "cf_midform1", "cf_midtrend13", "cf_ball_minus_strike"]
# --- PHASE 2 의 세 정보축 ---
GRP_A = ["cf_same_hand"]                                    # 플래툰
GRP_B = ["cf_form1", "cf_form3", "cf_form5", "cf_trend13",  # 폼 / 시간 편차
         "cf_trend35", "cf_midform1", "cf_midtrend13"]
GRP_C = ["cf_share_middle", "cf_share_ball", "cf_share_reverse",  # 실패구성·전략
         "cf_mid_vs_ball", "cf_mix_entropy", "cf_mix_dominant",
         "cf_non_fastball", "cf_ball_minus_strike"]


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
    PH, CNT = P * 10 + BH, BB * 4 + SS
    PHA = PH * 10 + (SS > BB).astype(np.int64)
    AXES = [(P, PH, 300), (PH, PHA, 2000),
            (PHA, PHA * 100 + CNT, 800), (PH, PH * 10 + OB, 2000)]

    base = np.column_stack([np.asarray(X[:, ixc[c]], dtype=np.float32)
                            for c in prod])
    F = build(C)
    hand_pair = (PHD * 2 + BH).astype(np.float32)[:, None]
    col = lambda ks: np.column_stack([F[k] for k in ks])

    CONFIGS = [
        ("A  대조 prod55", base),
        ("C  +CAAFE11", np.hstack([base, col(KEEP11)])),
        ("   +CAAFE10 (-hand)", np.hstack([base, col([k for k in KEEP11
                                                      if k != "cf_same_hand"])])),
        ("   +same_hand 만", np.hstack([base, col(GRP_A)])),
        ("G-A 플래툰(+쌍코드)", np.hstack([base, col(GRP_A), hand_pair])),
        ("G-B 폼/시간편차", np.hstack([base, col(GRP_B)])),
        ("G-C 실패구성·전략", np.hstack([base, col(GRP_C)])),
    ]
    print(f"설정 {len(CONFIGS)}개 x 폴드 {len(FOLDS)}개   "
          f"각 설정마다 raw / +후처리 둘 다 잰다\n")

    R = {nm: {} for nm, _ in CONFIGS}
    for f in FOLDS:
        tr, va = season < f, season == f
        yv = y[va]
        post = np.column_stack([lookup(*nested_dev(par[tr], ch[tr], y[tr], k),
                                       ch[va]) for par, ch, k in AXES]) @ W
        print(f"=== 폴드 {f}  학습 {int(tr.sum()):,}행 ===")
        for nm, M in CONFIGS:
            t = time.time()
            m = CatBoostClassifier(random_seed=42, **HP)
            m.fit(M[tr], y[tr].astype(int))
            p = m.predict_proba(M[va])[:, 1].astype(np.float64)
            r0 = 1e5 * np.corrcoef(p, yv)[0, 1] ** 2
            r1 = 1e5 * np.corrcoef(p + post, yv)[0, 1] ** 2
            R[nm][f] = (r0, r1)
            print(f"  {nm:<22}{M.shape[1]:>4}p  raw {r0:>9.1f}   "
                  f"+후처리 {r1:>9.1f}   (후처리 증분 {r1 - r0:>+7.1f})"
                  f"{time.time() - t:>6.0f}s", flush=True)

    def table(title, key, denom_key):
        print(f"\n=== {title} ===")
        print(f"  {'후보':<22}" + "".join(f"{f:>10}" for f in FOLDS)
              + f"{'기하평균':>10}{'3/3':>6}")
        out = {}
        for nm, _ in CONFIGS:
            ms = [R[nm][f][key] / R["A  대조 prod55"][f][denom_key]
                  for f in FOLDS]
            g = float(np.exp(np.mean(np.log(ms))))
            out[nm] = (ms, g)
            print(f"  {nm:<22}" + "".join(f"{m:>10.4f}" for m in ms)
                  + f"{g:>10.4f}{str(sum(x > 1 for x in ms)) + '/3':>6}")
        return out

    table("PHASE 1-a  후처리 없이 — 대조(raw) 대비 배수", 0, 0)
    print(f"\n  [참고] 기존 편차 후처리 단독: "
          + "".join(f"{R['A  대조 prod55'][f][1] / R['A  대조 prod55'][f][0]:>10.4f}"
                    for f in FOLDS))

    t2 = table("PHASE 1-b  🚩 후처리 위에 얹었을 때 — 대조+후처리 대비 배수", 1, 1)
    print(f"\n  판정: 배수 ~1.0000 이면 중복, ~1.0279(CAAFE 단독 이득)이면 완전 가산")

    print(f"\n=== 종합 (대조 raw 대비) ===")
    print(f"  {'후보':<22}{'raw 기하평균':>13}{'+후처리 기하평균':>16}"
          f"{'955 환산':>11}")
    b0 = float(np.exp(np.mean(np.log(
        [R["A  대조 prod55"][f][1] / R["A  대조 prod55"][f][0] for f in FOLDS]))))
    for nm, _ in CONFIGS:
        g0 = float(np.exp(np.mean(np.log(
            [R[nm][f][0] / R["A  대조 prod55"][f][0] for f in FOLDS]))))
        g1 = float(np.exp(np.mean(np.log(
            [R[nm][f][1] / R["A  대조 prod55"][f][0] for f in FOLDS]))))
        print(f"  {nm:<22}{g0:>13.4f}{g1:>16.4f}"
              f"{955.2193198652 * g1 / b0:>11.1f}")
    json.dump({nm: {str(f): list(v) for f, v in d.items()}
               for nm, d in R.items()},
              io.open(os.path.join(ROOT, "exp", "phase12_result.json"), "w",
                      encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
