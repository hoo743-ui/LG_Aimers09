r"""투수의 좌/우타자별 제구 편차(플래툰 스플릿). 학습 0회.

## 왜 여기인가

4-21 의 구조 진단은 `pitcher_id` 만 봤다. 그룹 오라클을 **교차적합**으로(칸이 잘아도
안 부푸는 형태로) 다시 재니 가장 큰 것은 `pitcher_id` 가 아니라
**`pitcher x 타자손`** 이었다.

```
                 fold2024   fold2022   fold2021
pitcher_id         +90.83     +20.91    +223.67
pitcher x 타자손    +230.92     +81.37    +263.10     <- 3/3, 가장 크다
```

그리고 **이 피처가 없다.** `asof_*` 는 전부 투수 단위 집계이고 `batter_hand` 는
따로 놓인 컬럼이라, "이 투수가 좌타자 상대로 얼마나 잘 던지는가"는 어디에도 없다.
트리가 `pitcher_id x batter_hand` 를 스스로 쪼개려면 투수 391명 x 2 를 분기로
만들어야 하는데 규제(l2=100, min_leaf=1000)가 그걸 막는다.

## 왜 이번엔 전이될 수 있는가

지금까지 다섯 번 죽은 것(4-22·4-26·4-27·4-28·4-29)은 전부 **수준값**이거나
**폴드에서 맞춘 계수**였다. 플래툰은 **편차**다.

    plat[p] = (p 의 좌타 상대 성공률) - (p 의 전체 성공률)

리그 전체가 해마다 내려가도(4-2) 그 이동은 **뺄셈에서 소거된다.** 4-9 에서 채택된
것도 수준값이 아니라 편차였고, 4-18 에서 CTR 이 실패한 이유도 낡은 수준값을 실어
왔기 때문이다. 그리고 학습 구간에서만 계산하므로 5) 원칙에 안전하다.

## 판정

계수는 **이전 폴드에서만** 맞춘다. 고정 계수로도 3/3 인지 함께 본다 (함정 ③).

    .\.venv\Scripts\python.exe exp\platoon_probe.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import level_probe as L                                   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "exp", "cache")
FOLDS = [2021, 2022, 2024]
KSHRINK = 300          # 표본이 이만큼 될 때 사전값과 반반


def platoon_table(pid, hand, y, k=KSHRINK):
    """학습 구간에서 투수별 '손 상대 편차' 를 만든다.

    plat[p][h] = shrink( rate(p,h) - rate(p) )

    투수 전체 성공률을 빼므로 투수의 수준은 소거되고 **손에 대한 상대적 강약만**
    남는다. 리그 수준의 연도 이동도 같이 소거된다.
    """
    out = {}
    order = np.argsort(pid, kind="stable")
    up, sp = np.unique(pid[order], return_index=True)
    for i, s in enumerate(sp):
        e = sp[i + 1] if i + 1 < len(sp) else len(order)
        idx = order[s:e]
        base = y[idx].mean()
        for h in np.unique(hand[idx]):
            m = hand[idx] == h
            n = int(m.sum())
            out[(up[i], h)] = (n * (y[idx][m].mean() - base)) / (n + k)
    return out


def lookup(table, pid, hand):
    return np.array([table.get((p, h), 0.0) for p, h in zip(pid, hand)])


def main():
    meta = json.load(open(f"{CACHE}/cols.json"))
    ix = {c: i for i, c in enumerate(meta["cols"])}
    X = np.load(f"{CACHE}/X.npy", mmap_mode="r")
    y_all = np.load(f"{CACHE}/y.npy")
    season = np.load(f"{CACHE}/season.npy")

    pid_all = np.asarray(X[:, ix["pitcher_id"]], dtype=np.int64)
    hand_all = np.asarray(X[:, ix["batter_hand"]], dtype=np.int64)

    print("투수 x 타자손 편차 — 학습 구간에서만 만든다 (5) 안전, 재학습 0회)\n")
    print(f"타자손 값: {sorted(set(hand_all.tolist()))}\n")

    D, feats = {}, {}
    for fold in FOLDS:
        m = season == fold
        y = y_all[m].astype(np.float64)
        p = L.model_preds(fold)
        tr = season < fold                       # 그 폴드의 학습 구간만
        tab = platoon_table(pid_all[tr], hand_all[tr],
                            y_all[tr].astype(np.float64))
        f = lookup(tab, pid_all[m], hand_all[m])
        D[fold] = (p, y)
        feats[fold] = f
        cov = (f != 0).mean()
        print(f"fold {fold}: 학습 {tr.sum():,}행에서 표 {len(tab):,}칸, "
              f"커버리지 {cov:.1%}, 편차 표준편차 {f.std():.5f}, "
              f"범위 {f.min():+.4f}~{f.max():+.4f}")

    print(f"\n{'폴드':>6s} {'모델':>9s} {'폴드내적합':>10s} {'워크포워드':>10s} {'WF증분':>8s}")
    print("-" * 50)
    fits = {}
    for i, fold in enumerate(FOLDS):
        p, y = D[fold]
        f = feats[fold]
        solo = 1e5 * L.r2_of(y, [p])
        ins = 1e5 * L.r2_of(y, [p, f])
        A = np.column_stack([np.ones(len(y)), p, f])
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        fits[fold] = beta
        if i == 0:
            print(f"{fold:6d} {solo:9.2f} {ins:10.2f} {'—':>10s} {'—':>8s}")
            continue
        b = np.mean([fits[g] for g in FOLDS[:i]], axis=0)
        q = A @ b
        wf = 1e5 * np.corrcoef(q, y)[0, 1] ** 2
        print(f"{fold:6d} {solo:9.2f} {ins:10.2f} {wf:10.2f} {wf-solo:+8.2f}")

    print("\n계수 (모델 가중으로 정규화한 편차 가중)")
    for fold in FOLDS:
        b = fits[fold]
        print(f"  {fold}  {b[2]/b[1]:+.4f}")

    print("\n고정 가중 (모델 + w * 편차) — 폴드에서 아무것도 고르지 않는다")
    print(f"{'w':>6s} {'평균':>8s} {'부호':>5s}   "
          + "  ".join(str(f).rjust(9) for f in FOLDS))
    for w in (0.25, 0.5, 0.75, 1.0, 1.5):
        row = []
        for fold in FOLDS:
            p, y = D[fold]
            a = 1e5 * np.corrcoef(p + w * feats[fold], y)[0, 1] ** 2
            row.append(a - 1e5 * np.corrcoef(p, y)[0, 1] ** 2)
        print(f"{w:6.2f} {np.mean(row):8.2f} {sum(v>0 for v in row)}/3   "
              + "  ".join(f"{v:+9.2f}" for v in row))


if __name__ == "__main__":
    main()
