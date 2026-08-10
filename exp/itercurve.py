r"""반복수를 캐시된 곡선에서 고른다. 재학습 0회.

`runner.py` 는 학습마다 반복수별 검증 점수 곡선을 저장한다. 그래서 "반복수를
1200 대신 700 으로 하면?" 은 **다시 학습하지 않고** 답할 수 있다.

## 왜 필요한가

CatBoost 의 곡선 최적점은 폴드마다 300~960 인데 제출 설정은 1200(전체 재학습
1320)이다. fold2024 seed42 에서 곡선 최적 725.72@660 vs 고정 712.29@1200 —
**13점을 과학습으로 버리고 있다.**

## 고르는 방법 — 워크포워드여야 한다

각 폴드에서 그 폴드 최적을 고르면 검증셋에 맞춘 낙관 편향이다 (4-6). 여기서는
**이전 폴드들에서 고른 반복수를 다음 폴드에 적용**해 실제 이득을 잰다.

학습량 보정도 같이 본다. 폴드마다 학습 행 수가 다르고 (2021 48만 / 2022 73만 /
2024 122만), 최적 반복수는 학습량과 함께 커지는 경향이 있다. 2025 는 147만이라
2024 보다 크므로, 고른 값을 그대로 쓰면 과소일 수 있다.

    .\.venv\Scripts\python.exe exp\itercurve.py --exp exp009 --tag cat_base
"""
import argparse
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "exp", "cache")
PREDS = os.path.join(ROOT, "exp", "preds")
LOG = os.path.join(ROOT, "experiment_log.jsonl")


def load_rows(exp, tag):
    keep = {}
    with open(LOG, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("exp") == exp and (tag is None or r.get("tag") == tag):
                keep[(r["exp"], r["key"])] = r
    return list(keep.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    rows = load_rows(a.exp, a.tag)
    if not rows:
        print("기록 없음")
        return
    folds = sorted({r["fold"] for r in rows})
    n_iter = rows[0]["hparams"]["n_iter"]
    model = rows[0]["model"]
    step = 1 if model == "hgb" else max(1, n_iter // 20)

    # 폴드별로 시드 곡선을 평균한다 (시드 하나의 봉우리를 쫓지 않도록)
    curves, ntrain = {}, {}
    for f in folds:
        cs = []
        for r in rows:
            if r["fold"] != f:
                continue
            cs.append(np.load(f"{PREDS}/{r['key']}.npz")["curve"])
            ntrain[f] = r["n_train"]
        curves[f] = np.mean(np.stack(cs), axis=0)

    iters = [(i + 1) * step for i in range(len(next(iter(curves.values()))))]
    print(f"{a.exp}/{a.tag} | {model} | 체크포인트 간격 {step}")
    print(f"\n{'iter':>6s} " + " ".join(f"{f:>10d}" for f in folds))
    for i, it in enumerate(iters):
        if it % (step * 2) and len(iters) > 12:
            continue
        print(f"{it:>6d} " + " ".join(f"{curves[f][i]:10.2f}" for f in folds))

    print(f"\n{'fold':>6s} {'학습행':>10s} {'최적iter':>9s} {'최적점수':>10s} "
          f"{'고정1200':>10s} {'차이':>8s}")
    best = {}
    for f in folds:
        b = int(np.argmax(curves[f]))
        best[f] = iters[b]
        print(f"{f:>6d} {ntrain[f]:>10,} {iters[b]:>9d} {curves[f][b]:10.2f} "
              f"{curves[f][-1]:10.2f} {curves[f][b]-curves[f][-1]:+8.2f}")

    print("\n=== 워크포워드 반복수 선택 (판단 기준) ===")
    print(f"{'적용폴드':>8s} {'쓴 iter':>8s} {'출처':>16s} {'그 iter 점수':>12s} "
          f"{'고정1200':>10s} {'델타':>8s}")
    deltas = []
    for t, f in enumerate(folds):
        if t == 0:
            continue
        prev = folds[:t]
        # 이전 폴드들의 최적을 학습량으로 스케일해 옮긴다
        scaled = np.mean([best[q] * (ntrain[f] / ntrain[q]) for q in prev])
        j = int(np.clip(round(scaled / step) - 1, 0, len(iters) - 1))
        s_sel, s_fix = curves[f][j], curves[f][-1]
        deltas.append(s_sel - s_fix)
        print(f"{f:>8d} {iters[j]:>8d} {str(prev):>16s} {s_sel:12.2f} "
              f"{s_fix:10.2f} {s_sel - s_fix:+8.2f}")
    if deltas:
        print(f"\n  워크포워드 평균 {np.mean(deltas):+.2f} "
              f"({sum(1 for d in deltas if d > 0)}/{len(deltas)} 승)")
    # 2025 로 옮길 때의 권고값
    n2025 = 1_475_092
    rec = np.mean([best[f] * (n2025 / ntrain[f]) for f in folds])
    print(f"\n  학습량 보정 권고 (2025, {n2025:,}행): "
          f"iter 약 {int(round(rec / 50) * 50)}")
    print(f"  폴드별 최적을 그대로 평균하면 {int(np.mean(list(best.values())))}")


if __name__ == "__main__":
    main()
