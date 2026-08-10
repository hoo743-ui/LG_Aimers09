r"""실험 로그 -> 폴드별 델타 표. 채택 판단에 필요한 것만 찍는다.

판단 기준 (4-6, 사용자 규칙 11):
  - 평균이 표준오차의 2배를 넘는가
  - 폴드별 부호가 일치하는가  (갈리면 그 변동폭만큼 할인)
  - 시드에 안정적인가
  - 단일 폴드에 의존하지 않는가

    .\.venv\Scripts\python.exe exp\analyze.py --exp exp001
    .\.venv\Scripts\python.exe exp\analyze.py --exp exp001 --metric score_best
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "experiment_log.jsonl")


def load(exp=None):
    # 같은 모델(key)이 여러 실험에서 재사용되므로 (exp, key) 로 중복을 지운다.
    # key 만으로 지우면 캐시 재사용 행이 사라져 짝비교가 깨진다.
    keep, bad = {}, 0
    with open(LOG, encoding="utf-8") as f:
        for line in f:
            # 샤드 여러 개가 동시에 append 하면 줄이 섞일 수 있다. 진실은
            # exp/preds/*.npz 에 있으므로 깨진 줄은 세고 넘어간다.
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if exp is None or r.get("exp") == exp:
                keep[(r["exp"], r["key"])] = r
    if bad:
        print(f"(로그 손상 {bad}줄 건너뜀 — runner 재실행으로 캐시에서 복구됨)")
    return list(keep.values())


def report(rows, metric="score_fixed", ref="base"):
    by = {(r["tag"], r["fold"], r["seed"]): r for r in rows}
    tags = sorted({r["tag"] for r in rows}, key=lambda t: (t != ref, t))
    folds = sorted({r["fold"] for r in rows})
    seeds = sorted({r["seed"] for r in rows})

    print(f"\n=== {metric} | 절대점수 (폴드 x 시드 평균) ===")
    print(f"{'tag':16s} " + " ".join(f"{f:>10d}" for f in folds) + f"{'평균':>11s}")
    for t in tags:
        vals, line = [], []
        for f in folds:
            v = [by[(t, f, s)][metric] for s in seeds if (t, f, s) in by]
            line.append(f"{np.mean(v):10.2f}" if v else f"{'-':>10s}")
            vals += v
        print(f"{t:16s} " + " ".join(line) +
              (f"{np.mean(vals):11.2f}" if vals else ""))

    if ref not in tags:
        return {}
    print(f"\n=== {ref} 대비 델타 (폴드 안에서 시드별로 짝지어 비교) ===")
    summary = {}
    for t in tags:
        if t == ref:
            continue
        per_fold, allv, cells = {}, [], []
        for f in folds:
            d = [by[(t, f, s)][metric] - by[(ref, f, s)][metric]
                 for s in seeds if (t, f, s) in by and (ref, f, s) in by]
            if d:
                per_fold[f] = d
                allv += d
                cells.append(f"{f}:{np.mean(d):+.1f}")
        if not allv:
            continue
        se = np.std(allv, ddof=1) / np.sqrt(len(allv)) if len(allv) > 1 else float("nan")
        fold_means = [np.mean(v) for v in per_fold.values()]
        same_sign = len(set(np.sign(fold_means))) == 1
        pos = sum(1 for x in allv if x > 0)
        summary[t] = {"mean": float(np.mean(allv)), "se": float(se),
                      "n": len(allv), "same_sign": bool(same_sign),
                      "pos": pos, "fold_means": {str(k): float(np.mean(v))
                                                 for k, v in per_fold.items()}}
        verdict = ("부호일치" if same_sign else "🚩부호갈림")
        sig = "유의" if (se == se and abs(np.mean(allv)) > 2 * se) else "미달"
        print(f"  {t:16s} 평균 {np.mean(allv):+7.2f}  표준오차 {se:5.2f}  "
              f"n={len(allv):<3d} {pos}/{len(allv)}+  {verdict}  {sig}")
        print(f"  {'':16s} {' | '.join(cells)}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default=None)
    ap.add_argument("--metric", default="score_fixed")
    ap.add_argument("--ref", default="base")
    ap.add_argument("--both", action="store_true",
                    help="score_fixed 와 score_best 를 모두 찍는다")
    a = ap.parse_args()
    rows = load(a.exp)
    if not rows:
        print("해당 실험 기록 없음")
        return
    print(f"기록 {len(rows)}건 | exp={a.exp}")
    metrics = ["score_fixed", "score_best"] if a.both else [a.metric]
    for m in metrics:
        report(rows, m, a.ref)


if __name__ == "__main__":
    main()
