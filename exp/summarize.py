r"""experiment_log.jsonl -> experiment_log.csv (실험 단위 요약).

jsonl 은 학습 1회 = 1줄이라 사람이 읽기 어렵다. 여기서는 (실험, 변형) 단위로
접어서 폴드별 점수·평균·표준편차·부호일치·델타를 한 줄에 담는다 (규칙 10).

    .\.venv\Scripts\python.exe exp\summarize.py
"""
import csv
import json
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "experiment_log.jsonl")
OUT = os.path.join(ROOT, "experiment_log.csv")

FOLDS = [2021, 2022, 2023, 2024]


def main():
    rows = {}
    with open(LOG, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows[(r["exp"], r["key"])] = r

    by = defaultdict(dict)
    for r in rows.values():
        by[(r["exp"], r["tag"])][(r["fold"], r["seed"])] = r

    out = []
    for (exp, tag), cells in sorted(by.items()):
        # 기준 변형은 'base'. 모델 비교 실험(exp006/008)은 HGB 가 기준이다.
        ref = by.get((exp, "base")) or by.get((exp, "hgb")) or {}
        is_ref = tag in ("base", "hgb") and cells is ref
        any_r = next(iter(cells.values()))
        per_fold, deltas = {}, []
        for f in FOLDS:
            v = [c["score_fixed"] for (ff, _), c in cells.items() if ff == f]
            if v:
                per_fold[f] = np.mean(v)
        for (f, s), c in cells.items():
            if not is_ref and (f, s) in ref:
                deltas.append(c["score_fixed"] - ref[(f, s)]["score_fixed"])
        fold_means = {}
        for f in FOLDS:
            d = [c["score_fixed"] - ref[(ff, s)]["score_fixed"]
                 for (ff, s), c in cells.items()
                 if ff == f and (ff, s) in ref and not is_ref]
            if d:
                fold_means[f] = np.mean(d)
        signs = {np.sign(v) for v in fold_means.values() if v}
        out.append({
            "experiment_id": exp,
            "variant": tag,
            "model": any_r["model"],
            "n_features": any_r["n_features"],
            "hyperparameters": json.dumps(any_r["hparams"], sort_keys=True),
            "feature_spec": json.dumps(any_r["features"], sort_keys=True),
            "seeds": ",".join(str(s) for s in sorted({s for _, s in cells})),
            "n_runs": len(cells),
            **{f"fold_{f}": (f"{per_fold[f]:.2f}" if f in per_fold else "")
               for f in FOLDS},
            "mean": f"{np.mean(list(per_fold.values())):.2f}" if per_fold else "",
            "delta_mean": f"{np.mean(deltas):+.2f}" if deltas else "",
            "delta_se": (f"{np.std(deltas, ddof=1)/np.sqrt(len(deltas)):.2f}"
                         if len(deltas) > 1 else ""),
            "delta_by_fold": " ".join(f"{f}:{v:+.1f}"
                                      for f, v in sorted(fold_means.items())),
            "sign_consistent": ("" if not fold_means
                                else ("yes" if len(signs) == 1 else "NO")),
            "pos_frac": (f"{sum(1 for d in deltas if d>0)}/{len(deltas)}"
                         if deltas else ""),
            "pred_mean": f"{np.mean([c['pred_mean'] for c in cells.values()]):.4f}",
            "pred_std": f"{np.mean([c['pred_std'] for c in cells.values()]):.4f}",
        })

    cols = list(out[0])
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(out)
    print(f"{OUT} 작성 ({len(out)}행)")
    for r in out:
        print(f"  {r['experiment_id']:8s} {r['variant']:14s} "
              f"delta {r['delta_mean']:>8s} se {r['delta_se']:>5s} "
              f"sign {r['sign_consistent']:>3s}  {r['delta_by_fold']}")


if __name__ == "__main__":
    main()
