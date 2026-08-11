r"""제출본에 박을 alpha 를 고른다. 재학습 0회.

## 무엇을 고르는가

`script.py` 는 예측을 `p' = center + alpha*(p - center)` 로 바꾼 뒤 clip 한다.
`alpha` 와 `center` 는 **학습 시점에 정해져 pkl 에 담기는 상수**이므로 행 독립이고
규칙 위반이 아니다 (4-3 의 세 번째 줄). 평가셋을 보고 정하는 것만 위반이다.

## 왜 평균 최적을 그대로 쓰지 않는가

폴드별 최적 alpha 가 1.0596 ~ 1.1236 로 흩어져 있다. 평균 최적을 박으면 평가
시즌이 fold2022 형(최적 1.06)일 때 손해를 본다. 그래서 두 가지를 같이 본다.

  - **평균 이득**   : 세 폴드 평균 델타가 가장 큰 alpha
  - **최악 이득**   : 세 폴드 중 최소 델타가 가장 큰 alpha (minimax)

손실함수가 최적점 근처에서 평평하다면 두 값이 가깝고, 그때 채택이 안전하다.
평평하지 않으면 minimax 쪽을 택한다 — 이 대회에서 우리는 평가 시즌의 형태를
모른다 (4-2).

    .\.venv\Scripts\python.exe exp\alpha_pick.py                 # 실험 경로 3폴드
    .\.venv\Scripts\python.exe exp\alpha_pick.py --npz exp\valpred_cat.npz
"""
import argparse
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "exp", "cache")
PREDS = os.path.join(ROOT, "exp", "preds")
LOG = os.path.join(ROOT, "experiment_log.jsonl")

GRID = np.round(np.arange(1.00, 1.205, 0.01), 3)


def score_of(y, p, base):
    return max(0.0, 100000 * (1 - ((p - y) ** 2).mean() / base))


def apply_alpha(p, c, a):
    return np.clip(c + a * (p - c), 0.0, 1.0)


def fold_ensembles(tag, folds):
    """폴드별 (시드 평균 예측, 정답, 학습구간 중심). 캐시에서만 읽는다."""
    y = np.load(f"{CACHE}/y.npy")
    season = np.load(f"{CACHE}/season.npy")
    rows = {}
    with open(LOG, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("tag") == tag and r.get("fold") in folds:
                rows[r["key"]] = r
    out = {}
    for fold in folds:
        ps = [np.load(f"{PREDS}/{r['key']}.npz")["p"]
              for r in rows.values() if r["fold"] == fold
              and os.path.exists(f"{PREDS}/{r['key']}.npz")]
        if not ps:
            continue
        yv = y[season == fold].astype(np.float64)
        out[fold] = (np.mean(ps, axis=0).astype(np.float64), yv,
                     float(y[season < fold].mean()), len(ps))
    return out


def table(items):
    """items: {라벨: (p, y, center)} -> alpha 격자 델타 표."""
    print(f"{'alpha':>7}", end="")
    for k in items:
        print(f"{str(k):>10}", end="")
    print(f"{'평균':>10}{'최악':>10}")
    best_mean = best_worst = None
    for a in GRID:
        deltas = []
        for k, (p, y, c) in items.items():
            r = y.mean()
            base = r * (1 - r)
            deltas.append(score_of(y, apply_alpha(p, c, a), base)
                          - score_of(y, p, base))
        mean, worst = float(np.mean(deltas)), float(np.min(deltas))
        mark = ""
        if best_mean is None or mean > best_mean[1]:
            best_mean = (a, mean)
        if best_worst is None or worst > best_worst[1]:
            best_worst = (a, worst)
        print(f"{a:>7.2f}", end="")
        for d in deltas:
            print(f"{d:>+10.2f}", end="")
        print(f"{mean:>+10.2f}{worst:>+10.2f}{mark}")
    print(f"\n  평균 최적 alpha {best_mean[0]:.2f} (평균 {best_mean[1]:+.2f})")
    print(f"  최악 최적 alpha {best_worst[0]:.2f} (최악 {best_worst[1]:+.2f})")
    return best_mean, best_worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="cat_tuned")
    ap.add_argument("--npz", default=None,
                    help="train_cat.py --save-val-pred 로 만든 제출 경로 예측")
    a = ap.parse_args()

    if a.npz:
        d = np.load(a.npz, allow_pickle=True)
        p, y = d["p"].astype(np.float64), d["y"].astype(np.float64)
        c = float(d["center_train"])
        print(f"제출 경로 예측 | n={len(y):,} | center(학습구간)={c:.4f} | "
              f"{str(d['config'])}\n")
        print("=== alpha 격자 (제출 경로 2024 홀드아웃) ===")
        table({"2024": (p, y, c)})
        return

    items = {}
    for fold, (p, y, c, k) in fold_ensembles(a.tag, [2021, 2022, 2024]).items():
        items[fold] = (p, y, c)
        print(f"fold{fold}: 시드 {k}개 앙상블 | center(학습구간)={c:.4f} | "
              f"예측평균 {p.mean():.4f} vs 실제 {y.mean():.4f}")
    print(f"\n=== alpha 격자 (실험 경로, tag={a.tag}) ===")
    table(items)


if __name__ == "__main__":
    main()
