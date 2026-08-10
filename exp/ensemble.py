r"""캐시된 검증 예측만으로 앙상블을 분석한다. 재학습 0회.

세 가지를 잰다.

1. **앙상블 크기 곡선** — 시드 k 개를 평균했을 때의 점수. 어떤 k 개를 뽑느냐에
   따라 운이 섞이므로 부분집합을 여러 번 뽑아 평균한다. "5 → 15 로 늘리면
   얼마나 버는가"에 답한다 (README 7장 2번).
2. **다양성** — 모델 간 예측 상관. 규칙 7: 단순히 모델이 다르다는 이유로
   앙상블하지 않는다. 상관이 0.99 면 섞어도 얻을 게 없다.
3. **가중치 최적화 — 반드시 워크포워드로.** 같은 폴드에서 가중치를 맞추고 그
   폴드로 채점하면 당연히 좋아진다. 이전 폴드에서 맞춘 가중치를 다음 폴드에
   적용해야 실제 이득이다 (4-3 이 중심 보정에서 배운 것과 같은 함정).

    .\.venv\Scripts\python.exe exp\ensemble.py --exp exp001 --tag base
"""
import argparse
import itertools
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "exp", "cache")
PREDS = os.path.join(ROOT, "exp", "preds")
LOG = os.path.join(ROOT, "experiment_log.jsonl")

RNG = np.random.default_rng(0)


def load_rows(exp=None, tag=None):
    keep = {}
    with open(LOG, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if exp and r["exp"] != exp:
                continue
            if tag and r["tag"] != tag:
                continue
            keep[(r["exp"], r["key"])] = r
    return list(keep.values())


def y_of(fold):
    y = np.load(f"{CACHE}/y.npy")
    season = np.load(f"{CACHE}/season.npy")
    yv = y[season == fold].astype(np.float64)
    return yv, float(yv.mean() * (1 - yv.mean()))


def p_of(key):
    return np.load(f"{PREDS}/{key}.npz")["p"].astype(np.float64)


def score(y, p, base):
    return max(0.0, 100000 * (1 - ((p - y) ** 2).mean() / base))


def size_curve(rows, folds, trials=40):
    print("\n=== 앙상블 크기 곡선 (시드 k 개 평균) ===")
    print(f"{'k':>3s} " + " ".join(f"{f:>9d}" for f in folds) +
          f"{'평균':>10s}{'vs k=5':>9s}")
    per_fold = {}
    for f in folds:
        yv, base = y_of(f)
        keys = [r["key"] for r in rows if r["fold"] == f]
        P = np.stack([p_of(k) for k in keys])
        per_fold[f] = (yv, base, P)
    ks = [1, 2, 3, 5, 7, 10]
    table = {}
    for k in ks:
        row = []
        for f in folds:
            yv, base, P = per_fold[f]
            if k > len(P):
                row.append(np.nan)
                continue
            if k == len(P):
                subs = [list(range(len(P)))]
            else:
                subs = [RNG.choice(len(P), k, replace=False)
                        for _ in range(trials)]
            row.append(float(np.mean([score(yv, P[list(s)].mean(0), base)
                                      for s in subs])))
        table[k] = row
    ref = np.nanmean(table[5])
    for k in ks:
        row = table[k]
        m = np.nanmean(row)
        print(f"{k:>3d} " + " ".join(f"{v:9.2f}" for v in row) +
              f"{m:10.2f}{m - ref:+9.2f}")
    return table


def diversity(groups, folds):
    """groups: {label: [key,...]}  — 그룹 평균 예측끼리 상관을 본다."""
    print("\n=== 예측 다양성 (그룹 평균 예측 간 상관, 폴드 평균) ===")
    labels = list(groups)
    if len(labels) < 2:
        print("  (모델 그룹이 하나뿐이라 생략)")
        return
    acc = np.zeros((len(labels), len(labels)))
    cnt = 0
    for f in folds:
        P = {}
        for L in labels:
            keys = [k for k, fl in groups[L] if fl == f]
            if keys:
                P[L] = np.stack([p_of(k) for k in keys]).mean(0)
        if len(P) < 2:
            continue
        cnt += 1
        for i, a in enumerate(labels):
            for j, b in enumerate(labels):
                if a in P and b in P:
                    acc[i, j] += np.corrcoef(P[a], P[b])[0, 1]
    if not cnt:
        return
    acc /= cnt
    print(f"{'':14s}" + " ".join(f"{L:>10s}" for L in labels))
    for i, a in enumerate(labels):
        print(f"{a:14s}" + " ".join(f"{acc[i, j]:10.4f}"
                                   for j in range(len(labels))))
    print("  상관 0.99 이상이면 섞어도 새 정보가 거의 없다 (규칙 7).")


def nnls_weights(P, y):
    """비음수 가중치 합=1 로 Brier 최소화. 좌표하강 몇 바퀴면 충분하다."""
    m = len(P)
    w = np.ones(m) / m
    for _ in range(200):
        p = w @ P
        g = np.array([((p - y) * (P[i] - p)).mean() for i in range(m)])
        i = int(np.argmin(g))
        step = 2.0 / (2.0 + _)
        w = (1 - step) * w
        w[i] += step
    return w


def weight_walkforward(groups, folds):
    """이전 폴드에서 맞춘 가중치를 다음 폴드에 적용한다."""
    labels = list(groups)
    if len(labels) < 2 or len(folds) < 2:
        return
    print("\n=== 앙상블 가중치 — 워크포워드 (규칙 8) ===")
    print(f"{'적용 폴드':>10s} {'균등':>9s} {'워크포워드':>11s} {'델타':>8s}"
          f"   (그 폴드 최적 = 낙관 상한)   가중치")
    for t, f in enumerate(folds):
        if t == 0:
            continue
        Pf, yv, base = {}, None, None
        yv, base = y_of(f)
        for L in labels:
            keys = [k for k, fl in groups[L] if fl == f]
            if not keys:
                Pf = None
                break
            Pf[L] = np.stack([p_of(k) for k in keys]).mean(0)
        if not Pf or len(Pf) < 2:
            continue
        # 이전 폴드들에서 가중치를 맞춘다
        Wacc = []
        for pf in folds[:t]:
            yp, _ = y_of(pf)
            Pp = {}
            ok = True
            for L in labels:
                keys = [k for k, fl in groups[L] if fl == pf]
                if not keys:
                    ok = False
                    break
                Pp[L] = np.stack([p_of(k) for k in keys]).mean(0)
            if ok:
                Wacc.append(nnls_weights(np.stack([Pp[L] for L in labels]), yp))
        if not Wacc:
            continue
        w = np.mean(Wacc, axis=0)
        M = np.stack([Pf[L] for L in labels])
        s_eq = score(yv, M.mean(0), base)
        s_wf = score(yv, w @ M, base)
        s_or = score(yv, nnls_weights(M, yv) @ M, base)
        ws = " ".join(f"{L}={x:.2f}" for L, x in zip(labels, w))
        print(f"{f:>10d} {s_eq:9.2f} {s_wf:11.2f} {s_wf - s_eq:+8.2f}"
              f"        {s_or:9.2f}          {ws}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--groups", default=None,
                    help="다양성/가중치용. 'label=exp:tag,label2=exp:tag2'")
    a = ap.parse_args()

    if a.groups:
        groups, folds = {}, set()
        for part in a.groups.split(","):
            label, _, ref = part.partition("=")
            e, _, t = ref.partition(":")
            rs = load_rows(e or None, t or None)
            groups[label] = [(r["key"], r["fold"]) for r in rs]
            folds |= {r["fold"] for r in rs}
        folds = sorted(folds)
        diversity(groups, folds)
        weight_walkforward(groups, folds)
        return

    rows = load_rows(a.exp, a.tag)
    if not rows:
        print("기록 없음")
        return
    folds = sorted({r["fold"] for r in rows})
    print(f"{a.exp}/{a.tag}: {len(rows)}건, 폴드 {folds}")
    size_curve(rows, folds)


if __name__ == "__main__":
    main()
