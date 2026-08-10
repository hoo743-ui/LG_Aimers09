r"""캘리브레이션 — 예측 퍼짐(기울기)을 워크포워드로 검증한다. 재학습 0회.

## 왜 다시 보는가

README 4-5 는 "축소 보정 불필요 — 최적 기울기가 1.01 근처"로 기각했다. 그런데
**상황 편차 피처를 넣은 뒤 7회차 학습 로그의 최적 기울기는 1.1233 이었다.**
방향이 뒤집혔다(축소가 아니라 **확장**). 기각의 근거가 된 관측 자체가 바뀌었으니
다시 잰다.

    p' = c + a*(p - c)        a > 1 이면 예측을 벌려야 이득

## 반드시 워크포워드로

그 폴드의 정답으로 a 를 맞추고 그 폴드로 채점하면 항상 이득이다 — 자유도를
하나 준 것뿐이다. 4-3 이 중심 보정에서 똑같은 함정을 겪었다. 여기서는
**이전 폴드들에서만 a 를 추정해 다음 폴드에 적용**한다. 그 a 는 학습 시점에
정해져 pkl 에 박히는 상수이므로 행 독립이고 규칙 위반이 아니다 (4-3 의 세 번째 줄).

중심 c 도 마찬가지로 **학습 데이터 성공률**만 쓴다. 평가셋을 보지 않는다.

    .\.venv\Scripts\python.exe exp\calib.py --exp exp001 --tag base
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
            if (exp is None or r["exp"] == exp) and (tag is None or r["tag"] == tag):
                keep[(r["exp"], r["key"])] = r
    return list(keep.values())


def score(y, p, base):
    return max(0.0, 100000 * (1 - ((p - y) ** 2).mean() / base))


def fit_slope(y, p, c):
    d = p - c
    v = (d ** 2).mean()
    return 1.0 if v <= 0 else float((d * (y - c)).mean() / v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="exp001")
    ap.add_argument("--tag", default="base")
    a = ap.parse_args()

    y_all = np.load(f"{CACHE}/y.npy").astype(np.float64)
    season = np.load(f"{CACHE}/season.npy")
    rows = load_rows(a.exp, a.tag)
    folds = sorted({r["fold"] for r in rows})

    ens, info = {}, {}
    for f in folds:
        keys = [r["key"] for r in rows if r["fold"] == f]
        P = np.stack([np.load(f"{PREDS}/{k}.npz")["p"].astype(np.float64)
                      for k in keys])
        yv = y_all[season == f]
        c = float(y_all[season < f].mean())      # 학습 데이터 성공률만 사용
        ens[f] = (P.mean(0), yv, c, float(yv.mean() * (1 - yv.mean())))
        info[f] = len(keys)

    print(f"{a.exp}/{a.tag} | 폴드별 시드 수 {info}")
    print("\n=== 폴드별 최적 기울기 (그 폴드 정답으로 사후 적합 — 관측용) ===")
    print(f"{'fold':>6s} {'중심c':>8s} {'예측평균':>9s} {'실제':>8s} "
          f"{'최적a':>8s} {'원본':>9s} {'a적용':>9s} {'이득':>8s}")
    slopes = {}
    for f in folds:
        p, yv, c, base = ens[f]
        aopt = fit_slope(yv, p, c)
        slopes[f] = aopt
        s0 = score(yv, p, base)
        s1 = score(yv, np.clip(c + aopt * (p - c), 0, 1), base)
        print(f"{f:>6d} {c:8.4f} {p.mean():9.4f} {yv.mean():8.4f} "
              f"{aopt:8.4f} {s0:9.2f} {s1:9.2f} {s1 - s0:+8.2f}")

    print("\n=== 워크포워드 — 이전 폴드들의 a 를 다음 폴드에 적용 (판단 기준) ===")
    print(f"{'fold':>6s} {'쓴 a':>8s} {'출처':>18s} {'원본':>9s} "
          f"{'적용후':>9s} {'델타':>8s}")
    deltas = []
    for t, f in enumerate(folds):
        if t == 0:
            continue
        prev = folds[:t]
        a_wf = float(np.mean([slopes[q] for q in prev]))
        p, yv, c, base = ens[f]
        s0 = score(yv, p, base)
        s1 = score(yv, np.clip(c + a_wf * (p - c), 0, 1), base)
        deltas.append(s1 - s0)
        print(f"{f:>6d} {a_wf:8.4f} {str(prev):>18s} {s0:9.2f} "
              f"{s1:9.2f} {s1 - s0:+8.2f}")
    if deltas:
        print(f"\n  워크포워드 평균 {np.mean(deltas):+.2f}  "
              f"({sum(1 for d in deltas if d > 0)}/{len(deltas)} 승)")
        print("  4-3 의 중심 보정은 여기서 평균 -3.55 로 기각됐다. 기울기도")
        print("  같은 잣대로 본다 — 평균이 음수거나 부호가 갈리면 채택하지 않는다.")


if __name__ == "__main__":
    main()
