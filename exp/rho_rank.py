r"""캐시된 모든 실험을 rho^2 로 다시 매긴다. 재학습 0회.

## 왜 지표를 바꾸는가

12회차 절편 탐침으로 평가셋이 풀리면서 **지표의 정체가 드러났다.**

    점수 = 1e5 * (1 - MSE/var(y)) = 1e5 * R^2

그리고 아핀 변환 `p' = A p + B` 를 최적으로 고르면 `R^2 = rho(p,y)^2` 다. 즉

    아핀 최적 점수 = 1e5 * rho^2        (A,B 와 무관)

**아핀은 이제 공짜다.** 12회차까지 LB 점 4개로 A,B 를 푸는 법을 알아냈으므로,
설정을 비교할 때 중심·퍼짐의 어긋남을 점수에 포함시킬 이유가 없다.

## 그런데 지금까지는 포함시키고 있었다

`postcal_rank.py` 는 `alpha` 만 격자탐색하고 `center` 는 학습구간 평균에 **고정**한다.
그래서 폴드별 중심 어긋남이 그대로 점수에 남는다 (4-21 의 오라클 중심교정:
2021 +35.01 / 2022 +4.99 / 2024 +34.38). 설정을 조금만 바꿔도 중심이 흔들리는데
그 흔들림이 30점씩 가산·감산됐다 — **"부호갈림" 판정의 상당 부분이 이것일 수 있다.**

rho^2 는 A,B 에 불변이므로 **순수 해상도**만 잰다. 4-21 의 Murphy 분해에서
reliability 를 0 으로 놓은 것과 같다.

## 주의 — 이 지표가 낙관인 지점

새 피처를 넣으면 평가셋에서의 `m`,`s^2`,`C` 가 달라지는데 그 값은 LB 점 없이는
모른다. `r`=0.4609 만 모델과 무관하게 유효하다. 따라서 rho^2 는 **도달 가능한
상한**이고, 실제 제출은 그 상한에 얼마나 가까이 A,B 를 맞추느냐가 남는다.
설정 **비교**에는 이게 옳은 지표지만, 절대 점수 예측으로 쓰면 안 된다.

    .\.venv\Scripts\python.exe exp\rho_rank.py
    .\.venv\Scripts\python.exe exp\rho_rank.py --seeds 3 --tags cat_tuned,sk200,oh_team
"""
import argparse
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "exp", "cache")
PREDS = os.path.join(ROOT, "exp", "preds")
LOG = os.path.join(ROOT, "experiment_log.jsonl")

FOLDS = [2021, 2022, 2024]
REF = "cat_tuned"


def raw_score(y, p, base):
    return max(0.0, 1e5 * (1 - ((p - y) ** 2).mean() / base))


def rho2_score(y, p):
    """아핀 최적 점수. A,B 를 최소제곱으로 고른 뒤의 1e5*R^2 = 1e5*rho^2."""
    s = p.std()
    if s < 1e-12:
        return 0.0
    return float(1e5 * np.corrcoef(p, y)[0, 1] ** 2)


def load_rows():
    rows = {}
    with open(LOG, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if os.path.exists(f"{PREDS}/{r['key']}.npz"):
                rows[r["key"]] = r
    return rows


def ensembles(rows, tag, seeds, y, season):
    out = {}
    for fold in FOLDS:
        keys = [k for k, v in rows.items()
                if v["tag"] == tag and v["fold"] == fold]
        if not keys:
            continue
        keys = sorted(keys, key=lambda k: rows[k]["seed"])[:seeds]
        ps = [np.load(f"{PREDS}/{k}.npz")["p"] for k in keys]
        out[fold] = (np.mean(ps, axis=0).astype(np.float64),
                     y[season == fold].astype(np.float64), len(ps))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tags", default=None)
    ap.add_argument("--min-delta", type=float, default=0.0,
                    help="이 값 이상 오른 것만 자세히 찍는다")
    a = ap.parse_args()

    y = np.load(f"{CACHE}/y.npy")
    season = np.load(f"{CACHE}/season.npy")
    rows = load_rows()
    tags = (a.tags.split(",") if a.tags
            else sorted({v["tag"] for v in rows.values()}))

    res = []
    for tag in tags:
        ens = ensembles(rows, tag, a.seeds, y, season)
        if len(ens) < len(FOLDS):
            continue
        raw, rho2 = {}, {}
        for fold in FOLDS:
            p, yv, k = ens[fold]
            r = yv.mean()
            raw[fold] = raw_score(yv, p, r * (1 - r))
            rho2[fold] = rho2_score(yv, p)
        res.append({"tag": tag, "seeds": min(ens[FOLDS[0]][2], a.seeds),
                    "raw": raw, "rho2": rho2,
                    "raw_mean": float(np.mean(list(raw.values()))),
                    "rho2_mean": float(np.mean(list(rho2.values())))})

    ref = next((r for r in res if r["tag"] == REF), None)
    if ref is None:
        raise SystemExit(f"기준 태그 {REF} 의 캐시가 없다")

    # 4-21 의 함정: 단일 폴드로 정렬하면 부호갈림을 1등으로 만든다. 평균으로 정렬.
    res.sort(key=lambda r: -r["rho2_mean"])

    print(f"시드 {a.seeds}개 앙상블 | 지표 = 1e5*rho^2 (아핀 최적, A/B 불변)")
    print(f"기준 {REF}: rho2평균 {ref['rho2_mean']:.2f}  "
          f"(보정전 원점수 평균 {ref['raw_mean']:.2f})\n")
    hdr = (f"{'tag':16s} {'rho2평균':>9s} {'기준대비':>9s} {'부호':>5s}  " +
           "  ".join(f"{f}".rjust(9) for f in FOLDS) +
           f" {'원점수평균':>10s} {'어긋남':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for r in res:
        d = {f: r["rho2"][f] - ref["rho2"][f] for f in FOLDS}
        dm = r["rho2_mean"] - ref["rho2_mean"]
        pos = sum(1 for f in FOLDS if d[f] > 0)
        gap = r["rho2_mean"] - r["raw_mean"]        # 아핀이 회수하는 몫
        star = " <-" if r["tag"] == REF else ""
        print(f"{r['tag']:16s} {r['rho2_mean']:9.2f} {dm:+9.2f} {pos}/{len(FOLDS)}  "
              + "  ".join(f"{d[f]:+9.2f}" for f in FOLDS)
              + f" {r['raw_mean']:10.2f} {gap:8.2f}{star}")

    print("\n원점수 순위와 뒤집힌 것 (중심 어긋남이 판정을 흔들던 설정)")
    by_raw = sorted(res, key=lambda r: -r["raw_mean"])
    rank_raw = {r["tag"]: i for i, r in enumerate(by_raw)}
    rank_rho = {r["tag"]: i for i, r in enumerate(res)}
    moved = sorted(res, key=lambda r: rank_raw[r["tag"]] - rank_rho[r["tag"]])
    for r in moved[:6] + moved[-6:]:
        j = rank_raw[r["tag"]] - rank_rho[r["tag"]]
        if j:
            print(f"  {r['tag']:16s} 원점수 {rank_raw[r['tag']]+1:2d}위 -> "
                  f"rho2 {rank_rho[r['tag']]+1:2d}위  ({j:+d})")


if __name__ == "__main__":
    main()
