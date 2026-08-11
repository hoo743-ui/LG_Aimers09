r"""보정(alpha) 이후의 설정 순위를 다시 낸다. 재학습 0회.

## 왜 다시 재는가

`border_count` 32 + `l2_leaf_reg` 100 은 **보정 없이** 잰 결과로 채택됐다 (4-11,
+28.27). 그런데 그 규제의 대가가 예측 과소분산이었고 alpha 가 그것을 되돌린다
(4-23). 그러면 **alpha 가 있는 지금은 최적 규제가 달라야 한다** — 모델이 스스로
보수적일 필요가 없다.

같은 논리가 다른 닫힌 축에도 적용된다. 앙상블(HGB+Cat)과 LightGBM 은 "예측이
중심으로 오므라든다"는 이유로 기각됐는데(4-5, 4-10), 그 수축이 바로 alpha 가
고치는 것이다.

## 판정 기준

`alpha` 는 **이전 폴드에서만** 추정해 다음 폴드에 적용한다 (워크포워드). 그 폴드의
정답으로 맞춘 alpha 는 낙관 상한이라 참고로만 찍는다. 순위는 워크포워드 점수로
매긴다 — 4-3 이 중심 보정에서 배운 함정과 같다.

    .\.venv\Scripts\python.exe exp\postcal_rank.py
    .\.venv\Scripts\python.exe exp\postcal_rank.py --seeds 3 --tags cat_tuned,border64
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
GRID = np.round(np.arange(1.00, 1.41, 0.01), 3)


def score_of(y, p, base):
    return max(0.0, 100000 * (1 - ((p - y) ** 2).mean() / base))


def load():
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


def ensembles(rows, tag, seeds):
    """tag -> {fold: (예측 평균, 정답, 학습구간 중심, 시드 수)}"""
    y = np.load(f"{CACHE}/y.npy")
    season = np.load(f"{CACHE}/season.npy")
    out = {}
    for fold in FOLDS:
        keys = [k for k, v in rows.items()
                if v["tag"] == tag and v["fold"] == fold]
        if not keys:
            continue
        keys = sorted(keys, key=lambda k: rows[k]["seed"])[:seeds]
        ps = [np.load(f"{PREDS}/{k}.npz")["p"] for k in keys]
        yv = y[season == fold].astype(np.float64)
        out[fold] = (np.mean(ps, axis=0).astype(np.float64), yv,
                     float(y[season < fold].mean()), len(ps))
    return out


def best_alpha(p, y, c):
    r = y.mean()
    base = r * (1 - r)
    vals = [score_of(y, np.clip(c + a * (p - c), 0, 1), base) for a in GRID]
    i = int(np.argmax(vals))
    return float(GRID[i]), vals[i], score_of(y, p, base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tags", default=None)
    a = ap.parse_args()

    rows = load()
    tags = (a.tags.split(",") if a.tags
            else sorted({v["tag"] for v in rows.values()}))

    res = []
    for tag in tags:
        ens = ensembles(rows, tag, a.seeds)
        if len(ens) < 3:
            continue
        raw, oracle, wf, alphas = {}, {}, {}, {}
        for fold in FOLDS:
            p, y, c, k = ens[fold]
            alphas[fold], oracle[fold], raw[fold] = best_alpha(p, y, c)
        # 워크포워드: 이전 폴드들의 alpha 평균을 다음 폴드에 쓴다
        for i, fold in enumerate(FOLDS[1:], start=1):
            prev = [alphas[f] for f in FOLDS[:i]]
            a_wf = float(np.mean(prev))
            p, y, c, k = ens[fold]
            r = y.mean()
            wf[fold] = score_of(y, np.clip(c + a_wf * (p - c), 0, 1),
                                r * (1 - r))
        res.append({
            "tag": tag, "seeds": min(ens[FOLDS[0]][3], a.seeds),
            "raw": raw, "oracle": oracle, "wf": wf, "alphas": alphas,
            "raw_2024": raw[2024], "wf_2024": wf[2024],
            "wf_mean": float(np.mean(list(wf.values()))),
            "raw_mean": float(np.mean(list(raw.values()))),
        })

    ref = next((r for r in res if r["tag"] == "cat_tuned"), None)
    # 🚩 fold2024 로 정렬하면 규칙 ⑤ 의 함정을 그대로 재현한다. 2026-08-11 에
    # 실제로 걸렸다 — oh_team 이 2024 에서 +25.0 으로 1위였는데 2021 에서
    # -47.0 이라 평균은 -0.4 였다. 폴드 평균으로 정렬한다.
    res.sort(key=lambda r: -r["wf_mean"])

    print(f"시드 {a.seeds}개 앙상블 | 워크포워드 alpha (이전 폴드 평균)\n")
    print(f"{'tag':16s} {'보정전 2024':>11s} {'보정후 2024':>11s} {'델타':>8s} "
          f"{'최적a 2024':>10s} {'WF평균':>9s} {'기준대비':>9s}")
    print("-" * 82)
    for r in res:
        d = r["wf_2024"] - r["raw_2024"]
        vs = (r["wf_2024"] - ref["wf_2024"]) if ref else 0.0
        star = " <-" if ref and r["tag"] == "cat_tuned" else ""
        print(f"{r['tag']:16s} {r['raw_2024']:11.2f} {r['wf_2024']:11.2f} "
              f"{d:+8.2f} {r['alphas'][2024]:10.2f} {r['wf_mean']:9.2f} "
              f"{vs:+9.2f}{star}")
    print("\n폴드별 최적 alpha (설정마다 다르다 — 규제가 약하면 1에 가까워야 한다)")
    for r in res[:8]:
        print(f"  {r['tag']:16s} " +
              " ".join(f"{f}:{r['alphas'][f]:.2f}" for f in FOLDS))


if __name__ == "__main__":
    main()
