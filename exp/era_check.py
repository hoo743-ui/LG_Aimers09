r"""연도 누설 스크리닝 — 캐시판. `era_screen.py` 와 같은 판정, CSV 재파싱 없음.

6회차에서 LB 29점을 잃고 얻은 검사다 (4-9). 피처 **하나만으로** "이 행이 Y년인가
Y-1년인가"를 가르는 AUC 를 순위통계로 잰다. 학습하지 않는다.

판정은 AUC 단독이 아니라 **AUC + 단조성**이다. 값이 '쌓여서' 커지는 양(표본 수,
누적 카운트)이 위험하다. 리그 국면이 실제로 이동해서 연도와 상관되는 것
(`asof_*` 비율)은 진짜 신호일 수 있다.

**새 피처는 반드시 이 검사를 먼저 통과해야 한다** (일하는 방식 ②).

    .\.venv\Scripts\python.exe exp\era_check.py --only tmc_,tmh_
"""
import argparse
import json
import os

import numpy as np
from scipy.stats import rankdata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "exp", "cache")
PAIRS = [(2020, 2021), (2021, 2022), (2022, 2023), (2023, 2024)]
SAMPLE = 400_000


def auc_by_rank(x, g):
    ok = ~np.isnan(x)
    x, g = x[ok], g[ok]
    n1, n0 = int(g.sum()), int((1 - g).sum())
    if n1 < 50 or n0 < 50:
        return np.nan
    r = rankdata(x)
    return (r[g == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--top", type=int, default=30)
    a = ap.parse_args()

    X = np.load(f"{CACHE}/X.npy", mmap_mode="r")
    season = np.load(f"{CACHE}/season.npy")
    meta = json.load(open(f"{CACHE}/cols.json"))
    cols = meta["cols"]
    idx = list(range(len(cols)))
    if a.only:
        pre = tuple(a.only.split(","))
        idx = [i for i in idx if cols[i].startswith(pre)]

    rng = np.random.default_rng(42)
    sel = rng.choice(len(season), min(SAMPLE, len(season)), replace=False)
    sel.sort()
    S = season[sel]

    rows = []
    for i in idx:
        if cols[i] == "season":
            continue
        v = np.asarray(X[sel, i], dtype=np.float64)
        aucs = []
        for p, q in PAIRS:
            m = (S == p) | (S == q)
            if m.sum() < 200:
                continue
            aucs.append(auc_by_rank(v[m], (S[m] == q).astype(int)))
        aucs = [x for x in aucs if x == x]
        if not aucs:
            continue
        dev = float(np.mean([abs(x - 0.5) for x in aucs]))
        mono = abs(sum(np.sign(x - 0.5) for x in aucs)) == len(aucs)
        med = [float(np.nanmedian(v[S == y])) for y in
               sorted(set(S.tolist()))]
        rows.append((cols[i], 0.5 + dev, mono, dev, med))

    rows.sort(key=lambda r: -r[3])
    print(f"{'피처':24s} {'평균|AUC|':>9s} {'단조':>5s}   시즌별 중앙값")
    print("-" * 96)
    for f, auc, mono, _, med in rows[:a.top]:
        mark = " 🚩시계 의심" if (auc > 0.60 and mono) else ""
        print(f"{f:24s} {auc:9.3f} {'예' if mono else '-':>5s}   "
              + " ".join(f"{v:>8.3g}" for v in med) + mark)
    print("\n🚩 = |AUC| 0.60 초과 + 매 연도쌍 같은 방향. 6회차를 죽인 tmc_n 이 0.602 였다.")


if __name__ == "__main__":
    main()
