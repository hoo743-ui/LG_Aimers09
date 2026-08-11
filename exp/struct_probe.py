r"""구조 진단 — 우리가 잃는 점수가 어디서 나오는가. 학습 0회.

## 왜 이걸 재는가

손잡이를 아홉 개 돌려 통과가 둘이었다 (2026-08-11). 1등까지 로컬 275점이 남았는데
그 크기는 손잡이로 닿지 않는다. 그러면 **남은 점수가 어떤 종류인지** 알아야 한다.
Brier 는 Murphy 분해로 세 조각이 된다.

    MSE = reliability - resolution + uncertainty
    BSS = (resolution - reliability) / uncertainty

  - `uncertainty` = r(1-r). 채점 기준선이다. 우리가 못 건드린다
  - `resolution`  = 예측이 실제로 갈라놓은 양. **진짜 신호다**
  - `reliability` = 어긋남 벌점. 예측 그룹의 평균이 그 그룹의 실제와 다른 만큼

resolution 이 작으면 신호를 더 찾아야 하고, reliability 가 크면 **가진 신호를
잘못 눕혀 놓은 것**이다. 후자는 성질이 다른 문제고, 4-3(국면 보정)이 규칙 위반으로
가져갔던 몫이 정확히 여기다. 합법적으로 얼마나 남았는지 재는 것이 이 파일이다.

    .\.venv\Scripts\python.exe exp\struct_probe.py
"""
import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "exp", "cache")
PREDS = os.path.join(ROOT, "exp", "preds")
LOG = os.path.join(ROOT, "experiment_log.jsonl")


def load_rows():
    keep = {}
    with open(LOG, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            keep[r["key"]] = r
    return keep


def pick(rows, tag, fold, seed, model="cat"):
    for r in rows.values():
        if (r["tag"] == tag and r["fold"] == fold and r["seed"] == seed
                and r["model"] == model
                and os.path.exists(f"{PREDS}/{r['key']}.npz")):
            return r
    return None


def murphy(y, p, bins=20):
    """reliability / resolution / uncertainty. 확률을 분위로 묶는다."""
    r = y.mean()
    unc = r * (1 - r)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    rel = res = 0.0
    n = len(y)
    for k in range(len(edges) - 1):
        m = idx == k
        nk = int(m.sum())
        if nk == 0:
            continue
        yk, pk = y[m].mean(), p[m].mean()
        rel += nk * (pk - yk) ** 2
        res += nk * (yk - r) ** 2
    return rel / n, res / n, unc


def score_of(y, p, base):
    return max(0.0, 100000 * (1 - ((p - y) ** 2).mean() / base))


def main():
    y = np.load(f"{CACHE}/y.npy")
    season = np.load(f"{CACHE}/season.npy")
    rows = load_rows()

    print("=== 1. 시즌별 성공률과 '중심만 어긋날 때' 비용 ===")
    print("전년 비율을 그대로 쓰면 몇 점을 잃는가. score 는 그 시즌 기준선으로 정규화된다.\n")
    print(f"{'시즌':>6} {'n':>10} {'성공률':>8} {'전년차':>8} {'중심비용':>9}")
    seasons = sorted(set(season.tolist()))
    rates = {S: float(y[season == S].mean()) for S in seasons}
    for i, S in enumerate(seasons):
        r = rates[S]
        base = r * (1 - r)
        if i == 0:
            print(f"{S:>6} {int((season==S).sum()):>10,} {r:>8.4f} {'-':>8} {'-':>9}")
            continue
        d = r - rates[seasons[i - 1]]
        cost = 100000 * (d ** 2) / base
        print(f"{S:>6} {int((season==S).sum()):>10,} {r:>8.4f} {d:>+8.4f} "
              f"{cost:>9.1f}")

    print("\n=== 2. Murphy 분해 — 우리 예측이 잃는 점수의 종류 ===")
    print("BSS x 100000 = (resolution - reliability) / uncertainty x 100000\n")
    print(f"{'폴드':>6} {'시드':>5} {'점수':>8} {'해상도':>9} {'어긋남':>9} "
          f"{'중심편차':>9} {'중심교정후':>10} {'교정이득':>9}")
    for fold in (2021, 2022, 2024):
        for seed in (42, 43):
            r = pick(rows, "cat_tuned", fold, seed)
            if r is None:
                continue
            d = np.load(f"{PREDS}/{r['key']}.npz", allow_pickle=True)
            p = d["p"].astype(np.float64)
            yv = y[season == fold].astype(np.float64)
            rate = yv.mean()
            base = rate * (1 - rate)
            rel, res, unc = murphy(yv, p)
            s = score_of(yv, p, base)
            # 중심만 실제 평균에 맞춘다 (오라클 — 규칙 위반. 상한을 재는 용도)
            s_shift = score_of(yv, np.clip(p - (p.mean() - rate), 0, 1), base)
            print(f"{fold:>6} {seed:>5} {s:>8.2f} {100000*res/unc:>9.1f} "
                  f"{100000*rel/unc:>9.1f} {p.mean()-rate:>+9.4f} "
                  f"{s_shift:>10.2f} {s_shift-s:>+9.2f}")

    print("\n=== 3. 그룹 평균의 신호 상한 (학습 없이) ===")
    print("학습 구간에서 그룹 평균을 구해 검증 시즌에 그대로 쓴다 (out).")
    print("검증 시즌 자신의 그룹 평균을 쓰면 상한이다 (in, 오라클).")
    print("out 과 in 의 차이가 그 피처에서 일어난 시즌 이동이다.\n")
    X = np.load(f"{CACHE}/X.npy", mmap_mode="r")
    meta = json.load(open(f"{CACHE}/cols.json"))
    ix = {c: i for i, c in enumerate(meta["cols"])}
    groups = ["base_state", "balls_before", "strikes_before", "outs_before",
              "inning", "top_bottom", "game_month", "pitcher_hand",
              "batter_hand", "pitcher_id"]
    fold = 2024
    tr, va = season < fold, season == fold
    yv = y[va].astype(np.float64)
    rate = yv.mean()
    base = rate * (1 - rate)
    print(f"{'그룹':>16} {'범주':>6} {'out':>9} {'in':>9} {'이동':>9}")
    ytr = y[tr].astype(np.float64)
    for g in groups:
        if g not in ix:
            continue
        col = np.asarray(X[:, ix[g]], dtype=np.float64)
        # bincount 로 그룹 평균을 한 번에 낸다 (범주마다 마스크를 돌면 pitcher_id
        # 792개 x 120만 행이라 분 단위로 느려진다)
        code = np.nan_to_num(col, nan=-1).astype(np.int64)
        code -= code.min()
        K = code.max() + 1
        ctr, cva = code[tr], code[va]
        s_tr, n_tr = np.bincount(ctr, ytr, K), np.bincount(ctr, None, K)
        s_va, n_va = np.bincount(cva, yv, K), np.bincount(cva, None, K)
        MIN = 200
        m_tr = np.where(n_tr >= MIN, s_tr / np.maximum(n_tr, 1), ytr.mean())
        m_va = np.where(n_va >= MIN, s_va / np.maximum(n_va, 1), rate)
        pout, pin = m_tr[cva], m_va[cva]
        so, si = score_of(yv, pout, base), score_of(yv, pin, base)
        print(f"{g:>16} {int((n_va>0).sum()):>6} {so:>9.1f} {si:>9.1f} "
              f"{si-so:>+9.1f}")

    print("\n=== 4. 오라클과 합법의 격차 — asof 는 그 신호를 얼마나 담고 있나 ===")
    print("1차원 비모수 적합: 학습 구간에서 분위 50칸의 평균을 구해 2024 에 쓴다.")
    print("pitcher_id 오라클(3번의 in)이 874.8 이었다. 그 중 얼마가 합법인가.\n")
    ytr = y[tr].astype(np.float64)
    print(f"{'컬럼':>44} {'점수':>9}")
    for c in ["asof_pitcher_success_rate",
              "asof_pitcher_prev5_game_success_rate",
              "asof_pitcher_prev1_game_success_rate",
              "asof_pitcher_middle_rate", "asof_batter_success_rate"]:
        if c not in ix:
            continue
        v = np.asarray(X[:, ix[c]], dtype=np.float64)
        vtr, vva = v[tr], v[va]
        ok = ~np.isnan(vtr)
        edges = np.unique(np.quantile(vtr[ok], np.linspace(0, 1, 51)))
        btr = np.clip(np.searchsorted(edges, vtr, "right") - 1, 0, len(edges) - 2)
        bva = np.clip(np.searchsorted(edges, vva, "right") - 1, 0, len(edges) - 2)
        K = len(edges) - 1
        s, n = np.bincount(btr[ok], ytr[ok], K), np.bincount(btr[ok], None, K)
        mean = np.where(n >= 200, s / np.maximum(n, 1), ytr.mean())
        p = mean[bva]
        p = np.where(np.isnan(vva), ytr.mean(), p)
        print(f"{c:>44} {score_of(yv, p, base):>9.1f}")


if __name__ == "__main__":
    main()
