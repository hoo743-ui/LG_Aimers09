r"""EXP007 — LEVEL 2 모델 다양성의 값. 학습 0회 (캐시된 예측만 쓴다).

## 무엇을 묻는가

`rho` 는 아핀 불변이라 스케일 보정으로는 못 오르지만, **오차가 서로 다른 모델을
섞는 것**은 다르다. 오차의 무상관 성분이 지워지면 신호 대비 잡음이 오른다.

Champion 은 캐시된 2024 예측 7개 중 **앞 2개 평균**만 쓴다. 남은 5개로
"같은 모델 계열 안에서 오차 무상관이 어디까지 남아 있는가"를 학습 없이 잰다.

    포화했다면    같은 계열의 다양성은 소진 -> 구조가 다른 학습기만 의미 있다
    안 했다면     남은 예측을 섞는 것만으로 점수가 오른다 (학습 0회)

## 왜 이게 LEVEL 2 인가

LEVEL 2 는 model space 다. 다양성/스태킹은 그 중 **가장 값싼 면**이고,
결과가 다른 모든 LEVEL 2 후보(대체 트리 구조·GAM·전문가 구조)의 기대값을
직접 규정한다 — 다양성 이득의 상한이 곧 그것들이 노릴 수 있는 몫이다.

    .\.venv\Scripts\python.exe -u research\exp007_diversity.py
"""
import json
import os
import sys
import itertools

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "exp"))
import engine as E                                          # noqa: E402
from path_alloc import build_df                             # noqa: E402
from resid_table import post_for                            # noqa: E402
from traj_probe import r2                                   # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

EXP = "EXP007"


def main():
    E.start_experiment(EXP, "L2-A", "python research/exp007_diversity.py", "load")
    tr = build_df()
    season = tr["season"].to_numpy()
    y = tr["control_success"].to_numpy(np.float64)
    P = tr["pitcher_id"].to_numpy(np.int64)

    def g(c):
        return tr[c].to_numpy(np.float64)

    AX = {"hand": ((g("pitcher_hand") == g("batter_hand")).astype(int), 1000),
          "2S": ((g("strikes_before").astype(int) == 2).astype(int), 1000),
          "runner": ((g("num_runners_on") > 0).astype(int), 2000)}
    res0 = {}
    for f in (2020, 2021, 2022, 2023):
        m = season == f
        res0[f] = y[m] - (np.load(os.path.join(ROOT, "exp", f"prod_champ_{f}.npy"))[:2].mean(0)
                          + post_for(tr, y, season < f, m))

    def dvec(ctx, src, k):
        p = np.concatenate([P[season == f] for f in src])
        c = np.concatenate([ctx[season == f] for f in src])
        r = np.concatenate([res0[f] for f in src])
        gg = pd.DataFrame({"p": p, "c": c, "r": r}).groupby(["p", "c"])["r"].agg(
            ["mean", "size"]).unstack()
        n0, n1 = gg[("size", 0)].fillna(0), gg[("size", 1)].fillna(0)
        ne = (n0 * n1) / (n0 + n1).replace(0, np.nan)
        return ((gg[("mean", 1)] - gg[("mean", 0)]) * ne / (ne + k)).dropna()

    for fold in (2022, 2023, 2024):
        m = season == fold
        A = np.load(os.path.join(ROOT, "exp", f"prod_champ_{fold}.npy"))
        post = post_for(tr, y, season < fold, m)
        src = {2022: (2020, 2021), 2023: (2021, 2022), 2024: (2022, 2023)}[fold]
        diff = np.zeros(int(m.sum()))
        for a, (ctx, k) in AX.items():
            diff += (pd.Series(P[m]).map(dvec(ctx, src, k)).fillna(0.).to_numpy()
                     * np.where(ctx[m] == 1, .5, -.5))
        add = post + diff
        yy = y[m]
        print(f"\n{'='*72}\n폴드 {fold} — 캐시 예측 {A.shape[0]}개\n{'='*72}")
        solo = [r2(A[i] + add, yy) for i in range(A.shape[0])]
        print("  개별 점수  " + "  ".join(f"[{i}] {v:.1f}" for i, v in enumerate(solo)))
        if fold != 2024:
            continue
        E.beat("2024 다양성")

        # 오차 상관 — 얼마나 다른 실수를 하는가
        errs = np.array([A[i] - yy for i in range(A.shape[0])])
        C = np.corrcoef(errs)
        off = C[np.triu_indices(len(C), 1)]
        print(f"\n  오차 상관  중앙 {np.median(off):.4f}  최소 {off.min():.4f}"
              f"  최대 {off.max():.4f}")
        Cp = np.corrcoef(np.array([A[i] for i in range(A.shape[0])]))
        print(f"  예측 상관  중앙 {np.median(Cp[np.triu_indices(len(Cp),1)]):.4f}")

        # 개수별 평균 — 포화 곡선 (조합 평균으로 순서 편향 제거)
        print(f"\n  {'섞는 개수':<10}{'평균 점수':>11}{'최선':>10}{'현행(앞2) 대비':>15}")
        cur = r2(A[:2].mean(0) + add, yy)
        curve = {}
        for n in range(1, A.shape[0] + 1):
            combos = list(itertools.combinations(range(A.shape[0]), n))
            if len(combos) > 25:
                rng = np.random.default_rng(0)
                combos = [combos[i] for i in
                          rng.choice(len(combos), 25, replace=False)]
            vs = [r2(A[list(c)].mean(0) + add, yy) for c in combos]
            curve[n] = dict(mean=float(np.mean(vs)), max=float(max(vs)))
            print(f"  {n:<10}{np.mean(vs):>11.1f}{max(vs):>10.1f}"
                  f"{np.mean(vs) - cur:>+15.1f}")
        full = r2(A.mean(0) + add, yy)
        print(f"\n  현행 앞2 평균 {cur:.1f}   전체 {A.shape[0]}개 평균 {full:.1f}"
              f"   차이 {full - cur:+.1f}")

        out = dict(solo=[round(v, 1) for v in solo],
                   err_corr_median=float(np.median(off)),
                   err_corr_min=float(off.min()), curve=curve,
                   current=cur, full=full, gain=full - cur)
        gain = full - cur
        dec = "PROMISING" if gain >= 4.8 else "REJECTED"
        E.set_hypothesis_status("L2-A", "PROMISING" if dec == "PROMISING" else "CLOSED",
                                level=2, hypothesis="같은 계열 모델 다양성(예측 혼합)",
                                result=round(gain, 2))
        E.finish_experiment(dict(
            experiment_id=EXP, hypothesis_id="L2-A", type="MODEL", level=2,
            started_at=E.read(E.CKPT)["start_time"],
            n_cached=int(A.shape[0]), err_corr_median=float(np.median(off)),
            current=round(cur, 1), full_blend=round(full, 1), gain=round(gain, 1),
            decision=dec, artifact=None,
            what_we_learned=(f"캐시 예측 {A.shape[0]}개 오차상관 중앙 "
                             f"{np.median(off):.4f}. 전부 섞으면 {gain:+.1f}")))
        json.dump(out, open(os.path.join(ROOT, "exp", "exp007_diversity.json"), "w"),
                  indent=1, default=float)
        print(f"\nDECISION = {dec}")


if __name__ == "__main__":
    main()
