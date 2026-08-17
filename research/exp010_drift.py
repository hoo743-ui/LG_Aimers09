r"""EXP010 — 개념 표류를 정보로 쓸 수 있는가. 학습 0회.

## 이 실험이 나온 경로 (연구 트리)

    FAILED       과거 시즌에서 학습한 잔차 학습기 (EXP009, 전이 상관 +0.0003)
    WHY FAILED   잔차 구조가 시즌마다 다르다
    MECHANISM    개념 표류 — 특징->타깃 사상이 시즌 간에 이동한다
    ALTERNATIVE  표류가 **잡음이 아니라 방향을 가진다면** 그 자체가 정보다

## 세 갈래를 한 번에 잰다

### 1. 표류의 구조 — 무작위인가 방향이 있는가

시즌별로 `corr(특징_j, 잔차_f)` 벡터 `c_f` 를 만든다. 그러면

    c_f 끼리의 상관이 높다      표류가 느리다 -> 과거 구조가 아직 쓸 만하다
    인접 시즌만 높다           표류가 빠르다 -> 최신 창만 의미 있다
    전부 0 근처                구조가 없다 (전부 잡음)

### 2. 표류가 단조인가 — 2025 로 외삽 가능한가

특징별로 `c_f` 를 시즌 인덱스에 회귀해 기울기를 잰다. 위약은 같은 절차를
셔플된 시즌 순서로 돌린 것이다. **단조 성분이 위약을 넘어야** 외삽에 의미가 있다.

### 3. 학습 창의 최신성 — 모델에도 적용되는가

30회차는 **표**에서 "최신성 > 표본량"을 증명했다(2시즌 > 5시즌, LB −9.95).
그런데 **모델 학습 창**에서는 검증된 적이 없다. 캐시된 `pred24_from*` 로
학습 없이 잰다.

    .\.venv\Scripts\python.exe -u research\exp010_drift.py
"""
import glob
import json
import os
import sys

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

EXP = "EXP010"
SEASONS = (2020, 2021, 2022, 2023, 2024)
PREV2 = {2022: (2020, 2021), 2023: (2021, 2022), 2024: (2022, 2023)}
SKIP = {"control_success", "season", "pitcher_id", "batter_id",
        "pitcher_team_id", "batter_team_id", "game_id", "game_date"}


def main():
    E.start_experiment(EXP, "L3-A", "python research/exp010_drift.py", "load")
    tr = build_df()
    season = tr["season"].to_numpy()
    y = tr["control_success"].to_numpy(np.float64)
    P = tr["pitcher_id"].to_numpy(np.int64)

    def g(c):
        return tr[c].to_numpy(np.float64)

    AX = {"hand": ((g("pitcher_hand") == g("batter_hand")).astype(int), 1000),
          "2S": ((g("strikes_before").astype(int) == 2).astype(int), 1000),
          "runner": ((g("num_runners_on") > 0).astype(int), 2000)}
    pv0, res0 = {}, {}
    for f in SEASONS:
        m = season == f
        pv0[f] = (np.load(os.path.join(ROOT, "exp", f"prod_champ_{f}.npy"))[:2].mean(0)
                  + post_for(tr, y, season < f, m))
        res0[f] = y[m] - pv0[f]

    num = [c for c in tr.columns if c not in SKIP
           and pd.api.types.is_numeric_dtype(tr[c])]
    X = np.nan_to_num(np.column_stack([g(c) for c in num]),
                      nan=0.0, posinf=0.0, neginf=0.0)

    # ---- 1. 표류의 구조 ----
    print("=" * 84)
    print("1. 시즌별 잔차-특징 상관 벡터 c_f 의 상호 상관 — 표류가 느린가 빠른가")
    print("=" * 84)
    C = {}
    for f in SEASONS:
        m = season == f
        r = res0[f]
        Xf = X[m]
        sd = Xf.std(0)
        ok = sd > 1e-9
        c = np.zeros(X.shape[1])
        c[ok] = ((Xf[:, ok] - Xf[:, ok].mean(0)) * (r - r.mean())[:, None]).mean(0) \
            / (sd[ok] * r.std())
        C[f] = c
        E.beat(f"c_{f}")
    M = pd.DataFrame(C)
    R = M.corr()
    print(f"  {'':<8}" + "".join(f"{s:>9}" for s in SEASONS))
    for i in SEASONS:
        print(f"  {i:<8}" + "".join(f"{R.loc[i, j]:>+9.3f}" for j in SEASONS))
    adj = [R.loc[SEASONS[i], SEASONS[i + 1]] for i in range(len(SEASONS) - 1)]
    far = [R.loc[SEASONS[i], SEASONS[j]] for i in range(len(SEASONS))
           for j in range(i + 2, len(SEASONS))]
    print(f"\n  인접 시즌 평균 {np.mean(adj):+.3f}   2시즌 이상 떨어진 평균 "
          f"{np.mean(far):+.3f}")
    print(f"  |c_f| 평균 크기 {np.abs(M.to_numpy()).mean():.4f} "
          f"(잡음 대역 1SE ≈ {1/np.sqrt(250000):.4f})")

    # ---- 2. 단조 표류 ----
    print("\n" + "=" * 84)
    print("2. 표류가 단조인가 — 시즌 인덱스에 대한 기울기 vs 셔플 위약")
    print("=" * 84)
    t = np.arange(len(SEASONS), dtype=float)
    t = (t - t.mean())
    A = M.to_numpy()
    slope = (A * t).sum(1) / (t ** 2).sum()
    rng = np.random.default_rng(0)
    ps = []
    for s in range(200):
        pt = rng.permutation(t)
        ps.append(np.abs((A * pt).sum(1) / (pt ** 2).sum()))
    ps = np.array(ps)
    frac = float(np.mean(np.abs(slope)[None, :] > ps))
    print(f"  |기울기| 가 셔플 위약보다 큰 비율 {frac:.1%}  (무작위면 50%)")
    top = np.argsort(-np.abs(slope))[:8]
    print(f"\n  {'특징':<40}{'기울기':>10}{'위약95%':>10}{'초과':>7}")
    q95 = np.percentile(ps, 95, axis=0)
    n_sig = 0
    for i in top:
        s_ = "예" if abs(slope[i]) > q95[i] else "아니오"
        n_sig += abs(slope[i]) > q95[i]
        print(f"  {num[i]:<40}{slope[i]:>+10.5f}{q95[i]:>10.5f}{s_:>7}")
    n_all = int(np.sum(np.abs(slope) > q95))
    print(f"\n  전체 {len(num)}열 중 위약 95% 초과 {n_all}열 "
          f"(우연 기대 {0.05*len(num):.1f}열)")

    # ---- 3. 학습 창의 최신성 ----
    print("\n" + "=" * 84)
    print("3. 모델 학습 창의 최신성 — 캐시된 pred24_from* 로 학습 없이")
    print("=" * 84)
    m24 = season == 2024
    C3add = np.zeros(int(m24.sum()))

    def dvec(ctx, src, k):
        p = np.concatenate([P[season == f] for f in src])
        c = np.concatenate([ctx[season == f] for f in src])
        r = np.concatenate([res0[f] for f in src])
        gg = pd.DataFrame({"p": p, "c": c, "r": r}).groupby(["p", "c"])["r"].agg(
            ["mean", "size"]).unstack()
        n0, n1 = gg[("size", 0)].fillna(0), gg[("size", 1)].fillna(0)
        ne = (n0 * n1) / (n0 + n1).replace(0, np.nan)
        return ((gg[("mean", 1)] - gg[("mean", 0)]) * ne / (ne + k)).dropna()

    for a, (ctx, k) in AX.items():
        C3add += (pd.Series(P[m24]).map(dvec(ctx, (2022, 2023), k)).fillna(0.).to_numpy()
                  * np.where(ctx[m24] == 1, .5, -.5))
    post24 = post_for(tr, y, season < 2024, m24)
    full = np.load(os.path.join(ROOT, "exp", "prod_champ_2024.npy"))
    win = {"전체 (2019~2023, 현행)": full[:2].mean(0)}
    for p in sorted(glob.glob(os.path.join(ROOT, "exp", "pred24_from*.npy"))):
        tag = os.path.basename(p).replace("pred24_from", "").replace(".npy", "")
        a = np.load(p)
        win[f"{tag}~2023 학습"] = a.mean(0) if a.ndim > 1 else a
    print(f"  {'학습 창':<28}{'C3 점수':>11}{'현행 대비':>11}")
    cur = None
    wout = {}
    for nm, pv in win.items():
        if len(pv) != int(m24.sum()):
            print(f"  {nm:<28}{'길이 불일치 — 건너뜀':>22}")
            continue
        v = r2(pv + post24 + C3add, y[m24])
        if cur is None:
            cur = v
        wout[nm] = v
        print(f"  {nm:<28}{v:>11.1f}{v - cur:>+11.1f}")

    out = dict(cf_corr=R.round(4).to_dict(), adj=float(np.mean(adj)),
               far=float(np.mean(far)), slope_frac=frac, n_sig=n_all,
               windows={k: round(v, 1) for k, v in wout.items()})
    best_win = max(wout, key=wout.get) if wout else "n/a"
    gain = (wout[best_win] - cur) if wout else 0.0
    dec = "PROMISING" if gain >= 4.8 else "REJECTED"
    E.set_hypothesis_status("L3-A", "PROMISING" if dec == "PROMISING" else "CLOSED",
                            level=3, hypothesis="개념 표류: 구조·단조성·학습 창 최신성",
                            result=round(gain, 2))
    E.finish_experiment(dict(
        experiment_id=EXP, hypothesis_id="L3-A", type="DRIFT", level=3,
        started_at=E.read(E.CKPT)["start_time"], results=out,
        decision=dec, artifact=None,
        what_we_learned=(f"c_f 인접상관 {np.mean(adj):+.3f} vs 원거리 "
                         f"{np.mean(far):+.3f}; 단조 초과 {n_all}/{len(num)}열 "
                         f"(우연 {0.05*len(num):.1f}); 최선 학습창 {best_win} {gain:+.1f}")))
    json.dump(out, open(os.path.join(ROOT, "exp", "exp010_drift.json"), "w"),
              indent=1, default=float)
    print(f"\nDECISION = {dec}")


if __name__ == "__main__":
    main()
