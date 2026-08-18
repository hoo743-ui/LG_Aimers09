r"""EXP038 — 시즌 감쇠를 **모든 후처리 축**에 적용한다. 학습 0회.

## 왜 여기로 왔는가

EXP037 에서 타자 수준 표에 지수 감쇠를 걸어 2024 +2.3 을 얻었다. 그 과정에서
30회차 실패(표 원천 5시즌, LB −9.95)의 원인이 **창 폭이 아니라 균등 가중**임을
확인했다 — 균등 4시즌은 −3.4 인데 감쇠 4시즌은 +1.6 이다.

그렇다면 같은 결함이 다른 축에도 있다.

    편차 4축   `post_for` 가 `season < f` 전체를 **균등 가중**으로 쓴다 (최대 5시즌)
               그런데 로컬 기여 **+22.7** 로 후처리 최대 성분이다
    대비 3축   하드 2시즌. 30회차 교훈이 창 폭 탓이었다면 감쇠로 넓힐 수 있다

## 주의 — 사슬이 얽혀 있다

편차 축을 바꾸면 잔차 `res0 = y − 모델 − 편차@W` 가 바뀌고, 그 위에 세운 대비·수준
표가 전부 바뀐다. 그래서 매 설정마다 **전체 사슬을 다시 계산**한다.

## 가중 nested_dev

원본은 표본수 `n` 으로 축소한다. 가중판은 **유효 가중합**으로 축소한다.

    dev = W_cell (cell_mean − parent_mean) / (W_cell + k)

    .\.venv\Scripts\python.exe -u research\exp038_decay_all.py
"""
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
import build_asof as ba                                     # noqa: E402
from path_alloc import build_df                             # noqa: E402
from traj_probe import r2                                   # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

EXP = "EXP038"
PREV2 = {2022: (2020, 2021), 2023: (2021, 2022), 2024: (2022, 2023)}
ALL = {2022: (2020, 2021), 2023: (2020, 2021, 2022), 2024: (2020, 2021, 2022, 2023)}


def nested_dev_w(parent, child, y, w, k):
    """가중판 nested_dev. 표본수 대신 유효 가중합으로 축소한다."""
    o = np.argsort(child, kind="stable")
    Ys, Ws, Ps, Cs = (y * w)[o], w[o], parent[o], child[o]
    u, s = np.unique(Cs, return_index=True)
    wsum = np.add.reduceat(Ws, s)
    cell = np.add.reduceat(Ys, s) / wsum
    par = Ps[s]
    op = np.argsort(parent, kind="stable")
    Yp, Wp, Pp = (y * w)[op], w[op], parent[op]
    pu, ps = np.unique(Pp, return_index=True)
    pw = np.add.reduceat(Wp, ps)
    pmean = np.add.reduceat(Yp, ps) / pw
    return u, wsum * (cell - pmean[np.searchsorted(pu, par)]) / (wsum + k)


def main():
    E.start_experiment(EXP, "D-ALL", "python research/exp038_decay_all.py", "load")
    tr = build_df()
    season = tr["season"].to_numpy()
    y = tr["control_success"].to_numpy(np.float64)
    P = tr["pitcher_id"].to_numpy(np.int64)
    B = tr["batter_id"].to_numpy(np.int64)
    BH = tr["batter_hand"].to_numpy(np.int64)
    BB = tr["balls_before"].to_numpy(np.int64)
    SS = tr["strikes_before"].to_numpy(np.int64)
    OB = (tr["num_runners_on"].to_numpy(np.int64) > 0).astype(np.int64)
    PH = P * 10 + BH
    PHA = PH * 10 + (SS > BB).astype(np.int64)
    AXD = [(P, PH), (PH, PHA), (PHA, PHA * 100 + (BB * 4 + SS)), (PH, PH * 10 + OB)]

    def g(c):
        return tr[c].to_numpy(np.float64)

    SAME = (g("pitcher_hand") == g("batter_hand")).astype(int)
    TWO = (SS == 2).astype(int)
    AXC = [(SAME, 1000), (TWO, 1000), (OB, 2000)]
    MODEL = {f: np.load(os.path.join(ROOT, "exp", f"prod_champ_{f}.npy"))[:2].mean(0)
             for f in (2020, 2021, 2022, 2023, 2024)}

    def devmat(f, gd):
        """폴드 f 의 편차 4열. gd 가 있으면 시즌 지수 감쇠."""
        m_tr, m_va = season < f, season == f
        w = (np.ones(int(m_tr.sum())) if gd is None
             else gd ** (f - 1 - season[m_tr].astype(float)))
        return np.column_stack([
            ba.look(*nested_dev_w(p[m_tr], c[m_tr], y[m_tr], w, k), c[m_va])
            for (p, c), k in zip(AXD, ba.KSH)])

    def chain(gd, gc, gb, wp, wb, wc=0.65):
        """편차 감쇠 gd, 대비 감쇠 gc, 타자수준 감쇠 gb 로 전체를 다시 세운다."""
        DEV = {f: devmat(f, gd) for f in (2020, 2021, 2022, 2023, 2024)}
        res = {f: y[season == f] - (MODEL[f] + DEV[f] @ np.array(ba.WPOST))
               for f in (2020, 2021, 2022, 2023, 2024)}

        def wtab(key, ctx, f, k, gamma, contrast):
            src = PREV2[f] if gamma is None else ALL[f]
            T = max(src)
            parts = []
            for s in src:
                m = season == s
                wt = 1.0 if gamma is None else gamma ** (T - s)
                d = {"k": key[m], "sr": res[s] * wt,
                     "n": np.full(int(m.sum()), wt)}
                if contrast:
                    d["c"] = ctx[m]
                parts.append(pd.DataFrame(d))
            D = pd.concat(parts)
            if not contrast:
                q = D.groupby("k")[["sr", "n"]].sum()
                return (q["sr"] / q["n"]) * q["n"] / (q["n"] + k)
            q = D.groupby(["k", "c"])[["sr", "n"]].sum().unstack()
            if ("n", 0) not in q or ("n", 1) not in q:
                return None
            n0, n1 = q[("n", 0)].fillna(0), q[("n", 1)].fillna(0)
            m0 = q[("sr", 0)] / n0.replace(0, np.nan)
            m1 = q[("sr", 1)] / n1.replace(0, np.nan)
            ne = (n0 * n1) / (n0 + n1).replace(0, np.nan)
            return ((m1 - m0) * ne / (ne + k)).dropna()

        out = {}
        for f in (2022, 2023, 2024):
            m = season == f
            v = MODEL[f] + DEV[f] @ np.array(ba.WPOST)
            for ctx, k in AXC:
                t = wtab(P, ctx, f, k, gc, True)
                v = v + wc * (pd.Series(P[m]).map(t).fillna(0.).to_numpy()
                              * np.where(ctx[m] == 1, .5, -.5))
            v = v + wp * pd.Series(P[m]).map(
                wtab(P, None, f, 50000, None, False)).fillna(0.).to_numpy()
            v = v + wb * pd.Series(B[m]).map(
                wtab(B, None, f, 20000, gb, False)).fillna(0.).to_numpy()
            out[f] = r2(v, y[m])
        return out

    E.beat("기준선")
    cur = chain(None, None, None, 1.0, 2.5)
    best_known = chain(None, None, 0.5, 2.0, 1.75)
    print(f"현행(감쇠 없음)      2022 {cur[2022]:.1f}  2023 {cur[2023]:.1f}  "
          f"2024 {cur[2024]:.1f}")
    print(f"EXP037 최선(타자만)  " + "  ".join(
        f"{f} {best_known[f] - cur[f]:+.1f}" for f in (2022, 2023, 2024)) + "\n")

    print("=" * 82)
    print("A. 편차 4축에 감쇠 (타자 수준은 EXP037 최선 유지)")
    print("=" * 82)
    print(f"{'편차 γ':>8}{'21→22':>10}{'22→23':>10}{'23→24':>10}{'2022&2024':>12}")
    A = {}
    for gd in (None, 0.5, 0.7, 0.85, 0.95):
        E.beat(f"dev {gd}")
        o = chain(gd, None, 0.5, 2.0, 1.75)
        gs = [o[f] - cur[f] for f in (2022, 2023, 2024)]
        A[str(gd)] = gs
        lab = "없음" if gd is None else f"{gd:.2f}"
        print(f"{lab:>8}" + "".join(f"{v:>+10.1f}" for v in gs)
              + f"{np.mean([gs[0], gs[2]]):>12.1f}"
              + ("  ★" if all(x > 0 for x in gs) else ""))

    print("\n" + "=" * 82)
    print("B. 대비 3축에 감쇠 (하드 2시즌 -> 감쇠 다중 시즌)")
    print("=" * 82)
    print(f"{'대비 γ':>8}{'21→22':>10}{'22→23':>10}{'23→24':>10}{'2022&2024':>12}")
    Bd = {}
    for gc in (None, 0.4, 0.55, 0.7):
        E.beat(f"ctr {gc}")
        o = chain(None, gc, 0.5, 2.0, 1.75)
        gs = [o[f] - cur[f] for f in (2022, 2023, 2024)]
        Bd[str(gc)] = gs
        lab = "하드2" if gc is None else f"{gc:.2f}"
        print(f"{lab:>8}" + "".join(f"{v:>+10.1f}" for v in gs)
              + f"{np.mean([gs[0], gs[2]]):>12.1f}"
              + ("  ★" if all(x > 0 for x in gs) else ""))

    E.finish_experiment(dict(
        experiment_id=EXP, hypothesis_id="D-ALL", type="B", level=3,
        started_at=E.read(E.CKPT)["start_time"], dev_decay=A, contrast_decay=Bd,
        decision="측정",
        what_we_learned="시즌 감쇠를 편차 4축과 대비 3축으로 확장"))
    json.dump({"dev": A, "ctr": Bd},
              open(os.path.join(ROOT, "exp", "exp038.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
