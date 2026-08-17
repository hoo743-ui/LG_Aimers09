r"""EXP020 — 물리 기제로 고른 **결합 국면**. 전수가 아니라 가설 6개. 학습 0회.

## 왜 여기가 빈칸인가

EXP003 의 246개 전수는 `build_contexts` 가 만든 **단일 컬럼** 국면뿐이다.
결합(conjunction)은 한 번도 열거되지 않았고, EXP001 은 이미 채택된 3축의
쌍 세 개만 봤다.

그리고 EXP019 가 확정한 원리가 **어떤 결합을 봐야 하는지** 지목한다.

    "이 국면에서 투수의 동작이나 선택이 물리적으로 바뀌는가?"

전수로 훑지 않는다 — 다중검정이 246개에서 어떻게 무너지는지 EXP014 에서
봤다(p=0.960). 기제가 명확한 **6개만** 본다.

## 후보와 그 물리적 근거

    P1 3루주자 & 2아웃미만   블로킹 실패가 곧 실점이다. 낮은 변화구를 못 던진다
    P2 만루                 세트포지션이되 **견제 부담이 없다**. 일반 주자와 다르다
    P3 1루주자만 & 2아웃미만  견제와 병살 유도. 퀵모션 부담이 가장 크다
    P4 3볼 & 주자있음        볼넷이 곧 진루다. 스트라이크를 넣어야 한다
    P5 2S & 3루주자          결정구를 던지고 싶은데 떨어뜨릴 수 없다 (상충)
    P6 초구(0-0) & 주자있음   타순 시작 + 세트포지션

P2 와 P5 는 **상충하는 요구**가 걸리는 국면이라 특히 흥미롭다 — 단일 국면의
선형 합으로는 표현되지 않는다.

## 판정 (5-c 규칙 전부 적용)

오라클(경기분할) + 위약 · 구성차 · 지속성 + 라벨섞기 · 3폴드 전이 · 기존 3축 겹침.

    .\.venv\Scripts\python.exe -u research\exp020_conj.py
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
from path_alloc import build_df                             # noqa: E402
from resid_table import post_for                            # noqa: E402
from traj_probe import cells, look, r2                      # noqa: E402
from game_decomp import games                               # noqa: E402
from exp016_persist import wcorr                            # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

EXP = "EXP020"
SEASONS = (2020, 2021, 2022, 2023, 2024)
PAIRS = [(2020, 2021), (2021, 2022), (2022, 2023), (2023, 2024)]
PREV2 = {2022: (2020, 2021), 2023: (2021, 2022), 2024: (2022, 2023)}
KGRID = [500, 1000, 2000, 5000, 20000]
MIN_NE = 20.0


def main():
    E.start_experiment(EXP, "L0-U", "python research/exp020_conj.py", "load")
    tr = build_df()
    season = tr["season"].to_numpy()
    y = tr["control_success"].to_numpy(np.float64)
    P = tr["pitcher_id"].to_numpy(np.int64)

    def g(c):
        return tr[c].to_numpy(np.float64)

    SAME = (g("pitcher_hand") == g("batter_hand")).astype(int)
    TWO = (g("strikes_before").astype(int) == 2).astype(int)
    RUN = (g("num_runners_on") > 0).astype(int)
    BB, SS = g("balls_before").astype(int), g("strikes_before").astype(int)
    OUT = g("outs_before").astype(int)
    R1, R2, R3 = (g("runner_on_1b") > 0), (g("runner_on_2b") > 0), (g("runner_on_3b") > 0)
    AX = {"hand": (SAME, 1000), "2S": (TWO, 1000), "runner": (RUN, 2000)}

    pv0, res0 = {}, {}
    for f in SEASONS:
        m = season == f
        pv0[f] = (np.load(os.path.join(ROOT, "exp", f"prod_champ_{f}.npy"))[:2].mean(0)
                  + post_for(tr, y, season < f, m))
        res0[f] = y[m] - pv0[f]

    def dtab(ctx, src, k):
        p = np.concatenate([P[season == f] for f in src])
        c = np.concatenate([ctx[season == f] for f in src])
        r = np.concatenate([res0[f] for f in src])
        ok = np.isin(c, (0, 1))
        gg = pd.DataFrame({"p": p[ok], "c": c[ok], "r": r[ok]}).groupby(
            ["p", "c"])["r"].agg(["mean", "size"]).unstack()
        if ("size", 0) not in gg or ("size", 1) not in gg:
            return None
        n0, n1 = gg[("size", 0)].fillna(0), gg[("size", 1)].fillna(0)
        ne = (n0 * n1) / (n0 + n1).replace(0, np.nan)
        return ((gg[("mean", 1)] - gg[("mean", 0)]) * ne / (ne + k)).dropna()

    def ap(t, ctx, m):
        if t is None:
            return np.zeros(int(m.sum()))
        h = np.where(ctx[m] == 1, .5, np.where(ctx[m] == 0, -.5, 0.))
        return pd.Series(P[m]).map(t).fillna(0.).to_numpy() * h

    C3, resC = {}, {}
    for f in (2022, 2023, 2024):
        m = season == f
        v = pv0[f].copy()
        for a, (ctx, k) in AX.items():
            v += ap(dtab(ctx, PREV2[f], k), ctx, m)
        C3[f], resC[f] = v, y[m] - v
    m24 = season == 2024
    base = r2(C3[2024], y[m24])
    print(f"C3 기준선 2024 = {base:.1f}\n")

    # 결합 국면 — 비교군은 '주자 있는 행 안에서' 로 좁혀 주자 축과 분리한다
    ex = lambda cond, pool: np.where(pool, cond.astype(np.int64), -1)
    CAND = {
        "P1 3루주자 & 2아웃미만": ex(R3 & (OUT < 2), RUN == 1),
        "P2 만루": ex(R1 & R2 & R3, RUN == 1),
        "P3 1루주자만 & 2아웃미만": ex(R1 & ~R2 & ~R3 & (OUT < 2), RUN == 1),
        "P4 3볼 & 주자있음": ex(BB == 3, RUN == 1),
        "P5 2S & 3루주자": ex((SS == 2) & R3, RUN == 1),
        "P6 초구 & 주자있음": ex((BB == 0) & (SS == 0), RUN == 1),
        "[대조] 같은손": SAME,
    }

    GID = games(P[m24], g("asof_pitcher_n")[m24],
                g("asof_pitcher_prev1_game_success_rate")[m24],
                g("asof_pitcher_prev1_game_middle_rate")[m24])
    rng = np.random.default_rng(0)
    u, inv = np.unique(GID, return_inverse=True)
    half = (rng.random(len(u)) < 0.5)[inv]

    def oracle(key):
        b = -1e9
        for k in (0, 150, 500, 2000, 10000):
            add = np.zeros(int(m24.sum()))
            for m in (half, ~half):
                uu, tb, _ = cells(key[m], resC[2024][m], k)
                add[~m] = look(uu, tb, key[~m])
            b = max(b, r2(C3[2024] + add, y[m24]) - base)
        return b

    def persist(ctx, shuffle=False):
        per = {}
        for f in SEASONS:
            m = season == f
            p_ = P[m]
            if shuffle:
                p_ = np.random.default_rng(f).permutation(p_)
            c, r = ctx[m], res0[f]
            ok = np.isin(c, (0, 1))
            gg = pd.DataFrame({"p": p_[ok], "c": c[ok], "r": r[ok]}).groupby(
                ["p", "c"])["r"].agg(["mean", "size"]).unstack()
            if ("size", 0) not in gg or ("size", 1) not in gg:
                continue
            n0, n1 = gg[("size", 0)].fillna(0), gg[("size", 1)].fillna(0)
            ne = (n0 * n1) / (n0 + n1).replace(0, np.nan)
            per[f] = pd.DataFrame({"d": gg[("mean", 1)] - gg[("mean", 0)],
                                   "ne": ne}).dropna()
        v, ns = [], []
        for a, b in PAIRS:
            if a not in per or b not in per:
                continue
            j = per[a].join(per[b], lsuffix="a", rsuffix="b", how="inner")
            j = j[(j["nea"] >= MIN_NE) & (j["neb"] >= MIN_NE)]
            if len(j) < 30:
                continue
            v.append(wcorr(j["da"].to_numpy(), j["db"].to_numpy(),
                           (j["nea"] * j["neb"] / (j["nea"] + j["neb"])).to_numpy()))
            ns.append(len(j))
        if not v:
            return dict(cross=np.nan, z=np.nan, n=0)
        cr, n = float(np.nanmean(v)), int(np.mean(ns))
        return dict(cross=cr, n=n, z=cr * np.sqrt(max(n - 3, 1)))

    HD = ap(dtab(SAME, (2022, 2023), 1000), SAME, m24)
    TS = ap(dtab(TWO, (2022, 2023), 1000), TWO, m24)
    RN = ap(dtab(RUN, (2022, 2023), 2000), RUN, m24)
    plac = oracle(P[m24] * 10 + rng.integers(0, 2, int(m24.sum())))

    print("=" * 110)
    print(f"{'국면':<26}{'=1비율':>8}{'제외':>7}{'오라클':>8}{'위약':>7}"
          f"{'지속z':>8}{'섞기z':>8}{'21→22':>8}{'22→23':>8}{'23→24':>8}"
          f"{'k':>7}{'C3증분':>8}")
    print("=" * 110)
    out = {}
    for nm, ctx in CAND.items():
        E.beat(nm)
        c24 = ctx[m24]
        orc = oracle(P[m24] * 10 + np.clip(c24, 0, 1))
        pr, sh = persist(ctx), persist(ctx, shuffle=True)
        G = {k: [] for k in KGRID}
        for a, b in ((2021, 2022), (2022, 2023), (2023, 2024)):
            mb = season == b
            bs = r2(C3[b], y[mb])
            for k in KGRID:
                G[k].append(r2(C3[b] + ap(dtab(ctx, (a,), k), ctx, mb), y[mb]) - bs)
        kb = max(G, key=lambda k: np.mean(G[k][:2]))
        add24 = ap(dtab(ctx, (2022, 2023), kb), ctx, m24)
        inc = r2(C3[2024] + add24, y[m24]) - base
        ov = [float(np.corrcoef(add24, v)[0, 1]) if add24.std() > 0 else 0.
              for v in (HD, TS, RN)]
        out[nm] = dict(frac=float(np.mean(c24 == 1)), excl=float(np.mean(c24 == -1)),
                       oracle=orc, placebo=plac, z=pr["z"], shuffle_z=sh["z"],
                       gains=G[kb], k=kb, inc=inc, overlap=ov)
        print(f"{nm:<26}{np.mean(c24==1):>8.1%}{np.mean(c24==-1):>7.0%}"
              f"{orc:>8.1f}{plac:>7.1f}{pr['z']:>8.2f}{sh['z']:>8.2f}"
              + "".join(f"{v:>+8.1f}" for v in G[kb]) + f"{kb:>7}{inc:>+8.1f}")

    print(f"\n{'국면':<26}{'hand':>8}{'2S':>8}{'runner':>9}  겹침 상관")
    for nm in CAND:
        o = out[nm]["overlap"]
        print(f"{nm:<26}{o[0]:>+8.2f}{o[1]:>+8.2f}{o[2]:>+9.2f}")

    new = [nm for nm in CAND if not nm.startswith("[대조]")
           and out[nm]["inc"] >= 4.8 and all(v > 0 for v in out[nm]["gains"])
           and out[nm]["oracle"] > out[nm]["placebo"] + 3]
    dec = "PROMISING" if new else "REJECTED"
    E.set_hypothesis_status("L0-U", "PROMISING" if new else "CLOSED", level=0,
                            hypothesis="물리 기제 결합 국면 6종", result=len(new))
    E.finish_experiment(dict(
        experiment_id=EXP, hypothesis_id="L0-U", type="A", level=0,
        started_at=E.read(E.CKPT)["start_time"], n_hypotheses=6,
        multiple_testing_control="전수가 아니라 기제로 고른 6개만 (EXP014 교훈)",
        results={k: {kk: (round(vv, 4) if isinstance(vv, float)
                          else ([round(x, 2) for x in vv] if isinstance(vv, list) else vv))
                     for kk, vv in v.items()} for k, v in out.items()},
        decision=dec, artifact=None,
        what_we_learned=("단일 컬럼 전수(246)가 못 본 결합 국면을 물리 기제로 "
                         f"6개 골라 검사. 통과 {len(new)}개")))
    json.dump(out, open(os.path.join(ROOT, "exp", "exp020_conj.json"), "w"),
              indent=1, default=float)
    print(f"\nDECISION = {dec}")


if __name__ == "__main__":
    main()
