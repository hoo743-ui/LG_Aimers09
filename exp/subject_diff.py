r"""NON-PITCHER SUBJECT DIFFERENTIAL — 차등의 주체를 타자·팀으로 바꾼다. 학습 0회.

투수 주체 차등은 소진됐다 (손/2S/주자 셋만 살아 C3 에 채택, 연속 기울기는 위약
수준). 마지막으로 **주체를 바꿔** 최소 비용으로 확인한다.

    d_subject = 잔차평균(주체, 조건=1) - 잔차평균(주체, 조건=0)

주체 = 타자 / 투수팀 / 타자팀.  조건은 그 행의 이산 국면.

## 강한 종료 규칙 (사용자 지정)

    오라클 <= 위약        -> 즉시 종료
    오라클 < +3           -> 우선순위 매우 낮음
    전이 0 / C3 증분 <= +0.5 -> 종료

## 참고 (경기 단위 분할 기준)

    타자 주효과 오라클 +2.2   투수팀 −6.8   타자팀 +11.6(전이 −9.7)

    .\.venv\Scripts\python.exe -u exp\subject_diff.py
"""
import io
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "exp"))
from path_alloc import build_df                            # noqa: E402
from resid_table import post_for                           # noqa: E402
from traj_probe import cells, look, r2                     # noqa: E402
from game_decomp import games                              # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

KGRID = [500, 1000, 2000, 5000, 20000]
PREV2 = {2022: (2020, 2021), 2023: (2021, 2022), 2024: (2022, 2023)}


def main():
    tr = build_df()
    season = tr["season"].to_numpy()
    y = tr["control_success"].to_numpy(np.float64)
    P = tr["pitcher_id"].to_numpy(np.int64)
    B = tr["batter_id"].to_numpy(np.int64)
    PT = tr["pitcher_team_id"].to_numpy(np.int64)
    BT = tr["batter_team_id"].to_numpy(np.int64)
    g = lambda c: tr[c].to_numpy(np.float64)
    SS = g("strikes_before").astype(int)
    SAME = (g("pitcher_hand") == g("batter_hand")).astype(int)
    TWO = (SS == 2).astype(int)
    RUN = (g("num_runners_on") > 0).astype(int)
    ISF = (tr["game_type"].to_numpy() == "F").astype(int)

    pv0, res0 = {}, {}
    for f in (2020, 2021, 2022, 2023, 2024):
        m = season == f
        pv0[f] = (np.load(os.path.join(ROOT, "exp", f"prod_champ_{f}.npy"))[:2].mean(0)
                  + post_for(tr, y, season < f, m))
        res0[f] = y[m] - pv0[f]

    def dtab(subj, ctx, src, k, rmap):
        s = np.concatenate([subj[season == f] for f in src])
        c = np.concatenate([ctx[season == f] for f in src])
        r = np.concatenate([rmap[f] for f in src])
        gg = pd.DataFrame({"s": s, "c": c, "r": r}).groupby(["s", "c"])["r"].agg(
            ["mean", "size"]).unstack()
        if ("size", 0) not in gg or ("size", 1) not in gg:
            return None
        n0, n1 = gg[("size", 0)].fillna(0), gg[("size", 1)].fillna(0)
        d = gg[("mean", 1)] - gg[("mean", 0)]
        ne = (n0 * n1) / (n0 + n1).replace(0, np.nan)
        return (d * ne / (ne + k)).dropna()

    def ap(t, subj, ctx, m):
        if t is None:
            return np.zeros(int(m.sum()))
        return (pd.Series(subj[m]).map(t).fillna(0.0).to_numpy()
                * np.where(ctx[m] == 1, .5, -.5))

    C3, resC = {}, {}
    for f in (2022, 2023, 2024):
        m = season == f
        s = PREV2[f]
        C3[f] = (pv0[f] + ap(dtab(P, SAME, s, 1000, res0), P, SAME, m)
                 + ap(dtab(P, TWO, s, 1000, res0), P, TWO, m)
                 + ap(dtab(P, RUN, s, 2000, res0), P, RUN, m))
        resC[f] = y[m] - C3[f]
    m24 = season == 2024
    base = r2(C3[2024], y[m24])
    print(f"  C3 기준선 (폴드 2024) = {base:.1f}\n")

    GID = games(P[m24], g("asof_pitcher_n")[m24],
                g("asof_pitcher_prev1_game_success_rate")[m24],
                g("asof_pitcher_prev1_game_middle_rate")[m24])
    rng = np.random.default_rng(0)
    u, inv = np.unique(GID, return_inverse=True)
    half = (rng.random(len(u)) < 0.5)[inv]
    HD = ap(dtab(P, SAME, (2022, 2023), 1000, res0), P, SAME, m24)
    TS = ap(dtab(P, TWO, (2022, 2023), 1000, res0), P, TWO, m24)
    RN = ap(dtab(P, RUN, (2022, 2023), 2000, res0), P, RUN, m24)

    SUBJ = {"타자": B, "투수팀": PT, "타자팀": BT}
    CTX = {"상대 같은손": SAME, "2스트라이크": TWO, "주자 있음": RUN, "F 경기": ISF}

    def oracle(key, plac=False):
        k_use = rng.permutation(key) if plac else key
        best = -1e9
        for k in (0, 150, 500, 2000, 10000):
            add = np.zeros(int(m24.sum()))
            for m in (half, ~half):
                uu, tb, _ = cells(k_use[m], resC[2024][m], k)
                add[~m] = look(uu, tb, k_use[~m])
            best = max(best, r2(C3[2024] + add, y[m24]) - base)
        return best

    print("=" * 100)
    print(f"{'주체 x 조건':<24}{'셀':>8}{'오라클':>8}{'위약':>8}{'21→22':>8}"
          f"{'22→23':>8}{'23→24':>8}{'k':>7}{'C3증분':>8}{'C3잔차상관':>11}")
    print("=" * 100)
    out = {}
    for sn, subj in SUBJ.items():
        for cn, ctx in CTX.items():
            key24 = subj[m24] * 10 + ctx[m24]
            orc = oracle(key24)
            plc = oracle(key24, plac=True)
            if orc <= plc:                       # 강한 종료 규칙
                print(f"{sn+' x '+cn:<24}{len(np.unique(key24)):>8,}{orc:>8.1f}"
                      f"{plc:>8.1f}{'—':>8}{'—':>8}{'—':>8}{'—':>7}{'—':>8}"
                      f"{'즉시 종료':>11}")
                out[f"{sn}|{cn}"] = dict(oracle=orc, placebo=plc, closed=True)
                continue
            G = {k: [] for k in KGRID}
            for a, b in ((2021, 2022), (2022, 2023), (2023, 2024)):
                mb = season == b
                bs = r2(C3[b], y[mb])
                for k in KGRID:
                    G[k].append(r2(C3[b] + ap(dtab(subj, ctx, (a,), k, res0),
                                              subj, ctx, mb), y[mb]) - bs)
            kb = max(G, key=lambda k: np.mean(G[k][:2]))
            add24 = ap(dtab(subj, ctx, (2022, 2023), kb, res0), subj, ctx, m24)
            inc = r2(C3[2024] + add24, y[m24]) - base
            cr = (float(np.corrcoef(add24, resC[2024])[0, 1])
                  if add24.std() > 0 else 0.0)
            out[f"{sn}|{cn}"] = dict(oracle=orc, placebo=plc, k=kb,
                                     gains=G[kb], inc=inc, corr_resid=cr)
            print(f"{sn+' x '+cn:<24}{len(np.unique(key24)):>8,}{orc:>8.1f}"
                  f"{plc:>8.1f}" + "".join(f"{v:>+8.1f}" for v in G[kb])
                  + f"{kb:>7}{inc:>+8.1f}{cr:>+11.4f}")

    json.dump(out, io.open(os.path.join(ROOT, "exp", "subject_diff.json"), "w",
                           encoding="utf-8"), indent=1, ensure_ascii=False,
              default=float)
    print("\n  오라클/위약 = 경기 단위 분할. k 는 과거 2전이로만 선택.")


if __name__ == "__main__":
    main()
