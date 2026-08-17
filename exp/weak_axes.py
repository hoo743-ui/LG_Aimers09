r"""persistent weak signal 탐색 — hand_diff 와 같은 틀을 다른 맥락 축에 적용.

## 통일된 정의

모든 축을 **투수별 상태 차등**으로 쓴다.

    d_p = 잔차평균(맥락=1) - 잔차평균(맥락=0)
    n_eff = n1*n0/(n1+n0),   보정 = ±0.5 * d_p * n_eff/(n_eff+k)

잔차는 **strictly out-of-fold** (그 시즌을 학습하지 않은 모델). 표는 목표 폴드
직전 두 시즌으로만 만들고, `k` 는 **과거 전이에서만** 고른다.

## 단계

    1 오라클     **경기 단위 분할** + 위약 대조 (투구 단위 금지)
    2 지속성     d_p 의 21->22 / 22->23 / 23->24 상관과 진폭
    3 전이       k 격자, 과거 전이로 k 선택
    4 생산       현행 4축 + hand_diff **위에** 가산, 시드 42/43
    5 중복       hand_diff 및 현행 잔차와의 상관, 양방향 증분

    .\.venv\Scripts\python.exe -u exp\weak_axes.py
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

KGRID = [500, 1000, 1500, 2000, 3000, 5000]
FOLDS = (2022, 2023, 2024)


def main():
    tr = build_df()
    season = tr["season"].to_numpy()
    y = tr["control_success"].to_numpy(np.float64)
    P = tr["pitcher_id"].to_numpy(np.int64)
    g = lambda c: tr[c].to_numpy(np.float64)
    BB, SS = g("balls_before").astype(int), g("strikes_before").astype(int)
    CTX = {
        "같은손 (hand_diff, 참조)": (g("pitcher_hand") == g("batter_hand")).astype(int),
        "카운트 우위 (S>B)": (SS > BB).astype(int),
        "2스트라이크": (SS == 2).astype(int),
        "3볼": (BB == 3).astype(int),
        "주자 있음": (g("num_runners_on") > 0).astype(int),
        "2아웃": (g("outs_before") == 2).astype(int),
        "시즌 후반 (7월~)": (g("game_month") >= 7).astype(int),
        "F 경기": (tr["game_type"].to_numpy() == "F").astype(int),
    }
    pv, res, GID = {}, {}, {}
    for f in (2021,) + FOLDS:
        m = season == f
        pv[f] = (np.load(os.path.join(ROOT, "exp", f"prod_champ_{f}.npy"))[:2].mean(0)
                 + post_for(tr, y, season < f, m))
        res[f] = y[m] - pv[f]
        GID[f] = games(P[m], g("asof_pitcher_n")[m],
                       g("asof_pitcher_prev1_game_success_rate")[m],
                       g("asof_pitcher_prev1_game_middle_rate")[m])
    m24 = season == 2024
    base24 = r2(pv[2024], y[m24])
    print(f"기준선 2024 = {base24:.1f}\n")

    def diff_tab(src, ctx, k):
        p = np.concatenate([P[season == f] for f in src])
        c = np.concatenate([ctx[season == f] for f in src])
        r = np.concatenate([res[f] for f in src])
        gg = pd.DataFrame({"p": p, "c": c, "r": r}).groupby(["p", "c"])["r"].agg(
            ["mean", "size"]).unstack()
        if ("size", 0) not in gg or ("size", 1) not in gg:
            return None
        n0, n1 = gg[("size", 0)].fillna(0), gg[("size", 1)].fillna(0)
        d = gg[("mean", 1)] - gg[("mean", 0)]
        ne = (n0 * n1) / (n0 + n1).replace(0, np.nan)
        return (d * ne / (ne + k)).dropna()

    def apply_t(tab, ctx, m):
        if tab is None:
            return np.zeros(int(m.sum()))
        h = np.where(ctx[m] == 1, 0.5, -0.5)
        return pd.Series(P[m]).map(tab).fillna(0.0).to_numpy() * h

    # hand_diff 기준 신호 (k=1000, 직전 두 시즌)
    HD = apply_t(diff_tab((2022, 2023), CTX["같은손 (hand_diff, 참조)"], 1000),
                 CTX["같은손 (hand_diff, 참조)"], m24)
    pv_hd = pv[2024] + HD
    base_hd = r2(pv_hd, y[m24])
    print(f"hand_diff 적용 후 기준선 = {base_hd:.1f} ({base_hd-base24:+.1f})\n")

    rng = np.random.default_rng(0)
    u, inv = np.unique(GID[2024], return_inverse=True)
    half = (rng.random(len(u)) < 0.5)[inv]

    out = {}
    print("=" * 104)
    print(f"{'축':<22}{'오라클':>8}{'위약':>7}{'21→22':>8}{'22→23':>8}{'23→24':>8}"
          f"{'지속성':>9}{'최적k':>7}{'2024':>8}{'+hd 위':>8}{'hd상관':>8}")
    print("=" * 104)
    for name, ctx in CTX.items():
        key = P * 10 + ctx
        # 1 오라클 (경기 분할) + 위약
        best = -1e9
        for k in (0, 150, 500, 2000, 10000):
            add = np.zeros(len(y[m24]))
            for m in (half, ~half):
                uu, tab, _ = cells(key[m24][m], res[2024][m], k)
                add[~m] = look(uu, tab, key[m24][~m])
            best = max(best, r2(pv[2024] + add, y[m24]) - base24)
        pl = P[m24] * 10 + rng.integers(0, 2, int(m24.sum()))
        bp = -1e9
        for k in (0, 150, 500, 2000, 10000):
            add = np.zeros(len(y[m24]))
            for m in (half, ~half):
                uu, tab, _ = cells(pl[m], res[2024][m], k)
                add[~m] = look(uu, tab, pl[~m])
            bp = max(bp, r2(pv[2024] + add, y[m24]) - base24)
        # 2 지속성
        dd = {}
        for f in (2021,) + FOLDS:
            t = diff_tab((f,), ctx, 0)
            dd[f] = t
        cors = []
        for a, b in ((2021, 2022), (2022, 2023), (2023, 2024)):
            if dd[a] is None or dd[b] is None:
                cors.append(np.nan)
                continue
            J = pd.concat([dd[a], dd[b]], axis=1, join="inner").dropna()
            cors.append(float(np.corrcoef(J.iloc[:, 0], J.iloc[:, 1])[0, 1])
                        if len(J) > 30 else np.nan)
        # 3 전이 — k 는 과거 두 전이로 선택
        gains = {k: [] for k in KGRID}
        for a, b in ((2021, 2022), (2022, 2023), (2023, 2024)):
            mb = season == b
            bs = r2(pv[b], y[mb])
            for k in KGRID:
                gains[k].append(r2(pv[b] + apply_t(diff_tab((a,), ctx, k), ctx, mb),
                                   y[mb]) - bs)
        past = {k: np.mean(gains[k][:2]) for k in KGRID}
        kb = max(past, key=past.get)
        t24 = gains[kb][2]
        # 4 생산 (직전 두 시즌 표) + hd 위에 가산
        add24 = apply_t(diff_tab((2022, 2023), ctx, kb), ctx, m24)
        g_alone = r2(pv[2024] + add24, y[m24]) - base24
        g_on_hd = r2(pv_hd + add24, y[m24]) - base_hd
        c_hd = float(np.corrcoef(add24, HD)[0, 1]) if add24.std() > 0 else 0.0
        out[name] = dict(oracle=best, placebo=bp, persist=cors, k=kb,
                         t2122=gains[kb][0], t2223=gains[kb][1], t2324=t24,
                         alone=g_alone, on_hd=g_on_hd, corr_hd=c_hd)
        pc = np.nanmean(cors)
        print(f"{name:<22}{best:>8.1f}{bp:>7.1f}"
              + "".join(f"{v:>+8.1f}" for v in
                        (gains[kb][0], gains[kb][1], t24))
              + f"{pc:>+9.3f}{kb:>7}{g_alone:>+8.1f}{g_on_hd:>+8.1f}{c_hd:>+8.2f}")

    json.dump(out, io.open(os.path.join(ROOT, "exp", "weak_axes.json"), "w",
                           encoding="utf-8"), indent=1, ensure_ascii=False,
              default=float)
    print("\n  오라클/위약은 경기 단위 분할. k 는 과거 두 전이로만 선택.")
    print("  '+hd 위' = hand_diff 를 이미 얹은 뒤의 추가 증분 (중복 제거된 값)")


if __name__ == "__main__":
    main()
