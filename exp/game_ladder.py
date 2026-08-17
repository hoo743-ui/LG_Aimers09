r"""GAME-ORACLE 분해 진단 — 큰 오라클의 정보가 어느 수준에 있는가. 학습 0회.

**제출 피처를 만들지 않는다.** 오라클의 구조를 알아내는 진단 전용이다.

## 1 분할 사다리

같은 오라클 정의를 분할 단위만 바꿔 반복한다.

    투구 무작위 / 타석 / 경기 / 시간(시즌 전반 경기 -> 후반 경기)

키가 경기 자체인 오라클은 경기 분할이 **구조적으로 불가능**하다(한 경기가 한쪽
절반에만 들어가면 반대쪽에서 조회할 셀이 없다). 그 사실도 기록한다.

## 2 반복 홀드아웃

경기 분할을 50회 반복해 분포를 본다 — "+187 이 드물게 나오는 값인가"를 가른다.

## 3 경기 내 예보

같은 경기의 **앞 절반 -> 뒤 절반**. 시간 순서를 지키므로 미래 누수가 없다.
다만 2025 에서는 같은 경기의 다른 행을 못 보므로 **제출에 쓸 수 없다**(규정 4).

## 4 경기 간 지속성

경기 잔차평균의 lag-1(다음 경기) · 같은 시즌 임의쌍 · 다음 시즌 상관.

## 6 계층 분산분해

각 수준의 관측 분산에서 이항 잡음 `p(1-p)/n` 을 빼 **진짜 성분**을 남긴다.

    .\.venv\Scripts\python.exe -u exp\game_ladder.py
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

KS = [0, 150, 500, 2000, 10000]


def main():
    tr = build_df()
    season = tr["season"].to_numpy()
    y = tr["control_success"].to_numpy(np.float64)
    P = tr["pitcher_id"].to_numpy(np.int64)
    B = tr["batter_id"].to_numpy(np.int64)
    g = lambda c: tr[c].to_numpy(np.float64)
    m24 = season == 2024
    pv = (np.load(os.path.join(ROOT, "exp", "prod_champ_2024.npy"))[:3].mean(0)
          + post_for(tr, y, season < 2024, m24))
    y24, r24 = y[m24], y[m24] - pv
    base = r2(pv, y24)
    pid, bid = P[m24], B[m24]
    gid = games(pid, g("asof_pitcher_n")[m24],
                g("asof_pitcher_prev1_game_success_rate")[m24],
                g("asof_pitcher_prev1_game_middle_rate")[m24])
    bb, ss = g("balls_before").astype(int)[m24], g("strikes_before").astype(int)[m24]
    o = np.lexsort((g("asof_pitcher_n")[m24], pid))
    aid = np.zeros(len(o), np.int64)
    idx = np.zeros(len(o), np.int64)
    A, pg, pos = 0, -1, {}
    for i in o:
        if gid[i] != pg or (bb[i] == 0 and ss[i] == 0):
            A += 1
        aid[i] = A
        pos[gid[i]] = pos.get(gid[i], 0) + 1
        idx[i] = pos[gid[i]]
        pg = gid[i]
    print(f"기준선 {base:.1f}  경기 {len(np.unique(gid)):,}  타석 {len(np.unique(aid)):,}")

    def oracle(key, split, rng):
        u, inv = np.unique(split, return_inverse=True)
        # 2분할 변수(시간 등)는 그 자체를 절반으로 쓴다. 무작위로 나누면
        # 양쪽이 같은 쪽으로 몰려 빈 절반이 생긴다.
        h = (inv == 0) if len(u) == 2 else (rng.random(len(u)) < 0.5)[inv]
        if h.all() or (~h).all():
            return float("nan")
        best = -1e9
        for k in KS:
            add = np.zeros(len(y24))
            for m in (h, ~h):
                if m.sum() == 0 or (~m).sum() == 0:
                    continue
                uu, tb, _ = cells(key[m], r24[m], k)
                if len(uu) == 0:
                    continue
                add[~m] = look(uu, tb, key[~m])
            best = max(best, r2(pv + add, y24) - base)
        return best

    print("\n" + "=" * 92)
    print("1. 분할 사다리 (같은 오라클, 분할 단위만 변경)")
    print("=" * 92)
    rng = np.random.default_rng(0)
    row = np.arange(len(y24))
    # 시간 분할 — 시즌 전반/후반 경기
    gmean_month = pd.Series(g("game_month")[m24]).groupby(gid).transform("mean")
    tsplit = (gmean_month.to_numpy() >= 7).astype(int)
    KEYS = {"(투수,경기)": gid, "(투수,타자) 매치업": pid * 100000 + bid,
            "투수": pid, "(투수,경기,타석)": aid}
    SPL = {"투구 무작위": row, "타석": aid, "경기": gid, "시간(전/후반)": tsplit}
    print(f"{'키':<20}" + "".join(f"{s:>14}" for s in SPL))
    out = {}
    for kn, key in KEYS.items():
        line = f"{kn:<20}"
        for sn, sp in SPL.items():
            if sn == "타석" and kn == "(투수,경기,타석)":
                line += f"{'구조상 불가':>14}"
                continue
            if sn == "경기" and kn in ("(투수,경기)", "(투수,경기,타석)"):
                line += f"{'구조상 불가':>14}"
                continue
            v = oracle(key, sp, np.random.default_rng(1))
            out[f"{kn}|{sn}"] = v
            line += f"{v:>+14.1f}"
        print(line)
    print("\n  '구조상 불가' = 그 키의 셀이 한쪽 절반에만 들어가 조회할 것이 없다")

    print("\n" + "=" * 92)
    print("2. 경기 분할 반복 홀드아웃 (50회)")
    print("=" * 92)
    for kn, key in (("(투수,타자) 매치업", pid * 100000 + bid), ("투수", pid)):
        vals = [oracle(key, gid, np.random.default_rng(s)) for s in range(50)]
        q = np.percentile(vals, [5, 10, 25, 50, 75, 90, 95])
        out[f"rep|{kn}"] = dict(median=float(np.median(vals)),
                                p05=float(q[0]), p95=float(q[6]),
                                mn=float(min(vals)), mx=float(max(vals)))
        print(f"  {kn:<18} 중앙 {np.median(vals):>+6.1f}  5~95% "
              f"{q[0]:>+6.1f}~{q[6]:>+6.1f}  최소 {min(vals):>+6.1f} "
              f"최대 {max(vals):>+6.1f}  sd {np.std(vals):.2f}")

    print("\n" + "=" * 92)
    print("3. 경기 내 예보 — 앞 절반으로 뒤 절반을 맞히는가 (진단 전용)")
    print("=" * 92)
    n_in_game = pd.Series(idx).groupby(gid).transform("max").to_numpy()
    firsth = idx <= n_in_game / 2
    for k in (0, 50, 150, 500):
        u, tb, _ = cells(gid[firsth], r24[firsth], k)
        add = np.zeros(len(y24))
        add[~firsth] = look(u, tb, gid[~firsth])
        sel = ~firsth
        v = (1e5 * np.corrcoef(pv[sel] + add[sel], y24[sel])[0, 1] ** 2
             - 1e5 * np.corrcoef(pv[sel], y24[sel])[0, 1] ** 2)
        print(f"  k={k:<5} 뒤 절반({int(sel.sum()):,}행)에서의 이득 {v:>+7.1f}")
    print("  주의 — 2025 에서는 같은 경기의 다른 행을 못 본다 (규정 4). 제출 불가.")

    print("\n" + "=" * 92)
    print("4. 경기 효과의 지속성")
    print("=" * 92)
    gm = pd.DataFrame({"g": gid, "p": pid, "r": r24, "n": 1}).groupby("g").agg(
        p=("p", "first"), r=("r", "mean"), n=("n", "size"))
    gm = gm[gm["n"] >= 15]
    seq = []
    for p_, grp in gm.groupby("p"):
        v = grp["r"].to_numpy()
        if len(v) >= 2:
            seq.append((v[:-1], v[1:]))
    a = np.concatenate([s[0] for s in seq])
    b = np.concatenate([s[1] for s in seq])
    print(f"  같은 투수 연속 경기 (n={len(a):,})   lag-1 상관 {np.corrcoef(a,b)[0,1]:+.4f}")
    rng2 = np.random.default_rng(3)
    perm = rng2.permutation(len(b))
    print(f"  같은 투수 임의 두 경기                 상관 {np.corrcoef(a,b[perm])[0,1]:+.4f}")
    print(f"  (참고) 같은 경기 내 절반끼리          상관 +0.129~+0.249 (앞선 측정)")
    print(f"  (참고) 투수 효과의 다음 시즌 상관      +0.193 / 감쇠보정 +0.774")

    print("\n" + "=" * 92)
    print("6. 계층 분산분해 — 이항 잡음을 뺀 진짜 성분")
    print("=" * 92)
    print(f"  {'수준':<16}{'셀':>9}{'셀당':>7}{'관측 sd':>10}{'잡음 sd':>10}"
          f"{'진짜 sd':>10}{'신호비':>8}")
    for nm, key in (("투수", pid), ("(투수,경기)", gid), ("(투수,타석)", aid)):
        d = pd.DataFrame({"k": key, "y": y24}).groupby("k")["y"].agg(["mean", "size"])
        d = d[d["size"] >= 2]
        w = d["size"] / d["size"].sum()
        mu = float((w * d["mean"]).sum())
        vo = float((w * (d["mean"] - mu) ** 2).sum())
        vn = float((w * (d["mean"] * (1 - d["mean"]) / d["size"])).sum())
        vs = max(vo - vn, 0.0)
        print(f"  {nm:<16}{len(d):>9,}{d['size'].mean():>7.1f}{vo**.5:>10.4f}"
              f"{vn**.5:>10.4f}{vs**.5:>10.4f}{vs/vo if vo else 0:>8.3f}")
        out[f"var|{nm}"] = dict(obs=vo, noise=vn, sig=vs)
    json.dump(out, io.open(os.path.join(ROOT, "exp", "game_ladder.json"), "w",
                           encoding="utf-8"), indent=1, ensure_ascii=False,
              default=float)


if __name__ == "__main__":
    main()
