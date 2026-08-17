r"""EXP016 — 지속성 전수 측정. 이득이 아니라 d_p 의 시즌 간 상관을 잰다.

## 왜 다시 훑는가 (반복이 아니다)

EXP003/EXP014 는 246개 국면을 **전이 이득**으로 쟀다. 그건 한 개의 요약수이고
폴드 2024 의 시드 잡음이 ±7.2 라, +2~5 짜리 진짜 신호는 **원리적으로 검출
불가능**하다. 실제로 포트폴리오는 −16.9 로 무너졌다.

살아남은 3축(손·2S·주자)의 공통점은 크기가 아니라 **지속성**이었다.
그렇다면 지속성을 직접 재야 한다.

    d_p(s)      시즌 s 에서 추정한 그 투수의 차등
    지속성      corr_p( d_p(s) , d_p(s+1) )     투수 ~400명에 걸친 상관

자유도가 400 이라 검정력이 두 자릿수 높다. 이득으로는 안 보이던 지속 신호가
여기서는 보일 수 있다.

## 잡음 감쇠를 어떻게 다루는가

`d_p` 는 추정오차가 있어 상관이 **감쇠**한다. 셀이 작을수록 더 감쇠하므로
국면끼리 원값을 그대로 비교하면 불공평하다. 두 가지로 대응한다.

    1  n_eff 가중 상관을 쓴다 (정보량이 큰 투수에 무게)
    2  **반분 신뢰도로 감쇠를 보정한다** — 같은 시즌을 경기 단위로 반 갈라
       두 반쪽의 d_p 상관 r_hh 를 구하면, 그 시즌 추정의 신뢰도가 나온다.
       보정 지속성 = r_cross / sqrt(rel(s) * rel(s+1))

이러면 "작은 셀이라 상관이 낮게 나온 것"과 "진짜 안 지속되는 것"이 갈린다.

## 귀무 분포

투수 라벨을 시즌 안에서 섞으면 지속성은 0 이어야 한다. 그것으로 z 를 만든다.

    .\.venv\Scripts\python.exe -u research\exp016_persist.py
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
from game_decomp import games                               # noqa: E402
from exp003_sweep import build_contexts                     # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

EXP = "EXP016"
SEASONS = (2020, 2021, 2022, 2023, 2024)
PAIRS = [(2020, 2021), (2021, 2022), (2022, 2023), (2023, 2024)]
MIN_NE = 20.0            # 이 정보량 미만 투수는 상관 계산에서 뺀다


def wcorr(a, b, w):
    """가중 상관."""
    w = w / w.sum()
    ma, mb = (w * a).sum(), (w * b).sum()
    va = (w * (a - ma) ** 2).sum()
    vb = (w * (b - mb) ** 2).sum()
    if va <= 0 or vb <= 0:
        return np.nan
    return float((w * (a - ma) * (b - mb)).sum() / np.sqrt(va * vb))


def main():
    E.start_experiment(EXP, "L0-Q", "python research/exp016_persist.py", "load")
    tr = build_df()
    season = tr["season"].to_numpy()
    y = tr["control_success"].to_numpy(np.float64)
    P = tr["pitcher_id"].to_numpy(np.int64)

    def g(c):
        return tr[c].to_numpy(np.float64)

    res0, halves = {}, {}
    rng = np.random.default_rng(0)
    for f in SEASONS:
        m = season == f
        res0[f] = y[m] - (np.load(os.path.join(ROOT, "exp", f"prod_champ_{f}.npy"))[:2].mean(0)
                          + post_for(tr, y, season < f, m))
        gid = games(P[m], g("asof_pitcher_n")[m],
                    g("asof_pitcher_prev1_game_success_rate")[m],
                    g("asof_pitcher_prev1_game_middle_rate")[m])
        u, inv = np.unique(gid, return_inverse=True)
        halves[f] = (rng.random(len(u)) < 0.5)[inv]
    E.beat("잔차·경기분할 준비")

    def dser(ctx, f, sub=None, shuffle=False):
        """시즌 f 의 (d_p, n_eff). sub 는 행 부분집합(반분용)."""
        m = season == f
        p, c, r = P[m], ctx[m], res0[f]
        if sub is not None:
            p, c, r = p[sub], c[sub], r[sub]
        if shuffle:
            p = np.random.default_rng(abs(hash((f, len(p)))) % 2**31).permutation(p)
        ok = np.isin(c, (0, 1))
        gg = pd.DataFrame({"p": p[ok], "c": c[ok], "r": r[ok]}).groupby(
            ["p", "c"])["r"].agg(["mean", "size"]).unstack()
        if ("size", 0) not in gg or ("size", 1) not in gg:
            return None
        n0, n1 = gg[("size", 0)].fillna(0), gg[("size", 1)].fillna(0)
        ne = (n0 * n1) / (n0 + n1).replace(0, np.nan)
        return pd.DataFrame({"d": gg[("mean", 1)] - gg[("mean", 0)], "ne": ne}).dropna()

    def persistence(ctx, shuffle=False):
        """인접 시즌 상관(평균), 반분 신뢰도, 감쇠보정 지속성."""
        cross, rel, npair = [], [], []
        for a, b in PAIRS:
            da, db = dser(ctx, a, shuffle=shuffle), dser(ctx, b, shuffle=shuffle)
            if da is None or db is None:
                continue
            j = da.join(db, lsuffix="a", rsuffix="b", how="inner")
            j = j[(j["nea"] >= MIN_NE) & (j["neb"] >= MIN_NE)]
            if len(j) < 40:
                continue
            w = (j["nea"] * j["neb"]) / (j["nea"] + j["neb"])
            cross.append(wcorr(j["da"].to_numpy(), j["db"].to_numpy(), w.to_numpy()))
            npair.append(len(j))
        for f in SEASONS:
            h = halves[f]
            d1, d2 = dser(ctx, f, sub=h, shuffle=shuffle), dser(ctx, f, sub=~h,
                                                               shuffle=shuffle)
            if d1 is None or d2 is None:
                continue
            j = d1.join(d2, lsuffix="1", rsuffix="2", how="inner")
            j = j[(j["ne1"] >= MIN_NE / 2) & (j["ne2"] >= MIN_NE / 2)]
            if len(j) < 40:
                continue
            w = (j["ne1"] * j["ne2"]) / (j["ne1"] + j["ne2"])
            rel.append(wcorr(j["d1"].to_numpy(), j["d2"].to_numpy(), w.to_numpy()))
        if not cross or not rel:
            return None
        cr, rl = float(np.nanmean(cross)), float(np.nanmean(rel))
        n = int(np.mean(npair))
        adj = cr / np.sqrt(rl) if rl > 0.02 else np.nan
        return dict(cross=cr, rel=rl, adj=adj, n=n, z=cr * np.sqrt(max(n - 3, 1)))

    CTX = build_contexts(tr, season)
    CTX["[대조] 같은손"] = (g("pitcher_hand") == g("batter_hand")).astype(int)
    CTX["[대조] 2스트라이크"] = (g("strikes_before").astype(int) == 2).astype(int)
    CTX["[대조] 주자있음"] = (g("num_runners_on") > 0).astype(int)
    print(f"국면 {len(CTX)}개 · 인접쌍 {len(PAIRS)} · 반분 신뢰도 {len(SEASONS)}시즌\n")

    rows = []
    for i, (nm, ctx) in enumerate(sorted(CTX.items())):
        if i % 20 == 0:
            E.beat(f"persist {i}/{len(CTX)}")
        r = persistence(ctx)
        if r:
            rows.append((nm, r))
    print(f"측정 완료 {len(rows)}개\n")

    # 귀무 — 투수 라벨 섞기 (대조 3축으로 확인)
    null = []
    for nm in ("[대조] 같은손", "[대조] 2스트라이크", "[대조] 주자있음"):
        r = persistence(CTX[nm], shuffle=True)
        if r:
            null.append(r["cross"])
    print(f"귀무(투수 라벨 섞기) 인접 상관 = {np.mean(null):+.4f}  "
          f"(0 이어야 정상)\n")

    rows.sort(key=lambda t: -t[1]["z"])
    print("=" * 96)
    print(f"{'국면 (지속성 상위 25)':<44}{'인접상관':>10}{'반분신뢰':>10}"
          f"{'감쇠보정':>10}{'투수n':>8}{'z':>8}")
    print("=" * 96)
    for nm, r in rows[:25]:
        mark = "  ◀ 채택축" if nm.startswith("[대조]") else ""
        print(f"{nm:<44}{r['cross']:>+10.4f}{r['rel']:>10.4f}"
              f"{r['adj'] if np.isfinite(r['adj']) else float('nan'):>10.3f}"
              f"{r['n']:>8}{r['z']:>8.2f}{mark}")

    ctrl = {nm: r for nm, r in rows if nm.startswith("[대조]")}
    print("\n채택된 3축의 순위")
    for nm, r in ctrl.items():
        rank = [i for i, (n2, _) in enumerate(rows) if n2 == nm][0] + 1
        print(f"  {nm:<24} {rank:>3}위 / {len(rows)}   z={r['z']:.2f}"
              f"   인접상관 {r['cross']:+.4f}")

    zc = min(r["z"] for r in ctrl.values())
    cand = [(nm, r) for nm, r in rows if not nm.startswith("[대조]") and r["z"] >= zc]
    print(f"\n채택축 최저 z = {zc:.2f} 이상인 미채택 국면 {len(cand)}개")
    for nm, r in cand[:15]:
        print(f"    {nm:<44} z={r['z']:>6.2f}  인접상관 {r['cross']:+.4f}")

    dec = "PROMISING" if cand else "REJECTED"
    E.set_hypothesis_status("L0-Q", "PROMISING" if cand else "CLOSED", level=0,
                            hypothesis="지속성 전수 측정", result=len(cand))
    E.finish_experiment(dict(
        experiment_id=EXP, hypothesis_id="L0-Q", type="A", level=0,
        started_at=E.read(E.CKPT)["start_time"],
        n_contexts=len(rows), null_cross=float(np.mean(null)),
        control={k: {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                     for kk, vv in v.items()} for k, v in ctrl.items()},
        n_candidates=len(cand),
        candidates=[(nm, round(r["z"], 2), round(r["cross"], 4)) for nm, r in cand[:20]],
        decision=dec, artifact=None,
        what_we_learned=(f"이득 대신 d_p 시즌간 상관으로 {len(rows)}개를 재측정. "
                         f"채택 3축 최저 z={zc:.2f}, 그 이상인 미채택 {len(cand)}개")))
    json.dump({"rows": [(nm, r) for nm, r in rows[:60]], "null": float(np.mean(null))},
              open(os.path.join(ROOT, "exp", "exp016_persist.json"), "w"),
              indent=1, default=float)
    print(f"\nDECISION = {dec}")


if __name__ == "__main__":
    main()
