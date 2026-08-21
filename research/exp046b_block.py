r"""EXP046b — PM 블록 전체의 **다중상관 상한**과 기존 열과의 중복.

단일 열의 `1e5*c^2` 는 한 방향의 상한이다. 블록으로 쓰면 상한은 `1e5*R^2` 다.
그리고 기존 TrackMan 4열을 **부분화(partial out)** 한 뒤의 증분 R^2 가
진짜로 새로 들어오는 몫이다.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "exp"))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from exp046_pitchmix import pitch_events, ctx_dev, look, TYPES  # noqa: E402
import numpy as np                                              # noqa: E402


def r2(X, r):
    X = np.column_stack([np.ones(len(r)), X])
    b, *_ = np.linalg.lstsq(X, r, rcond=None)
    return 1.0 - np.sum((r - X @ b) ** 2) / np.sum((r - r.mean()) ** 2)


def main():
    ev = pitch_events()
    from path_alloc import build_df
    tr = build_df()
    y = tr["control_success"].to_numpy(float)
    season = tr["season"].to_numpy()
    P = tr["pitcher_id"].to_numpy(np.int64)
    BB = tr["balls_before"].to_numpy(np.int64)
    SS = tr["strikes_before"].to_numpy(np.int64)
    SH = (tr["pitcher_hand"].to_numpy() == tr["batter_hand"].to_numpy()).astype(np.int64)
    OB = (tr["num_runners_on"].to_numpy(np.int64) > 0).astype(np.int64)
    CTX = {"cnt": BB * 4 + SS, "sh": SH, "two": (SS == 2).astype(np.int64), "onb": OB}
    TMC = ["tmc_fastball_dev", "tmc_breaking_dev", "tmc_offspeed_dev", "tmc_speed_dev",
           "tmh_fastball_dev", "tmh_breaking_dev", "tmh_offspeed_dev", "tmh_speed_dev"]

    for fold in (2022, 2023, 2024):
        src = (season < fold) & (season >= fold - 2)
        m = season == fold
        n = int(m.sum())
        res = y[m] - np.load(os.path.join(ROOT, "exp", f"prod_champ_{fold}.npy")).mean(0)

        cols, names = [], []
        for cname, ctx in CTX.items():
            for t in TYPES:
                ok = ~np.isnan(ev[t])
                tbl, _ = ctx_dev(P, ctx, np.nan_to_num(ev[t]), src & ok, 300)
                cols.append(look(tbl, None, P[m], ctx[m])); names.append(f"pm_{cname}_{t}")
            tot = np.zeros(n)
            for t in TYPES:
                ok = ~np.isnan(ev[t])
                tb, _ = ctx_dev(P, ctx, np.nan_to_num(ev[t]), src & ok, 300)
                st, _ = ctx_dev(P, np.nan_to_num(ev[t]).astype(np.int64), y, src & ok, 300)
                tot += look(tb, None, P[m], ctx[m]) * look(st, None, P[m], np.ones(n, np.int64))
            cols.append(tot); names.append(f"pm_xsucc_{cname}")
        PM = np.column_stack(cols)
        TM = np.nan_to_num(np.column_stack([tr[c].to_numpy(float)[m] for c in TMC]))

        rp, rt, rb = r2(PM, res), r2(TM, res), r2(np.column_stack([PM, TM]), res)
        print(f"\n=== 폴드 {fold}  n={n:,} ===")
        print(f"  PM 16열 단독            R^2 {rp:.6f}   상한 {1e5*rp:8.2f} 점")
        print(f"  기존 TrackMan 8열 단독   R^2 {rt:.6f}   상한 {1e5*rt:8.2f} 점")
        print(f"  둘 다                   R^2 {rb:.6f}   상한 {1e5*rb:8.2f} 점")
        print(f"  -> PM 의 **증분** (TM 부분화 후)        {1e5*(rb-rt):8.2f} 점")
        print(f"  -> TM 의 **증분** (PM 부분화 후)        {1e5*(rb-rp):8.2f} 점")
        xs = [i for i, nm in enumerate(names) if nm.startswith("pm_xsucc")]
        print(f"  pm_xsucc 4열만          R^2 {r2(PM[:, xs], res):.6f}   "
              f"상한 {1e5*r2(PM[:, xs], res):8.2f} 점")
        rr = np.corrcoef(np.column_stack([PM, TM]).T)[:16, 16:]
        print(f"  PM x TrackMan 최대 |r| = {np.abs(rr).max():.3f}")
        print("  " + "  ".join(f"{nm}:{np.corrcoef(PM[:,i],res)[0,1]:+.4f}"
                               for i, nm in enumerate(names) if i in xs))


if __name__ == "__main__":
    main()
