r"""차등 3축의 **축소 상수 k** 를 바꿔 제출 후보를 만든다. 학습 0회 · Champion 불변.

## w 와 k 는 방향이 다르다

    w   모든 투수의 보정을 **균일하게** 줄인다
    k   표본이 적은 투수의 보정을 **선택적으로** 더 줄인다  (n_eff/(n_eff+k))

LB 측정으로 `b=0.650` 이 나왔다 — 보정이 2025 에서 65% 강도로만 듣는다. 그
감쇠가 **모든 투수에 균일한지, 표본 적은 투수에 몰려 있는지**는 w 로는 알 수
없다. k 를 바꿔야 갈린다. 후자라면 k 를 키우는 쪽이 w 를 줄이는 것보다 낫다.

`CLAUDE.md` 5-b 의 관찰과도 맞물린다 — 전이 최적 k 는 적률법(EB) 값의 10~100배다.
현행 1000/1000/2000 은 과거 폴드 전이로 골랐고 **LB 로는 한 번도 재지 않았다.**

## 사용법

    python exp\build_k.py --name cand_k25 --kmul 2.5 --w 0.65
    python exp\build_k.py --name cand_k05 --kmul 0.5 --w 0.65

표는 `cat_asof_xl.pkl`(Champion 모델) 위에 다시 얹는다. 원천은 **직전 두 시즌
(2023·2024)의 strictly out-of-fold 잔차**로 고정 — 30회차에서 5시즌이 LB −9.95 였다.
"""
import argparse
import hashlib
import os
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "exp"))
from path_alloc import build_df                             # noqa: E402
from resid_table import post_for                            # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

K0 = dict(hand=1000.0, two=1000.0, run=2000.0)      # 현행
SRC = (2023, 2024)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--kmul", type=float, default=1.0, help="세 축 k 공통 배수")
    ap.add_argument("--khand", type=float, default=None)
    ap.add_argument("--ktwo", type=float, default=None)
    ap.add_argument("--krun", type=float, default=None)
    ap.add_argument("--w", type=float, default=0.65, help="차등 3축 전역 가중")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    K = {k: (v * a.kmul) for k, v in K0.items()}
    if a.khand is not None:
        K["hand"] = a.khand
    if a.ktwo is not None:
        K["two"] = a.ktwo
    if a.krun is not None:
        K["run"] = a.krun
    print(f"  k = hand {K['hand']:.0f}  2S {K['two']:.0f}  runner {K['run']:.0f}"
          f"   (현행 1000/1000/2000)   w = {a.w:g}")

    tr = build_df()
    season = tr["season"].to_numpy()
    y = tr["control_success"].to_numpy(np.float64)
    PID = tr["pitcher_id"].to_numpy(np.int64)
    PH = tr["pitcher_hand"].to_numpy(np.int64)
    BH = tr["batter_hand"].to_numpy(np.int64)
    SS = tr["strikes_before"].to_numpy(np.int64)
    NR = tr["num_runners_on"].to_numpy(np.int64)

    res = {}
    for f in SRC:
        m = season == f
        res[f] = y[m] - (np.load(os.path.join(ROOT, "exp", f"prod_champ_{f}.npy"))[:2].mean(0)
                         + post_for(tr, y, season < f, m))
    msrc = np.isin(season, SRC)
    rsrc = np.concatenate([res[f] for f in SRC])

    def diff(ctx, k):
        gg = pd.DataFrame({"p": PID[msrc], "c": ctx[msrc], "r": rsrc}).groupby(
            ["p", "c"])["r"].agg(["mean", "size"]).unstack()
        n0, n1 = gg[("size", 0)].fillna(0), gg[("size", 1)].fillna(0)
        ne = (n0 * n1) / (n0 + n1).replace(0, np.nan)
        return ((gg[("mean", 1)] - gg[("mean", 0)]) * ne / (ne + k)).dropna()

    hand_d = diff((PH == BH).astype(int), K["hand"])
    two_d = diff((SS == 2).astype(int), K["two"])
    run_d = diff((NR > 0).astype(int), K["run"])
    print(f"  |d| 중앙  손 {hand_d.abs().median():.5f}  "
          f"2S {two_d.abs().median():.5f}  주자 {run_d.abs().median():.5f}")

    tab_hand = {(int(p), ph, bh): float(0.5 * d if bh == ph else -0.5 * d)
                for p, d in hand_d.items() for ph in (1, 2) for bh in (1, 2)}
    tab_two = {(int(p), s): float(0.5 * d if s == 2 else -0.5 * d)
               for p, d in two_d.items() for s in (0, 1, 2)}
    tab_run = {(int(p), n): float(0.5 * d if n > 0 else -0.5 * d)
               for p, d in run_d.items() for n in range(4)}

    base = joblib.load(os.path.join(ROOT, "model_cand", "cat_asof_xl.pkl"))
    specs = list(base["platoon"]) + [
        {"w": a.w, "cols": ["pitcher_id", "pitcher_hand", "batter_hand"],
         "table": tab_hand, "note": f"손 차등 k={K['hand']:.0f}"},
        {"w": a.w, "cols": ["pitcher_id", "strikes_before"],
         "table": tab_two, "note": f"2S 차등 k={K['two']:.0f}"},
        {"w": a.w, "cols": ["pitcher_id", "num_runners_on"],
         "table": tab_run, "note": f"주자 차등 k={K['run']:.0f}"}]
    b = dict(base)
    b["platoon"] = specs
    b["note"] = (str(base.get("note", ""))
                 + f" | K: {K['hand']:.0f}/{K['two']:.0f}/{K['run']:.0f} w={a.w:g}")

    zp = os.path.join(ROOT, "submissions", f"{a.name}.zip")
    if os.path.exists(zp) and not a.force:
        raise SystemExit(f"이미 있다 — 덮어쓰지 않는다: {zp}")
    pkl = os.path.join(ROOT, "model_cand", f"{a.name}.pkl")
    joblib.dump(b, pkl, compress=3)
    r = subprocess.run(
        [sys.executable, os.path.join(ROOT, "make_submit.py"),
         "--model", os.path.relpath(pkl, ROOT),
         "--requirements", "requirements_cat.txt",
         "--out", os.path.relpath(zp, ROOT)],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if not os.path.exists(zp):
        raise SystemExit("빌드 실패\n" + r.stdout[-400:] + r.stderr[-800:])
    print(f"  {a.name}.zip  {os.path.getsize(zp)/1e6:.2f} MB")
    print(f"  sha256 {hashlib.sha256(open(zp,'rb').read()).hexdigest()}")


if __name__ == "__main__":
    main()
