r"""HAND-DIFFERENTIAL 정밀 검증 — 새 축인가, 현행 축의 k 문제인가. 학습 0회.

## 구조적 사실부터

현행 후처리 1번축은 `nested_dev(부모=투수, 자식=투수x손)` 이고 정의상

    dev = cnt * (셀평균 - 투수평균) / (cnt + k)

즉 **이미 손 차등을 축소한 값**이다 (k=300, w=0.20). 내가 만든 hand_diff 와
같은 구조이므로, 먼저 **중복 검사**로 그것을 확인하고 그렇다면 문제는
"새 축"이 아니라 **"현행 k 가 덜 축소돼 있는가"** 로 환원된다.

후처리는 재학습이 필요 없다 — 모델 예측은 고정하고 축만 바꿔 전부 잰다.

## 규약

    k 선택은 **과거 전이에서만**. 2024 를 보고 고르지 않는다.
    폴드별 생산 경로 예측(prod_champ_*.npy, 2시드)을 그대로 쓴다.

    .\.venv\Scripts\python.exe -u exp\hand_diff.py
"""
import io
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "exp"))
import build_asof as ba                                    # noqa: E402
from path_alloc import build_df                            # noqa: E402
from traj_probe import r2                                  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

KGRID = [300, 500, 1000, 1500, 2000, 3000, 5000, 10000, 20000]
FOLDS = (2021, 2022, 2023, 2024)


def main():
    tr = build_df()
    season = tr["season"].to_numpy()
    y = tr["control_success"].to_numpy(np.float64)
    P = tr["pitcher_id"].to_numpy(np.int64)
    BH = tr["batter_hand"].to_numpy(np.int64)
    BB = tr["balls_before"].to_numpy(np.int64)
    SS = tr["strikes_before"].to_numpy(np.int64)
    OB = (tr["num_runners_on"].to_numpy(np.int64) > 0).astype(np.int64)
    PH = P * 10 + BH
    PHA = PH * 10 + (SS > BB).astype(np.int64)
    AX = [(P, PH), (PH, PHA), (PHA, PHA * 100 + (BB * 4 + SS)), (PH, PH * 10 + OB)]
    SAME = (tr["pitcher_hand"].to_numpy(np.int64) == BH).astype(np.int64)

    def axes(m_tr, m_va, k1=None):
        """축 4개를 따로 돌려준다. k1 을 주면 1번축 k 만 바꾼다."""
        ks = list(ba.KSH)
        if k1 is not None:
            ks[0] = k1
        return [ba.look(*ba.nested_dev(p[m_tr], c[m_tr], y[m_tr], k), c[m_va])
                for (p, c), k in zip(AX, ks)]

    def hand_diff(m_tr, m_va, k):
        """차등 추정 — 같은손/반대손 잔차평균 차이를 유효표본으로 축소."""
        t = pd.DataFrame({"p": P[m_tr], "s": SAME[m_tr], "y": y[m_tr]})
        gg = t.groupby(["p", "s"])["y"].agg(["mean", "size"]).unstack()
        n0 = gg[("size", 0)].fillna(0)
        n1 = gg[("size", 1)].fillna(0)
        d = (gg[("mean", 1)] - gg[("mean", 0)])
        ne = (n0 * n1) / (n0 + n1).replace(0, np.nan)
        ds = (d * ne / (ne + k)).dropna()
        h = np.where(SAME[m_va] == 1, 0.5, -0.5)
        return pd.Series(P[m_va]).map(ds).fillna(0.0).to_numpy() * h

    MODEL = {}
    for f in FOLDS:
        m = season == f
        MODEL[f] = np.load(os.path.join(ROOT, "exp", f"prod_champ_{f}.npy")
                           )[:2].mean(0)

    print("=" * 92)
    print("STEP 3 먼저 — 중복 검사 (현행 1번축 vs 차등 추정)")
    print("=" * 92)
    m_tr, m_va = season < 2024, season == 2024
    A4 = axes(m_tr, m_va)
    hd = hand_diff(m_tr, m_va, 2000)
    base_pred = MODEL[2024] + np.column_stack(A4) @ ba.WPOST
    base = r2(base_pred, y[m_va])
    print(f"  corr(dev_platoon, hand_diff)          "
          f"{np.corrcoef(A4[0], hd)[0,1]:+.4f}")
    print(f"  corr(w1*dev_platoon, hand_diff)       "
          f"{np.corrcoef(ba.WPOST[0]*A4[0], hd)[0,1]:+.4f}")
    print(f"  corr(전체 후처리 합, hand_diff)         "
          f"{np.corrcoef(np.column_stack(A4) @ ba.WPOST, hd)[0,1]:+.4f}")
    res_after = y[m_va] - base_pred
    print(f"  corr(현행 잔차, hand_diff)             "
          f"{np.corrcoef(res_after, hd)[0,1]:+.4f}")
    print(f"  두 벡터의 sd: dev_platoon {A4[0].std():.5f} / hand_diff {hd.std():.5f}")
    print("\n  -> 상관이 매우 높으면 같은 정보이고, 문제는 k 하나로 환원된다")

    print("\n" + "=" * 92)
    print("STEP 1+2 — 1번축 k 만 바꾼다 (w=0.20 고정, 나머지 3축 고정)")
    print("=" * 92)
    out = {"overlap": float(np.corrcoef(A4[0], hd)[0, 1])}
    print(f"{'k':>7}" + "".join(f"{f:>10}" for f in FOLDS)
          + f"{'평균':>9}{'최악':>9}{'부호':>7}")
    rows = {}
    for k in KGRID:
        gains = []
        for f in FOLDS:
            mt, mv = season < f, season == f
            a0 = axes(mt, mv)
            ak = axes(mt, mv, k1=k)
            b0 = r2(MODEL[f] + np.column_stack(a0) @ ba.WPOST, y[mv])
            bk = r2(MODEL[f] + np.column_stack(ak) @ ba.WPOST, y[mv])
            gains.append(bk - b0)
        rows[k] = gains
        sgn = sum(1 for v in gains if v > 0)
        print(f"{k:>7}" + "".join(f"{v:>+10.1f}" for v in gains)
              + f"{np.mean(gains):>+9.1f}{min(gains):>+9.1f}{sgn:>5}/4")
    out["k_curve"] = {str(k): rows[k] for k in KGRID}

    past = {k: np.mean(rows[k][:3]) for k in KGRID}       # 2021~2023 만으로 선택
    kbest = max(past, key=past.get)
    print(f"\n  과거 3폴드(2021~2023) 평균으로 고른 k = {kbest}"
          f"   -> 2024 실측 {rows[kbest][3]:+.1f}"
          f" ({rows[kbest][3]/943.8*100:+.2f}%)")
    out["k_best_past"] = kbest
    out["gain_2024_at_kbest"] = rows[kbest][3]

    print("\n" + "=" * 92)
    print("STEP 2 — 구성 비교 (2024, 생산 경로)")
    print("=" * 92)
    cfg = {
        "A 현행 4축 (k1=300)": np.column_stack(A4) @ ba.WPOST,
        f"D k1 만 {kbest} 로 교체": np.column_stack(axes(m_tr, m_va, k1=kbest)) @ ba.WPOST,
        "B 1번축 제거 + 차등(k=2000)": (np.column_stack(A4)[:, 1:] @ ba.WPOST[1:]) + hd,
        "C 4축 유지 + 차등 추가": (np.column_stack(A4) @ ba.WPOST) + hd,
        "  참고: 1번축 제거만": np.column_stack(A4)[:, 1:] @ ba.WPOST[1:],
    }
    print(f"{'구성':<30}{'2024 rho^2':>12}{'vs 현행':>10}")
    for n, add in cfg.items():
        v = r2(MODEL[2024] + add, y[m_va])
        print(f"  {n:<28}{v:>12.1f}{v-base:>+10.1f}")
        out[f"cfg|{n}"] = v - base

    print("\n" + "=" * 92)
    print("STEP 5 — 손 차등의 해석 (조건별 안정성)")
    print("=" * 92)
    t = pd.DataFrame({"p": P[m_va], "s": SAME[m_va], "y": y[m_va],
                      "gt": tr["game_type"].to_numpy()[m_va],
                      "cs": tr["cur_succ"].to_numpy(np.float64)[m_va]})
    ov = t.groupby("s")["y"].mean()
    print(f"  전체: 같은손 {ov.get(1, np.nan):.4f} vs 반대손 {ov.get(0, np.nan):.4f}"
          f"  차이 {ov.get(1,0)-ov.get(0,0):+.4f}")
    for name, sub in (("R 경기", t[t.gt == "R"]), ("F 경기", t[t.gt == "F"]),
                      ("cur_succ 상위", t[t.cs > t.cs.median()]),
                      ("cur_succ 하위", t[t.cs <= t.cs.median()])):
        o = sub.groupby("s")["y"].mean()
        print(f"  {name:<14} 차이 {o.get(1,np.nan)-o.get(0,np.nan):+.4f}"
              f"   (n={len(sub):,})")
    dp = t[t.s == 1].groupby("p")["y"].mean() - t[t.s == 0].groupby("p")["y"].mean()
    print(f"  투수별 차등의 sd {dp.std():.4f}  (전체 평균 차이 "
          f"{ov.get(1,0)-ov.get(0,0):+.4f})  -> 투수마다 다른가: "
          f"{'예' if dp.std() > abs(ov.get(1,0)-ov.get(0,0)) else '아니오'}")

    json.dump(out, io.open(os.path.join(ROOT, "exp", "hand_diff.json"), "w",
                           encoding="utf-8"), indent=1, ensure_ascii=False,
              default=float)


if __name__ == "__main__":
    main()
