r"""신규 기전 검증 — 2진 편차를 num_runners_on 0/1/2/3 으로 펼친 조회가
실험 경로와 **같은 값**을 내는가. 그리고 커버리지.

16회차에서 했던 "추론 경로 == 실험 경로 최대차 0.000e+00" 검사의 신규 축 버전이다.
script.py 의 platoon_adjust 를 그대로 불러 쓴다 (재구현하지 않는다).

    .\.venv\Scripts\python.exe exp\verify_runner.py
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from script import platoon_adjust          # noqa: E402  추론 경로 그대로

CACHE = os.path.join(ROOT, "exp", "cache")
PKL = os.path.join(ROOT, "model_cand", "cat_plat_cnt_runner.pkl")
KSH1, KSH2, KSH3 = 300, 800, 2000


def nested_dev(parent, child, y, k):
    o = np.argsort(child, kind="stable")
    Ys, Ps, Cs = y[o], parent[o], child[o]
    u, s = np.unique(Cs, return_index=True)
    cnt = np.diff(np.append(s, len(Cs)))
    cell = np.add.reduceat(Ys, s) / cnt
    par = Ps[s]
    op = np.argsort(parent, kind="stable")
    Yp, Pp = y[op], parent[op]
    pu, ps = np.unique(Pp, return_index=True)
    pc = np.diff(np.append(ps, len(Pp)))
    pmean = np.add.reduceat(Yp, ps) / pc
    return u, cnt * (cell - pmean[np.searchsorted(pu, par)]) / (cnt + k)


def lookup(u, dev, keys):
    out = np.zeros(len(keys), dtype=np.float64)
    hit = np.zeros(len(keys), dtype=bool)
    ix = np.clip(np.searchsorted(u, keys), 0, len(u) - 1)
    ok = u[ix] == keys
    out[ok], hit[ok] = dev[ix[ok]], True
    return out, hit


def main():
    meta = json.load(open(f"{CACHE}/cols.json"))
    ixc = {c: i for i, c in enumerate(meta["cols"])}
    X = np.load(f"{CACHE}/X.npy", mmap_mode="r")
    y = np.load(f"{CACHE}/y.npy").astype(np.float64)
    season = np.load(f"{CACHE}/season.npy")

    def col(n):
        return np.asarray(X[:, ixc[n]], dtype=np.float64)

    P = col("pitcher_id").astype(np.int64)
    BH = col("batter_hand").astype(np.int64)
    BB, SS = col("balls_before").astype(np.int64), col("strikes_before").astype(np.int64)
    NR = col("num_runners_on").astype(np.int64)
    OB = (NR > 0).astype(np.int64)
    PH = P * 10 + BH
    PHC, PHO = PH * 100 + (BB * 4 + SS), PH * 10 + OB

    # 실험 경로 — 제출본과 같이 학습 전체로 표를 만든다
    u1, t1 = nested_dev(P, PH, y, KSH1)
    u2, t2 = nested_dev(PH, PHC, y, KSH2)
    u3, t3 = nested_dev(PH, PHO, y, KSH3)

    va = season == 2024                      # 평가셋 대역
    e1, h1 = lookup(u1, t1, PH[va])
    e2, h2 = lookup(u2, t2, PHC[va])
    e3, h3 = lookup(u3, t3, PHO[va])

    b = joblib.load(PKL)
    W = [float(s["w"]) for s in b["platoon"]]
    print(f"pkl 가중  w1={W[0]}  w2={W[1]:.4f}  w3={W[2]}")
    print(f"alpha={b['alpha']:.6f}  center={b['center']:.6f}")

    exp_total = W[0] * e1 + W[1] * e2 + W[2] * e3

    # 추론 경로 — script.py 의 함수를 그대로 호출
    Xdf = pd.DataFrame({"pitcher_id": P[va], "batter_hand": BH[va],
                        "balls_before": BB[va], "strikes_before": SS[va],
                        "num_runners_on": NR[va]})
    inf_total = platoon_adjust(b, Xdf)

    d = np.abs(np.asarray(inf_total) - exp_total)
    print(f"\n=== 추론 경로 vs 실험 경로 ({va.sum():,}행) ===")
    print(f"  최대차 {d.max():.3e}   평균차 {d.mean():.3e}   "
          f"{'통과' if d.max() < 1e-12 else '불일치'}")

    print(f"\n=== 커버리지 (2024 대역) ===")
    for nm, h in [("플래툰", h1), ("플래툰x카운트", h2), ("플래툰x주자유무", h3)]:
        print(f"  {nm:<16} {100 * h.mean():5.1f}%")

    print(f"\n=== 주자유무 편차 분포 ===")
    print(f"  표 {len(u3):,}칸  값 {t3.min():+.5f}~{t3.max():+.5f}  "
          f"|평균| {np.abs(t3).mean():.5f}")
    ob_va = OB[va]
    print(f"  적용값  주자없음 평균 {e3[ob_va == 0].mean():+.6f}  "
          f"주자있음 평균 {e3[ob_va == 1].mean():+.6f}")
    print(f"  펼침 검사 — 주자 1/2/3 명의 적용값이 모두 같은가: ", end="")
    same = True
    for pid, hand in [(int(k // 100), int((k // 10) % 10)) for k in u3[:200]]:
        sel = (P[va] == pid) & (BH[va] == hand) & (ob_va == 1)
        if sel.sum() > 1 and len(np.unique(e3[sel])) != 1:
            same = False
            break
    print("통과" if same else "불일치")


if __name__ == "__main__":
    main()
