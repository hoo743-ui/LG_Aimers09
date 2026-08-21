r"""EXP049 — 🚩 **인수분해 타깃(FACTORED TARGET)**. 라벨의 버려진 정보를 되찾는다.

## 발견 (EXP046 의 복원으로 처음 보였다)

    middle  = 1  ->  y = 0   n=220,460   평균 정확히 0.0000
    reverse = 1  ->  y = 0   n=337,496   평균 정확히 0.0000

    y = (1 - H) * S     H = middle OR reverse   기저율 0.345  (결정론적 0)
                        S = 조건부 성공          기저율 0.799
        S 안에서도 ball=1 이면 0.591, 아니면 0.93~0.96

즉 `control_success` 는 합성 사건이고, 지금 모델은 **12칸 라벨을 이진으로 뭉갠
것**을 학습한다. y=0 인 두 행이 "가운데로 몰렸다"인지 "볼이 됐다"인지 구별하지
못한다. 타깃 rho 천장이 0.10 인 문제에서 이 정보 손실은 크다.

## 판본

    base   현행 — y 에 직접 이진 적합
    ft2    (1 - Â) * B̂        A: H 이진,  B: H=0 행만으로 조건부 성공 이진
    ft12   Σ_c P̂(c|row) q_c   c = (ball,strike,middle,reverse) 12칸 다항, q_c 는 학습 상수

## ⚠️ TYPE 판정 — 이것은 **B (모델 적합 방식)** 다

새 정보축이 아니다. 같은 피처로 같은 타깃을 **다르게 추정**한다. 전이 위계
(§8)에서 모델 적합은 **금지** 등급이고 단조 제약이 −65.73 을 냈다. 그러므로
로컬이 크게 나와도 그것만으로 제출하지 않는다. §5-a 의 다섯 조건을 전부 넘어야
하고, 특히 **in-sample 이 함께 오르면 경고 신호**다 (단조 제약의 기제).

로컬 이득이 크면 그때 그 다섯 조건을 설계한다. 지금은 크기부터 잰다.

    .\.venv\Scripts\python.exe -u research\exp049_factored.py --folds 2024 --seeds 42
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "exp"))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import build_asof as ba                                   # noqa: E402
from exp046_pitchmix import pitch_events                  # noqa: E402


def cells(ev):
    """(ball, strike, middle, reverse) -> 0..11. 결측은 -1."""
    b, s = ev["ball"], ev["strike"]
    mi, rv = ev["middle"], ev["reverse"]
    ok = ~np.isnan(b)
    c = np.full(len(b), -1, np.int64)
    bs = np.where(np.nan_to_num(b) > .5, 2, np.where(np.nan_to_num(s) > .5, 1, 0))
    c[ok] = (bs * 4 + np.nan_to_num(mi) * 2 + np.nan_to_num(rv))[ok].astype(np.int64)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="2024")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--variants", default="aux,ft2,ftq")
    ap.add_argument("--out", default=os.path.join(ROOT, "exp", "exp049_ft.json"))
    a = ap.parse_args()
    folds = [int(x) for x in a.folds.split(",")]
    seeds = [int(x) for x in a.seeds.split(",")]
    vs = [v for v in a.variants.split(",") if v]

    ev = pitch_events()
    from path_alloc import build_df
    tr = build_df()
    import zipfile, io, joblib
    with zipfile.ZipFile(os.path.join(ROOT, "submissions", "cand_h1.zip")) as z:
        BASE = joblib.load(io.BytesIO(z.read("model/rf.pkl")))["features"]

    season = tr["season"].to_numpy()
    y = tr["control_success"].to_numpy(np.float64)
    C = cells(ev)
    H = ((np.nan_to_num(ev["middle"]) > .5) |
         (np.nan_to_num(ev["reverse"]) > .5)).astype(int)
    H[C < 0] = -1                                     # 복원 결측 (투수별 마지막 행)
    P = tr["pitcher_id"].to_numpy(np.int64)
    BH = tr["batter_hand"].to_numpy(np.int64)
    BB = tr["balls_before"].to_numpy(np.int64)
    SS = tr["strikes_before"].to_numpy(np.int64)
    OB = (tr["num_runners_on"].to_numpy(np.int64) > 0).astype(np.int64)
    PH = P * 10 + BH
    PHA = PH * 10 + (SS > BB).astype(np.int64)
    AX = [(P, PH), (PH, PHA), (PHA, PHA * 100 + (BB * 4 + SS)), (PH, PH * 10 + OB)]
    print(f"기본 {len(BASE)}열 · 복원 결측 {int((C<0).sum()):,}행 "
          f"({(C<0).mean():.4%}) — 학습에서만 제외한다")

    AUXC = os.path.join(ROOT, "exp", "cache", "exp049_aux_{f}_{t}.npz")

    def aux_col(tr, BASE, mt, mv, name, tgt, good, seed=42, hp=None):
        """OOF 보조 예측. 학습 행은 2겹 OOF, 검증 행은 두 모델 평균. 폴드별 캐시."""
        p = AUXC.format(f=fold, t=name + ("" if hp is None else "_deep"))
        if os.path.exists(p):
            z = np.load(p)
            return z["oof"], z["val"]
        idx = np.arange(int(mt.sum()))
        half = idx % 2 == 0
        oof = np.full(len(idx), np.nan)
        parts = []
        for hsel in (half, ~half):
            fitm = hsel & good
            t0 = time.time()
            if hp is None:
                ma = ba.pipeline(BASE, seed)
            else:
                keep = ba.HP.copy(); ba.HP.clear(); ba.HP.update({**keep, **hp})
                ma = ba.pipeline(BASE, seed)
                ba.HP.clear(); ba.HP.update(keep)
            ma.fit(tr.loc[mt, BASE][fitm], tgt[mt][fitm].astype(int))
            oof[~hsel] = ma.predict_proba(tr.loc[mt, BASE][~hsel])[:, 1]
            parts.append(ma.predict_proba(tr.loc[mv, BASE])[:, 1])
            print(f"    aux[{name}] 반쪽 학습 {time.time()-t0:.0f}s", flush=True)
        val = np.mean(parts, 0)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        np.savez_compressed(p, oof=oof, val=val)
        return oof, val

    res = {}
    for fold in folds:
        mt, mv = season < fold, season == fold
        post = np.column_stack([
            ba.look(*ba.nested_dev(p[mt], c[mt], y[mt], k), c[mv])
            for (p, c), k in zip(AX, ba.KSH)]) @ ba.WPOST
        Xtr, Xva, yv = tr.loc[mt, BASE], tr.loc[mv, BASE], y[mv]
        good = (C[mt] >= 0)
        print(f"\n=== 폴드 {fold} === 학습 {mt.sum():,} (복원가능 {good.sum():,}) "
              f"검증 {mv.sum():,}", flush=True)

        for v in ["base"] + vs:
            for s in seeds:
                t0 = time.time()
                if v == "base":
                    mm = ba.pipeline(BASE, s)
                    mm.fit(Xtr, y[mt].astype(int))
                    pv = mm.predict_proba(Xva)[:, 1]
                    ins = mm.predict_proba(Xtr)[:, 1]
                elif v.startswith("aux"):
                    # 🚩 TYPE A — 주 모델의 **타깃은 그대로** 두고 성분 확률을
                    # 피처로 넣는다. 새 라벨에서 뽑은 정보를 주입하는 것이라
                    # 원장 정의상 '새 정보축 추가'이고 전이 등급이 두 단계 낫다.
                    A3 = {"middle": ev["middle"], "reverse": ev["reverse"],
                          "ball": ev["ball"]}
                    A7 = {**A3, "strike": ev["strike"], "fb": ev["fb"],
                          "br": ev["br"], "os": ev["os"]}
                    tg = {"aux": {"H": H.astype(float)}, "aux3": A3,
                          "aux7": A7, "aux8": {**A7, "H": H.astype(float)},
                          "auxdeep": A3}[v]
                    # 🚩 보조 모델만 용량을 키운다. 보조 타깃은 y 보다 신호가
                    # 3배라 주 모델용 HP(잡음 타깃에 맞춰 규제됨)로는 덜 뽑는다.
                    # **주 모델의 적합 절차는 불변** -> 전이 위계상 여전히 TYPE A.
                    hpx = ({"depth": 8, "iterations": 2000, "l2_leaf_reg": 30.0}
                           if v == "auxdeep" else None)
                    Xt2, Xv2, add = Xtr.copy(), Xva.copy(), []
                    for nm2, t2 in tg.items():
                        o2, v2 = aux_col(tr, BASE, mt, mv, nm2,
                                         np.nan_to_num(t2), good, hp=hpx)
                        Xt2[f"aux_{nm2}"] = o2
                        Xv2[f"aux_{nm2}"] = v2
                        add.append(f"aux_{nm2}")
                    mm = ba.pipeline(BASE + add, s)
                    mm.fit(Xt2, y[mt].astype(int))
                    pv = mm.predict_proba(Xv2)[:, 1]
                    ins = mm.predict_proba(Xt2)[:, 1]
                elif v == "ftq":
                    # 소프트 라벨 — 셀 기대값 q_c 로 회귀 (CrossEntropy 는 [0,1]
                    # 실수 타깃을 받는다). H=0 셀 안의 잔여 x 의존(S 채널 19%)은
                    # 버려지므로 그 손실과 분산 감소(1.49x)의 순효과를 본다.
                    hp = dict(ba.HP); hp["loss_function"] = "CrossEntropy"
                    keep = ba.HP.copy(); ba.HP.clear(); ba.HP.update(hp)
                    mm = ba.pipeline(BASE, s)
                    ba.HP.clear(); ba.HP.update(keep)
                    cc = C[mt][good]; yg = y[mt][good]
                    qmap = {int(k): float(yg[cc == k].mean()) for k in np.unique(cc)}
                    tgt = np.array([qmap[int(k)] for k in cc])
                    mm.fit(Xtr[good], tgt)
                    pv = mm.predict_proba(Xva)[:, 1]
                    ins = mm.predict_proba(Xtr)[:, 1]
                elif v == "ft2":
                    ma = ba.pipeline(BASE, s)
                    ma.fit(Xtr[good], H[mt][good])
                    mb = ba.pipeline(BASE, s)
                    sub = good & (H[mt] == 0)
                    mb.fit(Xtr[sub], y[mt][sub].astype(int))
                    pv = (1 - ma.predict_proba(Xva)[:, 1]) * mb.predict_proba(Xva)[:, 1]
                    ins = ((1 - ma.predict_proba(Xtr)[:, 1]) *
                           mb.predict_proba(Xtr)[:, 1])
                else:
                    hp = dict(ba.HP); hp["loss_function"] = "MultiClass"
                    keep = ba.HP.copy()
                    ba.HP.clear(); ba.HP.update(hp)
                    mm = ba.pipeline(BASE, s)
                    ba.HP.clear(); ba.HP.update(keep)
                    cc = C[mt][good]
                    yg = y[mt][good]
                    qmap = {int(k): float(yg[cc == k].mean())
                            for k in np.unique(cc)}
                    mm.fit(Xtr[good], cc)
                    qv = np.array([qmap[int(k)] for k in mm.classes_])
                    pv = mm.predict_proba(Xva) @ qv       # Σ_c P(c|row) q_c
                    ins = mm.predict_proba(Xtr) @ qv
                r = 1e5 * np.corrcoef(pv + post, yv)[0, 1] ** 2
                ri = 1e5 * np.corrcoef(ins, y[mt])[0, 1] ** 2
                res.setdefault(f"{fold}:{v}", []).append(r)
                res.setdefault(f"{fold}:{v}:in", []).append(ri)
                print(f"  {v:5s} seed{s}  홀드아웃 {r:8.2f}   in-sample {ri:8.2f}   "
                      f"({time.time()-t0:.0f}s)", flush=True)
                json.dump(res, open(a.out, "w"), indent=1)
        b = np.array(res[f"{fold}:base"]); bi = np.array(res[f"{fold}:base:in"])
        for v in vs:
            g = np.array(res[f"{fold}:{v}"]); gi = np.array(res[f"{fold}:{v}:in"])
            print(f"  -> {v:5s} 홀드아웃 {g.mean():8.2f}  대조 대비 "
                  f"{(g-b).mean():+7.2f} ({(g.mean()/b.mean()-1)*100:+.3f}%)   "
                  f"in-sample {gi.mean():8.2f} ({(gi-bi).mean():+7.2f})", flush=True)
            ri_ = gi.mean() / bi.mean() - 1
            rh_ = g.mean() / b.mean() - 1
            if ri_ > rh_ + 0.002:
                print(f"     ⚠️ in-sample 상승률 {ri_:+.3%} > 홀드아웃 {rh_:+.3%} "
                      f"— 단조 제약과 같은 과적합 서명", flush=True)
            else:
                print(f"     in-sample {ri_:+.3%} vs 홀드아웃 {rh_:+.3%} — 건강",
                      flush=True)


if __name__ == "__main__":
    main()
