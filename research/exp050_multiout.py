r"""EXP050 — 🚩 **다중 결과변수 전수 재검.** 상쇄된 축을 찾는다.

## 왜 다시 여는가

8개월간 60여 개 축을 닫았고 그 판정은 **전부 `y` 하나**로 했다 — 오라클 진단,
243개 이진 국면 전수(EXP003), 잔차 상관, 전부. 그런데 2026-08-21 에 타깃이
합성 사건임이 밝혀졌다.

    y  ≈  (1 − middle) · (1 − reverse) · (0.95 − 0.36·ball)
    middle=1 -> y=0 (n=220,460, 정확히 0)   reverse=1 -> y=0 (n=337,496)

셋 다 학습셋에서 100% 복원된다 (EXP046). 그러므로 **결과변수가 넷이 됐다.**

## 가설

    축 A 가 P(middle) 을 올리면서 동시에 P(ball) 을 낮추면
      y 에 대한 순효과 ~ 0  ->  전수 스윕에서 "정보 없음" 으로 닫힌다
      성분별로는 둘 다 강한 신호다

**완전 상쇄라면 y 는 정말 영향을 안 받으므로 성분을 알아도 소용없다.** 이득은
상쇄가 **맥락 의존적**일 때만 나온다. 그래서 이 스윕이 찾는 것은 "성분에 강한
축" 이 아니라 **"성분별 효과의 부호가 갈리는 축"** 이다. 그런 축이 있으면
인수분해/AUX 모형이 직접 적합보다 적은 유효 모수로 같은 함수를 표현한다.

## 절차

    1  성분별 기준선 모델을 만든다 (middle · reverse · ball, 82열 동일 HP)
       -> 잔차 = "그 성분에 대해 **현행 피처로 설명되지 않는 부분**"
    2  EXP003 의 열거를 그대로 재사용해 이진 국면마다 투수별 차등열을 만든다
    3  각 국면 x 각 결과변수의 잔차 상관을 잰다
    4  |성분 상관| 이 |y 상관| 보다 크게 큰 축, 특히 **성분 간 부호가 갈리는** 축을 뽑는다

    .\.venv\Scripts\python.exe -u research\exp050_multiout.py --fold 2024
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "exp"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import build_asof as ba                                      # noqa: E402
from exp003_sweep import build_contexts                      # noqa: E402
from exp046_pitchmix import pitch_events, ctx_dev, look      # noqa: E402

OUTS = ["y", "middle", "reverse", "ball"]
BASE_CACHE = os.path.join(ROOT, "exp", "cache", "exp050_base_{f}.npz")


def baselines(tr, BASE, season, fold, targets):
    """성분별 기준선 예측. 캐시한다 (학습 3회 x 약 3분)."""
    p = BASE_CACHE.format(f=fold)
    if os.path.exists(p):
        z = np.load(p)
        return {k: z[k] for k in z.files}
    mt, mv = season < fold, season == fold
    out = {}
    for nm, t in targets.items():
        ok = mt & ~np.isnan(t)
        t0 = time.time()
        mm = ba.pipeline(BASE, 42)
        mm.fit(tr.loc[ok, BASE], t[ok].astype(int))
        out[nm] = mm.predict_proba(tr.loc[mv, BASE])[:, 1]
        print(f"  기준선 {nm:8s} 학습 {time.time()-t0:.0f}s  "
              f"corr {np.corrcoef(out[nm], t[mv])[0,1]:+.4f}", flush=True)
    np.savez_compressed(p, **out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=2024)
    ap.add_argument("--k", type=float, default=1000.0)
    ap.add_argument("--top", type=int, default=30)
    a = ap.parse_args()

    ev = pitch_events()
    from path_alloc import build_df
    tr = build_df()
    import zipfile, io, joblib
    with zipfile.ZipFile(os.path.join(ROOT, "submissions", "cand_h1.zip")) as z:
        BASE = joblib.load(io.BytesIO(z.read("model/rf.pkl")))["features"]
    season = tr["season"].to_numpy()
    y = tr["control_success"].to_numpy(np.float64)
    fold = a.fold
    mt, mv = season < fold, season == fold

    T = {"middle": ev["middle"], "reverse": ev["reverse"], "ball": ev["ball"]}
    bl = baselines(tr, BASE, season, fold, T)
    bl["y"] = np.load(os.path.join(ROOT, "exp", f"prod_champ_{fold}.npy")).mean(0)

    RES, sd = {}, {}
    good = ~np.isnan(ev["middle"][mv])
    for o in OUTS:
        t = y[mv] if o == "y" else ev[o][mv]
        RES[o] = np.where(good, np.nan_to_num(t) - bl[o], np.nan)
        sd[o] = np.nanstd(RES[o])
        print(f"  잔차 {o:8s} sd {sd[o]:.5f}")

    CTX = build_contexts(tr, season)
    print(f"\n이진 국면 {len(CTX)}개 x 결과 {len(OUTS)}개  (폴드 {fold}, k={a.k:g})")
    P = tr["pitcher_id"].to_numpy(np.int64)
    src = (season < fold) & (season >= fold - 2)
    n = int(mv.sum())
    se = 1.0 / np.sqrt(good.sum())

    rows = []
    for i, (nm, ctx) in enumerate(sorted(CTX.items())):
        okc = ctx >= 0
        cc = np.where(okc, ctx, 0).astype(np.int64)
        r = {"axis": nm}
        for o in OUTS:
            t = y if o == "y" else np.nan_to_num(ev[o])
            tb, _ = ctx_dev(P, cc, t, src & okc & ~np.isnan(ev["middle"]), a.k)
            v = look(tb, None, P[mv], cc[mv])
            v = np.where(okc[mv] & good, v, 0.0)
            m2 = good & (v != 0)
            r[o] = (float(np.corrcoef(v[m2], RES[o][m2])[0, 1])
                    if m2.sum() > 5000 and v[m2].std() > 0 else 0.0)
        r["cover"] = float((okc & mv).sum() / n)
        rows.append(r)
        if (i + 1) % 40 == 0:
            print(f"    {i+1}/{len(CTX)}", flush=True)

    D = pd.DataFrame(rows)
    D["max_comp"] = D[["middle", "reverse", "ball"]].abs().max(1)
    D["lift"] = D["max_comp"] - D["y"].abs()
    D["split"] = D[["middle", "reverse", "ball"]].max(1) - \
        D[["middle", "reverse", "ball"]].min(1)          # 성분 간 부호 갈림 폭
    D["z_y"] = D["y"].abs() / se
    D["z_comp"] = D["max_comp"] / se

    print(f"\n=== y 대비 성분 신호가 큰 축 상위 {a.top} (lift 순) ===")
    print(f"{'축':34s} {'y':>8s} {'mid':>8s} {'rev':>8s} {'ball':>8s} "
          f"{'lift':>7s} {'z_comp':>7s}")
    for _, r in D.sort_values("lift", ascending=False).head(a.top).iterrows():
        print(f"{r.axis[:34]:34s} {r.y:+8.4f} {r.middle:+8.4f} {r.reverse:+8.4f} "
              f"{r.ball:+8.4f} {r.lift:+7.4f} {r.z_comp:7.2f}")
    print(f"\n=== 🚩 성분 간 **부호가 갈리는** 축 상위 {a.top} (split 순) ===")
    print(f"{'축':34s} {'y':>8s} {'mid':>8s} {'rev':>8s} {'ball':>8s} {'split':>7s}")
    for _, r in D.sort_values("split", ascending=False).head(a.top).iterrows():
        print(f"{r.axis[:34]:34s} {r.y:+8.4f} {r.middle:+8.4f} {r.reverse:+8.4f} "
              f"{r.ball:+8.4f} {r.split:7.4f}")
    print(f"\n[대조] 같은손 계열 · 2스트라이크 · 주자유무 (LB 로 검증된 3축)")
    for _, r in D[D.axis.str.contains("batter_hand|strikes_before==2|"
                                      "num_runners_on==0")].iterrows():
        print(f"{r.axis[:34]:34s} {r.y:+8.4f} {r.middle:+8.4f} {r.reverse:+8.4f} "
              f"{r.ball:+8.4f} {r.split:7.4f}")
    D.to_csv(os.path.join(ROOT, "exp", f"exp050_multiout_{fold}.csv"),
             index=False, encoding="utf-8")
    print(f"\n-> exp/exp050_multiout_{fold}.csv   (잔차상관 SE = {se:.5f})")


if __name__ == "__main__":
    main()
