r"""FINAL SCORE MAXIMIZATION — 검증된 신호만으로 표를 더 좋게 만든다. 학습 0회.

새 신호를 찾지 않는다. C3 의 세 표(손/2S/주자)를 만드는 **방식**만 개선한다.

    A 원천 시즌 수     1개 / 2개 / 3개
    B cold-start 대체  표에 없는 투수를 0 대신 리그 평균 차등으로
    C 전역 가중        세 보정에 공통 스케일 (k 를 키우는 것과 다른 조작)

**선택은 전부 과거 전이(21->22, 22->23)로만** 한다. 2024 는 평가에만 쓴다.
"""
import io, json, os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "exp"))
from path_alloc import build_df
from resid_table import post_for
from traj_probe import r2
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

tr = build_df(); season = tr["season"].to_numpy()
y = tr["control_success"].to_numpy(np.float64)
P = tr["pitcher_id"].to_numpy(np.int64)
g = lambda c: tr[c].to_numpy(np.float64)
CTX = {"hand": ((g("pitcher_hand") == g("batter_hand")).astype(int), 1000),
       "2S": ((g("strikes_before").astype(int) == 2).astype(int), 1000),
       "run": ((g("num_runners_on") > 0).astype(int), 2000)}
pv0, res0 = {}, {}
for f in (2020, 2021, 2022, 2023, 2024):
    m = season == f
    pv0[f] = (np.load(os.path.join(ROOT, "exp", f"prod_champ_{f}.npy"))[:2].mean(0)
              + post_for(tr, y, season < f, m))
    res0[f] = y[m] - pv0[f]

def build(ctx, src, k, fallback=False):
    p = np.concatenate([P[season == f] for f in src])
    c = np.concatenate([ctx[season == f] for f in src])
    r = np.concatenate([res0[f] for f in src])
    gg = pd.DataFrame({"p": p, "c": c, "r": r}).groupby(["p", "c"])["r"].agg(
        ["mean", "size"]).unstack()
    n0, n1 = gg[("size", 0)].fillna(0), gg[("size", 1)].fillna(0)
    d = gg[("mean", 1)] - gg[("mean", 0)]
    ne = (n0 * n1) / (n0 + n1).replace(0, np.nan)
    tab = (d * ne / (ne + k)).dropna()
    glob = 0.0
    if fallback:      # 리그 전체 차등을 대체값으로 (표본 가중)
        w = (ne / (ne + k)).reindex(tab.index)
        glob = float((tab * 0 + d.reindex(tab.index)).mul(w).sum() / w.sum())
    return tab, glob

def apply(tab, glob, ctx, m, w=1.0):
    v = pd.Series(P[m]).map(tab)
    v = v.fillna(glob).to_numpy()
    return w * v * np.where(ctx[m] == 1, .5, -.5)

def score(src_n, fallback, w, folds=(2022, 2023, 2024)):
    out = {}
    for f in folds:
        m = season == f
        src = tuple(s for s in range(f - src_n, f) if s >= 2020)
        if len(src) < src_n:
            out[f] = np.nan; continue
        pred = pv0[f].copy()
        for nm, (ctx, k) in CTX.items():
            tab, gl = build(ctx, src, k, fallback)
            pred = pred + apply(tab, gl, ctx, m, w)
        out[f] = r2(pred, y[m]) - r2(pv0[f], y[m])
    return out

print("=" * 84); print("A. 원천 시즌 수 (fallback 없음, w=1.0)"); print("=" * 84)
print(f"{'원천':>6}{'2022':>10}{'2023':>10}{'2024':>10}{'과거 2폴드 평균':>16}")
best_n, best_v = None, -9e9
for n in (1, 2, 3):
    s = score(n, False, 1.0)
    past = np.nanmean([s[2022], s[2023]])
    print(f"{n:>6}{s[2022]:>+10.1f}{s[2023]:>+10.1f}{s[2024]:>+10.1f}{past:>+16.1f}")
    if past > best_v: best_n, best_v = n, past
print(f"  -> 과거로 고른 원천 시즌 수 = {best_n}")

print("\n" + "=" * 84); print("B. cold-start 대체 (리그 평균 차등)"); print("=" * 84)
print(f"{'fallback':>10}{'2022':>10}{'2023':>10}{'2024':>10}{'과거평균':>12}")
best_fb, bv = None, -9e9
for fb in (False, True):
    s = score(best_n, fb, 1.0)
    past = np.nanmean([s[2022], s[2023]])
    print(f"{str(fb):>10}{s[2022]:>+10.1f}{s[2023]:>+10.1f}{s[2024]:>+10.1f}{past:>+12.1f}")
    if past > bv: best_fb, bv = fb, past
print(f"  -> 과거로 고른 fallback = {best_fb}")

print("\n" + "=" * 84); print("C. 전역 가중"); print("=" * 84)
print(f"{'w':>6}{'2022':>10}{'2023':>10}{'2024':>10}{'과거평균':>12}")
best_w, bw = None, -9e9
for w in (0.6, 0.8, 1.0, 1.2, 1.5):
    s = score(best_n, best_fb, w)
    past = np.nanmean([s[2022], s[2023]])
    print(f"{w:>6.1f}{s[2022]:>+10.1f}{s[2023]:>+10.1f}{s[2024]:>+10.1f}{past:>+12.1f}")
    if past > bw: best_w, bw = w, past
print(f"  -> 과거로 고른 w = {best_w}")

print("\n" + "=" * 84); print("최종 조합 (전부 과거로만 선택)"); print("=" * 84)
cur = score(2, False, 1.0)
fin = score(best_n, best_fb, best_w)
print(f"  현행 C3 (원천 2, fallback X, w=1.0)   2022 {cur[2022]:+.1f}  2023 {cur[2023]:+.1f}  2024 {cur[2024]:+.1f}")
print(f"  최적  (원천 {best_n}, fallback {best_fb}, w={best_w})   2022 {fin[2022]:+.1f}  2023 {fin[2023]:+.1f}  2024 {fin[2024]:+.1f}")
print(f"  2024 차이 {fin[2024]-cur[2024]:+.2f}")
json.dump({"best_n": best_n, "fallback": bool(best_fb), "w": best_w,
           "cur": {str(k): v for k, v in cur.items()},
           "fin": {str(k): v for k, v in fin.items()}},
          io.open(os.path.join(ROOT, "exp", "final_opt.json"), "w", encoding="utf-8"),
          indent=1, default=float)
