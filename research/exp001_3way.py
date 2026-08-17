r"""EXP001 — 투수 x 맥락 x 맥락 (2차 상호작용 차등). TYPE A.

이미 채택된 세 축(손·2S·주자)은 각각 1차 차등이다. 그 **교차**는 미검증이다.

    d2_p = [r(a=1,b=1) - r(a=0,b=1)] - [r(a=1,b=0) - r(a=0,b=0)]
    보정 = 0.25 * d2_p * (+1 if a==b else -1)
    n_eff = 4 / (1/n00 + 1/n01 + 1/n10 + 1/n11)

기준선 C3. 오라클은 경기 단위 분할 + 위약. k 는 과거 2전이로만 선택.
"""
import os, sys, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "exp"))
import engine as E
from path_alloc import build_df
from resid_table import post_for
from traj_probe import cells, look, r2
from game_decomp import games
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

EXP, KGRID = "EXP001", [500, 1000, 2000, 5000, 20000]
PREV2 = {2022: (2020, 2021), 2023: (2021, 2022), 2024: (2022, 2023)}
E.start_experiment(EXP, "H001-H003", "python research/exp001_3way.py", "load")

tr = build_df(); season = tr["season"].to_numpy()
y = tr["control_success"].to_numpy(np.float64)
P = tr["pitcher_id"].to_numpy(np.int64); g = lambda c: tr[c].to_numpy(np.float64)
SAME = (g("pitcher_hand") == g("batter_hand")).astype(int)
TWO = (g("strikes_before").astype(int) == 2).astype(int)
RUN = (g("num_runners_on") > 0).astype(int)
pv0, res0 = {}, {}
for f in (2020, 2021, 2022, 2023, 2024):
    m = season == f
    pv0[f] = (np.load(os.path.join(ROOT, "exp", f"prod_champ_{f}.npy"))[:2].mean(0)
              + post_for(tr, y, season < f, m))
    res0[f] = y[m] - pv0[f]
E.beat("C3 기준선 구성")

def d1(ctx, src, k):
    p = np.concatenate([P[season == f] for f in src])
    c = np.concatenate([ctx[season == f] for f in src])
    r = np.concatenate([res0[f] for f in src])
    gg = pd.DataFrame({"p": p, "c": c, "r": r}).groupby(["p", "c"])["r"].agg(
        ["mean", "size"]).unstack()
    n0, n1 = gg[("size", 0)].fillna(0), gg[("size", 1)].fillna(0)
    d = gg[("mean", 1)] - gg[("mean", 0)]
    ne = (n0 * n1) / (n0 + n1).replace(0, np.nan)
    return (d * ne / (ne + k)).dropna()

def ap1(t, ctx, m):
    return pd.Series(P[m]).map(t).fillna(0.).to_numpy() * np.where(ctx[m] == 1, .5, -.5)

C3, resC = {}, {}
for f in (2022, 2023, 2024):
    m = season == f; s = PREV2[f]
    C3[f] = (pv0[f] + ap1(d1(SAME, s, 1000), SAME, m) + ap1(d1(TWO, s, 1000), TWO, m)
             + ap1(d1(RUN, s, 2000), RUN, m))
    resC[f] = y[m] - C3[f]
m24 = season == 2024; base = r2(C3[2024], y[m24])
print(f"C3 기준선 2024 = {base:.1f}")

def d2(a, b, src, k):
    """2차 상호작용 차등."""
    p = np.concatenate([P[season == f] for f in src])
    ca = np.concatenate([a[season == f] for f in src])
    cb = np.concatenate([b[season == f] for f in src])
    r = np.concatenate([res0[f] for f in src])
    key = ca * 2 + cb
    gg = pd.DataFrame({"p": p, "k": key, "r": r}).groupby(["p", "k"])["r"].agg(
        ["mean", "size"]).unstack()
    need = [("size", i) for i in range(4)]
    if any(c not in gg for c in need):
        return None
    n = [gg[("size", i)].fillna(0) for i in range(4)]
    mu = [gg[("mean", i)] for i in range(4)]
    inter = (mu[3] - mu[1]) - (mu[2] - mu[0])
    inv = sum(1.0 / x.replace(0, np.nan) for x in n)
    ne = 4.0 / inv
    return (inter * ne / (ne + k)).dropna()

def ap2(t, a, b, m):
    if t is None: return np.zeros(int(m.sum()))
    sg = np.where(a[m] == b[m], .25, -.25)
    return pd.Series(P[m]).map(t).fillna(0.).to_numpy() * sg

GID = games(P[m24], g("asof_pitcher_n")[m24],
            g("asof_pitcher_prev1_game_success_rate")[m24],
            g("asof_pitcher_prev1_game_middle_rate")[m24])
rng = np.random.default_rng(0); u, inv = np.unique(GID, return_inverse=True)
half = (rng.random(len(u)) < 0.5)[inv]
HD = ap1(d1(SAME, (2022, 2023), 1000), SAME, m24)
TS = ap1(d1(TWO, (2022, 2023), 1000), TWO, m24)
RN = ap1(d1(RUN, (2022, 2023), 2000), RUN, m24)

CAND = {"H001 손 x 2S": (SAME, TWO), "H002 손 x 주자": (SAME, RUN),
        "H003 2S x 주자": (TWO, RUN)}
print(f"\n{'가설':<18}{'오라클':>8}{'위약':>7}{'21→22':>8}{'22→23':>8}{'23→24':>8}"
      f"{'k':>7}{'C3증분':>8}{'hand/2S/run':>16}")
out = {}
for name, (a, b) in CAND.items():
    E.beat(f"probe {name}")
    key = (P[m24] * 4 + a[m24] * 2 + b[m24])
    bo = -1e9
    for k in (0, 150, 500, 2000, 10000):
        add = np.zeros(int(m24.sum()))
        for m in (half, ~half):
            uu, tb, _ = cells(key[m], resC[2024][m], k)
            add[~m] = look(uu, tb, key[~m])
        bo = max(bo, r2(C3[2024] + add, y[m24]) - base)
    pl = P[m24] * 4 + rng.integers(0, 4, int(m24.sum()))
    bp = -1e9
    for k in (0, 150, 500, 2000, 10000):
        add = np.zeros(int(m24.sum()))
        for m in (half, ~half):
            uu, tb, _ = cells(pl[m], resC[2024][m], k)
            add[~m] = look(uu, tb, pl[~m])
        bp = max(bp, r2(C3[2024] + add, y[m24]) - base)
    G = {k: [] for k in KGRID}
    for s_, t_ in ((2021, 2022), (2022, 2023), (2023, 2024)):
        mb = season == t_; bs = r2(C3[t_], y[mb])
        for k in KGRID:
            G[k].append(r2(C3[t_] + ap2(d2(a, b, (s_,), k), a, b, mb), y[mb]) - bs)
    kb = max(G, key=lambda k: np.mean(G[k][:2]))
    add24 = ap2(d2(a, b, (2022, 2023), kb), a, b, m24)
    inc = r2(C3[2024] + add24, y[m24]) - base
    ov = [float(np.corrcoef(add24, v)[0, 1]) if add24.std() > 0 else 0. for v in (HD, TS, RN)]
    out[name] = dict(oracle=bo, placebo=bp, k=kb, gains=G[kb], inc=inc, overlap=ov)
    print(f"{name:<18}{bo:>8.1f}{bp:>7.1f}" + "".join(f"{v:>+8.1f}" for v in G[kb])
          + f"{kb:>7}{inc:>+8.1f}   {ov[0]:+.2f}/{ov[1]:+.2f}/{ov[2]:+.2f}")

best = max(out, key=lambda n: out[n]["inc"])
dec = ("PROMISING" if out[best]["inc"] >= 4.8 and all(v > 0 for v in out[best]["gains"])
       else "REJECTED")
E.finish_experiment(dict(
    experiment_id=EXP, hypothesis_id="H001-H003", type="A",
    started_at=E.read(E.CKPT)["start_time"],
    local_result={k: round(v["inc"], 2) for k, v in out.items()},
    transfer_result={k: [round(x, 1) for x in v["gains"]] for k, v in out.items()},
    oracle={k: round(v["oracle"], 1) for k, v in out.items()},
    placebo={k: round(v["placebo"], 1) for k, v in out.items()},
    redundancy={k: [round(x, 3) for x in v["overlap"]] for k, v in out.items()},
    decision=dec, artifact=None,
    what_we_learned=("이미 채택된 1차 차등 축들의 2차 상호작용은 "
                     + ("신호가 있다" if dec == "PROMISING" else
                        "정보가 없다 — 1차 차등이 이미 다 가져갔고 4셀로 쪼개면 셀당 표본이 1/4 로 준다"))))
for h, n in (("H001", "H001 손 x 2S"), ("H002", "H002 손 x 주자"), ("H003", "H003 2S x 주자")):
    E.set_hypothesis_status(h, "PROBE_FAIL" if out[n]["inc"] < 4.8 else "PROMISING",
                            result=round(out[n]["inc"], 2))
json.dump(out, open(os.path.join(ROOT, "exp", "exp001_3way.json"), "w"), indent=1, default=float)
print(f"\nDECISION = {dec}   (best {best} {out[best]['inc']:+.1f})")
