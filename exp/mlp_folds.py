r"""혼합 이득이 **시즌을 넘는지** 잰다. 2024 한 폴드의 이득은 낙관 상한이다.

## 왜 이 검사가 필수인가

§6-p 가 FM 을 기각하며 명시한 함정 그대로다 — "혼합 이득 평균 +4.14 는 **대상
폴드에서 가중을 고른 낙관 상한**"이다. `exp/mlp_probe.py` 의 +4.8 도 2024 에서
`w` 를 골라 2024 를 잰 값이라 같은 성질을 갖는다.

여기서는 두 가지를 분리한다.

  (1) **오라클** — 그 폴드에서 `w` 를 고른 값. 상한.
  (2) **정직한 값** — `w` 를 **다른 폴드들에서** 고르고 이 폴드에 적용한 값.
      실제 제출이 할 수 있는 최선이다. 이것이 판정 기준이다.

세기비와 상관도 폴드마다 다시 잰다. 2024 에서 c=0.6917 이 나온 것이 그 폴드
고유의 사정인지, 계열의 성질인지가 여기서 갈린다.

## 주의 — 폴드마다 학습량이 다르다

    <2022 = 728,588행   <2023 = 976,060행   <2024 = 1,221,585행

MLP 는 데이터량에 GBDT 보다 민감하므로 이른 폴드의 세기비는 과소평가일 수 있다.
추세(2022 -> 2024)가 증가면 그 자체가 정보다.

    .\.venv\Scripts\python.exe -u exp\mlp_folds.py --drop 0.3 --wd 1e-4 --epochs 8 --emul 0.25
"""
import argparse
import io
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "exp", "cache")
FOLDS = (2022, 2023, 2024)
BASE_EMB = {"pitcher_id": 32, "batter_id": 32, "pitcher_team_id": 8,
            "batter_team_id": 8, "base_state": 6, "top_bottom": 2,
            "game_type": 2, "pitcher_hand": 2, "batter_hand": 2}
S_CUR = 955.2193198652


class PLR(nn.Module):
    def __init__(self, n, k, d, sigma=0.1):
        super().__init__()
        self.c = nn.Parameter(torch.randn(n, k) * sigma)
        self.w = nn.Parameter(torch.randn(n, 2 * k, d) / np.sqrt(2 * k))
        self.b = nn.Parameter(torch.zeros(n, d))

    def forward(self, z):
        v = 2 * np.pi * self.c.unsqueeze(0) * z.unsqueeze(-1)
        e = torch.cat([torch.sin(v), torch.cos(v)], -1)
        return torch.relu(torch.einsum("bnk,nkd->bnd", e, self.w)
                          + self.b).flatten(1)


class Net(nn.Module):
    def __init__(self, cards, n_num, n_flag, k, d, hid, drop):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(c, e) for c, e in cards])
        self.plr = PLR(n_num, k, d)
        din = sum(e for _, e in cards) + n_num * d + n_num + n_flag
        L, p = [], din
        for h in hid:
            L += [nn.Linear(p, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(drop)]
            p = h
        self.mlp, self.head = nn.Sequential(*L), nn.Linear(p, 1)

    def forward(self, xc, xn, xf):
        e = [m(xc[:, i]) for i, m in enumerate(self.emb)]
        return self.head(self.mlp(torch.cat(e + [self.plr(xn), xn, xf],
                                            1))).squeeze(1)


def fold_data(val, meta, X, y, season):
    """폴드마다 다시 만든다 — 표준화 통계도 그 폴드의 학습 구간에서만."""
    ixc = {c: i for i, c in enumerate(meta["cols"])}
    prod = meta["prod"]
    catn = [c for c in prod if c in BASE_EMB]
    numn = [c for c in prod if c not in BASE_EMB and c != "season"]
    tr = season < val
    XC = np.zeros((len(y), len(catn)), dtype=np.int64)
    ncat = []
    for j, c in enumerate(catn):
        v = np.asarray(X[:, ixc[c]])
        u = np.unique(v[tr & ~np.isnan(v)])
        idx = np.clip(np.searchsorted(u, v), 0, len(u) - 1)
        ok = (~np.isnan(v)) & (u[idx] == v)
        XC[:, j] = np.where(ok, idx + 1, 0)
        ncat.append(len(u) + 1)
    XN = np.zeros((len(y), len(numn)), dtype=np.float32)
    flags = []
    for j, c in enumerate(numn):
        v = np.asarray(X[:, ixc[c]], dtype=np.float64)
        q1, med, q3 = np.nanpercentile(v[tr], [25, 50, 75])
        XN[:, j] = np.clip(np.nan_to_num((v - med) / max((q3 - q1) / 1.349,
                                                         1e-6), nan=0.0), -5, 5)
        if np.isnan(v[tr]).any():
            flags.append(np.isnan(v).astype(np.float32))
    XF = (np.stack(flags, 1) if flags
          else np.zeros((len(y), 0), dtype=np.float32))
    return catn, ncat, XC, XN, XF


def train_fold(val, a, meta, X, y, season):
    catn, ncat, XC, XN, XF = fold_data(val, meta, X, y, season)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    keep = [i for i, c in enumerate(catn)
            if a.emul > 0 or c not in ("pitcher_id", "batter_id")]
    cards = [(ncat[i], max(2, int(round(BASE_EMB[catn[i]] * a.emul)))
              if catn[i] in ("pitcher_id", "batter_id") else BASE_EMB[catn[i]])
             for i in keep]
    tr, va = season < val, season == val
    ntr, nva = int(tr.sum()), int(va.sum())
    tc = torch.from_numpy(XC[tr][:, keep])
    tn, tf = torch.from_numpy(XN[tr]), torch.from_numpy(XF[tr])
    ty = torch.from_numpy(y[tr])
    vc = torch.from_numpy(XC[va][:, keep])
    vn, vf = torch.from_numpy(XN[va]), torch.from_numpy(XF[va])
    yv = y[va].astype(np.float64)
    hid = [int(h) for h in a.hid.split(",")]
    net = Net(cards, XN.shape[1], XF.shape[1], a.k, a.d, hid, a.drop)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    B = 8192
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt, a.lr, total_steps=a.epochs * ((ntr + B - 1) // B))
    lossf = nn.BCEWithLogitsLoss()
    for _ in range(a.epochs):
        perm = torch.randperm(ntr)
        for i in range(0, ntr, B):
            b = perm[i:i + B]
            opt.zero_grad()
            lossf(net(tc[b], tn[b], tf[b]), ty[b]).backward()
            opt.step()
            sch.step()
    net.eval()
    with torch.no_grad():
        p = np.concatenate([torch.sigmoid(
            net(vc[i:i + 16384], vn[i:i + 16384],
                vf[i:i + 16384])).numpy() for i in range(0, nva, 16384)])
    return p.astype(np.float64), yv


def blend_curve(pc, pm, yv):
    """w(챔피언 가중) 격자 위의 rho^2 곡선."""
    zc = (pc - pc.mean()) / pc.std()
    zm = (pm - pm.mean()) / pm.std()
    w = np.linspace(0, 1, 1001)
    return w, np.array([1e5 * np.corrcoef(x * zc + (1 - x) * zm, yv)[0, 1] ** 2
                        for x in w])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--drop", type=float, default=0.3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--emul", type=float, default=0.25)
    ap.add_argument("--hid", default="512,256")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--d", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    meta = json.load(io.open(f"{CACHE}/cols.json", encoding="utf-8"))
    X = np.load(f"{CACHE}/X.npy", mmap_mode="r")
    y = np.load(f"{CACHE}/y.npy").astype(np.float32)
    season = np.load(f"{CACHE}/season.npy")
    print(f"설정  epochs={a.epochs} drop={a.drop} wd={a.wd} lr={a.lr} "
          f"emul={a.emul} hid={a.hid}\n")

    res = {}
    for f in FOLDS:
        t = time.time()
        pm, yv = train_fold(f, a, meta, X, y, season)
        pc = np.load(os.path.join(ROOT, "exp", f"champ_oof_{f}.npy"))
        rc = 1e5 * np.corrcoef(pc, yv)[0, 1] ** 2
        rm = 1e5 * np.corrcoef(pm, yv)[0, 1] ** 2
        c = float(np.corrcoef(pm, pc)[0, 1])
        w, cur = blend_curve(pc, pm, yv)
        res[f] = dict(rc=rc, rm=rm, c=c, w=w, cur=cur,
                      wopt=float(w[int(np.argmax(cur))]))
        print(f"  {f}  학습 {int((season < f).sum()):>9,}행   "
              f"Champ rho^2 {rc:>7.1f}   MLP rho^2 {rm:>7.1f}   "
              f"세기비 {np.sqrt(rm / rc):.3f}   c={c:.4f}   "
              f"w*={res[f]['wopt']:.3f}   {time.time() - t:.0f}s")

    print(f"\n=== 오라클 vs 정직한 값 ===")
    print(f"  {'폴드':>6}{'Champ':>10}{'오라클':>10}{'오라클증분':>11}"
          f"{'타폴드w':>9}{'정직한값':>10}{'정직한증분':>11}")
    hon = []
    for f in FOLDS:
        r = res[f]
        orc = float(np.max(r["cur"]))
        others = [res[g]["wopt"] for g in FOLDS if g != f]
        wo = float(np.mean(others))
        j = int(np.argmin(np.abs(r["w"] - wo)))
        h = float(r["cur"][j])
        hon.append(h / r["rc"])
        print(f"  {f:>6}{r['rc']:>10.1f}{orc:>10.1f}{orc - r['rc']:>+11.1f}"
              f"{wo:>9.3f}{h:>10.1f}{h - r['rc']:>+11.1f}")

    g = float(np.mean(hon))
    print(f"\n  정직한 배수 평균 {g:.4f}   최악 {min(hon):.4f}")
    print(f"  -> 955.22 환산   평균 **{S_CUR * g:.1f}**   최악 {S_CUR * min(hon):.1f}")
    print(f"  (개선 폴드 {sum(x > 1 for x in hon)}/3)")


if __name__ == "__main__":
    main()
