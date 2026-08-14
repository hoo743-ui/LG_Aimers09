r"""MLP 의 **세기**를 올린다. 상관은 이미 통과했다.

## 1차 탐침이 남긴 것 (`exp/mlp_probe.log`)

    CatBoost rho^2 780.2   MLP rho^2 430.4   세기비 0.743   상관 c = 0.6917

`exp/blend_spec.py` 의 조건은 `세기비 > 상관` 이고, 1120 은
`세기비 1.00 + c <= 0.706` 이다. **상관 0.6917 은 이미 그 조건을 만족한다.**
막힌 것은 세기 하나뿐이다.

    세기비 0.743, c=0.69  ->   960.0   <- 지금
    세기비 0.90,  c=0.70  ->  1030.1
    세기비 1.00,  c=0.70  ->  1123.8

## 그리고 세기가 안 나온 이유가 곡선에 그대로 있다

    ep1 348.5   ep2 430.4 (최고)   ep5 312.3   ep11 175.7   ep15 138.8
    학습 loss 는 0.694 -> 0.653 으로 계속 내려간다

**15에폭 중 2에폭이 최고** = 규제가 전혀 안 맞았다. 범인 후보는 ID 임베딩이다
(투수 792 / 타자 830 을 32차원으로 주면 외우기 충분하다). 이 스윕은 규제 축을
훑는다 — 임베딩 차원, dropout, weight decay, 에폭수, 폭.

데이터 준비는 한 번만 하고 설정만 갈아끼운다.

    .\.venv\Scripts\python.exe -u exp\mlp_sweep.py
"""
import io
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "exp", "cache")
VAL = 2024
BASE_EMB = {"pitcher_id": 32, "batter_id": 32, "pitcher_team_id": 8,
            "batter_team_id": 8, "base_state": 6, "top_bottom": 2,
            "game_type": 2, "pitcher_hand": 2, "batter_hand": 2}

# (이름, 배수, dropout, wd, epochs, hidden, lr, k, d)
CONFIGS = [
    ("기준(1차 재현)",   1.00, 0.10, 1e-5, 15, (512, 256), 2e-3, 8, 12),
    ("에폭 3",          1.00, 0.10, 1e-5,  3, (512, 256), 2e-3, 8, 12),
    ("ID축소 4x",       0.25, 0.10, 1e-5,  6, (512, 256), 2e-3, 8, 12),
    ("ID축소 8x",       0.125, 0.10, 1e-5, 6, (512, 256), 2e-3, 8, 12),
    ("ID제거",          0.00, 0.10, 1e-5,  6, (512, 256), 2e-3, 8, 12),
    ("ID축소+drop0.3",  0.25, 0.30, 1e-4,  8, (512, 256), 1e-3, 8, 12),
    ("ID제거+drop0.3",  0.00, 0.30, 1e-4,  8, (512, 256), 1e-3, 8, 12),
    ("ID제거+wd1e-3",   0.00, 0.20, 1e-3, 10, (256, 128), 1e-3, 8, 12),
    ("ID제거+넓게",      0.00, 0.30, 1e-4, 10, (1024, 512), 1e-3, 16, 16),
]


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


def prep():
    meta = json.load(io.open(f"{CACHE}/cols.json", encoding="utf-8"))
    ixc = {c: i for i, c in enumerate(meta["cols"])}
    prod = meta["prod"]
    X = np.load(f"{CACHE}/X.npy", mmap_mode="r")
    y = np.load(f"{CACHE}/y.npy").astype(np.float32)
    season = np.load(f"{CACHE}/season.npy")
    catn = [c for c in prod if c in BASE_EMB]
    numn = [c for c in prod if c not in BASE_EMB and c != "season"]
    tr = season < VAL
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
    return catn, ncat, XC, XN, XF, y, season


def run(cfg, data, seed=42):
    name, emul, drop, wd, epochs, hid, lr, k, d = cfg
    catn, ncat, XC, XN, XF, y, season = data
    torch.manual_seed(seed)
    np.random.seed(seed)
    keep = [i for i, c in enumerate(catn)
            if emul > 0 or c not in ("pitcher_id", "batter_id")]
    cards = [(ncat[i], max(2, int(round(BASE_EMB[catn[i]] * emul)))
              if catn[i] in ("pitcher_id", "batter_id") else BASE_EMB[catn[i]])
             for i in keep]
    tr, va = season < VAL, season == VAL
    ntr, nva = int(tr.sum()), int(va.sum())
    tc = torch.from_numpy(XC[tr][:, keep])
    tn, tf = torch.from_numpy(XN[tr]), torch.from_numpy(XF[tr])
    ty = torch.from_numpy(y[tr])
    vc = torch.from_numpy(XC[va][:, keep])
    vn, vf = torch.from_numpy(XN[va]), torch.from_numpy(XF[va])
    yv = y[va].astype(np.float64)
    net = Net(cards, XN.shape[1], XF.shape[1], k, d, hid, drop)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    B = 8192
    steps = epochs * ((ntr + B - 1) // B)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
    lossf = nn.BCEWithLogitsLoss()
    best, bp = -1e18, None
    for ep in range(epochs):
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
        net.train()
        r2 = 1e5 * np.corrcoef(p.astype(np.float64), yv)[0, 1] ** 2
        if r2 > best:
            best, bp = r2, p.astype(np.float64)
    return best, bp, yv


def main():
    t0 = time.time()
    data = prep()
    print(f"준비 {time.time() - t0:.0f}s   학습 {int((data[6] < VAL).sum()):,}행"
          f" / 검증 {int((data[6] == VAL).sum()):,}행\n")
    pc = np.load(os.path.join(ROOT, "exp",
                              "valpred_cat_s3.npz"))["p"].astype(np.float64)
    S = 955.2193198652
    print(f"  {'설정':<18}{'rho^2':>9}{'세기비':>8}{'상관c':>8}"
          f"{'w(cat)':>8}{'혼합rho^2':>10}{'환산점수':>10}{'초':>6}")
    rows = []
    for cfg in CONFIGS:
        t = time.time()
        best, bp, yv = run(cfg, data)
        rc = 1e5 * np.corrcoef(pc, yv)[0, 1] ** 2
        c = float(np.corrcoef(bp, pc)[0, 1])
        w = np.linspace(0, 1, 2001)
        zc = (pc - pc.mean()) / pc.std()
        zm = (bp - bp.mean()) / bp.std()
        rb = np.array([np.corrcoef(x * zc + (1 - x) * zm, yv)[0, 1] ** 2
                       for x in w])
        j = int(np.argmax(rb))
        sc = S * 1e5 * rb[j] / rc
        rows.append((sc, cfg[0], best, c))
        print(f"  {cfg[0]:<18}{best:>9.1f}{np.sqrt(best / rc):>8.3f}{c:>8.4f}"
              f"{w[j]:>8.3f}{1e5 * rb[j]:>10.1f}{sc:>10.1f}"
              f"{time.time() - t:>6.0f}")
        np.savez_compressed(
            os.path.join(ROOT, "exp", "preds",
                         f"mlp_{cfg[0].replace(' ', '_')}.npz"),
            p=bp.astype(np.float32))
    print(f"\n=== 순위 ===")
    for sc, nm, b, c in sorted(rows, reverse=True):
        print(f"  {sc:>8.1f}  {nm:<18} rho^2={b:>7.1f}  c={c:.4f}")


if __name__ == "__main__":
    main()
