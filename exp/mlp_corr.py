r"""목적함수를 바꾼다 — Logloss 가 아니라 **상관을 직접 최대화**한다.

## 근거

우리 점수는 아핀 최적 후 정확히 `1e5 * corr(p, y)^2` 이고 (13회차가 소수점
6자리까지 확인), `corr` 은 **아핀 불변**이다. 즉 예측의 **보정 성분은 점수에
1 도 기여하지 않는다** — 뒤에 붙는 `alpha`/`center` 가 어차피 지운다.

그런데 Logloss 는 판별력과 보정을 **함께** 최적화한다. 모델 용량의 일부가
채점되지 않는 곳에 쓰이고 있다는 뜻이다. 목적함수를 `-corr` 로 바꾸면 그 용량이
판별력으로 간다.

`Opt & DFL` 덱의 논지("proxy-objective 가 아니라 실제 목적을 최적화하라")를 이
문제에 맞게 옮긴 것이다. 덱의 DFL 자체는 예측이 하류 최적화 문제로 들어갈 때의
얘기라 우리에겐 직접 적용되지 않는다 — 우리는 예측 자체로 채점받는다.

## 왜 다양성에도 유리한가

`exp/mlp_sweep.py` 가 보여준 프론티어가 문제였다 — 규제를 바꿔 세기를 올리면
상관이 같이 올라 `세기비/상관` 이 1.01~1.07 에 갇혔다. 규제는 **같은 목적함수
위에서** 해를 옮길 뿐이라 프론티어를 못 벗어난다. 목적함수를 바꾸면 해가 놓이는
곳 자체가 달라지므로 프론티어가 이동할 수 있다. CatBoost 는 상관 목적함수를
지원하지 않으므로(EXP018 은 RMSE 까지만 봤다) 이 축은 GBDT 가 못 가는 곳이다.

## 손실

    corr = <p - mean(p), y - mean(y)> / (||p - mean(p)|| * ||y - mean(y)||)
    loss = -corr

배치 안에서 계산하므로 배치가 클수록 추정이 안정적이다. `--loss mix` 는
`BCE + lam * (-corr)` 로, 상관 손실이 척도를 자유롭게 두는 것을 BCE 가 잡아준다.

    .\.venv\Scripts\python.exe -u exp\mlp_corr.py
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
S_CUR = 955.2193198652
BASE_EMB = {"pitcher_id": 32, "batter_id": 32, "pitcher_team_id": 8,
            "batter_team_id": 8, "base_state": 6, "top_bottom": 2,
            "game_type": 2, "pitcher_hand": 2, "batter_hand": 2}

# (이름, loss, lam, batch, epochs, drop, wd, lr, emul, seeds)
CONFIGS = [
    ("BCE 기준(대조)",   "bce",  0.0,  8192, 15, 0.10, 1e-5, 2e-3, 1.00, 1),
    ("corr b8192",     "corr", 0.0,  8192, 15, 0.10, 1e-5, 2e-3, 1.00, 1),
    ("corr b32768",    "corr", 0.0, 32768, 15, 0.10, 1e-5, 2e-3, 1.00, 1),
    ("corr b32768 규제", "corr", 0.0, 32768, 15, 0.30, 1e-4, 2e-3, 0.25, 1),
    ("mix lam=1",      "mix",  1.0,  8192, 15, 0.10, 1e-5, 2e-3, 1.00, 1),
    ("mix lam=4",      "mix",  4.0,  8192, 15, 0.10, 1e-5, 2e-3, 1.00, 1),
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


def neg_corr(p, y):
    pm, ym = p - p.mean(), y - y.mean()
    return -(pm * ym).sum() / (pm.norm() * ym.norm() + 1e-8)


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


def run(cfg, data, seed):
    _, loss_kind, lam, B, epochs, drop, wd, lr, emul, _ = cfg
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
    net = Net(cards, XN.shape[1], XF.shape[1], 8, 12, [512, 256], drop)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt, lr, total_steps=epochs * ((ntr + B - 1) // B))
    bce = nn.BCEWithLogitsLoss()
    best, bp = -1e18, None
    for _ in range(epochs):
        perm = torch.randperm(ntr)
        for i in range(0, ntr, B):
            b = perm[i:i + B]
            if len(b) < 64:
                continue
            opt.zero_grad()
            z = net(tc[b], tn[b], tf[b])
            if loss_kind == "bce":
                l = bce(z, ty[b])
            elif loss_kind == "corr":
                l = neg_corr(torch.sigmoid(z), ty[b])
            else:
                l = bce(z, ty[b]) + lam * neg_corr(torch.sigmoid(z), ty[b])
            l.backward()
            opt.step()
            sch.step()
        net.eval()
        with torch.no_grad():
            p = np.concatenate([torch.sigmoid(
                net(vc[i:i + 16384], vn[i:i + 16384],
                    vf[i:i + 16384])).numpy() for i in range(0, nva, 16384)])
        net.train()
        p = p.astype(np.float64)
        r2 = 1e5 * np.corrcoef(p, yv)[0, 1] ** 2
        if r2 > best:
            best, bp = r2, p
    return best, bp, yv


def judge(pm, pc, yv):
    rc = 1e5 * np.corrcoef(pc, yv)[0, 1] ** 2
    rm = 1e5 * np.corrcoef(pm, yv)[0, 1] ** 2
    c = float(np.corrcoef(pm, pc)[0, 1])
    zc = (pc - pc.mean()) / pc.std()
    zm = (pm - pm.mean()) / pm.std()
    w = np.linspace(0, 1, 2001)
    rb = np.array([np.corrcoef(x * zc + (1 - x) * zm, yv)[0, 1] ** 2
                   for x in w])
    j = int(np.argmax(rb))
    return rm, np.sqrt(rm / rc), c, float(w[j]), S_CUR * 1e5 * rb[j] / rc


def main():
    data = prep()
    pc = np.load(os.path.join(ROOT, "exp",
                              "valpred_cat_s3.npz"))["p"].astype(np.float64)
    print(f"학습 {int((data[6] < VAL).sum()):,}행 / 검증 "
          f"{int((data[6] == VAL).sum()):,}행\n")
    print(f"  {'설정':<18}{'rho^2':>9}{'세기비':>8}{'상관c':>8}"
          f"{'비/상관':>9}{'w(cat)':>8}{'환산점수':>10}{'초':>6}")
    rows = []
    for cfg in CONFIGS:
        t = time.time()
        best, bp, yv = run(cfg, data, 42)
        rm, ratio, c, wj, sc = judge(bp, pc, yv)
        rows.append((sc, cfg[0], ratio, c))
        print(f"  {cfg[0]:<18}{rm:>9.1f}{ratio:>8.3f}{c:>8.4f}"
              f"{ratio / c:>9.3f}{wj:>8.3f}{sc:>10.1f}{time.time() - t:>6.0f}")
        np.savez_compressed(
            os.path.join(ROOT, "exp", "preds",
                         f"corr_{cfg[0].replace(' ', '_').replace('=', '')}.npz"),
            p=bp.astype(np.float32))
    print(f"\n=== 순위 ===")
    for sc, nm, r, c in sorted(rows, reverse=True):
        print(f"  {sc:>8.1f}  {nm:<18} 세기비={r:.3f} 상관={c:.4f} "
              f"비/상관={r / c:.3f}")


if __name__ == "__main__":
    main()
