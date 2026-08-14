r"""2번째 모델을 만든다 — 세기와 **다양성**을 동시에. 결정적 측정 1회.

## 왜 이걸 하는가

`exp/blend_spec.py` 가 낸 산술이 이 실험의 전부다.

    점수 = 1e5 * rho^2 이고 rho 는 아핀 불변 -> 보정으로는 1 도 못 움직인다.
    챔피언과 동급이고 상관 c 인 모델을 섞으면 점수 배수가 2/(1+c) 다.

        c=0.95 -> 979.7    c=0.90 -> 1005.5    c=0.85 -> 1032.7    c=0.706 -> 1120

우리 도전자들이 0 이었던 이유도 같은 표가 설명한다.

    LightGBM  세기 1.00  c=0.994  -> +2.9     (같은 모델을 두 번 돌린 셈)
    FM        세기 0.54  c=0.499  -> +1.7     (상관은 좋은데 너무 약하다)

**세기와 다양성을 동시에 가진 모델을 한 번도 안 만들었다.** 편차 계열이 제출
한 장당 +1.3 을 짜내는 동안 상관 0.90 짜리 동급 모델은 +50 이다.

## 이 MLP 가 CatBoost 와 다른 지점 (의도된 것)

  1. **ID 임베딩** — CatBoost 는 pitcher_id/batter_id 를 정수로 넣고, CTR 은
     -31.17 로 기각됐다 (EXP017). 임베딩은 셋 다와 다른 표현이다.
  2. **주기 임베딩 (PLR)** — 연속 피처를 스칼라가 아니라 벡터로 만든다.
     트리의 축평행 분할이 원리적으로 못 만드는 표현이다.
     (Gorishniy et al., On Embeddings for Numerical Features, NeurIPS 2022)
  3. **전역 상호작용** — 한 번에 모든 피처를 섞는다. 트리는 순차 분할로만 만든다.

## 판정 기준 — 이 한 번으로 축이 열리거나 닫힌다

    c >= 0.97           LightGBM 재판. 축 종결.
    c 0.85~0.95, 세기 >= 0.8   투자 확정 (+30~+80 구간)
    세기 < 0.5          FM 재판. 표현이 아니라 세기가 문제.

    .\.venv\Scripts\python.exe exp\mlp_probe.py --epochs 15
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
EMB = {"pitcher_id": 32, "batter_id": 32, "pitcher_team_id": 8,
       "batter_team_id": 8, "base_state": 6, "top_bottom": 2,
       "game_type": 2, "pitcher_hand": 2, "batter_hand": 2}


class PLR(nn.Module):
    """주기 임베딩 — 각 연속 피처를 2K 차원으로 펼친 뒤 피처별 선형사상."""

    def __init__(self, n_num, k=8, d=12, sigma=0.1):
        super().__init__()
        self.c = nn.Parameter(torch.randn(n_num, k) * sigma)
        self.w = nn.Parameter(torch.randn(n_num, 2 * k, d) / np.sqrt(2 * k))
        self.b = nn.Parameter(torch.zeros(n_num, d))
        self.d = d

    def forward(self, z):
        v = 2 * np.pi * self.c.unsqueeze(0) * z.unsqueeze(-1)
        e = torch.cat([torch.sin(v), torch.cos(v)], dim=-1)
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
        return self.head(self.mlp(torch.cat(
            e + [self.plr(xn), xn, xf], dim=1))).squeeze(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--d", type=int, default=12)
    ap.add_argument("--hid", default="512,256")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val", type=int, default=2024)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    meta = json.load(io.open(f"{CACHE}/cols.json", encoding="utf-8"))
    ixc = {c: i for i, c in enumerate(meta["cols"])}
    prod = meta["prod"]
    X = np.load(f"{CACHE}/X.npy", mmap_mode="r")
    y = np.load(f"{CACHE}/y.npy").astype(np.float32)
    season = np.load(f"{CACHE}/season.npy")
    catn = [c for c in prod if c in EMB]
    numn = [c for c in prod if c not in EMB and c != "season"]
    print(f"임베딩 {len(catn)}개 / 연속 {len(numn)}개  (season 은 제외 — "
          f"MLP 는 학습범위 밖으로 선형 외삽한다)")

    tr, va = season < a.val, season == a.val
    ntr, nva = int(tr.sum()), int(va.sum())

    # --- 범주형: 학습 구간에서 본 값만 인덱스화, 미지값은 0 번 ---
    XC = np.zeros((len(y), len(catn)), dtype=np.int64)
    cards = []
    for j, c in enumerate(catn):
        v = np.asarray(X[:, ixc[c]])
        u = np.unique(v[tr & ~np.isnan(v)])
        idx = np.searchsorted(u, v)
        idx = np.clip(idx, 0, len(u) - 1)
        ok = (~np.isnan(v)) & (u[idx] == v)
        XC[:, j] = np.where(ok, idx + 1, 0)
        cards.append((len(u) + 1, EMB[c]))

    # --- 연속형: 중앙값/IQR 로 robust 표준화 + 결측 지시자 ---
    XN = np.zeros((len(y), len(numn)), dtype=np.float32)
    flags = []
    for j, c in enumerate(numn):
        v = np.asarray(X[:, ixc[c]], dtype=np.float64)
        t = v[tr]
        m = np.isnan(v)
        q1, med, q3 = np.nanpercentile(t, [25, 50, 75])
        s = max((q3 - q1) / 1.349, 1e-6)
        XN[:, j] = np.clip(np.nan_to_num((v - med) / s, nan=0.0), -5, 5)
        if m[tr].any():
            flags.append(m.astype(np.float32))
    XF = (np.stack(flags, 1) if flags
          else np.zeros((len(y), 0), dtype=np.float32))
    print(f"결측 지시자 {XF.shape[1]}개   학습 {ntr:,}행 / 검증 {nva:,}행")

    dev = torch.device("cpu")
    tc, tn, tf = (torch.from_numpy(XC[tr]), torch.from_numpy(XN[tr]),
                  torch.from_numpy(XF[tr]))
    ty = torch.from_numpy(y[tr])
    vc, vn, vf = (torch.from_numpy(XC[va]), torch.from_numpy(XN[va]),
                  torch.from_numpy(XF[va]))
    yv = y[va].astype(np.float64)

    net = Net(cards, len(numn), XF.shape[1],
              a.k, a.d, [int(h) for h in a.hid.split(",")], a.drop).to(dev)
    npar = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * ((ntr + a.batch - 1) // a.batch)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps)
    lossf = nn.BCEWithLogitsLoss()
    print(f"파라미터 {npar:,}   총 {steps:,} step\n")

    def predict():
        net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, nva, 16384):
                out.append(torch.sigmoid(net(vc[i:i + 16384], vn[i:i + 16384],
                                             vf[i:i + 16384])).numpy())
        net.train()
        return np.concatenate(out).astype(np.float64)

    best, bp = -1e18, None
    for ep in range(a.epochs):
        t0 = time.time()
        perm = torch.randperm(ntr)
        tot = 0.0
        for i in range(0, ntr, a.batch):
            b = perm[i:i + a.batch]
            opt.zero_grad()
            l = lossf(net(tc[b], tn[b], tf[b]), ty[b])
            l.backward()
            opt.step()
            sch.step()
            tot += float(l) * len(b)
        p = predict()
        r2 = 1e5 * np.corrcoef(p, yv)[0, 1] ** 2
        if r2 > best:
            best, bp = r2, p
        print(f"  ep{ep + 1:>2}  loss {tot / ntr:.6f}  "
              f"rho^2 {r2:>8.1f}  {time.time() - t0:>5.1f}s")

    # --- 판정 ---
    pc = np.load(os.path.join(ROOT, "exp",
                              "valpred_cat_s3.npz"))["p"].astype(np.float64)
    rc = 1e5 * np.corrcoef(pc, yv)[0, 1] ** 2
    c = float(np.corrcoef(bp, pc)[0, 1])
    ratio = np.sqrt(best / rc)
    w = np.linspace(0, 1, 2001)
    zc = (pc - pc.mean()) / pc.std()
    zm = (bp - bp.mean()) / bp.std()
    rb = np.array([np.corrcoef(x * zc + (1 - x) * zm, yv)[0, 1] ** 2
                   for x in w])
    j = int(np.argmax(rb))
    print(f"\n=== 판정 ({a.val} 폴드) ===")
    print(f"  CatBoost rho^2 {rc:>8.1f}      MLP rho^2 {best:>8.1f}"
          f"      세기비 {ratio:.3f}")
    print(f"  상관 c = {c:.4f}")
    print(f"  혼합 최적 w(cat)={w[j]:.3f}  ->  rho^2 {1e5 * rb[j]:>8.1f}  "
          f"({1e5 * rb[j] - rc:+.1f} vs CatBoost)")
    print(f"  배수 {1e5 * rb[j] / rc:.4f}  ->  955.22 기준 "
          f"**{955.2193198652 * 1e5 * rb[j] / rc:.1f}**")
    if c >= 0.97:
        print("  -> LightGBM 재판. 축 종결.")
    elif ratio < 0.5:
        print("  -> FM 재판. 표현이 아니라 세기가 문제다.")
    else:
        print("  -> 투자 확정.")
    if a.out:
        np.savez_compressed(a.out, p=bp.astype(np.float32))
        print(f"  저장 {a.out}")


if __name__ == "__main__":
    main()
