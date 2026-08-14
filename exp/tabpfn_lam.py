r"""TabPFN 컨텍스트 앙상블의 **상한을 결정하는 단 하나의 숫자**를 잰다.

## 왜 이 숫자 하나인가

TabPFN 예측을 `p = S + N` 으로 본다 (S = 체계적 부분, N = 컨텍스트 표본 잡음).
서로 다른 컨텍스트는 같은 S 를 공유하고 N 만 독립이다. M개를 평균하면 N 만
`1/M` 로 줄므로

    corr(p_M, y)   = corr(S,y)   * sqrt( 1 / (lam + (1-lam)/M) ) * sqrt(lam)
    corr(p_M, cat) = corr(S,cat) * sqrt( 1 / (lam + (1-lam)/M) ) * sqrt(lam)

**둘이 같은 인자로 곱해지므로 `비/상관` 은 앙상블로 안 변한다** (1.178 고정).
그러나 혼합 이득은 세기비와 상관이 **함께 커질 때** 커진다 (`exp/blend_spec.py`
표에서 위쪽 행으로 올라간다). 그 상한을 정하는 것이

    lam = corr(p_i, p_j)      서로 다른 두 컨텍스트 예측 사이의 상관

하나다. 단일 컨텍스트 실측(세기비 0.244 / 상관 0.2074)에 대입한 상한표:

    lam 0.50 -> M=inf 에서 958.0        lam 0.20 -> 963.4
    lam 0.30 ->            960.2        lam 0.10 -> 977.7      lam 0.05 -> 1137.4

**`lam` >= 0.5 면 이 축은 죽는다. 0.1 이하면 살아난다.**

## `lam` 이 작을 수 있는 근거

TabPFN 의 `corr(p, y)` 가 **0.0222** 다 (CatBoost 0.0911). 이렇게 약하면 예측
분산의 대부분이 컨텍스트 표본 잡음일 수 있다. 122만 행 중 7천 행(0.57%)만
쓴다는 것을 생각하면 그럴듯하다.

## 설계

`lam` 은 두 예측 벡터 사이의 상관이라 `y` 와의 상관보다 훨씬 정밀하게 잡힌다.
`q`=2,000 이면 `SE ~ (1-lam^2)/sqrt(n)` = 0.022 이하로 충분하다 — §10 에서
`q`=2000 이 못 쓸 것이었던 이유(`rho`=0.022 가 SE 0.022 에 묻힘)는 여기 해당 없다.

컨텍스트 3개의 **쌍별 상관 3개**를 모두 내어 평균한다. 컨텍스트 사이에
`del` + `gc` 로 메모리를 비운다 (§10 에서 두 번째 컨텍스트가 OOM 났다).

    .\.venv\Scripts\python.exe -u exp\tabpfn_lam.py
"""
import gc
import io
import itertools
import json
import os
import time

import numpy as np
import torch
from tabpfn import TabPFNClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "exp", "cache")
S_CUR = 955.2193198652
N_CTX, N_QRY, N_ENS = 7000, 2000, 3


def mult(r, c):
    w = np.linspace(0, 1, 20001)
    return float(np.max((w + (1 - w) * r) ** 2
                        / (w ** 2 + (1 - w) ** 2 + 2 * w * (1 - w) * c)))


def main():
    meta = json.load(io.open(f"{CACHE}/cols.json", encoding="utf-8"))
    ix = {c: i for i, c in enumerate(meta["cols"])}
    prod = meta["prod"]
    X = np.load(f"{CACHE}/X.npy", mmap_mode="r")
    y = np.load(f"{CACHE}/y.npy").astype(np.float64)
    season = np.load(f"{CACHE}/season.npy")
    F = np.column_stack([np.asarray(X[:, ix[c]], dtype=np.float32)
                         for c in prod])
    tr = np.where(season < 2024)[0]
    va = np.where(season == 2024)[0]
    pc_full = np.load(os.path.join(ROOT, "exp",
                                   "valpred_cat_s3.npz"))["p"].astype(np.float64)

    rng = np.random.default_rng(11)
    qs = np.sort(rng.choice(len(va), N_QRY, replace=False))
    qi = va[qs]
    Fq, yq, pcq = F[qi], y[qi], pc_full[qs]
    rc = 1e5 * np.corrcoef(pcq, yq)[0, 1] ** 2
    print(f"컨텍스트 {N_CTX:,} x {N_ENS}개   질의 {N_QRY:,}   "
          f"CatBoost rho^2 {rc:.1f}\n")

    P = []
    for e in range(N_ENS):
        t = time.time()
        ctx = rng.choice(tr, N_CTX, replace=False)
        m = TabPFNClassifier(device="cpu", n_estimators=1,
                             ignore_pretraining_limits=True, random_state=e)
        m.fit(F[ctx], y[ctx].astype(int))
        p = m.predict_proba(Fq)[:, 1].astype(np.float64)
        P.append(p)
        del m
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        av = np.mean(P, axis=0)
        rm = 1e5 * np.corrcoef(av, yq)[0, 1] ** 2
        cc = float(np.corrcoef(av, pcq)[0, 1])
        r = np.sqrt(rm / rc)
        print(f"  ctx#{e + 1}  단일 rho^2 {1e5 * np.corrcoef(p, yq)[0, 1] ** 2:>7.1f}"
              f"   | 앙상블{e + 1}: rho^2 {rm:>7.1f}  세기비 {r:.3f}  "
              f"상관 {cc:.4f}  비/상관 {r / cc:.3f}  환산 "
              f"{mult(r, cc) * S_CUR:>7.1f}   {time.time() - t:>5.0f}s",
              flush=True)
        np.savez_compressed(os.path.join(ROOT, "exp", "preds",
                                         "tabpfn_lam.npz"),
                            p=np.array(P, dtype=np.float32),
                            idx=qi.astype(np.int64))

    print(f"\n=== lam — 서로 다른 컨텍스트 예측 사이의 상관 ===")
    ls = []
    for i, j in itertools.combinations(range(len(P)), 2):
        v = float(np.corrcoef(P[i], P[j])[0, 1])
        ls.append(v)
        print(f"  corr(ctx#{i + 1}, ctx#{j + 1}) = {v:.4f}")
    lam = float(np.mean(ls))
    se = (1 - lam ** 2) / np.sqrt(N_QRY)
    print(f"  평균 lam = {lam:.4f}   (SE ~ {se:.4f})")

    r1 = np.sqrt(1e5 * np.corrcoef(P[0], yq)[0, 1] ** 2 / rc)
    c1 = float(np.corrcoef(P[0], pcq)[0, 1])
    print(f"\n=== 이 lam 에서의 컨텍스트 앙상블 상한 ===")
    print(f"  단일 실측: 세기비 {r1:.3f}  상관 {c1:.4f}")
    print(f"  {'M':>8}{'세기비':>9}{'상관':>9}{'환산점수':>11}")
    for M in (1, 2, 4, 8, 16, 32, 64, 10 ** 9):
        f = np.sqrt(1.0 / (lam + (1 - lam) / M))
        r, c = r1 * f, min(c1 * f, 0.999)
        tag = "  <- 상한" if M > 1000 else ""
        lbl = "inf" if M > 1000 else str(M)
        print(f"  {lbl:>8}{r:>9.3f}{c:>9.3f}"
              f"{mult(r, c) * S_CUR:>11.1f}{tag}")
    print(f"\n  판정 — lam>=0.5 면 축 종결, <=0.1 이면 살아난다.")


if __name__ == "__main__":
    main()
