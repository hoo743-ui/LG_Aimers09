r"""TabPFN 재측정 — 1차 숫자를 믿을 수 없어서 제대로 잰다.

## 1차에서 나온 것 (`exp/tabpfn_probe.log`, 스모크)

    ctx=3000  rho^2  26.5      ctx=5000  rho^2  56.9      ctx=7000  rho^2 465.1
    CatBoost (2024 부분표본)   rho^2 826.9   ->  ctx=7000 의 세기비 0.750

세기비 0.750 은 필요치(0.8)에 거의 닿는 값이고, 그것도 122만 행 중 **7천 행
(0.57%)** 으로 나왔다. 그러나 두 가지 이유로 그대로 믿을 수 없다.

  1. 질의 2,000행은 `SE(corr)`=0.022 인데 `rho`=0.068 이다. 95% 구간이
     `rho^2` 로 [64, 1235] — 465.1 은 그 안 어디든 될 수 있다.
  2. 5000 -> 7000 이 8배는 매끄러운 곡선이 아니다. 표본 잡음일 공산이 크다.

그리고 **상관 `c` 를 아직 안 쟀다.** `세기비 > 상관` 이 판정 기준인데 절반만 있다.

## 설계

  - 컨텍스트 7,000 (로컬 RAM 상한. `ctx^2*24*4B` 가 10.6GB 를 못 넘는다)
  - 질의 6,000 을 2,000씩 3청크. **청크마다 누적 결과를 찍는다** — q=2000 ->
    4000 -> 6000 으로 신뢰구간이 좁혀지는 것을 보며 일찍 판단할 수 있다.
  - 독립 컨텍스트 2개. 같은 질의 행에 대해 예측하므로 앙상블도 잰다.

## 판정

    세기비 <= 상관        -> 최적 혼합가중 0. 축 종결.
    앙상블로 세기비만 오름 -> 진짜 다른 귀납편향. 축 열림.
    둘이 같이 오름        -> §9-a(MLP 15설정)의 재판. 잡음이었다는 뜻.

    .\.venv\Scripts\python.exe -u exp\tabpfn_probe2.py
"""
import io
import json
import os
import time

import numpy as np
from tabpfn import TabPFNClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "exp", "cache")
S_CUR = 955.2193198652
N_CTX, N_QRY, CHUNK, N_ENS = 7000, 6000, 2000, 2


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
    return rc, rm, np.sqrt(rm / rc), c, float(w[j]), S_CUR * rb[j] * 1e5 / rc


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

    rng = np.random.default_rng(7)
    qsel = np.sort(rng.choice(len(va), N_QRY, replace=False))
    qi = va[qsel]
    Fq, yq, pcq = F[qi], y[qi], pc_full[qsel]
    print(f"컨텍스트 {N_CTX:,} x {N_ENS}개   질의 {N_QRY:,} ({CHUNK}씩 "
          f"{N_QRY // CHUNK}청크)   학습가능 {len(tr):,}행 중 "
          f"{N_CTX / len(tr):.2%}\n")
    print(f"  {'구성':<20}{'q':>6}{'CatB':>8}{'TabPFN':>9}{'세기비':>8}"
          f"{'상관c':>8}{'비/상관':>9}{'환산':>9}{'초':>7}")

    preds = []
    for e in range(N_ENS):
        ctx = rng.choice(tr, N_CTX, replace=False)
        m = TabPFNClassifier(device="cpu", n_estimators=1,
                             ignore_pretraining_limits=True, random_state=e)
        m.fit(F[ctx], y[ctx].astype(int))
        got = np.zeros(0)
        for k in range(N_QRY // CHUNK):
            t = time.time()
            sl = slice(k * CHUNK, (k + 1) * CHUNK)
            got = np.concatenate([got,
                                  m.predict_proba(Fq[sl])[:, 1].astype(np.float64)])
            n = len(got)
            rc, rm, ratio, c, wj, sc = judge(got, pcq[:n], yq[:n])
            print(f"  {f'ctx#{e + 1} 누적':<20}{n:>6}{rc:>8.1f}{rm:>9.1f}"
                  f"{ratio:>8.3f}{c:>8.4f}{ratio / c:>9.3f}{sc:>9.1f}"
                  f"{time.time() - t:>7.0f}", flush=True)
        preds.append(got)
        np.savez_compressed(os.path.join(ROOT, "exp", "preds",
                                         "tabpfn2_2024_sub.npz"),
                            p=np.array(preds, dtype=np.float32),
                            idx=qi.astype(np.int64))
        if e:
            av = np.mean(preds, axis=0)
            rc, rm, ratio, c, wj, sc = judge(av, pcq, yq)
            print(f"  {f'앙상블 {e + 1}개':<20}{len(av):>6}{rc:>8.1f}{rm:>9.1f}"
                  f"{ratio:>8.3f}{c:>8.4f}{ratio / c:>9.3f}{sc:>9.1f}"
                  f"{0:>7.0f}", flush=True)

    av = np.mean(preds, axis=0)
    rc, rm, ratio, c, wj, sc = judge(av, pcq, yq)
    print(f"\n=== 판정 ===")
    print(f"  세기비 {ratio:.3f}   상관 {c:.4f}   비/상관 {ratio / c:.3f}")
    print(f"  혼합 최적 w(cat)={wj:.3f}  ->  환산 {sc:.1f} ({sc - S_CUR:+.1f})")
    print("  -> " + ("혼합 이득 있음. 축 열림." if ratio > c else
                     "세기비 <= 상관. 최적 혼합가중 0. 축 종결."))
    print(f"\n  주의 — 질의 {N_QRY:,}행의 SE(corr)={1 / np.sqrt(N_QRY):.4f} 이므로 "
          f"세기비 오차는 대략 +-{1 / np.sqrt(N_QRY) / (np.sqrt(rm / 1e5)) / 2:.2f} 다.")


if __name__ == "__main__":
    main()
