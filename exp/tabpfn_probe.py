r"""마지막 미측정 모델 계열 — TabPFN 을 **부분표본으로** 잰다.

## 왜 부분표본으로 충분한가

§8~9 가 확정한 판정 기준은 두 숫자뿐이다.

    세기비 = rho(TabPFN) / rho(CatBoost)      상관 c = corr(TabPFN, CatBoost)

둘 다 상관계수라서 25만 행이 필요 없다. `rho`~0.09 에서 `SE(corr) ~ 1/sqrt(n)`
이므로 `n`=10,000 이면 SE 0.010 — 세기비 0.5 와 0.7 을 가르기에 충분하다.
로컬 GPU 가 GT 1030(2GB)이라 전체 폴드를 못 돌린다는 것은 **측정을 못 한다는
뜻이 아니었다.**

## 판정 기준 (`exp/blend_spec.py`)

혼합이 이득이 되려면 `세기비 > 상관` 이어야 하고, 지금까지 관측된 상관은 전부
0.66~0.79 였다. 즉 **세기비가 최소 0.8 근처는 돼야 한다.**

    세기비 0.70 + 상관 0.50 -> 1006      세기비 0.75 + 상관 0.50 -> 1035
    세기비 0.50 + 상관 0.50 ->  955      (이득 0)

## 🚩 오늘 배운 함정이 여기에도 걸린다

데이터를 적게 써서 생기는 낮은 상관은 **다양성이 아니라 잡음**이고, 그때는 세기도
같이 떨어져 이득이 0 이다 (§9-a, MLP 15설정). TabPFN 은 122만 행 중 1만 행만
쓰므로 정확히 그 위험에 있다. 그래서 **컨텍스트 앙상블**을 같이 잰다 — 서로 다른
1만 행 컨텍스트를 여러 개 평균하면 잡음은 줄고 귀납편향은 남는다. 이때
세기비가 상관보다 빠르게 오르면 축이 열리고, 같이 오르면 §9-a 의 재판이다.

## 라이선스 (규정 1)

`tabpfn` 8.3.0 은 가중치 다운로드에 **브라우저 로그인 + 라이선스 동의**를 요구한다
("누구에게나 공개"와 부딪힐 소지). 강의 노트북이 지정한 **2.0.9** 는 게이트 없이
받아진다. 채택 시 라이선스 조항 확인이 별도로 필요하다.

    .\.venv\Scripts\python.exe -u exp\tabpfn_probe.py
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
N_CTX = 10000      # TabPFN v2 의 사전학습 상한
N_QRY = 10000      # SE(corr) ~ 0.010
N_ENS = 3          # 서로 다른 컨텍스트 개수


def blend(pm, pc, yv):
    rc = 1e5 * np.corrcoef(pc, yv)[0, 1] ** 2
    rm = 1e5 * np.corrcoef(pm, yv)[0, 1] ** 2
    c = float(np.corrcoef(pm, pc)[0, 1])
    zc = (pc - pc.mean()) / pc.std()
    zm = (pm - pm.mean()) / pm.std()
    w = np.linspace(0, 1, 2001)
    rb = np.array([np.corrcoef(x * zc + (1 - x) * zm, yv)[0, 1] ** 2
                   for x in w])
    j = int(np.argmax(rb))
    return rm, np.sqrt(rm / rc), c, float(w[j]), S_CUR * rb[j] * 1e5 / rc


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

    rng = np.random.default_rng(42)
    q = np.sort(rng.choice(len(va), N_QRY, replace=False))
    qi = va[q]
    Fq, yq, pcq = F[qi], y[qi], pc_full[q]
    rc = 1e5 * np.corrcoef(pcq, yq)[0, 1] ** 2
    print(f"질의 {N_QRY:,}행 (2024 폴드에서 추출)   "
          f"CatBoost rho^2 {rc:.1f}   SE(corr)~{1 / np.sqrt(N_QRY):.4f}")
    print(f"컨텍스트 {N_CTX:,}행 x {N_ENS}개  (학습 가능 {len(tr):,}행 중 "
          f"{N_CTX * N_ENS / len(tr):.1%})\n")

    print(f"  {'구성':<16}{'rho^2':>9}{'세기비':>8}{'상관c':>8}{'비/상관':>9}"
          f"{'w(cat)':>8}{'환산점수':>10}{'초':>7}")
    acc = np.zeros(N_QRY)
    for e in range(N_ENS):
        t = time.time()
        ctx = rng.choice(tr, N_CTX, replace=False)
        m = TabPFNClassifier(device="cpu", n_estimators=1,
                             ignore_pretraining_limits=True, random_state=e)
        m.fit(F[ctx], y[ctx].astype(int))
        p = m.predict_proba(Fq)[:, 1].astype(np.float64)
        acc += p
        for lbl, pp in ((f"단일 #{e + 1}", p),
                        (f"앙상블 {e + 1}개", acc / (e + 1))):
            rm, ratio, c, wj, sc = blend(pp, pcq, yq)
            print(f"  {lbl:<16}{rm:>9.1f}{ratio:>8.3f}{c:>8.4f}"
                  f"{ratio / c:>9.3f}{wj:>8.3f}{sc:>10.1f}"
                  f"{time.time() - t if lbl.startswith('단일') else 0:>7.0f}",
                  flush=True)
        np.savez_compressed(os.path.join(ROOT, "exp", "preds",
                                         "tabpfn_2024_sub.npz"),
                            p=(acc / (e + 1)).astype(np.float32),
                            idx=qi.astype(np.int64))

    print(f"\n=== 판정 ===")
    rm, ratio, c, wj, sc = blend(acc / N_ENS, pcq, yq)
    print(f"  세기비 {ratio:.3f}  상관 {c:.4f}  비/상관 {ratio / c:.3f}"
          f"  ->  {sc:.1f}")
    if ratio > c:
        print("  -> 혼합 이득 있음. 축 열림.")
    else:
        print("  -> 세기비 <= 상관. 최적 혼합가중 0. 축 종결.")


if __name__ == "__main__":
    main()
