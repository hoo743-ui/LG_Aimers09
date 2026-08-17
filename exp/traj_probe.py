r"""궤적(trajectory) 진단 — **같은 현재 상태라도 어떤 경로로 왔는가가 중요한가.**

학습 0회. 오라클(A) / 시즌 간 전이(B) / 모델 잔차(C)를 **분리해서** 잰다.

## 구조 — 네 점은 시간축의 점이 아니라 중첩된 창이다

    cur (시즌 전체 누적)  ⊃  prev5  ⊃  prev3  ⊃  prev1

최신성이 커지는 순서가 `cur -> p5 -> p3 -> p1` 이므로, 세 증분

    d1 = p5 - cur      최근 5경기가 시즌 평균과 얼마나 다른가
    d2 = p3 - p5       더 좁히면 더 좋아지는가
    d3 = p1 - p3       가장 최근 한 경기

의 **부호 패턴 8가지**가 경로 모양이다. `이산 전환은 통하고 매끄러운 곱은 안
통한다`(CLAUDE 4절)에 맞춰 연속 차분이 아니라 **모양 등급**을 1급 표현으로 둔다.

## 기존 기각과 무엇이 다른가

F(`prev - cur`)는 **증분 하나**를 연속값으로 준 것이고 LB 전이 −60% 였다.
CAAFE·TREND·W2/W3 도 전부 차분 하나 또는 창 확장이다. 여기서 묻는 것은

    같은 cur 에서, 올라오며 도달했는가 / 내려오며 도달했는가 / 평평했는가

이고, 검증도 값의 자기상관이 아니라 **관계의 재현**으로 한다 (7번 지침).

## 세 단계를 분리한다

    A 오라클     그 폴드 라벨로 교차적합한 상한 (실현 가능치 아님)
    B 전이       s 시즌에서 만든 [모양 -> 잔차] 표를 s+1 에 적용했을 때의 실제 이득
    C 잔차신호   연속 성분들과 모델 잔차의 상관 (SE 대비)

**B 가 충분하지 않으면 학습하지 않는다.**

    .\.venv\Scripts\python.exe -u exp\traj_probe.py
"""
import io
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "exp"))
from path_alloc import build_df                            # noqa: E402
from resid_table import post_for                           # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

KS = [0, 50, 200, 1000, 5000, 20000]
FOLDS = (2022, 2023, 2024)


def r2(p, y):
    return 1e5 * np.corrcoef(p, y)[0, 1] ** 2


def cells(keys, vals, k):
    u, inv = np.unique(keys, return_inverse=True)
    n = np.bincount(inv, minlength=len(u)).astype(np.float64)
    s = np.bincount(inv, weights=vals, minlength=len(u))
    return u, (s / np.maximum(n, 1)) * (n / (n + k)), n


def look(u, tab, keys):
    ix = np.clip(np.searchsorted(u, keys), 0, max(len(u) - 1, 0))
    ok = u[ix] == keys
    out = np.zeros(len(keys))
    out[ok] = tab[ix[ok]]
    return out


def cv2(keys, pred, y, half):
    """그 폴드 라벨로 교차적합한 상한. 축소를 훑어 최댓값을 쓴다."""
    res = y - pred
    b, best, bk = r2(pred, y), -1e9, None
    for k in KS:
        add = np.zeros(len(y))
        for m in (half, ~half):
            u, tab, _ = cells(keys[m], res[m], k)
            add[~m] = look(u, tab, keys[~m])
        v = r2(pred + add, y) - b
        if v > best:
            best, bk = v, k
    return best, bk, len(np.unique(keys))


def build_traj(tr, kind, lbl):
    """네 점 경로와 그 모양 성분. 결측은 별도 등급으로 뺀다."""
    g = lambda c: tr[c].to_numpy(np.float64)
    cur = g(f"cur_{lbl}")
    p5 = g(f"asof_pitcher_prev5_game_{kind}_rate")
    p3 = g(f"asof_pitcher_prev3_game_{kind}_rate")
    p1 = g(f"asof_pitcher_prev1_game_{kind}_rate")
    d1, d2, d3 = p5 - cur, p3 - p5, p1 - p3
    ok = np.isfinite(d1) & np.isfinite(d2) & np.isfinite(d3)
    # 모양 등급 — 세 증분의 부호 8가지. 결측은 8번.
    shape = np.where(ok, 4 * (d1 > 0) + 2 * (d2 > 0) + (d3 > 0), 8).astype(np.int64)
    mono = np.where(ok, np.where((d1 > 0) & (d2 > 0) & (d3 > 0), 2,
                                 np.where((d1 < 0) & (d2 < 0) & (d3 < 0), 0, 1)),
                    3).astype(np.int64)      # 단조상승 2 / 혼합 1 / 단조하락 0
    rev = np.where(ok, ((np.sign(d2) != np.sign(d3)) & (d2 != 0)
                        & (d3 != 0)).astype(np.int64), 2)   # 반전 여부
    P = np.column_stack([p5, p3, p1])
    vol = np.where(ok, np.nanstd(P, axis=1), np.nan)         # 변동성
    curv = np.where(ok, d3 - d2, np.nan)                     # 가속도
    slope = np.where(ok, p1 - cur, np.nan)                   # 전체 기울기 (= F)
    conc = np.where(ok, np.abs(d3) / (np.abs(d1) + np.abs(d2) + np.abs(d3) + 1e-9),
                    np.nan)                                  # 최근 변화 집중도
    return dict(cur=cur, shape=shape, mono=mono, rev=rev, vol=vol,
                curv=curv, slope=slope, conc=conc, ok=ok)


def dec(x, m, q=10):
    v = np.where(np.isfinite(x), x, np.nan)
    qs = np.nanquantile(v[m], np.linspace(0, 1, q + 1)[1:-1])
    return np.searchsorted(qs, np.nan_to_num(v, nan=np.nanmedian(v[m])))


def main():
    tr = build_df()
    season = tr["season"].to_numpy()
    y = tr["control_success"].to_numpy(np.float64)
    T = {lbl: build_traj(tr, kind, lbl)
         for kind, lbl in (("success", "succ"), ("middle", "mid"))}
    pv, res = {}, {}
    for f in FOLDS:
        m = season == f
        P = np.load(os.path.join(ROOT, "exp", f"prod_champ_{f}.npy"))
        pv[f] = P[:3].mean(0) + post_for(tr, y, season < f, m)
        res[f] = y[m] - pv[f]
        print(f"  폴드 {f} 기준선 rho^2 = {r2(pv[f], y[m]):.1f}", flush=True)
    print(f"  경로 결측률 (succ) {1 - T['succ']['ok'].mean():.1%}\n", flush=True)

    out = {}
    m24 = season == 2024
    rng = np.random.default_rng(0)
    half = rng.random(int(m24.sum())) < 0.5

    print("=" * 88)
    print("0. 경로 모양의 분포와 타깃 (폴드 2024) — 같은 cur 에서 갈리는가")
    print("=" * 88)
    S = T["succ"]
    sh24, cur24 = S["shape"][m24], S["cur"][m24]
    cd = dec(S["cur"], m24)[m24]
    print(f"{'모양(d1d2d3 부호)':<20}{'행 비율':>9}{'평균 y':>9}{'평균 cur':>10}"
          f"{'잔차평균':>10}")
    names = {0: "--- 하락일관", 1: "--+", 2: "-+-", 3: "-++",
             4: "+--", 5: "+-+", 6: "++-", 7: "+++ 상승일관", 8: "결측"}
    for s in range(9):
        m = sh24 == s
        if m.sum() < 500:
            continue
        print(f"  {names[s]:<18}{m.mean():>9.1%}{y[m24][m].mean():>9.4f}"
              f"{np.nanmean(cur24[m]):>10.4f}{res[2024][m].mean():>+10.4f}")

    print()
    print("=" * 88)
    print("A. 오라클 상한 (폴드 2024, 교차적합) — 정보가 있는가")
    print("=" * 88)
    GR = {
        "cur 십분위 (기준)": cd,
        "모양 8등급": sh24,
        "단조성 3등급": S["mono"][m24],
        "반전 지시자": S["rev"][m24],
        "cur 십분위 x 모양": cd * 10 + sh24,
        "cur 십분위 x 단조성": cd * 10 + S["mono"][m24],
        "기울기 십분위": dec(S["slope"], m24)[m24],
        "곡률 십분위": dec(S["curv"], m24)[m24],
        "변동성 십분위": dec(S["vol"], m24)[m24],
        "최근집중도 십분위": dec(S["conc"], m24)[m24],
        "mid 모양 8등급": T["mid"]["shape"][m24],
        "succ모양 x mid모양": sh24 * 10 + T["mid"]["shape"][m24],
    }
    print(f"{'그룹':<28}{'셀수':>7}{'cv2 상한':>11}{'최적 k':>8}")
    for n, k in GR.items():
        v, bk, nc = cv2(k, pv[2024], y[m24], half)
        out[f"A|{n}"] = dict(gain=float(v), k=bk, cells=nc)
        print(f"{n:<28}{nc:>7}{v:>+11.1f}{bk:>8}")

    print()
    print("=" * 88)
    print("B. 시즌 간 전이 — [모양 -> 잔차] 표가 다음 시즌에 통하는가")
    print("=" * 88)
    print("  (관계의 재현. 값의 자기상관이 아니다)")
    RUNS = [((2022,), 2023), ((2023,), 2024), ((2022, 2023), 2024)]
    KEYS = {"모양 8등급": S["shape"], "단조성 3등급": S["mono"],
            "cur 십분위 x 모양": dec(S["cur"], season < 2025) * 10 + S["shape"],
            "기울기 십분위": dec(S["slope"], season < 2025),
            "곡률 십분위": dec(S["curv"], season < 2025),
            "mid 모양 8등급": T["mid"]["shape"]}
    print(f"{'원천->목표':<16}{'그룹':<22}{'최적 k':>8}{'이득':>9}{'k=0':>9}")
    for src, tgt in RUNS:
        m_t = season == tgt
        base = r2(pv[tgt], y[m_t])
        rsrc = np.concatenate([res[s] for s in src])
        for n, keyall in KEYS.items():
            ks = np.concatenate([keyall[season == s] for s in src])
            best, bk, v0 = -1e9, None, None
            for k in KS:
                u, tab, _ = cells(ks, rsrc, k)
                v = r2(pv[tgt] + look(u, tab, keyall[m_t]), y[m_t]) - base
                if k == 0:
                    v0 = v
                if v > best:
                    best, bk = v, k
            out[f"B|{'+'.join(map(str, src))}->{tgt}|{n}"] = dict(
                gain=float(best), k=bk, k0=float(v0))
            print(f"{'+'.join(map(str, src)) + '->' + str(tgt):<16}{n:<22}"
                  f"{bk:>8}{best:>+9.1f}{v0:>+9.1f}")
        print()

    print("=" * 88)
    print("B-2. 관계 자체의 안정성 — 시즌별 [모양 -> 잔차] 벡터의 상관")
    print("=" * 88)
    tabs = {}
    for f in FOLDS:
        u, tab, n = cells(S["shape"][season == f], res[f], 200)
        tabs[f] = pd.Series(tab, index=u)
    print(f"  2022~2023 상관 {np.corrcoef(tabs[2022], tabs[2023])[0,1]:+.3f}"
          f"   2023~2024 {np.corrcoef(tabs[2023], tabs[2024])[0,1]:+.3f}"
          f"   2022~2024 {np.corrcoef(tabs[2022], tabs[2024])[0,1]:+.3f}")
    for f in FOLDS:
        print(f"  {f} 셀 잔차평균: "
              + " ".join(f"{v:+.4f}" for v in tabs[f].to_numpy()))

    print()
    print("=" * 88)
    print("C. 모델 잔차 신호 — 연속 성분 (SE 대비)")
    print("=" * 88)
    print(f"{'성분':<16}" + "".join(f"{f:>12}" for f in FOLDS))
    for nm in ("slope", "curv", "vol", "conc"):
        line = f"  {nm:<14}"
        for f in FOLDS:
            m = season == f
            x, rr = T["succ"][nm][m], res[f]
            o = np.isfinite(x)
            c = float(np.corrcoef(x[o], rr[o])[0, 1])
            se = 1 / np.sqrt(o.sum())
            line += f"{c:>+9.4f}({c/se:>+.1f})"[:12].rjust(12)
            out[f"C|{nm}|{f}"] = dict(corr=c, se_mult=c / se)
        print(line)

    json.dump(out, io.open(os.path.join(ROOT, "exp", "traj_probe.json"), "w",
                           encoding="utf-8"), indent=1, ensure_ascii=False)
    print("\n판정 — B(전이)가 2024 에서 손익분기(+19.8점) 대비 의미 있는 양수가 "
          "아니면 학습하지 않는다. A 는 실현 가능치가 아니다.")


if __name__ == "__main__":
    main()
