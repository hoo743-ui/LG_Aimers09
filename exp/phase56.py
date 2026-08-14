r"""PHASE 5-6 — 시즌 전이 구조와 **정보 상한**. 모델 학습 0회.

## Phase 5 가 답할 것

"2025 의 값을 직접 예측" 대신 "선수의 다음 시즌 변화량을 예측"으로 재정의할 수
있는가.

### 🚩 먼저 짚어야 할 구조적 사실

평가 시점에 2025 선수 상태로 쓸 수 있는 정보는 그 행의 `asof_*` 컬럼뿐인데,
**주최측이 그것을 2025 시즌 내부 as-of 로 이미 계산해서 준다.**

    TEST_000001  asof_pitcher_n=3465  asof_pitcher_success_rate=0.489466

즉 "선수의 2025 상태"는 **예측 대상이 아니라 관측값**이다. 다음 시즌 변화량을
예측하는 구조는 이미 주어진 것을 다시 추정하는 셈이라 이득이 없다.

그래서 Phase 5 의 실제 질문은 이렇게 바뀐다 —
**투수의 어떤 특성이 시즌을 넘어 지속되는가?** 지속되는 것만 과거 시즌에서
배울 가치가 있고, 지속되지 않는 것은 `asof_*` 로만 잡아야 한다.

## Phase 6 가 답할 것

955 -> 1150 의 격차가 ① 아직 안 쓴 정보 ② 잘못된 문제 정의 ③ 우리가 못 쓰는
추가 정보 중 무엇인가.

**in-fold 오라클**로 정보 상한을 잰다 — 2024 의 정답으로 2024 의 셀 평균을 만들어
2024 에 적용한다. 이는 그 분할이 담을 수 있는 **이론적 최대치**다 (실전에서는
불가능하다). 같은 분할을 out-of-fold(<2024)로 만든 것과 비교하면

    in-fold 오라클      그 분할이 담은 정보의 총량
    out-of-fold 실측    그중 시즌을 넘어 실제로 전이된 몫
    비율                전이 효율

로 분해된다. 격차가 **정보 부족**인지 **전이 실패**인지가 여기서 갈린다.

    .\.venv\Scripts\python.exe -u exp\phase56.py
"""
import io
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "exp", "cache")


def grp_mean(key, val, min_n=1):
    """키별 평균과 개수."""
    o = np.argsort(key, kind="stable")
    k, v = key[o], val[o]
    u, s = np.unique(k, return_index=True)
    cnt = np.diff(np.append(s, len(k)))
    mean = np.add.reduceat(v, s) / cnt
    return u, mean, cnt


def lookup(u, val, cnt, keys, fallback, min_n=0):
    ix = np.clip(np.searchsorted(u, keys), 0, len(u) - 1)
    ok = (u[ix] == keys) & (cnt[ix] >= min_n)
    out = np.full(len(keys), fallback, dtype=np.float64)
    out[ok] = val[ix[ok]]
    return out


def rho2(p, y):
    return 1e5 * np.corrcoef(p, y)[0, 1] ** 2


def main():
    meta = json.load(io.open(f"{CACHE}/cols.json", encoding="utf-8"))
    ixc = {c: i for i, c in enumerate(meta["cols"])}
    X = np.load(f"{CACHE}/X.npy", mmap_mode="r")
    y = np.load(f"{CACHE}/y.npy").astype(np.float64)
    season = np.load(f"{CACHE}/season.npy")
    C = lambda n: np.asarray(X[:, ixc[n]], dtype=np.float64)

    P = C("pitcher_id").astype(np.int64)
    BH = C("batter_hand").astype(np.int64)
    PHD = C("pitcher_hand").astype(np.int64)
    BB, SS = C("balls_before").astype(np.int64), C("strikes_before").astype(np.int64)
    OB = (C("num_runners_on") > 0).astype(np.int64)
    CNT = BB * 4 + SS
    PH = P * 10 + BH
    PHC = PH * 100 + CNT
    PHCO = PHC * 10 + OB
    HP4 = PHD * 2 + BH

    # ================= PHASE 5 =================
    print("=" * 78)
    print("PHASE 5 — 투수 특성은 시즌을 넘는가 (n>=200 투수만)")
    print("=" * 78)
    print("  두 시즌 모두 200구 이상인 투수의 특성값 상관.")
    print("  corr 이 높을수록 그 특성은 '투수의 정체성'이고 과거 시즌에서 배울 수 있다.")
    print("  낮으면 시즌마다 새로 생기는 것이라 asof_* 로만 잡아야 한다.\n")

    def per_season_stat(f, mask_extra=None):
        m = season == f
        if mask_extra is not None:
            m = m & mask_extra
        u, mean, cnt = grp_mean(P[m], y[m])
        return dict(zip(u[cnt >= 200], mean[cnt >= 200]))

    print(f"  {'전이':<14}{'투수수':>7}{'성공률 corr':>13}{'기울기':>9}"
          f"{'플래툰차 corr':>14}")
    for a, b in zip(range(2019, 2024), range(2020, 2025)):
        sa, sb = per_season_stat(a), per_season_stat(b)
        common = sorted(set(sa) & set(sb))
        if not common:
            continue
        va = np.array([sa[k] for k in common])
        vb = np.array([sb[k] for k in common])
        r = float(np.corrcoef(va, vb)[0, 1])
        slope = float(np.polyfit(va, vb, 1)[0])
        # 플래툰 차 = (같은손 상대 성공률) - (다른손 상대 성공률)
        pl = {}
        for f_, dst in ((a, "a"), (b, "b")):
            same = per_season_stat(f_, PHD == BH)
            diff = per_season_stat(f_, PHD != BH)
            pl[dst] = {k: same[k] - diff[k] for k in set(same) & set(diff)}
        cp = sorted(set(pl["a"]) & set(pl["b"]))
        rp = (float(np.corrcoef([pl["a"][k] for k in cp],
                                [pl["b"][k] for k in cp])[0, 1])
              if len(cp) > 5 else float("nan"))
        print(f"  {a}->{b:<9}{len(common):>7}{r:>13.4f}{slope:>9.4f}"
              f"{rp:>14.4f}")

    # ================= PHASE 6 =================
    print()
    print("=" * 78)
    print("PHASE 6 — 정보 상한 (검증 폴드 2024)")
    print("=" * 78)
    f = 2024
    tr, va = season < f, season == f
    yv = y[va]
    base = float(yv.mean())
    print(f"  2024 {int(va.sum()):,}행   성공률 {base:.6f}")
    print(f"  Champion(후처리) 796.8   최고후보(CAAFE11+후처리) 821.3\n")

    PARTS = [("투수", P), ("투수x타자손", PH), ("손조합4", HP4),
             ("투수x손x카운트", PHC), ("투수x손x카운트x주자", PHCO)]
    print(f"  {'분할':<22}{'셀수':>9}{'in-fold 오라클':>15}"
          f"{'out-of-fold':>13}{'전이효율':>10}")
    rows = []
    for nm, key in PARTS:
        u_i, m_i, c_i = grp_mean(key[va], yv)
        p_in = lookup(u_i, m_i, c_i, key[va], base)
        u_o, m_o, c_o = grp_mean(key[tr], y[tr])
        p_out = lookup(u_o, m_o, c_o, key[va], base, min_n=20)
        a, b = rho2(p_in, yv), rho2(p_out, yv)
        rows.append((nm, len(u_i), a, b))
        print(f"  {nm:<22}{len(u_i):>9,}{a:>15.1f}{b:>13.1f}"
              f"{b / a:>10.3f}")

    print(f"\n  * in-fold 오라클 = 2024 정답으로 2024 셀평균을 만들어 적용."
          f" 실전 불가, 상한값.")
    print(f"  * out-of-fold = 같은 분할을 <2024 로 만들어 적용. 실제로 전이된 몫.")

    print(f"\n=== 격차 분해 ===")
    best = max(rows, key=lambda r: r[2])
    print(f"  가장 정보량이 큰 분할: {best[0]}  in-fold {best[2]:.1f}")
    print(f"  현재 최고후보(CAAFE11+후처리)        821.3")
    print(f"  1150 이 요구하는 rho^2                1150.0")
    print(f"    (점수=1e5*rho^2 이고 rho 는 아핀 불변이므로 폴드 rho^2 와 직접 비교 가능)")
    for tgt in (1000, 1050, 1150):
        print(f"    {tgt} 까지 필요한 배수  {tgt / 821.3:.4f}")


if __name__ == "__main__":
    main()
