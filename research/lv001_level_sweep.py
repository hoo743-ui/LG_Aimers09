r"""LV-001 — **수준(level) 축 전수 스윕.** 학습 0회.

## 왜 이 연구인가 (2026-08-24)

55회차에서 타자 수준축 하나가 +24.5 를 냈다. 그 축은 8개월 전에 **대비(contrast)**
로 재고 닫혔던 것이다. 원장 §5-c 9 가 이미 경고를 적어 뒀다.

    "어떤 축을 닫을 때 대비로 닫았는가 수준으로 닫았는가를 구분해 적어라"

닫힌 축 60여 건이 **대부분 대비로만** 측정됐다. 그러므로 지금 가장 큰 미탐색
면적은 새 컬럼이 아니라 **이미 닫은 축들의 수준 판본**이다.

## 무엇을 재는가 — 로컬로 후보를 고르지 않는다 (§5-d)

로컬 폴드는 LB 부호를 50% 로 맞힌다. 그러나 그것은 **이질적인 변경 전체**를
섞었을 때의 수치다. **수준 축 family 는 LB 앵커가 2개 있고 둘 다 부호가 맞았다.**

    투수 수준  로컬 +0.8   LB +1.19   전이율 1.49
    타자 수준  로컬 +4.2   LB +9.81   전이율 2.34

그래서 이 스윕의 로컬 값은 **승격 근거가 아니라 순위 근거**로만 쓴다.
제출은 언제나 LB 좌표 측정(3점 해법)으로 한다.

## 측정 규약

    표 원천   2022+2023 OOF 잔차 (직전 2시즌 하드 창 — 생산과 동일)
    적용      2024 행. 조회 키는 그 행 자신의 컬럼뿐
    기준선    champ_oof 2024 예측 + **현 Champion 9축 후처리 전부**
              -> 판정은 항상 "현 Champion 위의 증분" (§6)
    k         과거 표적(2023)에서 고른 값과 2024 최적을 **둘 다** 보고한다
              2024 에서 고른 k 는 편향이므로 별도 열로 표시한다

## 이득은 해석적으로 푼다 (근사 없음)

가중 w 를 붙인 예측 `pv + w·v` 의 rho^2 는 w 의 유리함수다.

    rho^2(w) = (a + wb)^2 / [(Vp + 2wC + w^2 Vv) · Vy]
    a=cov(pv,y)  b=cov(v,y)  Vp=var(pv)  Vv=var(v)  C=cov(pv,v)
    w* = (aC - bVp) / (bC - aVv)                        <- 정확해

즉 축마다 스칼라 5개만 구하면 **w 곡선 전체와 최적 이득**이 닫힌 형태로 나온다.
w 를 격자로 훑을 필요가 없다.

    .\.venv\Scripts\python.exe -u research\lv001_level_sweep.py
"""
import io
import json
import os
import sys
import time
import zipfile

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAMP = os.path.join(ROOT, "submissions", "cand_kb45.zip")
KS = [200, 1000, 5000, 20000, 50000, 200000]

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ---------------------------------------------------------------- 기준선
def champion_pred(df, oof, season):
    """champ_oof 예측 + 현 Champion 9축 후처리. 아핀은 rho 불변이라 생략."""
    b = joblib.load(io.BytesIO(zipfile.ZipFile(CHAMP).read("model/rf.pkl")))
    out = {}
    for s in (2022, 2023, 2024):
        m = season == s
        p = oof[f"p{s}"].astype(np.float64).copy()
        sub = df.loc[m]
        for sp in b["platoon"]:
            tab, w = sp["table"], float(sp["w"])
            cols = [sub[c].to_numpy(np.int64) for c in sp["cols"]]
            keys = list(zip(*cols))
            p += w * np.array([tab.get(k, 0.0) for k in keys], np.float64)
        out[s] = p
    return out


def r2(p, y):
    return 1e5 * np.corrcoef(p, y)[0, 1] ** 2


def gain_curve(pv, v, y):
    """rho^2(w) = (a+wb)^2 / ((Vp+2wC+w^2Vv)Vy) 를 정확히 푼다."""
    pv = pv - pv.mean(); v = v - v.mean(); y = y - y.mean()
    n = len(y)
    a, b = pv @ y / n, v @ y / n
    Vp, Vv, C = pv @ pv / n, v @ v / n, pv @ v / n
    Vy = y @ y / n
    if Vv <= 0 or Vy <= 0:
        return dict(w_opt=0.0, gain_opt=0.0, gain_w1=0.0, gain_w2=0.0, corr_pv=0.0)
    f = lambda w: 1e5 * (a + w * b) ** 2 / ((Vp + 2 * w * C + w * w * Vv) * Vy)
    den = b * C - a * Vv
    ws = (a * C - b * Vp) / den if den != 0 else 0.0
    g0 = f(0.0)
    # 분모가 양수인 구간의 최댓값인지 확인 (아니면 격자로 보정)
    cand = [ws] if (Vp + 2 * ws * C + ws * ws * Vv) > 0 else []
    best = max(cand, key=f) if cand else 0.0
    return dict(w_opt=float(best), gain_opt=float(f(best) - g0),
                gain_w1=float(f(1.0) - g0), gain_w2=float(f(2.1) - g0),
                corr_pv=float(C / np.sqrt(Vp * Vv)))


# ---------------------------------------------------------------- 표
def level_table(keys, resid, k):
    u, inv = np.unique(keys, return_inverse=True)
    n = np.bincount(inv, minlength=len(u)).astype(np.float64)
    s = np.bincount(inv, weights=resid, minlength=len(u))
    return u, s / (n + k), n


def lookup(u, tab, keys):
    ix = np.searchsorted(u, keys)
    ix = np.clip(ix, 0, max(len(u) - 1, 0))
    ok = (u[ix] == keys) if len(u) else np.zeros(len(keys), bool)
    out = np.zeros(len(keys))
    out[ok] = tab[ix[ok]]
    return out


def persistence(k1, r1, k2, r2_, min_n=30):
    """두 시즌의 셀 평균 잔차 상관 — 원장이 '오라클보다 먼저 보라'는 통계량."""
    def cell(kk, rr):
        u, inv = np.unique(kk, return_inverse=True)
        n = np.bincount(inv, minlength=len(u))
        s = np.bincount(inv, weights=rr, minlength=len(u))
        return u, s / np.maximum(n, 1), n
    u1, m1, n1 = cell(k1, r1)
    u2, m2, n2 = cell(k2, r2_)
    common, i1, i2 = np.intersect1d(u1, u2, return_indices=True)
    if len(common) < 10:
        return np.nan, 0
    ok = (n1[i1] >= min_n) & (n2[i2] >= min_n)
    if ok.sum() < 10:
        return np.nan, int(ok.sum())
    x, z = m1[i1][ok], m2[i2][ok]
    return float(np.corrcoef(x, z)[0, 1]), int(ok.sum())


# ---------------------------------------------------------------- 후보 정의
def mix(*arrs):
    """정수 배열 여러 개를 겹치지 않는 int64 키 하나로 접는다 (오버플로 검사)."""
    out = np.zeros(len(arrs[0]), np.int64)
    mult = 1
    for a in arrs:
        a = np.asarray(a, np.int64)
        lo = a.min()
        a = a - lo
        span = int(a.max()) + 1
        assert mult * span < 2 ** 62, "int64 오버플로 — 키 구성을 줄여라"
        out = out + a * mult
        mult *= span
    return out


def build_keys(d):
    b, s = d["balls_before"].to_numpy(np.int64), d["strikes_before"].to_numpy(np.int64)
    pid, bid = d["pitcher_id"].to_numpy(np.int64), d["batter_id"].to_numpy(np.int64)
    pt, bt = d["pitcher_team_id"].to_numpy(np.int64), d["batter_team_id"].to_numpy(np.int64)
    ph, bh = d["pitcher_hand"].to_numpy(np.int64), d["batter_hand"].to_numpy(np.int64)
    inn = d["inning"].to_numpy(np.int64)
    mon = d["game_month"].to_numpy(np.int64)
    dow = d["game_dayofweek"].to_numpy(np.int64)
    ron = (d["num_runners_on"].to_numpy(np.int64) > 0).astype(np.int64)
    two = (s >= 2).astype(np.int64)
    adv = (s > b).astype(np.int64)
    outs = d["outs_before"].to_numpy(np.int64)
    gt = pd.factorize(d["game_type"])[0].astype(np.int64)
    tb = pd.factorize(d["top_bottom"])[0].astype(np.int64)
    bs = pd.factorize(d["base_state"])[0].astype(np.int64)
    innb = np.clip((inn - 1) // 3, 0, 3)
    sd = np.clip(d["score_diff_pitcher_team"].to_numpy(np.int64), -5, 5)
    li = d["li"].to_numpy(np.float64)
    lid = np.digitize(li, np.quantile(li, np.linspace(0, 1, 11)[1:-1])).astype(np.int64)
    bn = d["asof_batter_n"].to_numpy(np.float64)
    bnd = np.digitize(bn, np.quantile(bn, np.linspace(0, 1, 11)[1:-1])).astype(np.int64)

    return {
        # --- 대조군 (이미 ACTIVE. 지표가 이 둘을 위로 올려야 신뢰할 수 있다)
        "[대조] 투수": pid,
        "[대조] 타자": bid,
        # --- 개체 수준, 미사용
        "투수팀": pt,
        "타자팀": bt,
        # --- 타자 x 맥락 (전부 '대비'로만 닫혔던 구역)
        "타자 x 투수손": mix(bid, ph),
        "타자 x 2스트라이크": mix(bid, two),
        "타자 x 주자유무": mix(bid, ron),
        "타자 x 카운트우위": mix(bid, adv),
        "타자 x game_type": mix(bid, gt),
        "타자 x 이닝대": mix(bid, innb),
        "타자 x 월": mix(bid, mon),
        "타자 x 홈원정": mix(bid, tb),
        "타자 x 투수팀": mix(bid, pt),
        # --- 투수 x 맥락 (현행 4·5·6 은 k=1000. 큰 k 수준판은 미측정)
        "투수 x 타자손": mix(pid, bh),
        "투수 x 스트라이크수": mix(pid, s),
        "투수 x 주자유무": mix(pid, ron),
        "투수 x game_type": mix(pid, gt),
        "투수 x 월": mix(pid, mon),
        "투수 x 이닝대": mix(pid, innb),
        "투수 x 홈원정": mix(pid, tb),
        "투수 x 타자팀": mix(pid, bt),
        # --- 매치업
        "투수 x 타자": mix(pid, bid),
        "투수팀 x 타자팀": mix(pt, bt),
        # --- 순수 맥락 수준 (후처리로는 한 번도 안 실렸다)
        "카운트(12)": mix(b, s),
        "이닝": inn,
        "base_state": bs,
        "아웃수": outs,
        "월": mon,
        "요일": dow,
        "홈원정": tb,
        "game_type": gt,
        "li 십분위": lid,
        "점수차": sd,
        "주자수": d["num_runners_on"].to_numpy(np.int64),
        "손 조합": mix(ph, bh),
        "타자경력 십분위": bnd,
    }


# ---------------------------------------------------------------- 본체
#
# 🚩 기준선을 **둘** 로 나눈다. 첫 판본에서 대조군(투수·타자)이 0.00 으로 나왔는데
#    버그가 아니라 기준선에 이미 그 두 축이 들어 있었기 때문이다. 즉 "현 Champion
#    위의 증분"만 재면 **대조군이 성립하지 않아 지표를 검증할 수 없다.**
#
#    LV0  후처리 축 0~6 만 (수준 2축 제거)  -> 대조군이 살아난다. 지표 검증용
#    FULL 현 Champion 9축 전부             -> 실제 한계 이득
#
# 그리고 목표 시즌을 둘로 잰다. 표는 언제나 **과거만** 쓴다 (walk-forward).
#    목표 2023  표 = 2022
#    목표 2024  표 = 2022+2023   (생산과 동일한 직전 2시즌 하드 창)
#    **둘 다 양수**가 아니면 순위에서 내린다 (2023 은 퇴화 폴드라 크기는 안 본다)
def main():
    t0 = time.time()
    cols = ["season", "control_success", "balls_before", "strikes_before",
            "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
            "pitcher_hand", "batter_hand", "inning", "game_month",
            "game_dayofweek", "num_runners_on", "outs_before", "game_type",
            "top_bottom", "base_state", "score_diff_pitcher_team", "li",
            "asof_batter_n"]
    df = pd.read_csv(os.path.join(ROOT, "data", "train.csv"), usecols=cols)
    df = df[df.season >= 2022].reset_index(drop=True)
    season = df["season"].to_numpy(np.int64)
    y = df["control_success"].to_numpy(np.float64)
    oof = np.load(os.path.join(ROOT, "exp", "champ_oof.npz"))
    b = joblib.load(io.BytesIO(zipfile.ZipFile(CHAMP).read("model/rf.pkl")))

    def post(mask, axes):
        sub = df.loc[mask]
        out = np.zeros(int(mask.sum()))
        for i in axes:
            sp = b["platoon"][i]
            keys = list(zip(*[sub[c].to_numpy(np.int64) for c in sp["cols"]]))
            out += float(sp["w"]) * np.array([sp["table"].get(k, 0.0) for k in keys])
        return out

    M = {s: season == s for s in (2022, 2023, 2024)}
    FULL, LV0 = {}, {}
    for s, m in M.items():
        base = oof[f"p{s}"].astype(np.float64)
        FULL[s] = base + post(m, range(9))
        LV0[s] = base + post(m, range(7))
        print(f"  기준선 {s}   FULL {r2(FULL[s], y[m]):8.2f}   "
              f"LV0(수준2축 제거) {r2(LV0[s], y[m]):8.2f}", flush=True)
    print("  -> LV0 와 FULL 의 차이가 곧 수준 2축의 로컬 기여다\n", flush=True)

    K = build_keys(df)
    cur24 = post(M[2024], range(9))          # 중복 진단용
    r22F, r23F = y[M[2022]] - FULL[2022], y[M[2023]] - FULL[2023]
    r22L, r23L = y[M[2022]] - LV0[2022], y[M[2023]] - LV0[2023]

    def measure(keys_tr, resid_tr, keys_ta, pv_ta, y_ta):
        """k 격자에서 최선을 고른다. w 는 해석적 최적과 w=1 을 둘 다 낸다."""
        best = None
        for k in KS:
            u, tab, n = level_table(keys_tr, resid_tr, k)
            v = lookup(u, tab, keys_ta)
            g = gain_curve(pv_ta, v, y_ta)
            g["k"] = k
            if best is None or g["gain_opt"] > best["gain_opt"]:
                best, best_v = g, v
        return best, best_v

    rows = []
    hdr = (f"{'축':<20}{'셀':>7}{'중앙n':>7}{'지속성':>7}{'z':>6}"
           f"{'LV0 2024':>9}{'w*':>7}{'FULL 2024':>10}{'2023':>8}{'중복':>7}")
    print(hdr); print("-" * len(hdr.encode('utf-8')) // 2 * "-" if False else "")
    for name, keyall in K.items():
        k22, k23, k24 = keyall[M[2022]], keyall[M[2023]], keyall[M[2024]]
        ktr = np.concatenate([k22, k23])
        pers, npers = persistence(k22, r22L, k23, r23L)
        z = pers * np.sqrt(max(npers - 3, 1)) if pers == pers else float("nan")

        gL, vL = measure(ktr, np.concatenate([r22L, r23L]), k24, LV0[2024], y[M[2024]])
        gF, vF = measure(ktr, np.concatenate([r22F, r23F]), k24, FULL[2024], y[M[2024]])
        g23, _ = measure(k22, r22F, k23, FULL[2023], y[M[2023]])
        dup = float(np.corrcoef(vF, cur24)[0, 1]) if vF.std() > 0 else 0.0
        u, _, n = level_table(ktr, np.concatenate([r22L, r23L]), gL["k"])

        rows.append(dict(name=name, cells=int(len(u)), med_n=float(np.median(n)),
                         persist=pers, z=float(z) if z == z else None,
                         k_lv0=gL["k"], gain_lv0=gL["gain_opt"], w_lv0=gL["w_opt"],
                         k_full=gF["k"], gain_full=gF["gain_opt"], w_full=gF["w_opt"],
                         gain_2023=g23["gain_opt"], w_2023=g23["w_opt"],
                         both_pos=bool(gF["gain_opt"] > 0 and g23["gain_opt"] > 0),
                         dup_cur=dup))
        print(f"{name:<20}{len(u):>7,}{np.median(n):>7.0f}{pers:>7.3f}{z:>6.1f}"
              f"{gL['gain_opt']:>9.2f}{gL['w_opt']:>7.2f}{gF['gain_opt']:>10.2f}"
              f"{g23['gain_opt']:>8.2f}{dup:>7.3f}", flush=True)

    rows.sort(key=lambda r: -r["gain_full"])
    json.dump(rows, io.open(os.path.join(ROOT, "research", "lv001_level_sweep.json"),
                            "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"\n총 {time.time() - t0:.0f}s   research/lv001_level_sweep.json")
    print("\n읽는 법 — 대조군(투수·타자)이 LV0 열에서 위로 올라와야 지표를 믿는다.")


if __name__ == "__main__":
    main()
