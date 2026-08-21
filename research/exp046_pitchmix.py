r"""EXP046 — 🚩 **투구 단위 구종을 복원했다.** 맥락별 구종 성향을 메인 데이터로 만든다.

## 착상

`asof_pitcher_n` 은 투수별 통산 카운터이고 학습셋에서 **연속쌍 전부가 정확히 1씩**
오른다(1,474,300/1,475,092). 그러므로 같은 투수의 행을 그 카운터로 정렬하면
누적 비율의 증분이 **그 투구의 실제 결과**를 준다.

    복원한 success 가 실제 라벨과 일치율 **1.000000** (n=1,473,508)
    구종도 100% 복원 — fastball .541 / breaking .296 / offspeed .163

학습 데이터를 어떻게 쓰든 규정상 자유다. 추론은 그 행 자신의 컬럼만 본다.

## 왜 이것이 중요한가

원장 §7-b 는 이렇게 적혀 있다.

> 카운트별 구종 성향은 메인 데이터에 아예 없어서(`asof_pitcher_*_rate` 는 투수
> 평균 하나뿐) 못 만들고 대리도 없다.

그래서 우리는 그 신호를 **오염된 TrackMan 조인**으로만 얻고 있었다 —
F 행 결측 40% 대 R 행 18% 라서 `game_type` F 지시자가 딸려 들어온다(TMX/TMR 사망
원인). 이제 **결측 0% · 오염 0%** 로 같은 신호를 만든다.

## 확인된 구조 (전량 학습셋 집계)

    구종별 제구 성공률   FB .5451  OS .5135  BR .4903     격차 +3.16pp
    3-0 카운트 FB 93.8%  vs  0-2 카운트 FB 45.0%          49pp 스윙
    같은손 OS 9.1%  vs  다른손 OS 23.5%                    2.6배
    구종 x 카운트우위 성공률의 **부호가 갈린다** (FB +3.4pp / BR -1.3pp)

    .\.venv\Scripts\python.exe -u research\exp046_pitchmix.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "exp"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

CACHE = os.path.join(ROOT, "exp", "cache", "pitch_events.npz")
TYPES = ["fb", "br", "os"]


def pitch_events():
    """train.csv **원래 행 순서**로 정렬된 투구 단위 결과를 만든다."""
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        return {k: z[k] for k in z.files}
    cols = ["season", "pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate",
            "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
            "asof_pitcher_strike_rate", "asof_pitcher_reverse_rate",
            "asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate",
            "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
            "control_success"]
    d = pd.read_csv(os.path.join(ROOT, "data", "train.csv"), usecols=cols)
    order = np.lexsort((d["asof_pitcher_n"].to_numpy(), d["pitcher_id"].to_numpy()))
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    pid = d["pitcher_id"].to_numpy()[order]
    same_next = np.r_[pid[:-1] == pid[1:], False]        # 다음 행이 같은 투수인가

    def rec(rate_col, ncol="asof_pitcher_n"):
        c = (d[ncol].to_numpy(float) * d[rate_col].to_numpy(float))[order]
        inc = np.r_[c[1:] - c[:-1], np.nan]              # 행 t 의 결과
        inc = np.where(same_next, inc, np.nan)
        return np.round(inc)[inv]                        # 원래 순서로

    out = {"success": rec("asof_pitcher_success_rate"),
           "middle": rec("asof_pitcher_middle_rate"),
           "ball": rec("asof_pitcher_ball_rate"),
           "strike": rec("asof_pitcher_strike_rate"),
           "reverse": rec("asof_pitcher_reverse_rate"),
           "fb": rec("asof_pitcher_fastball_rate", "asof_pitcher_pitchmix_n"),
           "br": rec("asof_pitcher_breaking_rate", "asof_pitcher_pitchmix_n"),
           "os": rec("asof_pitcher_offspeed_rate", "asof_pitcher_pitchmix_n")}
    y = d["control_success"].to_numpy(float)
    ok = ~np.isnan(out["success"])
    agree = float(np.mean(out["success"][ok] == y[ok]))
    print(f"  복원 검증 — success 일치율 {agree:.6f}  (n={ok.sum():,}, "
          f"결측 {(~ok).sum():,})")
    assert agree == 1.0, "복원 실패 — 이 축을 쓰면 안 된다"
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, **{k: v.astype(np.float32) for k, v in out.items()})
    return out


def ctx_dev(pid, ctx, val, mask_src, K):
    """P(val | 투수, 국면) - P(val | 투수) 를 축소한다. 표는 mask_src 행으로만."""
    df = pd.DataFrame({"p": pid[mask_src], "c": ctx[mask_src], "v": val[mask_src]})
    g = df.groupby(["p", "c"])["v"].agg(["sum", "size"])
    h = df.groupby("p")["v"].agg(["sum", "size"])
    base = (h["sum"] / h["size"]).rename("base")
    j = g.join(base, on="p")
    dev = (j["sum"] / j["size"] - j["base"]) * j["size"] / (j["size"] + K)
    return dev.to_dict(), base.to_dict()


def look(tbl, base, pid, ctx):
    key = pd.MultiIndex.from_arrays([pid, ctx])
    return pd.Series(key.map(tbl)).fillna(0.0).to_numpy(float)


def main():
    ev = pitch_events()
    from path_alloc import build_df
    tr = build_df()
    y = tr["control_success"].to_numpy(float)
    assert np.array_equal(ev["success"][~np.isnan(ev["success"])],
                          y[~np.isnan(ev["success"])]), "정렬 불일치"
    print("  build_df 와 정렬 일치 확인")

    season = tr["season"].to_numpy()
    P = tr["pitcher_id"].to_numpy(np.int64)
    BB = tr["balls_before"].to_numpy(np.int64)
    SS = tr["strikes_before"].to_numpy(np.int64)
    SH = (tr["pitcher_hand"].to_numpy() == tr["batter_hand"].to_numpy()).astype(np.int64)
    OB = (tr["num_runners_on"].to_numpy(np.int64) > 0).astype(np.int64)
    CTX = {"cnt": BB * 4 + SS, "sh": SH, "two": (SS == 2).astype(np.int64), "onb": OB}

    for fold in (2022, 2024):
        src = (season < fold) & (season >= fold - 2)      # 직전 2시즌 (원장 고정값)
        m = season == fold
        pred = np.load(os.path.join(ROOT, "exp", f"prod_champ_{fold}.npy")).mean(0)
        res = y[m] - pred
        sd_r = res.std()
        n = int(m.sum())
        se = 1.0 / np.sqrt(n)
        print(f"\n=== 폴드 {fold}  n={n:,}  잔차상관 SE={se:.4f} "
              f"(원리적 상한 1e5*c^2) ===")
        print(f"{'후보 열':22s} {'K':>6s} {'corr':>9s} {'|c|/SE':>7s} {'상한(점)':>9s}")

        ref = {}
        for c in ("tmc_fastball_dev", "tmc_breaking_dev", "tmc_offspeed_dev",
                  "tmc_speed_dev", "asof_pitcher_n", "cur_succ"):
            v = tr[c].to_numpy(float)[m]
            v = np.nan_to_num(v, nan=0.0)
            cc = np.corrcoef(v, res)[0, 1] if v.std() > 0 else 0.0
            ref[c] = cc
            print(f"{'[기존] '+c:22s} {'-':>6s} {cc:+9.5f} {abs(cc)/se:7.2f} "
                  f"{1e5*cc*cc:9.2f}")

        best = {}
        for cname, ctx in CTX.items():
            for K in (100, 300, 1000):
                for t in TYPES:
                    val = np.nan_to_num(ev[t], nan=0.0)
                    ok = ~np.isnan(ev[t])
                    tbl, base = ctx_dev(P, ctx, val, src & ok, K)
                    v = look(tbl, base, P[m], ctx[m])
                    cc = np.corrcoef(v, res)[0, 1] if v.std() > 0 else 0.0
                    key = f"pm_{cname}_{t}"
                    if abs(cc) > abs(best.get(key, (0, 0))[0]):
                        best[key] = (cc, K)
        for k, (cc, K) in sorted(best.items(), key=lambda x: -abs(x[1][0])):
            print(f"{k:22s} {K:6d} {cc:+9.5f} {abs(cc)/se:7.2f} {1e5*cc*cc:9.2f}")

        # 합성 열 — 국면별 배합 이동 x 그 투수의 구종별 성공률 편차
        for K in (300, 1000):
            for cname, ctx in CTX.items():
                tot = np.zeros(n)
                for t in TYPES:
                    ok = ~np.isnan(ev[t])
                    tbl, _ = ctx_dev(P, ctx, np.nan_to_num(ev[t]), src & ok, K)
                    dmix = look(tbl, None, P[m], ctx[m])
                    st, _ = ctx_dev(P, np.nan_to_num(ev[t]).astype(np.int64),
                                    y, src & ok, K)     # P(succ|투수,구종)-P(succ|투수)
                    dsucc = look(st, None, P[m], np.ones(n, np.int64))
                    tot += dmix * dsucc
                cc = np.corrcoef(tot, res)[0, 1] if tot.std() > 0 else 0.0
                print(f"{'pm_xsucc_'+cname:22s} {K:6d} {cc:+9.5f} "
                      f"{abs(cc)/se:7.2f} {1e5*cc*cc:9.2f}")


if __name__ == "__main__":
    main()
