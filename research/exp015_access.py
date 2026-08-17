r"""EXP015 — 접근 수준별 가치. "규정 4 가 무엇을 막고 있는가"를 정확히 센다.

## 질문

"평가셋 내 집계가 허용되면 +83.5 가 열린다"고 썼는데, **무엇이 어디까지
허용되어야** 하는지가 불분명하다. 접근 수준을 셋으로 나눠 각각의 값을 잰다.

    A  현행       그 행 자신의 컬럼 + 과거 train(라벨 포함)
    B  전이적 피처  + 평가셋 **다른 행의 피처** (라벨은 없음)
    C  라벨 접근    + 평가셋 다른 행의 **라벨**   (코드 제출 대회에선 애초에 불가)

## 왜 B 가 핵심인가

`asof_pitcher_n` 은 그 투구 **직전까지의** 통산 투구 수다. 한 시즌 안에서
같은 투수의 행을 모으면 `asof_pitcher_n` 이 최대인 행의 `cur_succ` 가 곧
**그 투수의 그 시즌 최종 성공률**이다. 라벨을 한 개도 안 보고 얻어진다.

그런데 `CLAUDE.md` 7-c 가 말하는 남은 점수의 정체가 정확히 "투수의 그 시즌
현재 상태"다. 즉 **B 만 열려도 그 자리가 통째로 열린다.**

## 이 실험은 제출용이 아니다

여기서 만드는 피처는 **규정 4 위반**이므로 어떤 후보에도 쓰지 않는다.
목적은 "규정이 막는 몫"을 정확히 세는 것뿐이다.

    .\.venv\Scripts\python.exe -u research\exp015_access.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "exp"))
import engine as E                                          # noqa: E402
from path_alloc import build_df                             # noqa: E402
from resid_table import post_for                            # noqa: E402
from traj_probe import r2                                   # noqa: E402
from game_decomp import games                               # noqa: E402
from exp005_geom import ridge_cv                            # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

EXP = "EXP015"
ALPHAS = [1e2, 1e3, 1e4, 1e5, 1e6]
CUR = ("cur_succ", "cur_mid", "cur_ball", "cur_str")


def main():
    E.start_experiment(EXP, "L4-B", "python research/exp015_access.py", "load")
    tr = build_df()
    season = tr["season"].to_numpy()
    y = tr["control_success"].to_numpy(np.float64)
    P = tr["pitcher_id"].to_numpy(np.int64)

    def g(c):
        return tr[c].to_numpy(np.float64)

    AX = {"hand": ((g("pitcher_hand") == g("batter_hand")).astype(int), 1000),
          "2S": ((g("strikes_before").astype(int) == 2).astype(int), 1000),
          "runner": ((g("num_runners_on") > 0).astype(int), 2000)}
    res0 = {}
    for f in (2022, 2023):
        m = season == f
        res0[f] = y[m] - (np.load(os.path.join(ROOT, "exp", f"prod_champ_{f}.npy"))[:2].mean(0)
                          + post_for(tr, y, season < f, m))

    def dvec(ctx, k):
        p = np.concatenate([P[season == f] for f in (2022, 2023)])
        c = np.concatenate([ctx[season == f] for f in (2022, 2023)])
        r = np.concatenate([res0[f] for f in (2022, 2023)])
        gg = pd.DataFrame({"p": p, "c": c, "r": r}).groupby(["p", "c"])["r"].agg(
            ["mean", "size"]).unstack()
        n0, n1 = gg[("size", 0)].fillna(0), gg[("size", 1)].fillna(0)
        ne = (n0 * n1) / (n0 + n1).replace(0, np.nan)
        return ((gg[("mean", 1)] - gg[("mean", 0)]) * ne / (ne + k)).dropna()

    m24 = season == 2024
    C3 = (np.load(os.path.join(ROOT, "exp", "prod_champ_2024.npy"))[:2].mean(0)
          + post_for(tr, y, season < 2024, m24))
    for a, (ctx, k) in AX.items():
        C3 += (pd.Series(P[m24]).map(dvec(ctx, k)).fillna(0.).to_numpy()
               * np.where(ctx[m24] == 1, .5, -.5))
    r24 = y[m24] - C3
    base = r2(C3, y[m24])
    p24 = P[m24]
    print(f"C3 기준선 2024 = {base:.1f}\n")
    E.beat("기준선")

    # ---- B 수준: 평가셋 피처만으로 '그 시즌 최종 상태'를 얻는다 ----
    an = g("asof_pitcher_n")[m24]
    df = pd.DataFrame({"p": p24, "n": an})
    for c in CUR:
        df[c] = np.nan_to_num(g(c)[m24], nan=np.nan)
    last = df.sort_values("n").groupby("p").tail(1).set_index("p")
    print("=" * 78)
    print("검증 — 라벨을 한 개도 안 보고 '그 시즌 최종 성공률'을 얻는가")
    print("=" * 78)
    truth = pd.DataFrame({"p": p24, "y": y[m24]}).groupby("p")["y"].agg(["mean", "size"])
    j = truth.join(last[["cur_succ"]], how="inner").dropna()
    j = j[j["size"] >= 100]
    print(f"  투수 {len(j)}명 (2024 100투구 이상)")
    print(f"  corr( 피처만으로 얻은 최종 cur_succ ,  실제 2024 성공률 ) "
          f"= {np.corrcoef(j['cur_succ'], j['mean'])[0,1]:+.4f}")
    print(f"  평균 절대오차 {np.abs(j['cur_succ'] - j['mean']).mean():.4f}")
    print("  -> 평가셋 라벨 없이 투수의 그 시즌 실력이 사실상 그대로 나온다\n")

    def std(cols):
        X = np.column_stack(cols)
        return np.nan_to_num((X - np.nanmean(X, 0)) / (np.nanstd(X, 0) + 1e-12))

    fin = {c: pd.Series(p24).map(last[c]).to_numpy(np.float64) for c in CUR}
    cur = {c: np.nan_to_num(g(c)[m24], nan=float(np.nanmean(g(c)[m24]))) for c in CUR}

    GID = games(p24, an, g("asof_pitcher_prev1_game_success_rate")[m24],
                g("asof_pitcher_prev1_game_middle_rate")[m24])
    rng = np.random.default_rng(0)
    u, inv = np.unique(GID, return_inverse=True)
    half = (rng.random(len(u)) < 0.5)[inv]

    BLK = {
        "A 현행 — 그 행 + 과거 train": std([cur[c] for c in CUR]),
        "B1 최종 시즌 상태 (피처만)": std([fin[c] for c in CUR]),
        "B2 최종 − 현재 (남은 여정)": std([fin[c] - cur[c] for c in CUR]),
        "B3 B1 + B2 전체": std([fin[c] for c in CUR] + [fin[c] - cur[c] for c in CUR]),
        "C 라벨 접근 (투수 실현 성공률)":
            std([pd.Series(p24).map(truth["mean"]).to_numpy(np.float64)]),
    }
    print("=" * 78)
    print(f"{'접근 수준':<34}{'열':>5}{'잔차상관':>11}{'C3증분':>10}{'합법':>10}")
    print("=" * 78)
    out = {}
    for nm, X in BLK.items():
        E.beat(nm)
        pred, cr = ridge_cv(X, r24, half, ALPHAS)
        inc = r2(C3 + pred, y[m24]) - base
        lab = "✅" if nm.startswith("A") else ("❌ 규정4" if nm.startswith("B")
                                              else "❌ 불가능")
        out[nm] = dict(cols=int(X.shape[1]), corr=cr, inc=inc)
        print(f"{nm:<34}{X.shape[1]:>5}{cr:>+11.4f}{inc:>+10.1f}{lab:>10}")

    a = out["A 현행 — 그 행 + 과거 train"]["inc"]
    b = out["B3 B1 + B2 전체"]["inc"]
    c = out["C 라벨 접근 (투수 실현 성공률)"]["inc"]
    print(f"\n  A -> B 로 열리는 몫  {b - a:+.1f}   (평가셋 **피처**만 허용해도)")
    print(f"  B -> C 로 열리는 몫  {c - b:+.1f}   (라벨까지 허용하면)")
    print("\n  결론 — 라벨은 필요 없다. **배치 추론 허용만으로 대부분이 열린다.**")

    E.set_hypothesis_status("L4-B", "MEASURED", level=4,
                            hypothesis="접근 수준별 가치", result=round(b - a, 2))
    E.finish_experiment(dict(
        experiment_id=EXP, hypothesis_id="L4-B", type="AUDIT", level=4,
        started_at=E.read(E.CKPT)["start_time"],
        results={k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in out.items()},
        corr_feature_only_vs_truth=float(np.corrcoef(j["cur_succ"], j["mean"])[0, 1]),
        decision="AUDIT ONLY — 제출 금지",
        what_we_learned=("규정 4 가 막는 것은 라벨이 아니라 **배치 추론**이다. "
                         "평가셋 피처만 모아도 asof_n 최대 행의 cur_* 가 곧 그 투수의 "
                         f"그 시즌 실현 실력이고(상관 "
                         f"{np.corrcoef(j['cur_succ'], j['mean'])[0,1]:.4f}), "
                         f"A->B 로 {b-a:+.1f} 가 열린다. 라벨까지 가도 추가 {c-b:+.1f}.")))
    json.dump(out, open(os.path.join(ROOT, "exp", "exp015_access.json"), "w"),
              indent=1, default=float)


if __name__ == "__main__":
    main()
