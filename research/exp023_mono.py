r"""EXP023 — 단조 제약. 우리가 물리적으로 아는 방향을 모델에 그냥 준다.

## 왜 이게 아직 남아 있었는가

`CLAUDE.md` 7 의 닫힌 목록에 `capacity` · `objective` 는 있어도 **단조 제약은
없다.** CatBoost 파라미터 하나인데 한 번도 걸어본 적이 없다.

## 왜 통할 수 있는가

이 타깃은 신호 대 잡음이 극단적으로 낮다 (`rho` ≈ 0.10, 기저율 0.52).
그런 조건에서 트리는 잡음을 학습하기 쉽고, **올바른 방향 제약은 공짜 정칙화**다.
그리고 우리는 방향을 데이터가 아니라 **물리로** 안다.

    cur_succ  ↑   시즌내 제구 성공률이 높으면 이번 투구도 성공 확률이 높다
    cur_rev   ↓   반대 방향으로 간 비율이 높으면 제구가 나쁘다
    cur_mid   ↓   가운데로 몰린 비율이 높으면 제구가 나쁘다
    cur_ball  ↓   볼 비율이 높으면 제구가 나쁘다
    asof_pitcher_success_rate ↑   통산 제구 성공률

`success` 와 `reverse` 의 상관이 −0.865 라는 오늘의 감사 결과가 이 방향들을
뒷받침한다 (둘은 한 축의 양끝).

## 왜 해로울 수도 있는가 — 그래서 두 수준을 쟀다

제약은 상호작용을 막는다. 예컨대 `cur_succ` 의 효과가 카운트에 따라 부호를
바꾼다면 단조 제약이 그걸 죽인다. 그래서 **가장 확실한 3개만**과 **5개 전부**를
따로 잰다.

    .\.venv\Scripts\python.exe -u research\exp023_mono.py
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "exp"))
import engine as E                                          # noqa: E402
import build_asof as ba                                     # noqa: E402
from path_alloc import build_df                             # noqa: E402
from resid_table import post_for                            # noqa: E402
from traj_probe import r2                                   # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

EXP = "EXP023"
SEED = 42
CORE = {"cur_succ": 1, "cur_rev": -1, "cur_mid": -1}
FULL = dict(CORE, **{"cur_ball": -1, "asof_pitcher_success_rate": 1})


def main():
    E.start_experiment(EXP, "M1", "python research/exp023_mono.py", "load")
    ft, sc = ba.ft, ba.sc
    tr = build_df()
    season = tr["season"].to_numpy()
    y = tr["control_success"].to_numpy(np.float64)
    P = tr["pitcher_id"].to_numpy(np.int64)

    def g(c):
        return tr[c].to_numpy(np.float64)

    tc = pd.read_csv(os.path.join(ft.DATA_DIR, "test.csv"),
                     encoding="utf-8-sig", nrows=0).columns
    allf = [c for c in tc if c != ft.ID]
    ctxf = [c for c in ft.COUNT_FEATS + ft.HAND_FEATS if c not in ("tmc_n", "tmh_n")]
    CHAMP = list(allf) + ctxf + sc.ASOF_COLS + sc.CTX_COLS + sc.LVL_COLS

    # ColumnTransformer 는 [cat..., num...] 순서로 내보낸다 — 그 순서로 색인한다
    cat = [c for c in ft.CAT_COLS if c in CHAMP]
    num = [c for c in CHAMP if c not in cat]
    order = cat + num
    print(f"특징 {len(CHAMP)}열 (범주 {len(cat)} + 수치 {len(num)})")

    def mono_vec(spec):
        v = [0] * len(order)
        hit = []
        for name, s in spec.items():
            if name in order:
                v[order.index(name)] = s
                hit.append(f"{name}{'+' if s > 0 else '-'}")
        return v, hit

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
    m_tr = season < 2024
    add = post_for(tr, y, m_tr, m24)
    for a, (ctx, k) in AX.items():
        add += 0.65 * (pd.Series(P[m24]).map(dvec(ctx, k)).fillna(0.).to_numpy()
                       * np.where(ctx[m24] == 1, .5, -.5))
    ref = np.load(os.path.join(ROOT, "exp", "prod_champ_2024.npy"))
    print(f"참조 — 캐시 시드42 단독 {r2(ref[0] + add, y[m24]):.1f}"
          f"   2시드 {r2(ref[:2].mean(0) + add, y[m24]):.1f}\n")

    print("=" * 76)
    print(f"{'구성':<28}{'제약':<26}{'2024':>10}{'대비':>10}")
    print("=" * 76)
    out, base = {}, None
    for name, spec in (("무제약 (재현)", {}), ("핵심 3개", CORE), ("전체 5개", FULL)):
        E.beat(name)
        t0 = time.time()
        mdl = ba.pipeline(CHAMP, SEED)
        if spec:
            v, hit = mono_vec(spec)
            mdl.named_steps["clf"].set_params(monotone_constraints=v)
        else:
            hit = []
        mdl.fit(tr.loc[m_tr, CHAMP], y[m_tr].astype(int))
        pv = mdl.predict_proba(tr.loc[m24, CHAMP])[:, 1]
        s = r2(pv + add, y[m24])
        if base is None:
            base = s
        out[name] = dict(score=s, secs=time.time() - t0, spec=hit)
        print(f"{name:<28}{','.join(hit)[:24]:<26}{s:>10.1f}{s - base:>+10.1f}"
              f"   ({time.time()-t0:.0f}s)")
        np.save(os.path.join(ROOT, "exp", f"pred24_mono_{len(spec)}.npy"), pv)
        del mdl

    best = max(out, key=lambda k: out[k]["score"])
    gain = out[best]["score"] - base
    dec = "PROMISING" if (best != "무제약 (재현)" and gain > 2) else "REJECTED"
    print(f"\n최선 = {best}   무제약 대비 {gain:+.1f}")
    print("  시드 잡음 ±7.2 — 이 범위면 LB 로 판정한다 (로컬은 파국 검사만)")
    E.set_hypothesis_status("M1", "PROMISING" if dec == "PROMISING" else "CLOSED",
                            level=2, hypothesis="단조 제약", result=round(gain, 2))
    E.finish_experiment(dict(
        experiment_id=EXP, hypothesis_id="M1", type="MODEL", level=2,
        started_at=E.read(E.CKPT)["start_time"],
        results={k: {kk: (round(vv, 2) if isinstance(vv, float) else vv)
                     for kk, vv in v.items()} for k, v in out.items()},
        gain=round(gain, 2), decision=dec, artifact=None,
        what_we_learned=(f"물리로 아는 방향을 제약으로 준다. 최선 {best}, "
                         f"무제약 대비 {gain:+.1f}")))
    json.dump(out, open(os.path.join(ROOT, "exp", "exp023_mono.json"), "w"),
              indent=1, default=float)


if __name__ == "__main__":
    main()
