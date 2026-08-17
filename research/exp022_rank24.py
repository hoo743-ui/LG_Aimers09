r"""EXP022 — 246 국면을 **2024 이득**으로 줄 세운다. 홀드아웃 없이. 학습 0회.

## 왜 지금 이게 가능해졌는가 (그리고 왜 전엔 못 했는가)

EXP014 는 246개를 훑되 **2024 를 숨겨야** 했다. 로컬밖에 없었으니 판정용
홀드아웃이 필요했고, 그래서 선택은 2022·2023 으로만 했다. 결과는 실패였다
(숨긴 2024 에서 −16.9, p=0.960).

이제 **LB 가 심판**이다. 홀드아웃이 필요 없다. 그러면 선택에 2024 를 쓸 수 있고,
`CLAUDE.md` 5 의 관찰이 그걸 정당화한다 —

> 2024 폴드 부호가 LB 방향과 지금까지 예외 없이 일치한다.

## 그리고 기각 기준 하나가 스스로 깨졌다

EXP020 의 같은 실행에서 나온 대조군 숫자다.

    [대조] 같은손       오라클 +11.8  (위약 −1.9)   검출됨
    [대조] 2스트라이크    오라클  −0.2  (위약 −1.9)   **검출 실패**

2S 는 LB 로 검증된 진짜 축인데(C3−C2 증분 +3.7443) 오라클이 못 잡는다.
**즉 "오라클 <= 위약 -> 정보 없음"은 틀린 기각 규칙이다.** 그런데 EXP003 은
243개 중 오라클을 통과한 12개만 전이를 쟀다 — **231개는 전이 측정을 받은 적이
없다.** 이 실험이 그 231개를 포함해 전부 잰다.

## 다중검정은 어떻게 다루는가

246개 중 최대값은 잡음만으로도 크게 나온다. 그래서 **순위 자체를 결론으로 쓰지
않는다.** 상위 후보 중 (a) 3폴드 모두 양수 (b) 기존 3축과 겹침 낮음
(c) 투수 동작이 바뀌는 기제가 있음 을 만족하는 것만 LB 로 보낸다.
LB 오답의 비용은 0 이므로 최종 판정은 LB 가 한다.

    .\.venv\Scripts\python.exe -u research\exp022_rank24.py
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
from exp003_sweep import build_contexts                     # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

EXP = "EXP022"
KGRID = [500, 1000, 2000, 5000, 20000]
PREV2 = {2022: (2020, 2021), 2023: (2021, 2022), 2024: (2022, 2023)}
WD = 0.65          # 현 Champion 의 차등 전역 가중 (LB 로 확정)


def main():
    E.start_experiment(EXP, "L0-V", "python research/exp022_rank24.py", "load")
    tr = build_df()
    season = tr["season"].to_numpy()
    y = tr["control_success"].to_numpy(np.float64)
    P = tr["pitcher_id"].to_numpy(np.int64)

    def g(c):
        return tr[c].to_numpy(np.float64)

    AX = {"hand": ((g("pitcher_hand") == g("batter_hand")).astype(int), 1000),
          "2S": ((g("strikes_before").astype(int) == 2).astype(int), 1000),
          "runner": ((g("num_runners_on") > 0).astype(int), 2000)}
    pv0, res0 = {}, {}
    for f in (2020, 2021, 2022, 2023, 2024):
        m = season == f
        pv0[f] = (np.load(os.path.join(ROOT, "exp", f"prod_champ_{f}.npy"))[:2].mean(0)
                  + post_for(tr, y, season < f, m))
        res0[f] = y[m] - pv0[f]

    def stats(ctx, src):
        p = np.concatenate([P[season == f] for f in src])
        c = np.concatenate([ctx[season == f] for f in src])
        r = np.concatenate([res0[f] for f in src])
        ok = np.isin(c, (0, 1))
        gg = pd.DataFrame({"p": p[ok], "c": c[ok], "r": r[ok]}).groupby(
            ["p", "c"])["r"].agg(["mean", "size"]).unstack()
        if ("size", 0) not in gg or ("size", 1) not in gg:
            return None
        n0, n1 = gg[("size", 0)].fillna(0), gg[("size", 1)].fillna(0)
        ne = (n0 * n1) / (n0 + n1).replace(0, np.nan)
        return pd.DataFrame({"d": gg[("mean", 1)] - gg[("mean", 0)], "ne": ne}).dropna()

    def apply_(st, k, ctx, m, w=1.0):
        t = st["d"] * st["ne"] / (st["ne"] + k)
        h = np.where(ctx[m] == 1, .5, np.where(ctx[m] == 0, -.5, 0.))
        return w * pd.Series(P[m]).map(t).fillna(0.).to_numpy() * h

    # 현 Champion = 편차 4축(pv0 에 포함) + 차등 3축 x0.65
    C3, base = {}, {}
    for f in (2022, 2023, 2024):
        m = season == f
        v = pv0[f].copy()
        for a, (ctx, k) in AX.items():
            v += apply_(stats(ctx, PREV2[f]), k, ctx, m, WD)
        C3[f], base[f] = v, r2(v, y[m])
    print(f"Champion(w=0.65) 로컬  2022 {base[2022]:.1f}  2023 {base[2023]:.1f}  "
          f"2024 {base[2024]:.1f}")

    m24 = season == 2024
    HD = apply_(stats(AX["hand"][0], (2022, 2023)), 1000, AX["hand"][0], m24, WD)
    TS = apply_(stats(AX["2S"][0], (2022, 2023)), 1000, AX["2S"][0], m24, WD)
    RN = apply_(stats(AX["runner"][0], (2022, 2023)), 2000, AX["runner"][0], m24, WD)

    CTX = build_contexts(tr, season)
    print(f"국면 {len(CTX)}개 — 3폴드 전이 (k 는 2022·2023 으로만 선택)\n")
    rows = []
    for i, (nm, ctx) in enumerate(sorted(CTX.items())):
        if i % 25 == 0:
            E.beat(f"{i}/{len(CTX)}")
        per = {k: [] for k in KGRID}
        ok = True
        for f in (2022, 2023, 2024):
            st = stats(ctx, PREV2[f])
            if st is None or not len(st):
                ok = False
                break
            m = season == f
            for k in KGRID:
                per[k].append(r2(C3[f] + apply_(st, k, ctx, m), y[m]) - base[f])
        if not ok:
            continue
        kb = max(KGRID, key=lambda k: np.mean(per[k][:2]))   # k 는 과거로만
        g22, g23, g24 = per[kb]
        add = apply_(stats(ctx, (2022, 2023)), kb, ctx, m24)
        ov = max(abs(float(np.corrcoef(add, v)[0, 1])) if add.std() > 0 else 0.
                 for v in (HD, TS, RN))
        rows.append(dict(ctx=nm, k=kb, g22=g22, g23=g23, g24=g24, overlap=ov,
                         frac=float(np.mean(ctx[m24] == 1))))

    rows.sort(key=lambda d: -d["g24"])
    print("=" * 100)
    print(f"{'국면 (2024 이득 상위 20)':<42}{'=1비율':>8}{'21→22':>9}{'22→23':>9}"
          f"{'23→24':>9}{'k':>7}{'겹침':>8}")
    print("=" * 100)
    for d in rows[:20]:
        star = " ★" if (d["g22"] > 0 and d["g23"] > 0 and d["overlap"] < 0.3) else ""
        print(f"{d['ctx']:<42}{d['frac']:>8.1%}{d['g22']:>+9.1f}{d['g23']:>+9.1f}"
              f"{d['g24']:>+9.1f}{d['k']:>7}{d['overlap']:>8.2f}{star}")

    g24 = np.array([d["g24"] for d in rows])
    print(f"\n2024 이득 분포  평균 {g24.mean():+.2f}  sd {g24.std():.2f}  "
          f"최대 {g24.max():+.1f}  양수비율 {(g24 > 0).mean():.1%}")
    print(f"  잡음만으로 246개 중 최대 기대값 ≈ +{g24.std()*2.9:.1f} "
          f"(정규 근사) -> 이보다 큰 것만 의미가 있다")
    live = [d for d in rows if d["g24"] > 0 and d["g22"] > 0 and d["g23"] > 0
            and d["overlap"] < 0.3]
    print(f"\n★ 3폴드 모두 양수 + 겹침<0.3 : {len(live)}개 "
          f"(우연 기대 {len(rows)/8:.0f})")
    for d in live[:12]:
        print(f"    {d['ctx']:<42}{d['g22']:>+8.1f}{d['g23']:>+8.1f}{d['g24']:>+8.1f}")

    E.set_hypothesis_status("L0-V", "MEASURED", level=0,
                            hypothesis="246 국면 2024 이득 순위", result=len(live))
    E.finish_experiment(dict(
        experiment_id=EXP, hypothesis_id="L0-V", type="A", level=0,
        started_at=E.read(E.CKPT)["start_time"], n=len(rows),
        top=[{k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()}
             for d in rows[:25]],
        n_live=len(live), decision="LB 후보 선정용",
        what_we_learned=("오라클 기각이 2S·주자를 못 잡는다는 사실로 EXP003 의 231개 "
                         "미측정 국면을 전부 전이 측정. LB 가 심판이라 2024 를 "
                         "선택에 쓸 수 있다.")))
    json.dump(rows, open(os.path.join(ROOT, "exp", "exp022_rank24.json"), "w"),
              indent=1, default=float)


if __name__ == "__main__":
    main()
