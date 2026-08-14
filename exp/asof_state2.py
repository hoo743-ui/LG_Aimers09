r"""AS-OF 축 확장 — §17 이 연 축에서 남은 세 갈래를 **현행(D) 위에서** 잰다.

## 왜 D 위에서 재는가

§17 의 표는 전부 `A 기준 55p` 대비 배수였다. 그때는 D 가 후보였으니 맞다.
지금 D 는 **22회차 제출본(1040.8656)에 들어가 있다.** 새 축의 질문은
"기준보다 나은가"가 아니라 **"D 에 더했을 때 오르는가"** 다. §17-d 의 J(전부)가
D 보다 낮았던 것(1.0303 vs 1.0322)이 그 차이를 이미 보여준다 — 축을 더하는
것만으로는 오르지 않는다.

그래서 이 스크립트는 **D 대비 배수**를 판정값으로 낸다 (A 대비도 같이 찍는다).

## 세 갈래 (SETUP.md "다음에 할 것")

| | 내용 | 왜 |
|---|---|---|
| **F** | `prev1/3/5 게임 rate` **vs `cur_rate`** | CAAFE 는 `prev1 - 통산rate` 를 썼는데 척도가 다른 두 값을 뺀 것이었다. 이제 `cur_succ`(2025 시즌내)가 있으니 **같은 시즌 안에서** 최근 경기 vs 시즌누적을 비교한다 — 질적으로 다른 비교다 |
| **E** | `cur - prior` | §17-d 에서 min 1.0332 로 D(1.0322)보다 근소 우위였다. D 와 **가산인지** 확인 |
| **B** | 타자 쪽 확장 | 지금 타자는 `bsucc`/`bmid` 2개뿐이다. 이력·폼변화·투타 격차를 더한다 |

`B` 의 투타 격차(`cur_succ - cur_bsucc`)는 축평행 분할이 한 번에 못 만드는
형태다 (`caafe.py` L29 의 관찰과 같은 이유).

## 판정 게이트 (§15-c 의 교훈)

`min(폴드별 배수)` 를 **주 판정값**으로 쓴다. 3/3 부호는 크기를 무시하고
기하평균은 퇴화 폴드(2023, Champion `rho^2` 76.5)에 지배된다. CAAFE 는
min=1.0001 로 3/3 을 통과하고도 평가셋 이득이 0 이었다.

프로토콜은 §17 과 동일 — 폴드 `f` 의 상수는 시즌 `<f` 에서, 학습 행도 같은
규율(시즌 `g` 행은 `<g` 상수)로 분해한다. `asof_state.build_state` 를 그대로
import 해서 쓴다. 복붙하면 측정이 의미를 잃는다.

    .\.venv\Scripts\python.exe -u exp\asof_state2.py

메모리 — 이 PC 는 가용 3GB 다. 설정별 행렬을 **미리 다 만들지 않고** 폴드
슬라이스만 그때그때 합친다 (`asof_state.py` 는 5개를 동시에 들고 있었다).
"""
import io
import json
import os
import sys
import time

import numpy as np
from catboost import CatBoostClassifier

from asof_state import HP, RATE_COLS, build_state    # 프로토콜 공유

sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # 콘솔은 cp949 다

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "exp", "cache")


def argv(flag, default):
    """--flag v  형태만 받는다. 기본값은 위 상수와 같아야 한다."""
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


FOLDS = tuple(int(v) for v in argv("--folds", "2022,2023,2024").split(","))
SEEDS = tuple(int(v) for v in argv("--seeds", "42,43").split(","))
ONLY = argv("--only", "")                    # 부분 문자열, 쉼표 구분
OUT = os.path.join(ROOT, "exp", argv("--out", "asof_state2.json"))

LBLS = [lb for _, _, lb in RATE_COLS]
NCOLS = ("asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n")


def main():
    meta = json.load(io.open(f"{CACHE}/cols.json", encoding="utf-8"))
    ixc = {c: i for i, c in enumerate(meta["cols"])}
    prod = meta["prod"]
    X = np.load(f"{CACHE}/X.npy", mmap_mode="r")
    y = np.load(f"{CACHE}/y.npy").astype(np.float64)
    season = np.load(f"{CACHE}/season.npy")
    C = lambda n: np.asarray(X[:, ixc[n]], dtype=np.float64)
    ID_P = C("pitcher_id").astype(np.int64)
    ID_B = C("batter_id").astype(np.int64)

    t = time.time()
    S = build_state(C, ID_P, ID_B, season)
    print(f"as-of 분해 생성 {time.time() - t:.0f}s", flush=True)

    L = lambda a: np.log1p(np.clip(a, 0, None)).astype(np.float32)
    F32 = lambda cols: np.column_stack(cols).astype(np.float32)

    # D — 22회차 제출본에 들어간 현행 13피처. 정의를 바꾸지 않는다.
    D = [S[f"cur_{lb}"] for lb in LBLS] + [L(S[f"cur_n_{n}"]) for n in NCOLS]

    # E — 현재 vs 이력 (§17-d 의 E 에서 log n 항은 D 와 중복이라 뺀다)
    E = [S[f"cur_{lb}"] - S[f"prior_{lb}"] for lb in LBLS]

    # F — 최근 경기 vs **시즌내 누적**. 여기가 CAAFE 와 갈리는 지점이다.
    p = [C(f"asof_pitcher_prev{k}_game_success_rate") for k in (1, 3, 5)]
    q = [C(f"asof_pitcher_prev{k}_game_middle_rate") for k in (1, 3, 5)]
    F = ([v - S["cur_succ"] for v in p] + [v - S["cur_mid"] for v in q])

    # B — 타자 쪽 확장: 이력 · 폼변화 · 투타 격차
    B = [S["prior_bsucc"], S["prior_bmid"],
         S["cur_bsucc"] - S["prior_bsucc"], S["cur_bmid"] - S["prior_bmid"],
         S["cur_succ"] - S["cur_bsucc"], S["cur_mid"] - S["cur_bmid"]]

    ADD = {"A 기준 55p": None, "D 현재(재현)": D,
           "D+E 차이": D + E, "D+F 최근vs시즌": D + F, "D+B 타자확장": D + B}
    if ONLY:
        keep = ONLY.split(",")
        ADD = {k: v for k, v in ADD.items()
               if any(s in k for s in keep)}
    for k, v in ADD.items():
        print(f"  {k:<16}{55 if v is None else 55 + len(v):>4}p", flush=True)

    base = np.column_stack([np.asarray(X[:, ixc[c]], dtype=np.float32)
                            for c in prod])
    EXTRA = {k: (None if v is None else F32(v)) for k, v in ADD.items()}
    del S, p, q, D, E, F, B

    R, done = {}, {}
    if os.path.exists(OUT):                      # 중단 시 이어서
        done = {k: {int(a): b for a, b in v.items()}
                for k, v in json.load(io.open(OUT, encoding="utf-8")).items()}
    for f in FOLDS:
        tr, va = season < f, season == f
        yv, ytr = y[va], y[tr].astype(int)
        print(f"\n=== 폴드 {f}  학습 {int(tr.sum()):,}행  검증 "
              f"{int(va.sum()):,}행 ===", flush=True)
        for nm, ex in EXTRA.items():
            if f in done.get(nm, {}):
                R.setdefault(nm, {})[f] = done[nm][f]
                print(f"  {nm:<16}(캐시) {done[nm][f]:>9.1f}", flush=True)
                continue
            t = time.time()
            Mtr = base[tr] if ex is None else np.hstack([base[tr], ex[tr]])
            Mva = base[va] if ex is None else np.hstack([base[va], ex[va]])
            acc = np.zeros(int(va.sum()))
            for sd in SEEDS:
                m = CatBoostClassifier(random_seed=sd, **HP)
                m.fit(Mtr, ytr)
                acc += m.predict_proba(Mva)[:, 1]
                del m
            r = 1e5 * np.corrcoef(acc / len(SEEDS), yv)[0, 1] ** 2
            R.setdefault(nm, {})[f] = r
            npf = Mtr.shape[1]
            del Mtr, Mva
            d0 = R.get("A 기준 55p", {}).get(f)
            print(f"  {nm:<16}{npf:>5}p{r:>10.1f}"
                  + (f"{r - d0:>+9.1f}" if d0 else " " * 9)
                  + f"{time.time() - t:>7.0f}s", flush=True)
            json.dump({k: {str(a): b for a, b in v.items()}
                       for k, v in R.items()},
                      io.open(OUT, "w", encoding="utf-8"), indent=1)

    den = R.get("D 현재(재현)") or R.get("A 기준 55p")
    if not den:
        return
    n = len(FOLDS)
    print(f"\n=== 배수표 (시드 {SEEDS} 평균, 분모 = 위 표의 첫 구성) ===")
    print(f"  {'구성':<16}" + "".join(f"{f:>10}" for f in FOLDS)
          + f"{'min':>10}{'3/3':>8}")
    for nm in R:
        d = [R[nm][f] / den[f] for f in FOLDS]
        print(f"  {nm:<16}" + "".join(f"{v:>10.4f}" for v in d)
              + f"{min(d):>10.4f}" + f"{str(sum(v > 1 for v in d))}/{n}".rjust(8))
    print("\n  판정 - vs D min > 1 이어야 채택 후보다. 3/3 만으로는 부족하다 "
          "(15-c: CAAFE 는 min 1.0001 로 3/3 통과했고 평가셋 이득 0).")


if __name__ == "__main__":
    main()
