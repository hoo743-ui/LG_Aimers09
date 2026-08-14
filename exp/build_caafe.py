r"""20회차 후보 빌드 — CAAFE11 + 기존 편차 후처리.

## 근거

3폴드 검증 (`exp/caafe_stack.py`, `exp/phase12.py`), 기존 편차 후처리 위 배수:

    2022 1.0001   2023 1.0465   2024 1.0308   기하평균 1.0256   3/3

CAAFE 단독 이득 1.0279 중 91% 가 후처리 위에서 보존된다 — 겹침이 아니라 새 정보다.
955.22 환산 **979.7 (+24.5)**. 최악 폴드가 1.0001 이므로 하방은 0 에 가깝다.

**규정 4) 와 무관하다** — CAAFE 11개는 전부 그 행 안의 값만 쓰는 산술이라
test.csv 에 그 행 하나만 있어도 값이 같다.

## 아핀 처리 — 새 모델은 평가셋 적률이 다르다

`alpha`/`center` 는 챔피언 예측의 평가셋 적률에서 나온 값이다 (LB 역산).
새 모델은 그 적률이 달라지므로 그대로 쓰면 어긋난다. 그래서 새 예측을
**로컬 2024 에서 챔피언의 평균·분산에 맞춰** 정합하고, 그 변환을 아핀에 흡수한다.

    p_adj = A0 + B0*p_new      B0 = std_champ/std_new,  A0 = mean_champ - B0*mean_new
    alpha' = alpha*B0
    center' = [center*(1-alpha) + alpha*A0] / (1 - alpha*B0)

이러면 **평균 조건 `A*m + B = r` 이 그대로 보존된다** (14회차에서 2.73점을 잃은
조건이다). 기울기 최적점은 rho 개선분(1.27%)만큼 어긋나지만 손실이
`(dA)^2*V/BASE*1e5` = **0.15점**이라 무시한다.

두 모델의 로컬->평가 drift 가 같은 인구 변화에서 오므로 차이분만 남는다는 가정
위에 있다. 이것이 이 후보의 유일한 미검증 위험이다.

    .\.venv\Scripts\python.exe -u exp\build_caafe.py            # 적률·아핀만
    .\.venv\Scripts\python.exe -u exp\build_caafe.py --build    # pkl/zip 까지
"""
import argparse
import importlib.util
import os
import subprocess
import sys
import time

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ft = _load("ft", "final_train.py")
sc = _load("sc", "script.py")

ALPHA0, CENTER0 = 1.089079, 0.620698      # 19회차 제출본
WPOST = np.array([0.20, 0.825, 0.280, 0.45])
KSH = [300, 2000, 800, 2000]
HP = dict(iterations=1200, learning_rate=0.02, depth=6, l2_leaf_reg=100.0,
          border_count=32, loss_function="Logloss", verbose=0,
          thread_count=16, allow_writing_files=False)
OUT_PKL = os.path.join(ROOT, "model_cand", "cat_caafe.pkl")
OUT_ZIP = os.path.join(ROOT, "submissions", "cand_caafe.zip")


def nested_dev(parent, child, y, k):
    o = np.argsort(child, kind="stable")
    Ys, Ps, Cs = y[o], parent[o], child[o]
    u, s = np.unique(Cs, return_index=True)
    cnt = np.diff(np.append(s, len(Cs)))
    cell = np.add.reduceat(Ys, s) / cnt
    par = Ps[s]
    op = np.argsort(parent, kind="stable")
    Yp, Pp = y[op], parent[op]
    pu, ps = np.unique(Pp, return_index=True)
    pc = np.diff(np.append(ps, len(Pp)))
    pmean = np.add.reduceat(Yp, ps) / pc
    return u, cnt * (cell - pmean[np.searchsorted(pu, par)]) / (cnt + k), cnt


def lookup(u, dev, keys):
    ix = np.clip(np.searchsorted(u, keys), 0, len(u) - 1)
    ok = u[ix] == keys
    out = np.zeros(len(keys))
    out[ok] = dev[ix[ok]]
    return out


def keys_of(df):
    P = df["pitcher_id"].to_numpy(np.int64)
    BH = df["batter_hand"].to_numpy(np.int64)
    BB = df["balls_before"].to_numpy(np.int64)
    SS = df["strikes_before"].to_numpy(np.int64)
    OB = (df["num_runners_on"].to_numpy(np.int64) > 0).astype(np.int64)
    PH = P * 10 + BH
    PHA = PH * 10 + (SS > BB).astype(np.int64)
    return [(P, PH), (PH, PHA), (PHA, PHA * 100 + (BB * 4 + SS)),
            (PH, PH * 10 + OB)]


def pipeline(features, seed):
    cat = [c for c in ft.CAT_COLS if c in features]
    num = [c for c in features if c not in cat]
    pre = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                                unknown_value=-1), cat),
         ("num", "passthrough", num)])
    return Pipeline([("pre", pre),
                     ("clf", CatBoostClassifier(random_seed=seed, **HP))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()

    t0 = time.time()
    test_cols = pd.read_csv(os.path.join(ft.DATA_DIR, "test.csv"),
                            encoding="utf-8-sig", nrows=0).columns
    allf = [c for c in test_cols if c != ft.ID]
    train = pd.read_csv(os.path.join(ft.DATA_DIR, "train.csv"),
                        encoding="utf-8-sig", usecols=allf + [ft.TARGET])
    tm = ft.load_trackman()
    train = ft.attach_ctx_train(train, tm)
    ctxf = [c for c in ft.COUNT_FEATS + ft.HAND_FEATS
            if c not in ("tmc_n", "tmh_n")]
    c_all, h_all = ft.ctx_tables(tm, 9999)
    ctx_pack = {"count": ft.pack_table(c_all, ft.COUNT_KEY),
                "hand": ft.pack_table(h_all, ft.HAND_KEY),
                "hand_map": {str(k): v for k, v in ft.HAND.items()},
                "count_key": ft.COUNT_KEY, "hand_key": ft.HAND_KEY}

    train = sc.attach_caafe(train)          # 추론과 **같은 함수**를 쓴다
    features = list(allf) + ctxf + sc.CAAFE_COLS
    y = train[ft.TARGET].to_numpy(np.float64)
    season = train["season"].to_numpy()
    print(f"피처 {len(features)}개 (기본 {len(allf)} + ctx {len(ctxf)} + "
          f"CAAFE {len(sc.CAAFE_COLS)})   {len(train):,}행   "
          f"{time.time() - t0:.0f}s")

    # ---- 1) 적률 정합용: 2024 홀드아웃 1시드 ----
    tr, va = season < 2024, season == 2024
    AX = keys_of(train)
    post = np.column_stack([
        lookup(*nested_dev(par[tr], ch[tr], y[tr], k)[:2], ch[va])
        for (par, ch), k in zip(AX, KSH)]) @ WPOST
    yv = y[va]
    # 대조군은 **같은 코드 경로의 55피처 판**이어야 한다. champ_oof_*.npy 는
    # 다른 설정으로 만든 아티팩트라 분모로 쓰면 배수가 부풀려진다.
    base_f = [c for c in features if c not in sc.CAAFE_COLS]
    P2 = {}
    for lbl, fs in (("대조 55p", base_f), ("신규 66p", features)):
        t = time.time()
        mm = pipeline(fs, 42)
        mm.fit(train.loc[tr, fs], y[tr].astype(int))
        P2[lbl] = mm.predict_proba(train.loc[va, fs])[:, 1] + post
        print(f"  {lbl} 학습 {time.time() - t:.0f}s", flush=True)
    p_ch, p_new = P2["대조 55p"], P2["신규 66p"]
    r_new = 1e5 * np.corrcoef(p_new, yv)[0, 1] ** 2
    r_ch = 1e5 * np.corrcoef(p_ch, yv)[0, 1] ** 2
    print(f"\n2024 홀드아웃 (둘 다 +후처리, 같은 시드/경로)")
    print(f"  대조 55p rho^2 {r_ch:>8.1f}   신규 66p {r_new:>8.1f}   "
          f"배수 {r_new / r_ch:.4f}")
    print(f"  평균  champ {p_ch.mean():.6f}  신규 {p_new.mean():.6f}")
    print(f"  표준편차 champ {p_ch.std():.6f}  신규 {p_new.std():.6f}")

    B0 = p_ch.std() / p_new.std()
    A0 = p_ch.mean() - B0 * p_new.mean()
    alpha = ALPHA0 * B0
    center = (CENTER0 * (1 - ALPHA0) + ALPHA0 * A0) / (1 - alpha)
    print(f"\n적률 정합  B0={B0:.6f}  A0={A0:+.6f}")
    print(f"  alpha {ALPHA0:.6f} -> {alpha:.6f}")
    print(f"  center {CENTER0:.6f} -> {center:.6f}")
    print(f"\n기대 LB  955.2193 x {r_new / r_ch:.4f} = "
          f"**{955.2193198652 * r_new / r_ch:.1f}**")
    if not a.build:
        print("\n(--build 를 주면 pkl/zip 을 만든다)")
        return

    # ---- 2) 전체 학습 + 편차표 (전 구간) ----
    models = []
    for s in range(42, 42 + a.seeds):
        t = time.time()
        mm = pipeline(features, s)
        mm.fit(train[features], y.astype(int))
        models.append(mm)
        print(f"  seed {s} 학습 {time.time() - t:.0f}s")
    # 편차표 4축 — 키 인코딩은 make_nest.py(19회차)와 동일해야 한다
    (u1, t1, c1), (uC, tC, cC), (uN, tN, cN), (u3, t3, c3) = [
        nested_dev(par, ch, y, k) for (par, ch), k in zip(AX, KSH)]
    tab1 = {(int(k // 10), int(k % 10)): float(x) for k, x in zip(u1, t1)}
    tabC = {}
    for k, x in zip(uC, tC):                    # ADV 를 (b,st) 12칸으로 펼친다
        pid, hand, adv = int(k // 100), int((k // 10) % 10), int(k % 10)
        for b in range(4):
            for st in range(3):
                if int(st > b) == adv:
                    tabC[(pid, hand, b, st)] = float(x)
    tabN = {}
    for k, x in zip(uN, tN):
        cnt, rest = int(k % 100), k // 100
        tabN[(int(rest // 100), int((rest // 10) % 10),
              int(cnt // 4), int(cnt % 4))] = float(x)
    tab3 = {}
    for k, x in zip(u3, t3):                    # 주자유무 -> num_runners_on
        pid, hand, ob = int(k // 100), int((k // 10) % 10), int(k % 10)
        for nr in ([0] if ob == 0 else [1, 2, 3]):
            tab3[(pid, hand, nr)] = float(x)
    CK = ["pitcher_id", "batter_hand", "balls_before", "strikes_before"]
    platoon = [
        {"w": float(WPOST[0]), "cols": ["pitcher_id", "batter_hand"],
         "table": tab1, "note": "dev(투수x타자손 | 부모=투수), n/(n+300)"},
        {"w": float(WPOST[1]), "cols": CK, "table": tabC,
         "note": "dev(플래툰x투수유리 | 부모=플래툰), n/(n+2000). 12칸 전개"},
        {"w": float(WPOST[2]), "cols": CK, "table": tabN,
         "note": "dev(플래툰x투수유리x카운트 | 부모=플래툰x투수유리), n/(n+800)"},
        {"w": float(WPOST[3]),
         "cols": ["pitcher_id", "batter_hand", "num_runners_on"],
         "table": tab3, "note": "dev(플래툰x주자유무 | 부모=플래툰), n/(n+2000)"},
    ]
    for nm, tb, cc in (("플래툰", tab1, c1), ("거친", tabC, cC),
                       ("카운트|거친", tabN, cN), ("주자유무", tab3, c3)):
        print(f"  {nm:<12}{len(tb):>9,}칸  중앙n {int(np.median(cc)):,}")

    b = {"models": models, "alpha": float(alpha), "center": float(center),
         "features": features, "spec": [f"cat-s{s}" for s in
                                        range(42, 42 + a.seeds)],
         "shift": None, "detrend": None, "ctx": ctx_pack,
         "platoon": platoon,
         "note": (f"catboost x{a.seeds}; CAAFE11 in-model; "
                  f"p += 편차4 -> center+{alpha:.6f}*(p-center) -> clip(0,1). "
                  f"3폴드 1.0256 (3/3), 기대 "
                  f"{955.2193198652 * r_new / r_ch:.1f}")}
    joblib.dump(b, OUT_PKL, compress=3)
    print(f"\n저장 {OUT_PKL} ({os.path.getsize(OUT_PKL) / 1e6:.1f} MB)")

    r = subprocess.run(
        [sys.executable, os.path.join(ROOT, "make_submit.py"),
         "--model", os.path.relpath(OUT_PKL, ROOT),
         "--requirements", "requirements_cat.txt",
         "--out", os.path.relpath(OUT_ZIP, ROOT)],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
        errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    print(r.stdout[-800:] if r.returncode == 0
          else r.stdout[-400:] + r.stderr[-800:])


if __name__ == "__main__":
    main()
