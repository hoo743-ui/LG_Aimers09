r"""22회차 후보 — AS-OF 현재상태 분해 (D). 오늘 유일하게 크기가 다른 축.

## 근거 (`exp/asof_state.py`, 시드 2개 평균, walk-forward)

| 구성 | 2022 | 2023 | 2024 | 기하평균 | **min gain** | 3/3 |
|---|---:|---:|---:|---:|---:|---|
| C 이력(prior) | 1.0005 | 1.0641 | 1.0047 | 1.0227 | 1.0005 | 3/3 |
| **D 현재(cur)** | **1.0322** | **2.3381** | **1.1291** | **1.3968** | **1.0322** | 3/3 |
| E 현재-이력 | 1.0332 | 1.7301 | 1.1070 | 1.2555 | 1.0332 | 3/3 |
| J 전부 | 1.0303 | 2.3376 | 1.1195 | 1.3918 | 1.0303 | 3/3 |

**게이트 B(min gain)가 이번엔 실질적이다** — CAAFE 는 min 1.0001 로 3/3 을 통과하고도
평가셋 이득이 0 이었다(§15-c). 여기는 min 1.0322 로 322배 크다.

`C 이력` 이 거의 0 인데 `D 현재` 가 크다는 대조가 핵심이다 — 새 정보는 "그 투수가
원래 어떤가"가 아니라 **"지금 어떤가"** 다.

## 무엇이 새 정보인가

모델이 보는 `asof_pitcher_success_rate` 는 **통산**이라 이력과 현재 폼이 섞여 있고,
모델은 그 투수의 직전 시즌말 통산을 모르므로 **원리적으로 못 가른다**.
학습 데이터에서 그 상수를 빼주면 갈린다.

    cur_n    = asof_n(행) - prior_n[선수]
    cur_rate = ((asof_n * asof_rate)(행) - prior_events[선수]) / cur_n

검증 — 2024 폴드에서 `cur_n` 이 실제 시즌내 순번과 **100.0000%** 일치(음수 0%),
`cur_rate` 복원 평균절대오차 **3.1e-6**. test 5행에서도 전부 정합
(예: 투수 21813 asof_n 3,465 - train 3,085 = 2025 시즌 380구, 7월).

## 규정 5)

행 자신의 공식 `asof_*` 컬럼(`data_description.md` L182 사용 허가 명시) +
학습 데이터만으로 만든 선수별 상수. 평가셋의 다른 행을 안 본다.

## 아핀 — 클린 노선

§16 에서 확인한 방식을 그대로 쓴다 (평가셋 정보 미사용).

    r_hat = 최근 3시즌 선형외삽
    m_hat = r_hat + mean(m - r)      워크포워드
    center = (r_hat - A*m_hat)/(1-A),  A = 1.09 (10회차 문서화 상수)

    .\venv_submit\Scripts\python.exe -u exp\build_asof.py            # 적률만
    .\venv_submit\Scripts\python.exe -u exp\build_asof.py --build    # pkl/zip
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

ALPHA = 1.09
WPOST = np.array([0.20, 0.825, 0.280, 0.45])
KSH = [300, 2000, 800, 2000]
HP = dict(iterations=1200, learning_rate=0.02, depth=6, l2_leaf_reg=100.0,
          border_count=32, loss_function="Logloss", verbose=0,
          thread_count=16, allow_writing_files=False)
OUT_PKL = os.path.join(ROOT, "model_cand", "cat_asof.pkl")
OUT_ZIP = os.path.join(ROOT, "submissions", "cand_asof.zip")
# --form 일 때 (23회차). 22회차 아티팩트를 절대 덮어쓰지 않는다.
OUT_PKL_F = os.path.join(ROOT, "model_cand", "cat_asof_f.pkl")
OUT_ZIP_F = os.path.join(ROOT, "submissions", "cand_asof_f.zip")


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
    return u, cnt * (cell - pmean[np.searchsorted(pu, par)]) / (cnt + k)


def look(u, d, keys):
    ix = np.clip(np.searchsorted(u, keys), 0, len(u) - 1)
    ok = u[ix] == keys
    out = np.zeros(len(keys))
    out[ok] = d[ix[ok]]
    return out


def prior_tables(df, mask):
    """`mask` 구간에서 선수별 (표본수, 사건수...) 를 만든다.

    표본수는 **행 수**다 (as-of 카운터가 정확히 이 행들을 센다 — 100.0000% 검증).
    사건수는 그 구간 마지막 행의 `n*rate` 최대값이다 (통산이 단조증가).
    """
    out = {}
    for kind, (ncol, idcol) in sc.ASOF_NCOL.items():
        ids = df.loc[mask, idcol].to_numpy(np.int64)
        n = df.loc[mask, ncol].to_numpy(np.float64)
        u, cnt = np.unique(ids, return_counts=True)
        cols = [rc for rc, _, _, k in sc.ASOF_SPEC if k == kind]
        ev = []
        o = np.argsort(ids, kind="stable")
        ks, s0 = np.unique(ids[o], return_index=True)
        for rc in cols:
            tot = (n * np.nan_to_num(df.loc[mask, rc].to_numpy(np.float64)))[o]
            ev.append(np.maximum.reduceat(tot, s0))
        out[kind] = {int(k): tuple([float(c)] + [float(e[i]) for e in ev])
                     for i, (k, c) in enumerate(zip(u, cnt))}
    return out


def add_state(df, tabs):
    """`tabs` 를 임시 번들로 삼아 추론과 **같은 함수**로 파생컬럼을 만든다."""
    return sc.attach_asof_state(df, {"asof_prior": tabs})


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
    ap.add_argument("--form", action="store_true",
                    help="F(최근경기 vs 시즌내 누적) 6개를 더한다 — 23회차")
    a = ap.parse_args()
    acols = sc.ASOF_COLS + (sc.FORM_COLS if a.form else [])
    out_pkl, out_zip = ((OUT_PKL_F, OUT_ZIP_F) if a.form else (OUT_PKL, OUT_ZIP))

    t0 = time.time()
    tc = pd.read_csv(os.path.join(ft.DATA_DIR, "test.csv"),
                     encoding="utf-8-sig", nrows=0).columns
    allf = [c for c in tc if c != ft.ID]
    tr = pd.read_csv(os.path.join(ft.DATA_DIR, "train.csv"),
                     encoding="utf-8-sig", usecols=allf + [ft.TARGET])
    tm = ft.load_trackman()
    tr = ft.attach_ctx_train(tr, tm)
    ctxf = [c for c in ft.COUNT_FEATS + ft.HAND_FEATS
            if c not in ("tmc_n", "tmh_n")]
    c_all, h_all = ft.ctx_tables(tm, 9999)
    ctx_pack = {"count": ft.pack_table(c_all, ft.COUNT_KEY),
                "hand": ft.pack_table(h_all, ft.HAND_KEY),
                "hand_map": {str(k): v for k, v in ft.HAND.items()},
                "count_key": ft.COUNT_KEY, "hand_key": ft.HAND_KEY}

    # --- 학습용: 시즌 g 의 행은 <g 상수로 분해 (추론 시점과 같은 형태) ---
    season = tr["season"].to_numpy()
    for c in acols:
        tr[c] = np.nan
    for g in sorted(np.unique(season)):
        m = season == g
        tabs = prior_tables(tr, season < g)
        part = add_state(tr.loc[m].copy(), tabs)
        for c in acols:
            tr.loc[m, c] = part[c].to_numpy()
    features = list(allf) + ctxf + acols
    y = tr[ft.TARGET].to_numpy(np.float64)
    print(f"피처 {len(features)}개 (기본 {len(allf)} + ctx {len(ctxf)} + "
          f"AS-OF {len(acols)})   {len(tr):,}행   {time.time()-t0:.0f}s")
    print(f"  cur_n>0 비율 {float((tr['cur_logn_pitch'] > 0).mean()):.1%}")

    # --- 편차 후처리 4축 ---
    P = tr["pitcher_id"].to_numpy(np.int64)
    BH = tr["batter_hand"].to_numpy(np.int64)
    BB = tr["balls_before"].to_numpy(np.int64)
    SS = tr["strikes_before"].to_numpy(np.int64)
    OB = (tr["num_runners_on"].to_numpy(np.int64) > 0).astype(np.int64)
    PH = P * 10 + BH
    PHA = PH * 10 + (SS > BB).astype(np.int64)
    AX = [(P, PH), (PH, PHA), (PHA, PHA * 100 + (BB * 4 + SS)), (PH, PH * 10 + OB)]

    # --- 2024 홀드아웃: 대조 vs 신규, 그리고 클린 아핀용 적률 ---
    m_tr, m_va = season < 2024, season == 2024
    post = np.column_stack([
        look(*nested_dev(p[m_tr], c[m_tr], y[m_tr], k), c[m_va])
        for (p, c), k in zip(AX, KSH)]) @ WPOST
    drop = sc.FORM_COLS if a.form else sc.ASOF_COLS
    base_f = [c for c in features if c not in drop]
    ref = f"대조 {len(base_f)}p" + (" (D=22회차)" if a.form else "")
    new = f"신규 {len(features)}p"
    P2 = {}
    for lbl, fs in ((ref, base_f), (new, features)):
        t = time.time()
        mm = pipeline(fs, 42)
        mm.fit(tr.loc[m_tr, fs], y[m_tr].astype(int))
        P2[lbl] = mm.predict_proba(tr.loc[m_va, fs])[:, 1] + post
        print(f"  {lbl} 학습 {time.time()-t:.0f}s", flush=True)
    yv = y[m_va]
    r0 = 1e5 * np.corrcoef(P2[ref], yv)[0, 1] ** 2
    r1 = 1e5 * np.corrcoef(P2[new], yv)[0, 1] ** 2
    print(f"\n2024 홀드아웃  대조 {r0:.1f}   신규 {r1:.1f}   배수 {r1/r0:.4f}")

    # --- 클린 아핀 (§16) ---
    print("\n=== 클린 아핀 (평가셋 정보 미사용) ===")
    ms, rr = [], []
    for f in (2022, 2023, 2024):
        mt, mv = season < f, season == f
        po = np.column_stack([
            look(*nested_dev(p[mt], c[mt], y[mt], k), c[mv])
            for (p, c), k in zip(AX, KSH)]) @ WPOST
        mm = pipeline(features, 42)
        mm.fit(tr.loc[mt, features], y[mt].astype(int))
        pv = mm.predict_proba(tr.loc[mv, features])[:, 1] + po
        ms.append(float(pv.mean()))
        rr.append(float(y[mv].mean()))
        print(f"  {f}  r={rr[-1]:.6f}  m={ms[-1]:.6f}  m-r={ms[-1]-rr[-1]:+.6f}",
              flush=True)
    d = float(np.mean(np.array(ms) - np.array(rr)))
    seasons = np.array(sorted(np.unique(season)))
    rate = np.array([float(y[season == s].mean()) for s in seasons])
    sl, ic = np.polyfit(seasons[-3:], rate[-3:], 1)
    r_hat = float(sl * 2025 + ic)
    m_hat = r_hat + d
    center = (r_hat - ALPHA * m_hat) / (1 - ALPHA)
    print(f"\n  mean(m-r)={d:.6f}   r_hat={r_hat:.6f}   m_hat={m_hat:.6f}")
    print(f"  alpha={ALPHA:.6f}   center={center:.6f}")
    # 기준선 — --form 은 22회차 실측(1040.8656) 위에 곱한다. 22회차는 2024
    # 배수의 78~87% 가 평가셋으로 넘어왔으므로(17-i) 그 구간도 같이 찍는다.
    b0 = 1040.8656 if a.form else 955.64
    g = r1 / r0 - 1.0
    print(f"  기대 LB  {b0:.4f} x {r1/r0:.4f} = **{b0*(1+g):.1f}**")
    if a.form:
        print(f"  전이 78~87% 가정  {b0*(1+g*0.78):.1f} ~ {b0*(1+g*0.87):.1f}")
    if not a.build:
        print("\n(--build 를 주면 pkl/zip 을 만든다)")
        return

    models = []
    for s in range(42, 42 + a.seeds):
        t = time.time()
        mm = pipeline(features, s)
        mm.fit(tr[features], y.astype(int))
        models.append(mm)
        print(f"  seed {s} 학습 {time.time()-t:.0f}s", flush=True)

    (u1, t1), (uC, tC), (uN, tN), (u3, t3) = [
        nested_dev(p, c, y, k) for (p, c), k in zip(AX, KSH)]
    tab1 = {(int(k // 10), int(k % 10)): float(x) for k, x in zip(u1, t1)}
    tabC = {}
    for k, x in zip(uC, tC):
        pid, hd, adv = int(k // 100), int((k // 10) % 10), int(k % 10)
        for b in range(4):
            for st in range(3):
                if int(st > b) == adv:
                    tabC[(pid, hd, b, st)] = float(x)
    tabN = {}
    for k, x in zip(uN, tN):
        cn, rest = int(k % 100), k // 100
        tabN[(int(rest // 100), int((rest // 10) % 10),
              int(cn // 4), int(cn % 4))] = float(x)
    tab3 = {}
    for k, x in zip(u3, t3):
        pid, hd, ob = int(k // 100), int((k // 10) % 10), int(k % 10)
        for nr in ([0] if ob == 0 else [1, 2, 3]):
            tab3[(pid, hd, nr)] = float(x)
    CK = ["pitcher_id", "batter_hand", "balls_before", "strikes_before"]
    b = {"models": models, "alpha": float(ALPHA), "center": float(center),
         "features": features, "spec": [f"cat-s{s}" for s in
                                        range(42, 42 + a.seeds)],
         "shift": None, "detrend": None, "ctx": ctx_pack,
         "asof_prior": prior_tables(tr, np.ones(len(tr), bool)),
         "platoon": [
             {"w": float(WPOST[0]), "cols": ["pitcher_id", "batter_hand"],
              "table": tab1, "note": "dev(투수x타자손|부모=투수), n/(n+300)"},
             {"w": float(WPOST[1]), "cols": CK, "table": tabC,
              "note": "dev(플래툰x투수유리|부모=플래툰), n/(n+2000), 12칸 전개"},
             {"w": float(WPOST[2]), "cols": CK, "table": tabN,
              "note": "dev(플래툰x투수유리x카운트|부모=플래툰x투수유리), n/(n+800)"},
             {"w": float(WPOST[3]),
              "cols": ["pitcher_id", "batter_hand", "num_runners_on"],
              "table": tab3, "note": "dev(플래툰x주자유무|부모=플래툰), n/(n+2000)"}],
         "note": (f"catboost x{a.seeds}; AS-OF 현재상태 분해 {len(acols)}개 "
                  f"in-model; p += 편차4 -> center+{ALPHA}*(p-center) -> clip. "
                  f"walk-forward 배수 1.3968 (min 1.0322, 3/3). "
                  f"클린 아핀 (r_hat 추세외삽 + m_hat 워크포워드), 평가셋 정보 미사용")}
    joblib.dump(b, out_pkl, compress=3)
    print(f"\n저장 {out_pkl} ({os.path.getsize(out_pkl)/1e6:.1f} MB)")
    r = subprocess.run(
        [sys.executable, os.path.join(ROOT, "make_submit.py"),
         "--model", os.path.relpath(out_pkl, ROOT),
         "--requirements", "requirements_cat.txt",
         "--out", os.path.relpath(out_zip, ROOT)],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
        errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    print(r.stdout[-800:] if r.returncode == 0
          else r.stdout[-400:] + r.stderr[-800:])


if __name__ == "__main__":
    main()
