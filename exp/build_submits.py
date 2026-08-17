r"""제출 아티팩트 3종 — Champion 모델 그대로, 후처리 표만 추가한다.

SUBMIT-1  Champion 그대로 (무결성 보존용 앵커)
SUBMIT-2  Champion + 잔차 기반 손 차등 (k=1000)
SUBMIT-3  SUBMIT-2 + 잔차 기반 2스트라이크 차등 (k=1000)

표는 **직전 두 시즌(2023, 2024)의 strictly out-of-fold 잔차**로만 만든다.
2023 잔차 <- 2019~2022 학습 모델,  2024 잔차 <- 2019~2023 학습 모델.
조회 키는 그 행 자신의 컬럼뿐이다 (규정 4 안전).
"""
import os, sys, shutil, hashlib, subprocess
import numpy as np, pandas as pd, joblib
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "exp"))
from path_alloc import build_df
from resid_table import post_for
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

tr = build_df(); season = tr["season"].to_numpy()
y = tr["control_success"].to_numpy(np.float64)
PID = tr["pitcher_id"].to_numpy(np.int64)
PH = tr["pitcher_hand"].to_numpy(np.int64)
BH = tr["batter_hand"].to_numpy(np.int64)
SS = tr["strikes_before"].to_numpy(np.int64)
res = {}
for f in (2023, 2024):
    m = season == f
    res[f] = y[m] - (np.load(os.path.join(ROOT, "exp", f"prod_champ_{f}.npy"))[:2].mean(0)
                     + post_for(tr, y, season < f, m))
SRC = (2023, 2024)
msrc = np.isin(season, SRC)
rsrc = np.concatenate([res[f] for f in SRC])

def diff(ctx, k):
    gg = pd.DataFrame({"p": PID[msrc], "c": ctx[msrc], "r": rsrc}).groupby(
        ["p", "c"])["r"].agg(["mean", "size"]).unstack()
    n0, n1 = gg[("size", 0)].fillna(0), gg[("size", 1)].fillna(0)
    d = gg[("mean", 1)] - gg[("mean", 0)]
    ne = (n0 * n1) / (n0 + n1).replace(0, np.nan)
    return (d * ne / (ne + k)).dropna()

SAME = (PH == BH).astype(int)
TWO = (SS == 2).astype(int)
RUN = (tr["num_runners_on"].to_numpy(np.int64) > 0).astype(int)
hand_d = diff(SAME, 1000)
two_d = diff(TWO, 1000)
run_d = diff(RUN, 2000)   # runner 축은 과거 전이에서 k=2000 이 최적이었다
phand = pd.Series(PH, index=PID).groupby(level=0).first()   # 투수별 손
print(f"  손 차등 표 {len(hand_d):,}명  |d| 중앙 {hand_d.abs().median():.5f}")
print(f"  2S 차등 표 {len(two_d):,}명  |d| 중앙 {two_d.abs().median():.5f}")

# 투수 4명이 양손 값(1,2)을 모두 갖는다 (스위치/기록 불일치, 39행).
# 그래서 투수손을 고정하지 않고 **키에 투수손을 포함**해 같은손 관계를 정확히 만든다.
tab_hand = {}
for pid, d in hand_d.items():
    for ph_ in (1, 2):
        for bh in (1, 2):
            tab_hand[(int(pid), ph_, bh)] = float(0.5 * d if bh == ph_ else -0.5 * d)
tab_two = {}
for pid, d in two_d.items():
    for s in (0, 1, 2):
        tab_two[(int(pid), s)] = float(0.5 * d if s == 2 else -0.5 * d)
print(f"  표 항목 수: 손 {len(tab_hand):,}  2S {len(tab_two):,}")

SPEC_HAND = {"w": 1.0, "cols": ["pitcher_id", "pitcher_hand", "batter_hand"], "table": tab_hand,
             "note": "잔차기반 손 차등 (OOF 2023+2024, k=1000, +-0.5d, 투수손 포함 키)"}
SPEC_TWO = {"w": 1.0, "cols": ["pitcher_id", "strikes_before"], "table": tab_two,
            "note": "잔차기반 2스트라이크 차등 (OOF 2023+2024, k=1000, +-0.5d)"}
tab_run = {}
for pid, d in run_d.items():
    for nr in range(0, 4):          # num_runners_on 은 0~3
        tab_run[(int(pid), nr)] = float(0.5 * d if nr > 0 else -0.5 * d)
SPEC_RUN = {"w": 1.0, "cols": ["pitcher_id", "num_runners_on"], "table": tab_run,
            "note": "잔차기반 주자유무 차등 (OOF 2023+2024, k=2000, +-0.5d)"}
print(f"  주자 차등 표 {len(run_d):,}명  항목 {len(tab_run):,}")

base = joblib.load(os.path.join(ROOT, "model_cand", "cat_asof_xl.pkl"))
print(f"  Champion 번들 로드: 모델 {len(base['models'])}개, 피처 {len(base['features'])}, "
      f"후처리 {len(base['platoon'])}축")

for name, extra in (("2", [SPEC_HAND]), ("3", [SPEC_HAND, SPEC_TWO, SPEC_RUN])):
    b = dict(base)
    b["platoon"] = list(base["platoon"]) + extra
    b["note"] = (str(base.get("note", "")) + f" | SUBMIT-{name}: "
                 + " + ".join(s["note"] for s in extra))
    pkl = os.path.join(ROOT, "model_cand", f"cat_submit_{name}.pkl")
    joblib.dump(b, pkl, compress=3)
    zp = os.path.join(ROOT, "submissions", f"cand_submit_{name}.zip")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "make_submit.py"),
                        "--model", os.path.relpath(pkl, ROOT),
                        "--requirements", "requirements_cat.txt",
                        "--out", os.path.relpath(zp, ROOT)],
                       cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    print(r.stdout[-300:] if r.returncode == 0 else r.stdout[-200:] + r.stderr[-500:])

# SUBMIT-1 = Champion 그대로 복사 (덮어쓰기 아님)
src = os.path.join(ROOT, "submissions", "cand_asof_xl.zip")
dst = os.path.join(ROOT, "submissions", "cand_submit_1.zip")
shutil.copy2(src, dst)
print(f"  SUBMIT-1 복사 {os.path.basename(dst)}")
print("\n=== SHA256 ===")
for n in ("cand_submit_1.zip", "cand_submit_2.zip", "cand_submit_3.zip"):
    p = os.path.join(ROOT, "submissions", n)
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    print(f"  {n:<22}{os.path.getsize(p)/1e6:>7.2f} MB  {h}")
