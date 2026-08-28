r"""10번째 후처리 축 — **잔차 기반 손 차등** (persistent weak signal 승격 탐침, 2026-08-29).

원장 HAND-DIFFERENTIAL 절 (exp/hand_diff.py) 의 보관 후보를 제출물로 만든다.

    표   원천 2023·2024 의 strictly OOF 잔차 (prod_champ_{f}.npy 2시드 평균 + 편차4축 후처리 뺀 것)
        d_p = mean(resid | 같은손) − mean(resid | 반대손)
        ds  = d_p · n_eff/(n_eff + k),   n_eff = n0·n1/(n0+n1),   k=1500 (과거 전이로 선택, 3/3 양수)
    적용  같은손 +0.5·ds, 반대손 −0.5·ds  ->  (pitcher_id, pitcher_hand, batter_hand) 표로 전개
    이력  23->24 실측 +8.4 (+0.89%), 시드폭 0.9, 자기흡수 보정 후 기대 +0.70%

학습 0회. 기존 script.py 의 platoon 리스트가 일반형이라 10번째 항목 추가로 끝난다.

    .\venv_submit\Scripts\python.exe -u exp\build_hd.py --w 1.0 --name cand_hd1
"""
import argparse, hashlib, io, os, sys, zipfile

import joblib
import numpy as np
import pandas as pd

import subname

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_ZIP = os.path.join(ROOT, "submissions", "260829_1_c4p3.zip")   # Champion 1105.9428
K = 1500

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def build(w, name, force=False, base=None):
    subname.check(name)
    sys.path.insert(0, os.path.join(ROOT, "exp"))
    import build_asof as ba
    from path_alloc import build_df

    base_zip = os.path.join(ROOT, base) if base else BASE_ZIP
    out = os.path.join(ROOT, "submissions", f"{name}.zip")
    assert force or not os.path.exists(out), f"이미 있다: {out}"

    tr = build_df()
    season = tr["season"].to_numpy()
    y = tr["control_success"].to_numpy(np.float64)
    P = tr["pitcher_id"].to_numpy(np.int64)
    BH = tr["batter_hand"].to_numpy(np.int64)
    PHa = tr["pitcher_hand"].to_numpy(np.int64)
    BB = tr["balls_before"].to_numpy(np.int64)
    SS = tr["strikes_before"].to_numpy(np.int64)
    OB = (tr["num_runners_on"].to_numpy(np.int64) > 0).astype(np.int64)
    PH = P * 10 + BH
    PHA = PH * 10 + (SS > BB).astype(np.int64)
    AX = [(P, PH), (PH, PHA), (PHA, PHA * 100 + (BB * 4 + SS)), (PH, PH * 10 + OB)]
    SAME = (PHa == BH).astype(np.int64)

    # strictly OOF 잔차 (모델 + 편차4축 후처리 뺀 것) — hand_diff.py 와 동일 구성
    parts = []
    for f in (2023, 2024):
        mt, mv = season < f, season == f
        pred = np.load(os.path.join(ROOT, "exp", f"prod_champ_{f}.npy"))[:2].mean(0)
        post = np.column_stack([ba.look(*ba.nested_dev(p[mt], c[mt], y[mt], k), c[mv])
                                for (p, c), k in zip(AX, ba.KSH)]) @ ba.WPOST
        assert len(pred) == mv.sum(), (f, len(pred), mv.sum())
        parts.append(pd.DataFrame({"p": P[mv], "s": SAME[mv],
                                   "r": y[mv] - (pred + post)}))
    t = pd.concat(parts)
    gg = t.groupby(["p", "s"])["r"].agg(["mean", "size"]).unstack()
    n0 = gg[("size", 0)].fillna(0)
    n1 = gg[("size", 1)].fillna(0)
    d = gg[("mean", 1)] - gg[("mean", 0)]
    ne = (n0 * n1) / (n0 + n1).replace(0, np.nan)
    ds = (d * ne / (ne + K)).dropna()
    print(f"  투수 {len(ds)}명  ds sd {ds.std():.6f}  n_eff 중앙 {ne.dropna().median():.0f}")

    hand = pd.DataFrame({"p": P, "ph": PHa}).drop_duplicates("p").set_index("p")["ph"]
    table = {}
    for p, v in ds.items():
        ph = int(hand[p])
        for bh in (1, 2):
            table[(int(p), ph, int(bh))] = float(0.5 * v if bh == ph else -0.5 * v)

    src = zipfile.ZipFile(base_zip)
    b = joblib.load(io.BytesIO(src.read("model/rf.pkl")))
    pl = b["platoon"]
    assert len(pl) == 9, len(pl)
    pl.append({"w": float(w), "cols": ["pitcher_id", "pitcher_hand", "batter_hand"],
               "table": table,
               "note": f"HD 잔차 손 차등 k={K} (원천 23·24 OOF 잔차, n_eff 축소), w={w:g}"})
    b["platoon"] = pl
    b["note"] = b["note"].split("|")[0] + f"| HD 잔차 손 차등 k={K} w={w:g} (10축), 기존 9가중 불변"

    buf = io.BytesIO()
    joblib.dump(b, buf, compress=3)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("model/rf.pkl", buf.getvalue())
        for nm in ("script.py", "requirements.txt"):
            z.writestr(nm, src.read(nm))
    h = hashlib.sha256(open(out, "rb").read()).hexdigest()
    print(f"{name}.zip   HD 10축 w={w:g}   기반 {os.path.basename(base_zip)}   셀 {len(table)}")
    print(f"  10가중 {[round(x['w'],4) for x in pl]}")
    print(f"  sha256 {h[:16]}   {os.path.getsize(out):,} bytes")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=float, default=1.0)
    ap.add_argument("--name", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--base", default=None)
    a = ap.parse_args()
    build(a.w, a.name, a.force, a.base)
