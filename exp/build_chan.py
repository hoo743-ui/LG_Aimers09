r"""채널 혼합 후보 — Champion 모델 불변 + 채널 모델 3개 가산 혼합 (2026-08-29, 고위험 트랙 1).

EXP059: 순수 3채널 곱은 참사(−270)지만 base 0.75 + chan 0.25 는 폴드 2024 +7.67.
Champion 의 y-모델 예측은 **비트 단위 그대로** 두고, 채널 모델 3개(middle·reverse·ball,
전체 train 학습)의 곱 공식 예측을 가중 w 로 혼합한다. 혼합 후 후처리·아핀은 기존 그대로.

    preds = (1−w)·mean(champion models) + w·(1−p_mid)(1−p_rev)(0.95−0.36·p_ball)
            + platoon_adjust + affine

w 는 LB 2점 포물선 좌표다 (w=0 이 Champion 실측점). 채널 라벨은 asof 카운터 역산으로
train 에서 100% 복원된 것 (exp/cache/pitch_labels.npz), 정렬은 (pitcher, asof_n) 유일 키.

    .\venv_submit\Scripts\python.exe -u exp\build_chan.py --w 0.25 --name cand_chan25
"""
import argparse, hashlib, io, os, sys, zipfile

import joblib
import numpy as np
import pandas as pd

import subname

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_ZIP = os.path.join(ROOT, "submissions", "260829_1_c4p3.zip")   # Champion 1105.9428

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PATCH_OLD = """    preds = acc / len(models)

    preds = preds + platoon_adjust(bundle, X)"""
PATCH_NEW = """    preds = acc / len(models)

    # 채널 혼합 (2026-08-29). 채널 모델 3개는 학습 데이터에서 역산 복원한
    # middle/reverse/ball 라벨로 학습된 상수 모델이고, 각 행은 자기 피처만으로
    # 예측된다 — 행 독립 (규정 4 안전). 곱 공식은 타깃의 합성 정의를 따른다.
    ch = bundle.get("chan")
    if ch:
        pm = ch["models"][0].predict_proba(X)[:, 1]
        pr = ch["models"][1].predict_proba(X)[:, 1]
        pb = ch["models"][2].predict_proba(X)[:, 1]
        pf = (1.0 - pm) * (1.0 - pr) * (0.95 - 0.36 * pb)
        cw = float(ch["w"])
        preds = (1.0 - cw) * preds + cw * pf

    preds = preds + platoon_adjust(bundle, X)"""


def build(w, name, force=False, base=None):
    subname.check(name)
    sys.path.insert(0, os.path.join(ROOT, "exp"))
    import build_asof as ba
    from path_alloc import build_df

    base_zip = os.path.join(ROOT, base) if base else BASE_ZIP
    out = os.path.join(ROOT, "submissions", f"{name}.zip")
    assert force or not os.path.exists(out), f"이미 있다: {out}"

    src = zipfile.ZipFile(base_zip)
    b = joblib.load(io.BytesIO(src.read("model/rf.pkl")))
    FEAT = list(b["features"])
    script = src.read("script.py").decode("utf-8")
    po, pn = PATCH_OLD, PATCH_NEW
    if po not in script:                       # zip 의 script.py 는 CRLF
        po, pn = po.replace("\n", "\r\n"), pn.replace("\n", "\r\n")
    assert po in script, "script.py 패치 지점을 못 찾았다"
    script = script.replace(po, pn)

    tr = build_df()
    season = tr["season"].to_numpy()

    lab = np.load(os.path.join(ROOT, "exp", "cache", "pitch_labels.npz"), allow_pickle=True)
    csv = pd.read_csv(os.path.join(ROOT, "data", "train.csv"),
                      usecols=["row_id", "pitcher_id", "asof_pitcher_n"])
    rid_pos = {r: j for j, r in enumerate(lab["row_id"])}
    csv_pos = np.array([rid_pos[r] for r in csv["row_id"]])
    key_csv = csv["pitcher_id"].to_numpy(np.int64) * 100000 + csv["asof_pitcher_n"].to_numpy(np.int64)
    key_map = dict(zip(key_csv, csv_pos))
    key_tr = tr["pitcher_id"].to_numpy(np.int64) * 100000 + tr["asof_pitcher_n"].to_numpy(np.int64)
    perm = np.array([key_map[k] for k in key_tr])
    assert (lab["season"][perm] == season).all(), "정렬 불일치"
    del csv, rid_pos, key_map

    X = tr[FEAT]
    chan_models = []
    for nm in ("middle", "reverse", "ball"):
        L = lab[nm].astype(np.float64)[perm]
        ok = np.isfinite(L)
        m = ba.pipeline(FEAT, 42)
        m.fit(X[ok], L[ok].astype(int))
        chan_models.append(m)
        print(f"  [{nm}] 학습 {ok.sum():,}행 완료", flush=True)

    b["chan"] = {"models": chan_models, "w": float(w)}
    b["note"] = b["note"].split("|")[0] + f"| CHAN 채널 혼합 w={w:g} (mid·rev·ball 전체학습, 곱 공식)"

    buf = io.BytesIO()
    joblib.dump(b, buf, compress=3)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("model/rf.pkl", buf.getvalue())
        z.writestr("script.py", script)
        z.writestr("requirements.txt", src.read("requirements.txt"))
    h = hashlib.sha256(open(out, "rb").read()).hexdigest()
    print(f"{name}.zip   CHAN w={w:g}   기반 {os.path.basename(base_zip)}")
    print(f"  sha256 {h[:16]}   {os.path.getsize(out):,} bytes")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=float, default=0.25)
    ap.add_argument("--name", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--base", default=None)
    a = ap.parse_args()
    build(a.w, a.name, a.force, a.base)
