r"""🚩 H-K1 — 수준 축의 **모양**(축소 상수 k)을 LB 로 직접 재는 후보를 만든다.

## 왜 이 좌표인가

수준 2축의 LB 최적 가중이 둘 다 2 근처다 (투수 1.985 · 타자 2.105). 표가
`n/(n+k)` 로 감쇠돼 있으면 스칼라 w 로 되돌리려 하는데, 감쇠는 개체마다
달라서(이질적) **스칼라로는 원리적으로 못 되돌린다.** 올바른 수정은 w 를
올리는 것이 아니라 k 를 내리는 것이다.

EXP041 이 이 좌표를 닫았으나 근거 둘 중 하나만 유효하다.

    (a) k >= 5000 에서 corr(c_k, c_20000) >= 0.9925 -> w 와 축퇴      [유효]
    (b) k = 500~2000 은 모양이 다른데 로컬 3폴드가 더 낮다            [무효 — EXP057]

로컬은 부호를 50% 로 맞힌다(EXP057). (b) 는 근거가 아니다.

## 표의 형태를 재현으로 확정했다 (2026-08-24)

원천 창을 열거로 맞췄다 — 2023+2024 에서 투수 510 · 타자 526 으로 표 키 집합과
**완전 일치**(누락 0, 여분 0, 최소 n 필터 없음). `champ_oof` 잔차로 다시 만들면

    투수 k=50000  corr +0.9976  기울기 0.9914
    타자 k=20000  corr +0.9983  기울기 1.0004

따라서 `v_e = Σresid_e / (n_e + k)` 이고, `S_e = v_e·(n_e + k)` 로 **정확히**
역산된다 (근사 없음 · 학습 0회 · 실제 생산 잔차 그대로).

## 모양만 움직인다 — 크기는 이미 최적이다

k 를 바꾸면 보정 벡터의 **모양과 크기가 동시에** 바뀐다. 그런데 크기(w)는
이미 3점 곡선으로 최적이 확인돼 있다 (EXP048: L_batter A=2.38 b=2.118±0.478,
현행 2.105 가 최적). 그래서 새 표를 **2024 조회 분포에서 행 sd 가 같아지도록**
정규화해 크기를 고정하고 모양만 바꾼다. 48회차 미러와 같은 규율이다.

    v_t[e] = v0[e] + t · ( alpha · v_k[e] − v0[e] ),   alpha = sd(c_v0)/sd(c_vk)

t=0 이 현행(측정됨 = 1076.2214). t=+1 을 찍고, 내려가면 t=−1 을 찍는다
(43->48회차가 정확히 그 경로였다). 9가중은 **한 개도 건드리지 않는다.**

## TYPE 판정

**TYPE B (표 제작 파라미터).** 5-a 는 기본을 "바꾸지 않는다"로 둔다. 다만
TYPE B 4연패(−9.95 / −65.73 / −4.93 / −7.07)는 **전부 대비 표**였고 **전부
로컬로 골랐다.** 여기서는 로컬로 아무것도 고르지 않고 좌표를 LB 로 직접 잰다 —
그 규율은 5/5 로 맞았다(39·40·41·45·48). 하방이 0 이므로 E[max(0,·)] > 0.

    .\venv_submit\Scripts\python.exe -u exp\build_lvlk.py --axis batter --k 2000 --t 1 --name cand_kbf
"""
import argparse
import hashlib
import io
import os
import sys
import zipfile

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_ZIP = os.path.join(ROOT, "submissions", "cand_mir.zip")   # --base 로 교체 가능
AXIS = {"pitcher": (7, "pitcher_id", 50000), "batter": (8, "batter_id", 20000)}
WIN = [2023, 2024]          # 표 원천 창 (키 집합 열거로 확정)
REF = 2024                  # 정규화용 조회 분포 (2025 대용)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def load_counts():
    df = pd.read_csv(os.path.join(ROOT, "data", "train.csv"),
                     usecols=["season", "pitcher_id", "batter_id"])
    return df[df.season.isin(WIN)], df[df.season == REF]


def build(axis, k, t, name, force=False, base=None):
    i, col, k0 = AXIS[axis]
    base_zip = os.path.join(ROOT, base) if base else BASE_ZIP
    out = os.path.join(ROOT, "submissions", f"{name}.zip")
    assert force or not os.path.exists(out), f"이미 있다: {out}"

    src = zipfile.ZipFile(base_zip)
    b = joblib.load(io.BytesIO(src.read("model/rf.pkl")))
    pl = b["platoon"]
    assert len(pl) == 9, len(pl)
    assert pl[i]["cols"] == [col], pl[i]["cols"]
    tab = {kk[0]: v for kk, v in pl[i]["table"].items()}
    assert f"k={k0}" in pl[i]["note"] or "H-K1" not in pl[i]["note"],         f"이 축의 표가 이미 이동했다 — k0 역산이 무효다: {pl[i]['note'][:120]}"

    win, ref = load_counts()
    n = win.groupby(col).size()
    ids = np.array(sorted(tab))
    v0 = np.array([tab[e] for e in ids], float)
    ne = n.reindex(ids).to_numpy(float)
    assert np.isfinite(ne).all() and (ne > 0).all(), "표 키가 원천 창 밖에 있다"

    S = v0 * (ne + k0)                       # 정확한 역산
    vk = S / (ne + k)                         # 새 k 의 표

    rid = ref[col].to_numpy()                 # 2024 조회 분포에서 행 sd
    pos = {e: j for j, e in enumerate(ids)}
    j = np.array([pos.get(e, -1) for e in rid])
    hit = j >= 0
    def rowsd(vv):
        o = np.zeros(len(rid)); o[hit] = vv[j[hit]]; return o.std()
    alpha = rowsd(v0) / rowsd(vk)
    vt = v0 + t * (alpha * vk - v0)

    # 진단
    def rowvec(vv):
        o = np.zeros(len(rid)); o[hit] = vv[j[hit]]; return o
    c0, ct = rowvec(v0), rowvec(vt)
    corr = float(np.corrcoef(c0, ct)[0, 1])
    sdr = float(ct.std() / c0.std())

    pl[i] = dict(pl[i], table={(e,): float(x) for e, x in zip(ids, vt)},
                 note=pl[i]["note"] + f" | H-K1 모양 k {k0}->{k} t={t:g} "
                      f"(행 sd 고정 alpha={alpha:.6f}), 가중 불변")
    b["platoon"] = pl
    b["note"] = (b["note"].split("|")[0]
                 + f"| H-K1 {axis} 수준축 모양 k={k0}->{k} t={t:g}, "
                   f"9가중 불변, 행 sd 고정")

    buf = io.BytesIO()
    joblib.dump(b, buf, compress=3)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("model/rf.pkl", buf.getvalue())
        for nm in ("script.py", "requirements.txt"):
            z.writestr(nm, src.read(nm))
    h = hashlib.sha256(open(out, "rb").read()).hexdigest()
    print(f"{name}.zip   축={axis}  k {k0} -> {k}   t={t:g}   기반 {os.path.basename(base_zip)}")
    print(f"  alpha(정규화)  {alpha:.6f}")
    print(f"  corr(c_t, c_0) {corr:.5f}   직교성분 {np.sqrt(max(0,1-corr**2))*100:.1f}%")
    print(f"  행 sd 비       {sdr:.6f}   (t=+1 에서 1.000000. 다른 t 는 직선 유지가 우선 — 3점 해법 성립 조건)")
    print(f"  9가중          {[round(s['w'],6) for s in pl]}")
    print(f"  sha256 {h[:16]}   {os.path.getsize(out):,} bytes")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", required=True, choices=list(AXIS))
    ap.add_argument("--k", type=float, required=True)
    ap.add_argument("--t", type=float, default=1.0)
    ap.add_argument("--name", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--base", default=None,
                    help="기반 zip (기본 submissions/cand_mir.zip). 다른 축이 이미 "
                         "이동한 후보 위에 얹을 때 쓴다")
    a = ap.parse_args()
    build(a.axis, a.k, a.t, a.name, a.force, a.base)


if __name__ == "__main__":
    main()
