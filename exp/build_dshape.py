r"""편차 4축(0 플래툰 · 1 우위 · 2 카운트 · 3 주자)의 **모양**(축소 상수 k) 좌표 (2026-08-28).

## 구조 (2026-08-28 열거로 확정 — 전부 정확 일치)

    축0  (투수,타자손) 1571셀 직접.  k=300   원천 전시즌.  n/(n+300) 중앙 0.592
    축1  (투수,손,우위) 3109그룹 -> 12칸 전개 (동일값 1.0000).  k=2000.  중앙 0.082
    축2  (투수,손,볼,스트) 17767셀 직접.  k=800.  중앙 0.035
    축3  (투수,손,주자유무) 3085그룹 -> 주자수 전개 (동일값 1.0000).  k=2000.  중앙 0.100

편차 표는 라벨 편차라 train.csv 만으로 corr 1.000000 재현이 확인돼 있다(표 제작 명세).
S_e = v0(n_e+k0) 역산 -> v_t = v0 + t(α·v_k' − v0), 2024 조회 분포 행 sd 고정, 9가중 불변.
우위 = strikes_before > balls_before, 주자유무 = num_runners_on > 0.

로컬 기여(생산 2024): dev_platoon +11.9 > dev_plat_cnt +7.4 > dev_runner +1.7 > dev_count +1.5
-> 슬롯 순서는 축0 -> 축1/2 -> 축3.

    .\venv_submit\Scripts\python.exe -u exp\build_dshape.py --axis plat --k 30 --t 3 --name cand_d0p3
"""
import argparse, hashlib, io, os, sys, zipfile

import joblib
import numpy as np
import pandas as pd

import subname

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_ZIP = os.path.join(ROOT, "submissions", "260827_5_plogn.zip")   # Champion 1099.0547
AXIS = {
    "plat": (0, ["pitcher_id", "batter_hand"], 300, None),
    "adv":  (1, ["pitcher_id", "batter_hand", "balls_before", "strikes_before"], 2000, "adv"),
    "cnt":  (2, ["pitcher_id", "batter_hand", "balls_before", "strikes_before"], 800, None),
    "run":  (3, ["pitcher_id", "batter_hand", "num_runners_on"], 2000, "onb"),
}
REF = 2024

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def build(axis, k, t, name, force=False, base=None):
    subname.check(name)
    i, cols, k0, gmode = AXIS[axis]
    base_zip = os.path.join(ROOT, base) if base else BASE_ZIP
    out = os.path.join(ROOT, "submissions", f"{name}.zip")
    assert force or not os.path.exists(out), f"이미 있다: {out}"

    src = zipfile.ZipFile(base_zip)
    b = joblib.load(io.BytesIO(src.read("model/rf.pkl")))
    pl = b["platoon"]
    assert len(pl) == 9 and pl[i]["cols"] == cols, (len(pl), pl[i]["cols"])
    assert "DSHAPE" not in pl[i]["note"], f"이미 이동한 표: {pl[i]['note'][:120]}"
    tab = dict(pl[i]["table"])

    df = pd.read_csv(os.path.join(ROOT, "data", "train.csv"),
                     usecols=["season"] + cols)
    win, ref = df, df[df.season == REF]          # 편차 4축 원천 = 전시즌

    if gmode is None:
        keys = sorted(tab)
        n = win.groupby(cols).size().reindex(keys)
        assert n.notna().all(), "표 키가 열거 밖"
        v0 = np.array([tab[kk] for kk in keys], float)
        ne = n.to_numpy(float)

        def expand(vv):
            return {kk: float(x) for kk, x in zip(keys, vv)}
    else:
        if gmode == "adv":
            gk = lambda kk: (kk[0], kk[1], kk[3] > kk[2])
            wg = win.assign(g=(win.strikes_before > win.balls_before))
        else:
            gk = lambda kk: (kk[0], kk[1], int(kk[2] > 0))
            wg = win.assign(g=(win.num_runners_on > 0).astype(int))
        groups = {}
        for kk in tab:
            groups.setdefault(gk(kk), []).append(kk)
        keys = sorted(groups)
        n = wg.groupby([cols[0], cols[1], "g"]).size().reindex(keys)
        assert n.notna().all(), "그룹이 열거 밖"
        v0 = np.array([tab[groups[kk][0]] for kk in keys], float)
        for kk in keys:                          # 전개 동일값 확인
            vs = {round(tab[m], 12) for m in groups[kk]}
            assert len(vs) == 1, (kk, vs)
        ne = n.to_numpy(float)

        def expand(vv):
            t2 = dict(tab)
            for kk, x in zip(keys, vv):
                for m in groups[kk]:
                    t2[m] = float(x)
            return t2

    S = v0 * (ne + k0)
    vk = S / (ne + k)

    def rowvec(t2):
        m = ref[cols].apply(tuple, axis=1).map(t2)
        return m.fillna(0.0).to_numpy(float)

    c0 = rowvec(expand(v0))
    ck = rowvec(expand(vk))
    alpha = c0.std() / ck.std()
    vt = v0 + t * (alpha * vk - v0)
    tab_t = expand(vt)
    ct = rowvec(tab_t)
    corr = float(np.corrcoef(c0, ck)[0, 1])
    corrt = float(np.corrcoef(c0, ct)[0, 1])

    pl[i] = dict(pl[i], table=tab_t,
                 note=pl[i]["note"] + f" | DSHAPE 모양 k {k0}->{k:g} t={t:g} "
                      f"(행 sd 고정 alpha={alpha:.6f}), 가중 불변")
    b["platoon"] = pl
    b["note"] = (b["note"].split("|")[0]
                 + f"| DSHAPE {axis} 편차축 모양 k={k0}->{k:g} t={t:g}, 9가중 불변")

    buf = io.BytesIO()
    joblib.dump(b, buf, compress=3)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("model/rf.pkl", buf.getvalue())
        for nm in ("script.py", "requirements.txt"):
            z.writestr(nm, src.read(nm))
    h = hashlib.sha256(open(out, "rb").read()).hexdigest()
    print(f"{name}.zip   축={axis}(#{i})  k {k0} -> {k:g}   t={t:g}   기반 {os.path.basename(base_zip)}")
    print(f"  alpha {alpha:.6f}   corr(c_k, c_0) {corr:.5f}  직교성분 {np.sqrt(max(0,1-corr**2))*100:.1f}%")
    print(f"  corr(c_t, c_0) {corrt:.5f}   행 sd 비 {ct.std()/c0.std():.6f}   셀 {len(keys)} (표 {len(tab)})")
    print(f"  9가중 {[round(x['w'],4) for x in pl]}")
    print(f"  sha256 {h[:16]}   {os.path.getsize(out):,} bytes")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", required=True, choices=list(AXIS))
    ap.add_argument("--k", type=float, required=True)
    ap.add_argument("--t", type=float, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--base", default=None)
    a = ap.parse_args()
    build(a.axis, a.k, a.t, a.name, a.force, a.base)
