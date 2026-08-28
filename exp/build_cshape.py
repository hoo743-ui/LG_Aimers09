r"""대비 3축(4 손 · 5 2S · 6 주자)의 **모양**(축소 상수 k)을 LB 로 직접 재는 후보 (2026-08-28).

## 왜 이 좌표인가

8/28 로 **크기(스칼라) 공간이 전부 소진**됐다 — 9가중(EXP048) + 투수 log n 정점 +
타자 수준 크기(m*=0.970) + 손 대비 크기(m*=1.045) 가 전부 "현행=정점"이다.
남은 미측정 면적은 표의 **모양**이고, 그 좌표는 타자 수준 표에서 +13.79(55회차),
순수 n 방향으로 +8.35(60회차)를 실측했다. 대비 3축(블록 +10.45)은 같은 수술이
**한 번도 안 갔다.**

## 대비 셀도 같은 과소축소 대역이다 (2026-08-28 실측)

    축4 (투수,타자손)   셀 n 중앙 278   n/(n+1000) 중앙 0.218   (타자 표 최대 0.210 과 같은 급)
    축5 (투수,스트)     셀 n 중앙 184   n/(n+1000) 중앙 ~0.16
    축6 (투수,주자유무) 셀 n 중앙 291   n/(n+2000) 중앙 ~0.13

## 표 구조 (2026-08-28 열거로 확정)

    축4  1996셀 = 499투수 x {투수손 2} x {타자손 2}.  실제 조회되는 것은 관측 손 쪽
         (996셀)뿐이고 비관측 손 991셀은 죽은 셀이다 — **건드리지 않는다**
    축5  1524셀 = 508투수 x {스트 0,1,2}.  전부 관측
    축6  1996셀 = 499투수 x {주자 0,1,2,3}.  주자 1/2/3 동일값 (주자유무 전개, 비율 1.0000)
         -> 주자유무 수준에서 역산·수술 후 재전개
    빠진 투수 (510-499=11, 2023-24 행수 9~27) 는 표에 없다 = 보정 0. 그대로 둔다

## 수술 — build_lvlk 와 동일 규율

    S_e = v0_e·(n_e + k0)   (정확 역산)      v_k = S_e/(n_e + k')
    v_t = v0 + t·(alpha·v_k − v0),   alpha = sd(c_v0)/sd(c_vk)   (2024 조회 분포, 행 sd 고정)

t=0 이 현행 Champion. 9가중 불변. TYPE B 지만 로컬로 아무것도 고르지 않고
LB 3점 해법으로 잰다 (그 규율은 55·61·63·71·73·75회차에서 전부 맞았다).

    .\venv_submit\Scripts\python.exe -u exp\build_cshape.py --axis hand --k 100 --t 3 --name cand_c4p3
"""
import argparse, hashlib, io, os, sys, zipfile

import joblib
import numpy as np
import pandas as pd

import subname

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_ZIP = os.path.join(ROOT, "submissions", "260827_5_plogn.zip")   # Champion 1099.0547
AXIS = {
    "hand":   (4, ["pitcher_id", "pitcher_hand", "batter_hand"], 1000),
    "strike": (5, ["pitcher_id", "strikes_before"],              1000),
    "runner": (6, ["pitcher_id", "num_runners_on"],              2000),
}
WIN, REF = [2023, 2024], 2024

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def build(axis, k, t, name, force=False, base=None):
    subname.check(name)
    i, cols, k0 = AXIS[axis]
    base_zip = os.path.join(ROOT, base) if base else BASE_ZIP
    out = os.path.join(ROOT, "submissions", f"{name}.zip")
    assert force or not os.path.exists(out), f"이미 있다: {out}"

    src = zipfile.ZipFile(base_zip)
    b = joblib.load(io.BytesIO(src.read("model/rf.pkl")))
    pl = b["platoon"]
    assert len(pl) == 9 and pl[i]["cols"] == cols, (len(pl), pl[i]["cols"])
    assert f"k={k0}" in pl[i]["note"], f"k0 역산 무효: {pl[i]['note'][:120]}"
    assert "CSHAPE" not in pl[i]["note"], f"이미 이동한 표: {pl[i]['note'][:120]}"
    tab = dict(pl[i]["table"])

    use = ["season"] + [c for c in ("pitcher_id", "pitcher_hand", "batter_hand",
                                    "strikes_before", "num_runners_on") if c in cols]
    df = pd.read_csv(os.path.join(ROOT, "data", "train.csv"), usecols=use)
    win, ref = df[df.season.isin(WIN)], df[df.season == REF]

    if axis == "runner":
        # 주자유무 수준 역산 -> 재전개
        wn = win.assign(onb=(win.num_runners_on > 0).astype(int))
        n = wn.groupby(["pitcher_id", "onb"]).size()
        pits = sorted({kk[0] for kk in tab})
        keys = [(p, o) for p in pits for o in (0, 1)]
        v0 = np.array([tab[(p, 0 if o == 0 else 1)] for p, o in keys], float)
        ne = n.reindex(keys).to_numpy(float)
        assert np.isfinite(ne).all() and (ne > 0).all(), "주자유무 셀이 원천 창 밖"
        S = v0 * (ne + k0)
        vk = S / (ne + k)
        live = np.ones(len(keys), bool)

        def expand(vv):
            m = {kk: x for kk, x in zip(keys, vv)}
            t2 = dict(tab)
            for p in pits:
                t2[(p, 0)] = float(m[(p, 0)])
                for r in (1, 2, 3):
                    if (p, r) in t2:
                        t2[(p, r)] = float(m[(p, 1)])
            return t2
        look_cols = ["pitcher_id", "num_runners_on"]
    else:
        n = win.groupby(cols).size()
        keys = sorted(tab)
        nser = n.reindex(keys)
        live = nser.notna().to_numpy()          # 축4 비관측 손 셀은 죽은 셀 -> 불변
        if axis == "strike":
            assert live.all(), "축5 에 원천 창 밖 셀"
        keys_l = [kk for kk, lv in zip(keys, live) if lv]
        v0 = np.array([tab[kk] for kk in keys_l], float)
        ne = nser[live].to_numpy(float)
        S = v0 * (ne + k0)
        vk = S / (ne + k)

        def expand(vv):
            t2 = dict(tab)
            for kk, x in zip(keys_l, vv):
                t2[kk] = float(x)
            return t2
        look_cols = cols

    # 2024 조회 분포에서 행 sd 정규화 (조회는 실제 표 그대로 — 죽은 셀 포함)
    def rowvec(t2):
        m = ref[look_cols].apply(tuple, axis=1).map(t2)
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
                 note=pl[i]["note"] + f" | CSHAPE 모양 k {k0}->{k:g} t={t:g} "
                      f"(행 sd 고정 alpha={alpha:.6f}), 가중 불변")
    b["platoon"] = pl
    b["note"] = (b["note"].split("|")[0]
                 + f"| CSHAPE {axis} 대비축 모양 k={k0}->{k:g} t={t:g}, 9가중 불변")

    buf = io.BytesIO()
    joblib.dump(b, buf, compress=3)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("model/rf.pkl", buf.getvalue())
        for nm in ("script.py", "requirements.txt"):
            z.writestr(nm, src.read(nm))
    h = hashlib.sha256(open(out, "rb").read()).hexdigest()
    print(f"{name}.zip   축={axis}(#{i})  k {k0} -> {k:g}   t={t:g}   기반 {os.path.basename(base_zip)}")
    print(f"  alpha {alpha:.6f}   corr(c_k, c_0) {corr:.5f}  직교성분 {np.sqrt(max(0,1-corr**2))*100:.1f}%")
    print(f"  corr(c_t, c_0) {corrt:.5f}   행 sd 비 {ct.std()/c0.std():.6f}   살아있는 셀 {int(live.sum())}/{len(tab)}")
    print(f"  9가중 {[round(s['w'],4) for s in pl]}")
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
