r"""타자 수준축의 **{1/(n+k)} family 밖** 방향을 탐침하는 후보를 만든다 (2026-08-25).

## 왜

55회차 정점(t=-4.5653)은 span{v_20000, v_2000} 2평면 안의 최적이다. 8/25 실측:
투수 모양 정점 +0.002, 카운트 축 -6.39 -> 남은 이득은 그 평면 **밖**에 있다.
2차식 기하로 잰 타자 크기 축 잔여는 +0.04 (c* 방향 기울기 P-Q≈-1.0, Q≈26.5).

## 8/25 실측: hi s=0.4 −1.49 · **n s=0.4 +8.35 (60회차, 새 Champion 1098.3639)**

## 방향 (원장 2026-08-24 "타자축에 남은 좌표")

    hi   r̄_e·1[n_e > N]      고-n 만 남기고 저-n 은 0      (평면 밖 44.8%)
                            "저-n 의 음수 가중이 정답인가 0 이 정답인가"
    n    (n_e − n̄)          잔차 없이 출장수만            (평면 밖 92.7%)   60회차 +8.35
    logn (log n_e − 평균) ⊥ {c*, n}   n 효과의 오목 성분.  OOF 잔차상관이 2022·2024 모두 선형 n 보다 큼 (|c| 1.5~1.9배)
                            +24.5 가 타자 수준인가 '주전 편향'인가

**가산** 직선이고 방향은 c* 에 **직교화**한다(크기 성분은 이미 측정됨) (t-선처럼 치환하면 corr 0.77/-0.14 라 +24.5 를 버린다).

    v_s = v* + s·alpha·v_dir,   alpha = sd(c_v*)/sd(c_dir)   (2024 조회 분포)

s=0 이 현행 kb45(1090.0095). 보폭 s=0.4 -> 곡률 Q≈26.5·0.16≈4 (카운트 탐침과
같은 급). 내려가면 s=−0.4 미러, 2점으로 포물선(정점 = P²/Q, 보폭 무관).
S_e·n_e 는 **원 표(cand_mir, k=20000)** 에서 역산한다 — kb45 의 표는 이미 v* 다.

    .\venv_submit\Scripts\python.exe -u exp\build_bdir.py --dir hi --s 0.4 --name 260825_5_kbhi
"""
import argparse, hashlib, io, os, sys, zipfile
import joblib, numpy as np, pandas as pd
import subname

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "submissions", "cand_kb45.zip")     # v* (t=-4.5)
ORIG = os.path.join(ROOT, "submissions", "cand_mir.zip")      # v0 (k=20000)
AXES = {"batter": (8, "batter_id", 20000, 1950), "pitcher": (7, "pitcher_id", 50000, 1950)}
WIN, REF, NTH = [2023, 2024], 2024, 1950

try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass


def load(z):
    return joblib.load(io.BytesIO(zipfile.ZipFile(z).read("model/rf.pkl")))


def build(terms, name, force=False, axis="batter", base=None):
    """terms = [(dir, s), ...]  v = v* + Σ s·alpha_dir·v_dir.  방향 정의는 고정(정준):
       hi, n 은 c* 에 직교, logn 은 {c*, n⊥} 에 직교 (비중심 내적)."""
    I, COL, K0, NTH = AXES[axis]
    subname.check(name)
    BASE = base or globals()["BASE"]           # --base 로 연쇄 빌드 (타자 축 위에 투수 축)
    out = os.path.join(ROOT, "submissions", f"{name}.zip")
    assert force or not os.path.exists(out), out
    b = load(BASE); pl = b["platoon"]; assert pl[I]["cols"] == [COL]
    o = load(ORIG)["platoon"][I]; assert o["cols"] == [COL] and f"k={K0}" in o["note"]
    tab_s = {k[0]: v for k, v in pl[I]["table"].items()}
    tab_0 = {k[0]: v for k, v in o["table"].items()}
    ids = np.array(sorted(tab_s)); assert set(ids) == set(tab_0)
    df = pd.read_csv(os.path.join(ROOT, "data", "train.csv"), usecols=["season", COL])
    n = df[df.season.isin(WIN)].groupby(COL).size().reindex(ids).to_numpy(float)
    assert np.isfinite(n).all()
    vs = np.array([tab_s[e] for e in ids]); v0 = np.array([tab_0[e] for e in ids])
    S = v0 * (n + K0); rbar = S / n
    rid = df[df.season == REF][COL].to_numpy()
    pos = {e: j for j, e in enumerate(ids)}
    j = np.array([pos.get(e, -1) for e in rid]); hit = j >= 0
    def rv(v):
        x = np.zeros(len(rid)); x[hit] = v[j[hit]]; return x
    cs = rv(vs)
    def orth(v, basis, centered=False):
        for bv in basis:
            a, c = rv(v), rv(bv)
            if centered:                      # rho 는 상수에 불변 -> 공분산 내적이 맞다
                a, c = a - a.mean(), c - c.mean()
            v = v - float(np.dot(a, c) / np.dot(c, c)) * bv
        return v
    raw = {"hi": np.where(n > NTH, rbar, 0.0), "n": n - n.mean(), "logn": np.log1p(n) - np.log1p(n).mean()}
    D = {}
    D["hi"] = orth(raw["hi"], [vs]); D["n"] = orth(raw["n"], [vs]); D["logn"] = orth(raw["logn"], [vs, D["n"]], centered=True)   # hi/n 은 60회차 직선 보존을 위해 비중심 유지
    vt = vs.copy(); tag = []
    print(f"{name}  axis={axis}  terms={terms}  기반 {os.path.basename(BASE)} (v*), 원 표 cand_mir (S_e 역산)")
    for d, s in terms:
        vd = D[d]; cd = rv(vd); alpha = cs.std() / cd.std()
        print(f"  [{d}] corr(c_dir, c*) {np.corrcoef(cd, cs)[0,1]:+.4f}   corr(c_dir, c_n) {np.corrcoef(cd, rv(D['n']))[0,1]:+.4f}   alpha {alpha:.6g}")
        vt = vt + s * alpha * vd; tag.append(f"{d} s={s:g}")
    ct = rv(vt)
    print(f"  corr(c_final, c*) {np.corrcoef(ct, cs)[0,1]:+.4f}   행 sd 비 {ct.std()/cs.std():.6f}   n>{NTH}: {(n>NTH).sum()}/{len(n)}")
    pl[I] = dict(pl[I], table={(e,): float(x) for e, x in zip(ids, vt)},
                 note=pl[I]["note"] + f" | BDIR {axis} {' + '.join(tag)} (행 sd 정규화 가산), 가중 불변")
    b["platoon"] = pl
    b["note"] = b["note"].split("|")[0] + f"| BDIR {axis} {' + '.join(tag)}, 9가중 불변"
    buf = io.BytesIO(); joblib.dump(b, buf, compress=3)
    src = zipfile.ZipFile(BASE)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("model/rf.pkl", buf.getvalue())
        for nm in ("script.py", "requirements.txt"):
            z.writestr(nm, src.read(nm))
    h = hashlib.sha256(open(out, "rb").read()).hexdigest()
    print(f"  9가중 {[round(x['w'],4) for x in pl]}")
    print(f"  sha256 {h[:16]}   {os.path.getsize(out):,} bytes")
    return h


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", choices=["hi", "n", "logn"])
    ap.add_argument("--s", type=float)
    ap.add_argument("--terms", help="예) n:0.8,logn:0.4  (--dir/--s 대신)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--axis", default="batter", choices=list(AXES))
    ap.add_argument("--base", help="시작 zip (기본 cand_kb45). 다른 축 위에 얹을 때")
    a = ap.parse_args()
    terms = [(t.split(':')[0], float(t.split(':')[1])) for t in a.terms.split(',')] if a.terms else [(a.dir, a.s)]
    build(terms, a.name, a.force, a.axis, a.base)
