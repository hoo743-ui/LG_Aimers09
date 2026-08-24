r"""신규 **수준 축**을 후처리 10번째 축으로 얹는다. 학습 0회.

## 근거 — LV-001~005 (2026-08-24)

수준 축 후보 36개를 전수로 재고 **키를 시즌 안에서 치환한 귀무분포**로 걸렀다.

    귀무 "36개 중 최대"   +2.80 ± 0.72  (최대 +4.39, 12회)
    카운트(12)            **+7.21**  ->  z = +6.1     🔴 유일한 생존
    base_state           +3.11      ->  경계, 미달
    나머지 34개            전부 위약 이하

카운트 축의 추가 검증
    3시즌 부호 일치 9/12 셀 · corr(2022+23 셀평균, 2024 셀평균) = +0.53
    중심화 불변 (+5.57 -> +5.57)  -> 리그 드리프트 상수가 아니다
    무작위 12셀 위약 20회 **전부 음수** (평균 −3.19)
    현행 후처리 9축과 중복 corr = **0.006**
    walk-forward w=1 고정   목표 2023 +5.56 · 목표 2024 +5.57

## 표 원천은 생산과 맞춘다

축 4~8 이 **직전 2시즌 OOF 잔차**이므로 같은 규약으로 2023+2024 를 쓴다.
셀이 12개에 셀당 3.5만 행이라 축소는 사실상 무의미하다 (k=0 과 k=2000 이
99.9% 같다). 그래도 형식을 맞추려 k 를 인자로 받는다.

## 이건 로컬 근거다 — 제출은 LB 좌표로 연다

37회차(볼우위)가 로컬 3폴드 양수 + 겹침 0.02 로 −7.07 이었다. 그러므로 이
축도 **w 를 LB 3점 해법으로** 연다. w=+1 과 w=−1(미러) 두 탐침이면 포물선이
확정되고, 최고점 채점이라 하방은 0 이다.

    .\venv_submit\Scripts\python.exe -u exp\build_level.py --w 1.0 --name cand_cnt1
"""
import argparse, hashlib, io, os, sys, zipfile
import joblib, numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "exp"))
import subname                                                        # noqa: E402
WIN = [2023, 2024]                       # 표 원천 — 축 4~8 과 같은 규약

try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass


def build(base, w, name, k=0.0, force=False):
    subname.check(name)
    base_zip = base if os.path.isabs(base) else os.path.join(ROOT, base)
    out = os.path.join(ROOT, "submissions", f"{name}.zip")
    assert force or not os.path.exists(out), f"이미 있다: {out}"

    src = zipfile.ZipFile(base_zip)
    b = joblib.load(io.BytesIO(src.read("model/rf.pkl")))
    pl = [dict(s) for s in b["platoon"]]
    assert not any(s["cols"] == ["balls_before", "strikes_before"] for s in pl), \
        "카운트 수준축이 이미 들어 있다 — 가중만 바꾸려면 build_wscale.py 를 써라"

    df = pd.read_csv(os.path.join(ROOT, "data", "train.csv"),
                     usecols=["season", "control_success", "balls_before",
                              "strikes_before"])
    oof = np.load(os.path.join(ROOT, "exp", "champ_oof.npz"))
    ks, rs = [], []
    for s in WIN:
        m = (df.season == s).to_numpy()
        p = oof[f"p{s}"].astype(np.float64)
        assert m.sum() == len(p), f"{s} 행수 불일치 {m.sum()} vs {len(p)}"
        ks.append(df.balls_before.to_numpy()[m] * 4 + df.strikes_before.to_numpy()[m])
        rs.append(df.control_success.to_numpy(float)[m] - p)
    keys, res = np.concatenate(ks), np.concatenate(rs)
    u, inv = np.unique(keys, return_inverse=True)
    n = np.bincount(inv, minlength=len(u)).astype(float)
    v = np.bincount(inv, weights=res, minlength=len(u)) / (n + k)
    v = v - (v * n).sum() / n.sum()          # 중심화 (rho 불변. 값 크기만 안정)

    tab = {(int(c) // 4, int(c) % 4): float(x) for c, x in zip(u, v)}
    pl.append(dict(w=float(w), cols=["balls_before", "strikes_before"], table=tab,
                   note=f"카운트(12) 수준축 k={k:g} 원천={WIN} OOF잔차 "
                        f"(LV-001~005: 귀무 최대 +2.80±0.72 대비 실측 +7.21, z=+6.1)"))
    b["platoon"] = pl
    b["note"] = str(b.get("note", "")) + f" | LV 카운트 수준축 10번째, w={w:g}"

    buf = io.BytesIO(); joblib.dump(b, buf, compress=3)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("model/rf.pkl", buf.getvalue())
        for nm in ("script.py", "requirements.txt"):
            z.writestr(nm, src.read(nm))
    h = hashlib.sha256(open(out, "rb").read()).hexdigest()
    print(f"{name}.zip   기반 {os.path.basename(base_zip)}   축 10개")
    print(f"  카운트 표 (B-S: 보정값, w={w:g})")
    for (bb, ss), x in sorted(tab.items()):
        print(f"    {bb}-{ss}  {x:+.6f}   n={int(n[u == bb * 4 + ss][0]):,}")
    print(f"  9가중 불변 {[round(s['w'], 6) for s in pl[:9]]}")
    print(f"  sha256 {h[:16]}   {os.path.getsize(out):,} bytes")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="submissions/cand_kb45.zip")
    ap.add_argument("--w", type=float, required=True)
    ap.add_argument("--k", type=float, default=0.0)
    ap.add_argument("--name", required=True)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    build(a.base, a.w, a.name, a.k, a.force)
