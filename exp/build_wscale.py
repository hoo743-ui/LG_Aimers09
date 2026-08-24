r"""기존 제출 zip 위에서 **지정 축의 가중만** 배수해 새 후보를 만든다.

## 왜 필요한가 (2026-08-24, 타자축 크기 좌표)

55회차까지의 타자 수준축 모양 이동(H-K1)은 `v_t = v0 + t(alpha·v_k − v0)` 에
`w = 2.105` 를 **고정한 채** 돌렸다. 두 기저의 계수 합이 항상 2.105 라는 뜻이고,
그 2.105 는 **옛 모양에서 잰 최적값**이다. 모양이 바뀌면서 2024 조회 분포의
행 sd 가 1.434 배로 커졌는데 크기는 한 번도 다시 재지 않았다.

    t-선 방향 vs 크기 방향  corr = -0.7205  ->  직교성분 69.3%

즉 크기는 잉여 측정이 아니라 **아직 안 잰 좌표**다. LB 로 가중을 잰 다섯 번이
전부 로컬 최적과 달랐다 (대비 0.80->0.65, 타자 1.70->2.10, 편차 1.00->0.48).

## 안전성

표와 조회 키를 전혀 건드리지 않고 스칼라만 곱한다. 행 독립성은 구조적으로
보존되지만 그래도 관문 3종은 매번 돌린다 (CLAUDE.md 예외 없음).

    .\venv_submit\Scripts\python.exe -u exp\build_wscale.py ^
        --base submissions\cand_kb45.zip --axis 8 --mult 0.6 --name cand_kbw06
"""
import argparse
import hashlib
import io
import os
import sys
import zipfile

import joblib

import subname

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def build(base, axis, mult, name, force=False):
    subname.check(name)
    base_zip = base if os.path.isabs(base) else os.path.join(ROOT, base)
    out = os.path.join(ROOT, "submissions", f"{name}.zip")
    assert force or not os.path.exists(out), f"이미 있다 — 덮어쓰지 않는다: {out}"

    src = zipfile.ZipFile(base_zip)
    b = joblib.load(io.BytesIO(src.read("model/rf.pkl")))
    pl = [dict(s) for s in b["platoon"]]
    assert 0 <= axis < len(pl), f"축 번호 범위 밖: {axis} / {len(pl)}"

    w0 = float(pl[axis]["w"])
    w1 = w0 * float(mult)
    tab0 = b["platoon"][axis]["table"]
    pl[axis]["w"] = w1
    pl[axis]["note"] = pl[axis]["note"] + f" | W-SCALE w {w0:.6g}->{w1:.6g} (x{mult:g})"
    # 표는 같은 객체를 그대로 넘긴다 — 값이 바뀌지 않았음을 뒤에서 검증한다
    assert pl[axis]["table"] is tab0 and pl[axis]["cols"] == b["platoon"][axis]["cols"]

    b["platoon"] = pl
    b["note"] = str(b.get("note", "")) + \
        f" | W-SCALE 축{axis} w {w0:.6g}->{w1:.6g}, 표·키 불변"

    buf = io.BytesIO()
    joblib.dump(b, buf, compress=3)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("model/rf.pkl", buf.getvalue())
        for nm in ("script.py", "requirements.txt"):
            z.writestr(nm, src.read(nm))

    # 되읽어 표가 비트 단위로 같은지 확인
    chk = joblib.load(io.BytesIO(zipfile.ZipFile(out).read("model/rf.pkl")))
    for i, (a, c) in enumerate(zip(b["platoon"], chk["platoon"])):
        assert a["cols"] == c["cols"], i
        assert a["table"] == c["table"], f"표가 변했다: 축 {i}"
    assert chk["platoon"][axis]["table"] == tab0, "대상 축 표가 변했다"

    h = hashlib.sha256(open(out, "rb").read()).hexdigest()
    print(f"{name}.zip   기반 {os.path.basename(base_zip)}   축 {axis} ({pl[axis]['cols']})")
    print(f"  가중  {w0:.6f} -> {w1:.6f}   (x{mult:g})")
    print(f"  9가중 {[round(s['w'], 6) for s in pl]}")
    print(f"  표 9개 전부 비트 단위 동일 확인")
    print(f"  sha256 {h[:16]}   {os.path.getsize(out):,} bytes")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="submissions/cand_kb45.zip")
    ap.add_argument("--axis", type=int, required=True)
    ap.add_argument("--mult", type=float, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    build(a.base, a.axis, a.mult, a.name, a.force)


if __name__ == "__main__":
    main()
