r"""후처리 9가중만 다시 써서 새 후보를 만든다. **재학습 없음.**

번들의 표(`platoon[i]['table']`)·모델·아핀은 그대로 두고 `platoon[i]['w']` 9개만
교체한다. 표가 같으므로 이것은 원장 §8 의 **안전 등급**(이미 LB 로 검증된 축의
가중 조정) 변경이다. 새 표를 만드는 것(TYPE B)과 혼동하지 말 것.

    .\venv_submit\Scripts\python.exe -u exp\reweight.py ^
        --src submissions\cand_h1.zip --out cand_rob1 ^
        --w 0.099,0.412,0.140,0.224,0.7037,0.7753,0.7753,1.985,2.5119 ^
        --tag "EXP044 강건최적 (앙상블 78 최악 +0.50)"
"""
import argparse
import io
import os
import re
import subprocess
import sys
import zipfile

import joblib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="원본 zip 또는 pkl")
    ap.add_argument("--w", required=True, help="9개 가중, 쉼표 구분")
    ap.add_argument("--out", required=True, help="후보 이름 (확장자 없이)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    w = [float(x) for x in a.w.split(",")]
    assert len(w) == 9, f"가중 9개가 필요하다 (받은 것 {len(w)})"

    if a.src.endswith(".zip"):
        with zipfile.ZipFile(a.src) as z:
            b = joblib.load(io.BytesIO(z.read("model/rf.pkl")))
    else:
        b = joblib.load(a.src)

    pl = b["platoon"]
    assert len(pl) == 9, f"platoon 이 9축이 아니다 ({len(pl)})"
    old = [float(d["w"]) for d in pl]
    for d, nw in zip(pl, w):
        d["w"] = float(nw)

    note = b.get("note", "")
    note = re.sub(r"\s*\|\s*w=\[[^\]]*\]", "", note)
    b["note"] = note + " | w=[" + ", ".join(f"{x:g}" for x in w) + "]" + \
                (f" | {a.tag}" if a.tag else "")

    print(f"원본 {a.src}")
    for i, (o, n) in enumerate(zip(old, w)):
        mark = "   " if abs(o - n) < 1e-9 else " <-"
        print(f"  [{i}] {o:9.6f} -> {n:9.6f}{mark}  {pl[i]['cols']}")

    out_pkl = os.path.join(ROOT, "model_cand", f"{a.out}.pkl")
    out_zip = os.path.join(ROOT, "submissions", f"{a.out}.zip")
    assert not os.path.exists(out_zip), f"이미 있다: {out_zip} (덮어쓰지 않는다)"
    os.makedirs(os.path.dirname(out_pkl), exist_ok=True)
    joblib.dump(b, out_pkl, compress=3)
    print(f"\n저장 {out_pkl} ({os.path.getsize(out_pkl)/1e6:.1f} MB)")

    r = subprocess.run(
        [sys.executable, os.path.join(ROOT, "make_submit.py"),
         "--model", os.path.relpath(out_pkl, ROOT),
         "--requirements", "requirements_cat.txt",
         "--out", os.path.relpath(out_zip, ROOT)],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
        errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    print(r.stdout[-1200:] if r.returncode == 0
          else "FAILED\n" + r.stdout[-600:] + r.stderr[-1200:])
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
