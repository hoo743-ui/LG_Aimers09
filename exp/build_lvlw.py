r"""수준 2축(투수·타자 주효과)의 **가중만** 바꿔 제출 후보를 만든다. 학습 0회.

## 왜 이 도구가 필요한가

34~36회차에서 수준 2축을 후처리 스택에 넣었는데, 그 후보들을 만든 빌더가
저장소에 남지 않았다. 38회차(감쇠 표 −4.93)로 탐색 공간이 닫히면서 남은 수가
**이 두 가중의 미세조정뿐**이 됐으므로 재현 가능한 경로를 만들어 둔다.

## 좌표 — Champion `cand_bw25.zip` 의 platoon 9축

    [0..3] 후처리 편차 4축   0.20 · 0.825 · 0.280 · 0.45
    [4..6] 잔차 차등 3축     0.65 · 0.65 · 0.65        (손 · 2S · 주자)
    [7]    투수 주효과       1.00     k=50000
    [8]    타자 주효과       2.50     k=20000

표와 조회 키는 손대지 않고 스칼라만 바꾼다. 그래서 행 독립성이 구조적으로 보존되고
모델 재학습이 없다(38회차 이후 유일하게 허용되는 변경 등급).

## LB 곡선 (2026-08-20 기준, `gain(w) = A(2bw − w²)`)

    타자 수준   A = 2.414   b = 2.105   3점 실측(0 / 1.5 / 2.5)  -> 최적 2.10
    투수 수준   A = 0.371   b = 2.105   2점 + b 차용             -> 최적 2.10 (가정)

## 사용법

    python exp\build_lvlw.py --name cand_bw21 --wb 2.10
    python exp\build_lvlw.py --name cand_pw17 --wp 1.75
"""
import argparse
import hashlib
import io
import os
import shutil
import sys
import zipfile

import joblib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_ZIP = os.path.join(ROOT, "submissions", "cand_bw25.zip")
I_P, I_B = 7, 8

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def build(name, wp=None, wb=None, dev=None, diff=None, force=False):
    out = os.path.join(ROOT, "submissions", f"{name}.zip")
    assert force or not os.path.exists(out), f"이미 있다: {out} (--force 로 덮어쓰기)"
    src = zipfile.ZipFile(BASE_ZIP)
    b = joblib.load(io.BytesIO(src.read("model/rf.pkl")))
    pl = b["platoon"]
    assert len(pl) == 9, f"축 수가 9가 아니다: {len(pl)}"
    assert pl[I_P]["cols"] == ["pitcher_id"], pl[I_P]["cols"]
    assert pl[I_B]["cols"] == ["batter_id"], pl[I_B]["cols"]

    before = [s["w"] for s in pl]
    if wp is not None:
        pl[I_P] = dict(pl[I_P], w=float(wp))
    if wb is not None:
        pl[I_B] = dict(pl[I_B], w=float(wb))
    if dev is not None:                       # 편차 4축 전역 배수
        for i in range(0, 4):
            pl[i] = dict(pl[i], w=pl[i]["w"] * float(dev))
    if diff is not None:                      # 차등(대비) 3축 전역 배수
        for i in range(4, 7):
            pl[i] = dict(pl[i], w=pl[i]["w"] * float(diff))
    after = [s["w"] for s in pl]
    assert before != after, "바뀐 가중이 없다"
    b["platoon"] = pl
    b["note"] = (b["note"].split("|")[0]
                 + "| bw25 스칼라 변경 w=[" + ", ".join(f"{x:g}" for x in after) + "]")

    buf = io.BytesIO()
    joblib.dump(b, buf, compress=3)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("model/rf.pkl", buf.getvalue())
        for n in ("script.py", "requirements.txt"):
            z.writestr(n, src.read(n))
    h = hashlib.sha256(open(out, "rb").read()).hexdigest()
    print(f"{name}.zip")
    print("  before " + ", ".join(f"{x:g}" for x in before))
    print("  after  " + ", ".join(f"{x:g}" for x in after))
    print(f"  sha256 {h[:16]}   {os.path.getsize(out):,} bytes")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--wp", type=float, default=None, help="투수 주효과 가중 (현행 1.0)")
    ap.add_argument("--wb", type=float, default=None, help="타자 주효과 가중 (현행 2.5)")
    ap.add_argument("--dev", type=float, default=None, help="편차 4축 전역 배수")
    ap.add_argument("--diff", type=float, default=None, help="차등 3축 전역 배수")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    assert any(x is not None for x in (a.wp, a.wb, a.dev, a.diff)), "바꿀 가중을 지정하라"
    build(a.name, a.wp, a.wb, a.dev, a.diff, a.force)


if __name__ == "__main__":
    main()
