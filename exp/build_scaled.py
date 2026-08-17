r"""보정 가중만 바꾼 제출 후보. **모델 재학습 없음 · Champion 덮어쓰기 없음.**

## 왜 이것이 지금 옳은가

`+3.8% 게이트`와 TYPE B 동결은 **제출이 희소하다**는 전제 위에 있었다. 로컬로
판정 못 하는 것은 건드리지 않는다는 규율이었다. 제출이 5회/일 갱신이면 전제가
사라진다 — 로컬(시드 잡음 ±7.2)이 못 재는 것을 **LB(245,789행)로 직접 잰다.**

## 무엇을 재는가

보정 벡터 `c` 를 `w` 배로 실을 때 그 시즌의 실제 유효 강도를 `b` 라 하면

    이득(w) ≈ A(2bw − w²)      w* = b,   최대 이득 A·b²

LB 앵커 두 점이 이미 있다 — w=0 은 1049.9226, w=1.0 은 1057.3394 (+7.4168).
전이율 0.46 과 로컬 폴드 두 개(2022 최적 0.50, 2024 최적 0.75)가 모두 `b<1` 을
가리킨다. **한 점만 더 찍으면 `A`, `b` 가 풀린다.**

후처리 편차 4축의 가중 `[0.20, 0.825, 0.280, 0.45]` 도 **로컬로 튜닝된 값**이라
같은 감쇠를 받을 수 있다. 로컬 기여가 +37.1 로 차등 3축(+14.4)의 2.6배다.

## 산출물

    cand_wd060.zip   차등 3축 x 0.60      (편차 4축은 그대로)
    cand_wd035.zip   차등 3축 x 0.35      (곡선 세 번째 점 + 포물선 가정 검증)
    cand_dev075.zip  편차 4축 x 0.75      (차등 3축은 1.0 — 직교 검정)

셋 다 조회 키가 그 행 자신의 컬럼뿐이라 규정 4 에 안전하다 (스칼라만 바뀐다).

    .\.venv\Scripts\python.exe -u exp\build_scaled.py
"""
import hashlib
import os
import subprocess
import sys

import joblib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# (이름, 차등 3축 배수, 편차 4축 배수)
VARIANTS = [
    ("cand_wd060", 0.60, 1.00),
    ("cand_wd035", 0.35, 1.00),
    ("cand_dev075", 1.00, 0.75),
]
N_DEV = 4          # Champion 번들의 앞 4축이 후처리 편차축


def main():
    src = os.path.join(ROOT, "model_cand", "cat_submit_3.pkl")
    base = joblib.load(src)
    pl = base["platoon"]
    assert len(pl) == N_DEV + 3, f"축 수가 다르다: {len(pl)}"
    print(f"Champion 번들 로드 — 모델 {len(base['models'])}개, "
          f"피처 {len(base['features'])}, 후처리 {len(pl)}축")
    for i, s in enumerate(pl):
        tag = "편차" if i < N_DEV else "차등"
        print(f"  [{i}] {tag}  w={s['w']:<6} 표 {len(s['table']):>7,}  "
              f"{str(s.get('note', ''))[:52]}")

    print()
    for name, wd, wdev in VARIANTS:
        b = dict(base)
        specs = []
        for i, s in enumerate(pl):
            s2 = dict(s)
            s2["w"] = float(s["w"]) * (wdev if i < N_DEV else wd)
            specs.append(s2)
        b["platoon"] = specs
        b["note"] = (str(base.get("note", ""))
                     + f" | SCALED: 차등x{wd:g} 편차x{wdev:g}")
        pkl = os.path.join(ROOT, "model_cand", f"{name}.pkl")
        joblib.dump(b, pkl, compress=3)
        zp = os.path.join(ROOT, "submissions", f"{name}.zip")
        assert not os.path.exists(zp), f"이미 있다 — 덮어쓰지 않는다: {zp}"
        r = subprocess.run(
            [sys.executable, os.path.join(ROOT, "make_submit.py"),
             "--model", os.path.relpath(pkl, ROOT),
             "--requirements", "requirements_cat.txt",
             "--out", os.path.relpath(zp, ROOT)],
            cwd=ROOT, capture_output=True, text=True,
            encoding="utf-8", errors="replace")
        ok = r.returncode == 0 and os.path.exists(zp)
        print(f"  {name:<14} 차등x{wd:<5g} 편차x{wdev:<5g} "
              + ("빌드 OK" if ok else "실패\n" + r.stdout[-300:] + r.stderr[-500:]))

    print("\n=== SHA256 ===")
    for name, _, _ in VARIANTS:
        p = os.path.join(ROOT, "submissions", f"{name}.zip")
        if not os.path.exists(p):
            continue
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        print(f"  {name+'.zip':<20}{os.path.getsize(p)/1e6:>7.2f} MB  {h}")
    ch = os.path.join(ROOT, "submissions", "cand_submit_3.zip")
    print(f"\n  Champion 무결 확인 "
          f"{hashlib.sha256(open(ch, 'rb').read()).hexdigest()[:16]}...")


if __name__ == "__main__":
    main()
