# make_submit.py
r"""제출용 submit.zip 생성.

Windows PowerShell 5.1의 Compress-Archive 는 zip 엔트리 경로를 역슬래시로
쓰기 때문에 리눅스 평가 서버에서 model/ 폴더가 만들어지지 않는다.
zipfile 로 직접 쓰면서 경로를 슬래시로 정규화한다.

사용법:
    .\.venv\Scripts\python.exe make_submit.py
    .\.venv\Scripts\python.exe make_submit.py --model model_cand/exp001.pkl `
        --out submissions/exp_001.zip

--model 을 주면 그 pkl 을 zip 안에서 model/rf.pkl 로 넣는다. 디스크의
model/ 폴더를 건드리지 않으므로 **베이스라인 제출물이 안전하다.** 후보를
만들 때는 반드시 이 형태를 쓸 것.
"""
import argparse
import os
import zipfile

OUT = "submit.zip"

# zip 최상단에 들어갈 항목. 파일은 그대로, 디렉터리는 하위 전체를 담는다.
ITEMS = ["model", "script.py", "requirements.txt"]


def iter_entries(items):
    """(디스크 경로, zip 내부 경로) 쌍을 순회. zip 경로는 항상 슬래시."""
    for item in items:
        if not os.path.exists(item):
            raise FileNotFoundError(f"제출 항목이 없음: {item}")
        if os.path.isfile(item):
            yield item, item.replace(os.sep, "/")
            continue
        for root, _, files in os.walk(item):
            for name in sorted(files):
                path = os.path.join(root, name)
                yield path, path.replace(os.sep, "/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="이 pkl 을 zip 안에서 model/rf.pkl 로 넣는다. "
                         "디스크의 model/ 폴더는 건드리지 않는다.")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--requirements", default="requirements.txt",
                    help="zip 안에 requirements.txt 로 넣을 파일. CatBoost "
                         "후보는 requirements_cat.txt 를 쓴다 — 저장소의 "
                         "requirements.txt 를 건드리면 베이스라인 재현이 깨진다.")
    a = ap.parse_args()

    out = a.out
    if a.model:
        # model/ 폴더 대신 지정한 pkl 하나만 model/rf.pkl 로 담는다.
        entries = [(a.model, "model/rf.pkl"),
                   ("script.py", "script.py"),
                   (a.requirements, "requirements.txt")]
    else:
        entries = list(iter_entries(ITEMS))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        total = 0
        for src, arc in entries:
            zf.write(src, arc)
            size = os.path.getsize(src)
            total += size
            print(f"  + {arc}  ({size/1e6:.2f} MB)")
    print(f"\n✅ {out} 생성 ({os.path.getsize(out)/1e6:.2f} MB, 원본 {total/1e6:.2f} MB)")

    # 검증 — 역슬래시가 섞였거나 항목이 빠지면 즉시 알림
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
    bad = [n for n in names if "\\" in n]
    if bad:
        raise SystemExit(f"❌ 역슬래시 경로 발견: {bad}")
    for required in ("model/rf.pkl", "script.py", "requirements.txt"):
        if required not in names:
            raise SystemExit(f"❌ 누락: {required} (실제: {names})")
    print("✅ 구조 검증 통과:", names)


if __name__ == "__main__":
    main()
