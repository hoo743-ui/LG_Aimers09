import sys, zipfile, os
sys.stdout.reconfigure(encoding="utf-8")
BS = chr(92)
for n in ("cand_submit_1.zip", "cand_submit_2.zip", "cand_submit_3.zip"):
    p = os.path.join("submissions", n)
    with zipfile.ZipFile(p) as z:
        names = z.namelist()
        bad = [x for x in names if BS in x]
        need = [r for r in ("model/rf.pkl", "script.py", "requirements.txt")
                if r not in names]
        ok = z.testzip()
    print(f"{n:<22} {names}")
    print(f"{'':<22} 역슬래시 {bad or '없음'} · 누락 {need or '없음'} · CRC {ok or 'OK'}")
