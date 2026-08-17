import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E
E.start_experiment("EXP_CRASHTEST", "H000", "crash test", step="probe")
E.beat("중간 단계 — 여기서 죽는다")
time.sleep(120)          # 여기서 강제 종료된다
