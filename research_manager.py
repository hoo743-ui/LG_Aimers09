import json
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
RESEARCH = ROOT / "research"

STATE_FILE = RESEARCH / "state.json"
CHECKPOINT_FILE = RESEARCH / "checkpoint.json"
HEARTBEAT_FILE = RESEARCH / "heartbeat.json"

LOG_DIR = RESEARCH / "manager_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

POLL_SECONDS = 10

# Claude 자동 실행 시 사용할 프롬프트
CLAUDE_PROMPT = r"""
너는 현재 LG Aimers 프로젝트의 MACHINE LEARNING RESEARCH MACHINE이다.

반드시 프로젝트의 다음 파일들을 먼저 읽어라:
- CLAUDE.md
- research/state.json
- research/checkpoint.json
- research/heartbeat.json
- research/hypotheses.jsonl
- research/experiments.jsonl
- research/closed_families.json
- reports/latest.md
- RELATION_LEDGER.md
- DEVIATION_LEDGER.md
- LB_LEDGER.csv

현재 연구 상태를 디스크에서 복구하라.

규칙:
1. 현재 Champion을 절대 덮어쓰지 않는다.
2. 이전 실험이 중단되었다면 checkpoint를 확인하고 안전하게 resume/retry한다.
3. 이미 끝난 실험을 다시 시작하지 않는다.
4. 현재 hypothesis queue에서 다음 연구를 선택한다.
5. 연구가 한 family에서 소진되면 다음 research level로 이동한다.
6. SIGNAL INNOVATION과 ESTIMATION CHANGE를 구분한다.
7. leakage / 규정 위반을 절대 허용하지 않는다.
8. 결과를 반드시 research/ 아래 상태 파일에 저장한다.
9. 실제 LB 제출은 하지 않는다.
10. 제출 artifact가 필요하면 artifact까지만 만든다.
11. 사용자의 추가 입력을 기다리지 말고 자율적으로 연구를 계속한다.

가장 먼저 현재 state와 checkpoint를 확인하고,
미완료 연구가 있으면 복구하고,
없으면 다음 hypothesis부터 진행하라.

중요:
현재 실행 중인 Python 실험 프로세스가 살아있다면
절대로 중복 실행하지 말라.
"""

def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] JSON read failed: {path} -> {exc}")
        return {}


def write_json_atomic(path: Path, data: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    os.replace(tmp, path)


def log(message: str):
    line = f"[{now()}] {message}"
    print(line)

    logfile = LOG_DIR / "manager.log"

    with logfile.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def process_exists(pid) -> bool:
    """
    Windows에서 PID가 살아있는지 확인.
    psutil 없이 tasklist 사용.
    """
    if not pid:
        return False

    try:
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {int(pid)}"],
            capture_output=True,
            text=True,
            encoding="cp949",
            errors="ignore",
            timeout=5,
        )

        return str(pid) in result.stdout

    except Exception as exc:
        log(f"[WARN] PID check failed: {exc}")
        return False


def current_status():
    state = read_json(STATE_FILE)
    checkpoint = read_json(CHECKPOINT_FILE)
    heartbeat = read_json(HEARTBEAT_FILE)

    return state, checkpoint, heartbeat


def print_status():
    state, checkpoint, heartbeat = current_status()

    champion = state.get("champion", {})

    print("\n========== RESEARCH MANAGER STATUS ==========")
    print(f"Champion     : {champion.get('lb_score')}")
    print(f"Artifact     : {champion.get('artifact')}")
    print(f"Experiment   : {state.get('current_experiment')}")
    print(f"Hypothesis   : {state.get('current_hypothesis')}")
    print(f"Checkpoint   : {checkpoint.get('status')}")
    print(f"Process ID   : {checkpoint.get('process_id')}")
    print(f"Heartbeat    : {heartbeat.get('last_heartbeat')}")
    print(f"Research lvl : {state.get('research_level')}")
    print(f"Next         : {state.get('next_experiment')}")
    print("============================================\n")


def claude_running() -> bool:
    """
    Claude CLI 자체가 이미 실행 중인지 대략 확인.
    중복 실행 방지용.
    """
    try:
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq claude.exe"],
            capture_output=True,
            text=True,
            encoding="cp949",
            errors="ignore",
            timeout=5,
        )
        return "claude.exe" in result.stdout.lower()

    except Exception:
        return False


def launch_claude():
    if claude_running():
        log("[SKIP] Claude already running.")
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stdout_log = LOG_DIR / f"claude_{timestamp}.log"

    log("[START] Launching Claude autonomous research...")

    try:
        with stdout_log.open("w", encoding="utf-8") as out:
            proc = subprocess.Popen(
                [
                    "claude",
                    "-p",
                    CLAUDE_PROMPT,
                    "--max-budget-usd",
                    "10",
                ],
                cwd=str(ROOT),
                stdout=out,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            )

        log(f"[OK] Claude launched. PID={proc.pid}")
        return proc

    except FileNotFoundError:
        log(
            "[ERROR] 'claude' command not found. "
            "Run 'claude --version' in PowerShell first."
        )
        return None

    except Exception as exc:
        log(f"[ERROR] Claude launch failed: {exc}")
        return None


def mark_interrupted():
    checkpoint = read_json(CHECKPOINT_FILE)

    if checkpoint.get("status") != "RUNNING":
        return

    checkpoint["status"] = "INTERRUPTED"
    checkpoint["interrupted_at"] = now()

    write_json_atomic(CHECKPOINT_FILE, checkpoint)

    log(
        f"[RECOVERY] Experiment "
        f"{checkpoint.get('current_experiment') or checkpoint.get('hypothesis')} "
        f"marked INTERRUPTED."
    )


def monitor_loop():
    log("=== RESEARCH MANAGER STARTED ===")
    print_status()

    while True:
        state, checkpoint, heartbeat = current_status()

        status = checkpoint.get("status")
        pid = checkpoint.get("process_id")

        # ----------------------------------------
        # 1) 현재 실험이 실행 중인 경우
        # ----------------------------------------
        if status == "RUNNING":
            if pid and process_exists(pid):
                log(
                    f"[MONITOR] Experiment process alive. "
                    f"PID={pid}"
                )
            else:
                log(
                    "[DETECT] Checkpoint says RUNNING "
                    "but process is dead."
                )

                mark_interrupted()

                # Claude가 이미 있다면 기다림
                if not claude_running():
                    launch_claude()

        # ----------------------------------------
        # 2) 실험이 완료되었거나 현재 실행이 없는 경우
        # ----------------------------------------
        elif status in ("COMPLETED", "FAILED", "INTERRUPTED", None):
            if not claude_running():
                log(
                    f"[IDLE] checkpoint={status}. "
                    "Starting/resuming autonomous research."
                )
                launch_claude()
            else:
                log("[WAIT] Claude already running.")

        # ----------------------------------------
        # 3) unknown 상태
        # ----------------------------------------
        else:
            log(f"[WARN] Unknown checkpoint status: {status}")

        time.sleep(POLL_SECONDS)


def main():
    if len(sys.argv) < 2:
        print(
            "Usage:\n"
            "  python research_manager.py status\n"
            "  python research_manager.py dry-run\n"
            "  python research_manager.py start\n"
            "  python research_manager.py monitor\n"
        )
        return

    command = sys.argv[1].lower()

    if command == "status":
        print_status()

    elif command == "dry-run":
        state, checkpoint, heartbeat = current_status()

        print("\n========== DRY RUN ==========")
        print(f"Champion    : {state.get('champion', {}).get('lb_score')}")
        print(f"Experiment  : {state.get('current_experiment')}")
        print(f"Hypothesis  : {state.get('current_hypothesis')}")
        print(f"Checkpoint  : {checkpoint.get('status')}")
        print(f"PID         : {checkpoint.get('process_id')}")
        print(f"PID alive   : {process_exists(checkpoint.get('process_id'))}")
        print(f"Claude alive: {claude_running()}")
        print(f"Next        : {state.get('next_experiment')}")
        print("==============================")
        print("DRY RUN: no process will be started or stopped.\n")

    elif command in ("start", "monitor"):
        monitor_loop()

    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    main()   