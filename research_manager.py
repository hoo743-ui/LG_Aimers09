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
CLAUDE_STATE_FILE = RESEARCH / "manager_claude.json"
CONTROL_FILE = RESEARCH / "control.json"

LOG_DIR = RESEARCH / "manager_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

POLL_SECONDS = 10

APPDATA = os.environ.get("APPDATA", "")
CLAUDE_CLI = Path(APPDATA) / "npm" / "node_modules" / "@anthropic-ai" / "claude-code" / "bin" / "claude.exe"
MANAGER_LOCK_FILE = RESEARCH / "manager.lock"

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

CRITICAL RESUME RULE

현재 state.json에서:
- current_experiment == null
- last_completed_experiment 존재
- next_experiment가 이미 완료된 experiment를 가리킴

이라면 next_experiment 값을 신뢰하지 말라.

반드시:
1. research/hypotheses.jsonl을 다시 읽는다.
2. 각 hypothesis의 최신 status를 재구성한다.
3. CANDIDATE / PROMISING 중 아직 TESTED/MEASURED/CLOSED되지 않은 후보를 찾는다.
4. priority가 가장 높은 유효 후보를 다음 연구로 선택한다.
5. state.json의 next_hypothesis / next_experiment를 갱신한다.
6. 그 후보의 실험을 실제로 시작한다.
7. 더 이상 후보가 없으면 현재 level이 정말 exhausted인지 확인하고 다음 research level로 승격한다.
8. 절대로 last_completed_experiment를 다시 실행하지 않는다.

연구가 끝났다고 판단하기 전에 반드시
"다음 hypothesis를 선택했는가?"
를 확인한다.

current_experiment = null 인 상태에서
무조건 종료하지 않는다.
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

def read_jsonl(path: Path):
    if not path.exists():
        return []

    out = []

    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    except Exception as exc:
        log(f"[WARN] JSONL read failed: {path} -> {exc}")

    return out

def write_json_atomic(path: Path, data: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    os.replace(tmp, path)

def read_control_action():
    if not CONTROL_FILE.exists():
        return None

    try:
        data = json.loads(
            CONTROL_FILE.read_text(encoding="utf-8")
        )
        return data.get("action")
    except Exception:
        return None


def clear_control_action():
    try:
        if CONTROL_FILE.exists():
            CONTROL_FILE.unlink()
    except Exception as exc:
        log(f"[WARN] control.json delete failed: {exc}")


def handle_control_action():
    """
    GUI -> Research Manager control bridge.

    PAUSE:
        현재 실행 중인 실험은 죽이지 않는다.
        현재 실험이 끝난 뒤 새 연구를 시작하지 않는다.

    RESUME:
        자율 연구 재개.

    STOP:
        Research Manager만 중단한다.
        현재 실험 Python은 죽이지 않는다.
    """

    action = read_control_action()

    if action == "PAUSE":
        state = read_json(STATE_FILE)
        state["manager_mode"] = "PAUSED"
        state["pause_requested_at"] = now()

        write_json_atomic(STATE_FILE, state)
        clear_control_action()

        log(
            "[CONTROL] PAUSE requested. "
            "Current experiment will NOT be killed."
        )

        return "PAUSE"

    if action == "RESUME":
        state = read_json(STATE_FILE)
        state["manager_mode"] = "RUNNING"
        state["resumed_at"] = now()

        write_json_atomic(STATE_FILE, state)
        clear_control_action()

        log("[CONTROL] RESUME requested.")

        return "RESUME"

    if action == "STOP":
        state = read_json(STATE_FILE)
        state["manager_mode"] = "STOPPING"
        state["stop_requested_at"] = now()

        write_json_atomic(STATE_FILE, state)
        clear_control_action()

        log(
            "[CONTROL] STOP requested. "
            "Manager will stop after current loop."
        )

        return "STOP"

    return None

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

def find_managed_claude_processes():
    """Return only the real Claude Code CLI processes, never desktop Claude."""
    try:
        result = subprocess.run(
            [
                "powershell", "-NoProfile", "-Command",
                "Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'claude.exe' -and $_.CommandLine -like '*@anthropic-ai\\claude-code\\bin\\claude.exe*' } | Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress"
            ],
            capture_output=True, text=True, encoding="utf-8", errors="ignore", timeout=8
        )
        raw = result.stdout.strip()
        if not raw:
            return []
        data = json.loads(raw)
        return [data] if isinstance(data, dict) else data
    except Exception as exc:
        log(f"[WARN] Claude Code discovery failed: {exc}")
        return []


def get_managed_claude_pid():
    data = read_json(CLAUDE_STATE_FILE)
    pid = data.get("pid")
    if not pid:
        return None
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return None
    for proc in find_managed_claude_processes():
        try:
            if int(proc.get("ProcessId", -1)) == pid:
                return pid
        except Exception:
            pass
    return None


def managed_claude_running():
    return get_managed_claude_pid() is not None


def clear_stale_claude_state():
    if CLAUDE_STATE_FILE.exists() and get_managed_claude_pid() is None:
        data = read_json(CLAUDE_STATE_FILE)
        if data:
            data["status"] = "STOPPED"
            data["stopped_at"] = now()
            write_json_atomic(CLAUDE_STATE_FILE, data)


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

def sync_next_hypothesis():
    """
    state.json의 next_hypothesis가 비어 있거나
    이미 완료된 hypothesis를 가리키면
    engine.next_candidate()로 최신 후보를 다시 선택한다.
    """
    try:
        import sys

        if str(RESEARCH) not in sys.path:
            sys.path.insert(0, str(RESEARCH))

        import engine

        state = read_json(STATE_FILE)

        current = state.get("current_experiment")
        last_completed = state.get("last_completed_experiment")
        current_hypothesis = state.get("current_hypothesis")
        next_hypothesis = state.get("next_hypothesis")

        # 현재 실험이 살아있으면 건드리지 않는다.
        if current:
            return next_hypothesis

        # 이미 유효한 다음 hypothesis가 있으면 유지.
        if next_hypothesis:
            latest = engine.latest_hypotheses()
            h = latest.get(next_hypothesis)

            if h and h.get("status") in ("CANDIDATE", "PROMISING"):
                return next_hypothesis

        # stale / missing state이면 최신 후보를 다시 선택
        nxt = engine.next_candidate()

        if not nxt:
            state["next_hypothesis"] = None
            state["next_hypothesis_priority"] = None
            state["next_experiment"] = None
            state["research_space_status"] = "CURRENT_LEVEL_EXHAUSTED"
            write_json_atomic(STATE_FILE, state)

            log("[SYNC] No valid next hypothesis.")
            return None

        # 완료된 hypothesis가 다시 선택되는 비정상 상태 방지
        if nxt["id"] == current_hypothesis or nxt["id"] == last_completed:
            log(
                f"[SYNC] Candidate {nxt['id']} matches a completed/current "
                "hypothesis. Searching next candidate."
            )

            latest = engine.latest_hypotheses()

            candidates = []
            for hid, h in latest.items():
                if hid in (current_hypothesis, last_completed):
                    continue

                if h.get("status") not in ("CANDIDATE", "PROMISING"):
                    continue

                candidates.append(
                    (
                        float(h.get("priority", 0)),
                        hid,
                        h
                    )
                )

            if not candidates:
                state["next_hypothesis"] = None
                state["next_hypothesis_priority"] = None
                state["next_experiment"] = None
                state["research_space_status"] = "CURRENT_LEVEL_EXHAUSTED"
                write_json_atomic(STATE_FILE, state)

                log("[SYNC] All remaining candidates are invalid/completed.")
                return None

            candidates.sort(key=lambda x: (-x[0], x[1]))
            priority, hid, _ = candidates[0]

            state["next_hypothesis"] = hid
            state["next_hypothesis_priority"] = priority
            state["next_experiment"] = None
            state["research_space_status"] = "READY_FOR_NEXT_HYPOTHESIS"

            write_json_atomic(STATE_FILE, state)

            log(
                f"[SYNC] next_hypothesis={hid}, "
                f"priority={priority}"
            )

            return hid

        state["next_hypothesis"] = nxt["id"]
        state["next_hypothesis_priority"] = nxt["priority"]
        state["next_experiment"] = None
        state["research_space_status"] = "READY_FOR_NEXT_HYPOTHESIS"

        write_json_atomic(STATE_FILE, state)

        log(
            f"[SYNC] next_hypothesis={nxt['id']}, "
            f"priority={nxt['priority']}"
        )

        return nxt["id"]

    except Exception as exc:
        log(f"[ERROR] next hypothesis sync failed: {exc}")
        return None    


def launch_claude():
    """Launch real Claude Code executable directly and track its real PID."""
    sync_next_hypothesis()
    if managed_claude_running():
        pid = get_managed_claude_pid()
        log(f"[SKIP] Managed Claude already running. PID={pid}")
        return None
    clear_stale_claude_state()
    if not CLAUDE_CLI.exists():
        log(f"[ERROR] Claude Code executable not found: {CLAUDE_CLI}")
        return None

    state = read_json(STATE_FILE)
    next_hypothesis_id = state.get("next_hypothesis")
    hypothesis_text = "없음"
    if next_hypothesis_id:
        latest = {}
        for h in read_jsonl(RESEARCH / "hypotheses.jsonl"):
            if h.get("id"):
                latest[h["id"]] = h
        if next_hypothesis_id in latest:
            hypothesis_text = json.dumps(latest[next_hypothesis_id], ensure_ascii=False, indent=2)

    prompt_to_run = f"""
{CLAUDE_PROMPT}

============================================================
CURRENT AUTONOMOUS RESEARCH TARGET
============================================================

현재 연구 엔진이 선택한 다음 hypothesis:

{hypothesis_text}

이번 세션의 최우선 과제는 위 hypothesis다.

반드시:
1. 기존 실험/원장에서 해당 hypothesis의 상태를 먼저 확인한다.
2. 이미 TESTED / MEASURED / CLOSED 된 hypothesis는 다시 실행하지 않는다.
3. 현재 Champion과의 중복 여부를 확인한다.
4. legality / leakage / data resolution을 먼저 검사한다.
5. cheap probe를 먼저 수행한다.
6. 필요하면 새로운 experiment script를 직접 작성한다.
7. 결과를 research/experiments.jsonl에 기록한다.
8. hypothesis 상태를 TESTED / PROMISING / CLOSED 중 하나로 갱신한다.
9. 실험 완료 시 engine.finish_experiment()를 사용한다.
10. 다음 hypothesis가 있으면 state.json에 반영한다.
11. current_experiment가 null이라고 해서 연구를 종료하지 않는다.
12. last_completed_experiment를 다시 실행하지 않는다.
13. Champion은 절대 덮어쓰지 않는다.
14. 실제 LB 제출은 하지 않는다.

연구 family가 소진되면 다음 research level로 이동한다.
"""

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    stdout_log = LOG_DIR / f"claude_{timestamp}.log"
    log(f"[START] Launching managed Claude Code directly: {CLAUDE_CLI}")
    try:
        out = stdout_log.open("w", encoding="utf-8")
        proc = subprocess.Popen(
            [str(CLAUDE_CLI), "-p", prompt_to_run, "--max-budget-usd", "10"],
            cwd=str(ROOT), stdout=out, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
        )
        write_json_atomic(CLAUDE_STATE_FILE, {
            "pid": proc.pid, "started_at": now(), "status": "RUNNING", "hypothesis": next_hypothesis_id
        })
        log(f"[OK] Managed Claude Code launched. PID={proc.pid}")
        return proc
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


def acquire_manager_lock():
    if MANAGER_LOCK_FILE.exists():
        data = read_json(MANAGER_LOCK_FILE)
        pid = data.get("pid") if data else None
        if pid and process_exists(pid):
            log(f"[STOP] Another Research Manager is already running. PID={pid}")
            return False
    write_json_atomic(MANAGER_LOCK_FILE, {"pid": os.getpid(), "started_at": now()})
    return True


def release_manager_lock():
    try:
        if MANAGER_LOCK_FILE.exists():
            data = read_json(MANAGER_LOCK_FILE)
            if data.get("pid") == os.getpid():
                MANAGER_LOCK_FILE.unlink()
    except Exception:
        pass


def monitor_loop():
    if not acquire_manager_lock():
        return

    try:
        log("=== RESEARCH MANAGER STARTED ===")
        print_status()
        clear_stale_claude_state()

        while True:
            action = handle_control_action()

            state = read_json(STATE_FILE)
            manager_mode = state.get("manager_mode", "RUNNING")

            if action == "STOP" or manager_mode == "STOPPING":
                log("[CONTROL] Manager stopping by user request.")
                break

            if manager_mode == "PAUSED":
                log(
                    "[CONTROL] Manager paused. "
                    "Monitoring only; no new Claude launch."
                )
                time.sleep(POLL_SECONDS)
                continue
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
                    if not managed_claude_running():
                        launch_claude()

            # ----------------------------------------
            # 2) 실험이 완료되었거나 현재 실행이 없는 경우
            # ----------------------------------------
            elif status in ("COMPLETED", "FAILED", "INTERRUPTED", None):
                if not managed_claude_running():
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

    finally:
        release_manager_lock()

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
        print(f"Claude alive: {managed_claude_running()}")
        print(f"Next        : {state.get('next_experiment')}")
        print("==============================")
        print("DRY RUN: no process will be started or stopped.\n")

    elif command in ("start", "monitor"):
        monitor_loop()

    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    main()   