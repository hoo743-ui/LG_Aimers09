r"""자율 연구 엔진 — 상태는 **디스크가 진실의 원천**이다.

세션이 끊기거나 재부팅돼도 `research/` 만 읽으면 복구된다.
모든 상태 쓰기는 tmp -> flush -> rename 의 원자적 교체다.
"""
import io, json, os, time, datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "research")
STATE = os.path.join(RES, "state.json")
CKPT = os.path.join(RES, "checkpoint.json")
BEAT = os.path.join(RES, "heartbeat.json")
HYPO = os.path.join(RES, "hypotheses.jsonl")
EXPS = os.path.join(RES, "experiments.jsonl")
CLOSED = os.path.join(RES, "closed_families.json")

def next_candidate():
    """
    hypotheses.jsonl의 최신 상태를 기준으로
    아직 실행할 가치가 있는 hypothesis 하나를 선택한다.
    """

    hypotheses = latest_hypotheses()

    candidates = []

    for hyp_id, h in hypotheses.items():
        status = h.get("status")

        if status not in ("CANDIDATE", "PROMISING"):
            continue

        # 이미 현재 실행 중이면 제외
        current = read(STATE, {}).get("current_experiment")
        if current and h.get("experiment_id") == current:
            continue

        priority = float(h.get("priority", 0))

        candidates.append((priority, hyp_id, h))

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: (-x[0], x[1])
    )

    _, hyp_id, hyp = candidates[0]

    return {
        "id": hyp_id,
        "hypothesis": hyp,
        "priority": float(hyp.get("priority", 0))
    }

def finish_experiment(rec, status="COMPLETED"):
    rec.setdefault("finished_at", now())
    append_jsonl(EXPS, rec)

    c = read(CKPT, {})
    c["status"] = status
    c["last_update"] = now()
    _atomic(CKPT, c)

    s = read(STATE, {})

    s["last_completed_experiment"] = rec.get("experiment_id")
    s["current_experiment"] = None
    s["experiments_completed"] = (
        int(s.get("experiments_completed", 0)) + 1
    )

    # 다음 hypothesis 자동 선택
    nxt = next_candidate()

    if nxt:
        s["next_hypothesis"] = nxt["id"]
        s["next_experiment"] = None
        s["next_hypothesis_priority"] = nxt["priority"]
        s["research_space_status"] = "READY_FOR_NEXT_HYPOTHESIS"
    else:
        s["next_hypothesis"] = None
        s["next_experiment"] = None
        s["research_space_status"] = "CURRENT_LEVEL_EXHAUSTED"

    s["updated_at"] = now()
    _atomic(STATE, s)

    beat("done")

def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def _atomic(path, obj):
    tmp = path + ".tmp"
    with io.open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read(path, default=None):
    if not os.path.exists(path):
        return default
    with io.open(path, encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    with io.open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                out.append(json.loads(ln))
    return out


def append_jsonl(path, obj):
    with io.open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def save_state(**kw):
    s = read(STATE, {})
    s.update(kw)
    s["updated_at"] = now()
    _atomic(STATE, s)
    return s


def start_experiment(exp_id, hyp_id, command, step="init"):
    save_state(current_experiment=exp_id, current_hypothesis=hyp_id)
    _atomic(CKPT, dict(current_experiment=exp_id, hypothesis=hyp_id,
                       process_id=os.getpid(), command=command,
                       start_time=now(), last_update=now(), status="RUNNING"))
    beat(step)


def beat(step):
    c = read(CKPT, {})
    _atomic(BEAT, dict(status=c.get("status", "RUNNING"),
                       experiment_id=c.get("current_experiment"),
                       started_at=c.get("start_time"), last_heartbeat=now(),
                       current_step=step, pid=os.getpid()))
    c["last_update"] = now()
    _atomic(CKPT, c)


def set_hypothesis_status(hyp_id, status, **kw):
    """hypotheses.jsonl 은 append-only 이므로 상태 변경도 append 한다."""
    append_jsonl(HYPO, dict(id=hyp_id, status=status, updated_at=now(), **kw))


def latest_hypotheses():
    """id 별 최신 레코드를 병합해 돌려준다."""
    out = {}
    for h in read_jsonl(HYPO):
        out.setdefault(h["id"], {}).update(h)
    return out


def pid_alive(pid):
    if not pid:
        return False
    try:
        import subprocess
        r = subprocess.run(["tasklist", "/FI", f"PID eq {int(pid)}"],
                           capture_output=True, text=True)
        return str(int(pid)) in r.stdout
    except Exception:
        return False


def resume_report():
    """재시작 시 첫 호출. 무엇을 해야 하는지 판정한다."""
    s, c, b = read(STATE, {}), read(CKPT, {}), read(BEAT, {})
    exps = read_jsonl(EXPS)
    status = c.get("status")
    verdict = "NO_STATE"
    if status == "RUNNING":
        verdict = ("CASE_B_RUNNING" if pid_alive(c.get("process_id"))
                   else "CASE_B_INTERRUPTED")
    elif status == "COMPLETED":
        verdict = "CASE_A_COMPLETED"
    elif status == "FAILED":
        verdict = "CASE_C_FAILED"
    return dict(verdict=verdict, state=s, checkpoint=c, heartbeat=b,
                n_experiments=len(exps),
                last=exps[-1] if exps else None)
