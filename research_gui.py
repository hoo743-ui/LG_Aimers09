import sys
import json
import os
import subprocess
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox

if getattr(sys, "frozen", False):
    ROOT = Path(sys.executable).resolve().parent
else:
    ROOT = Path(__file__).resolve().parent
RESEARCH = ROOT / "research"
STATE_FILE = RESEARCH / "state.json"
CHECKPOINT_FILE = RESEARCH / "checkpoint.json"
HEARTBEAT_FILE = RESEARCH / "heartbeat.json"
CONTROL_FILE = RESEARCH / "control.json"
EXPERIMENT_LOG = RESEARCH / "experiments.jsonl"
MANAGER_LOG = RESEARCH / "manager_logs" / "manager.log"
POLL_MS = 2000


def read_json(path: Path):
    try:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def count_log():
    if not EXPERIMENT_LOG.exists():
        return 0
    try:
        with EXPERIMENT_LOG.open("r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def last_experiments(limit=8):
    if not EXPERIMENT_LOG.exists():
        return []
    try:
        lines = EXPERIMENT_LOG.read_text(encoding="utf-8").splitlines()
        out = []
        for line in lines[-limit:]:
            try:
                out.append(json.loads(line))
            except Exception:
                continue
        return list(reversed(out))
    except Exception:
        return []


def tail_log(limit=18):
    if not MANAGER_LOG.exists():
        return []
    try:
        return MANAGER_LOG.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]
    except Exception:
        return []


def process_alive(pid):
    if not pid:
        return False
    try:
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {int(pid)}"],
            capture_output=True, text=True, encoding="cp949", errors="ignore", timeout=3,
        )
        return str(pid) in result.stdout
    except Exception:
        return False


class ResearchMachine(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("LG Aimers · Research Machine")
        self.geometry("1180x760")
        self.minsize(1000, 680)
        self.manager_process = None
        self._build_style()
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.refresh()

    def _build_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Title.TLabel", font=("Segoe UI", 20, "bold"))
        style.configure("Subtitle.TLabel", font=("Segoe UI", 10))
        style.configure("Card.TFrame", relief="solid", borderwidth=1)
        style.configure("CardTitle.TLabel", font=("Segoe UI", 9, "bold"))
        style.configure("CardValue.TLabel", font=("Segoe UI", 16, "bold"))
        style.configure("Small.TLabel", font=("Segoe UI", 9))
        style.configure("Status.TLabel", font=("Segoe UI", 12, "bold"))

    def _build_ui(self):
        root = ttk.Frame(self, padding=16)
        root.pack(fill="both", expand=True)

        header = ttk.Frame(root)
        header.pack(fill="x")
        ttk.Label(header, text="LG Aimers Research Machine", style="Title.TLabel").pack(side="left")
        ttk.Label(header, text="Autonomous ML research control center", style="Subtitle.TLabel").pack(side="left", padx=(12, 0), pady=(7, 0))
        self.status_var = tk.StringVar(value="UNKNOWN")
        ttk.Label(header, textvariable=self.status_var, style="Status.TLabel").pack(side="right")

        controls = ttk.Frame(root)
        controls.pack(fill="x", pady=(14, 10))
        ttk.Button(controls, text="▶ 연구 시작", command=self.start_manager).pack(side="left", padx=(0, 6))
        ttk.Button(controls, text="⏸ 일시중지 요청", command=lambda: self.write_control("PAUSE")).pack(side="left", padx=6)
        ttk.Button(controls, text="▶ 계속", command=lambda: self.write_control("RESUME")).pack(side="left", padx=6)
        ttk.Button(controls, text="■ 매니저 중단", command=self.stop_manager).pack(side="left", padx=6)
        ttk.Button(controls, text="↻ 새로고침", command=self.refresh).pack(side="left", padx=6)
        ttk.Button(controls, text="전체 결과", command=self.show_results).pack(side="left", padx=6)
        ttk.Button(controls, text="폴더 열기", command=self.open_folder).pack(side="right")

        cards = ttk.Frame(root)
        cards.pack(fill="x", pady=(0, 12))
        self.card_values = {}
        for i, (title, key) in enumerate([
            ("Champion", "champion"), ("현재 실험", "experiment"), ("가설", "hypothesis"),
            ("Research Level", "level"), ("Python PID", "pid"), ("Heartbeat", "heartbeat")
        ]):
            cards.columnconfigure(i, weight=1)
            frame = ttk.Frame(cards, padding=12, style="Card.TFrame")
            frame.grid(row=0, column=i, sticky="nsew", padx=4)
            ttk.Label(frame, text=title, style="CardTitle.TLabel").pack(anchor="w")
            var = tk.StringVar(value="-")
            self.card_values[key] = var
            ttk.Label(frame, textvariable=var, style="CardValue.TLabel").pack(anchor="w", pady=(4, 0))

        body = ttk.Panedwindow(root, orient="horizontal")
        body.pack(fill="both", expand=True)
        left = ttk.Frame(body, padding=8)
        right = ttk.Frame(body, padding=8)
        body.add(left, weight=1)
        body.add(right, weight=1)

        ttk.Label(left, text="최근 연구 결과", style="CardTitle.TLabel").pack(anchor="w")
        self.results_text = tk.Text(left, wrap="word", font=("Consolas", 10), height=24, state="disabled")
        self.results_text.pack(fill="both", expand=True, pady=(8, 0))

        ttk.Label(right, text="Manager 로그", style="CardTitle.TLabel").pack(anchor="w")
        self.log_text = tk.Text(right, wrap="none", font=("Consolas", 9), height=24, state="disabled")
        self.log_text.pack(fill="both", expand=True, pady=(8, 0))

        self.footer = tk.StringVar(value="Ready")
        ttk.Label(root, textvariable=self.footer, style="Small.TLabel").pack(anchor="w", pady=(8, 0))

    def write_control(self, action):
        write_json(CONTROL_FILE, {"action": action, "requested_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
        self.footer.set(f"control request: {action}")
        self.refresh()

    def open_folder(self):
        try:
            os.startfile(str(ROOT))
        except Exception as exc:
            messagebox.showerror("오류", str(exc))

    def start_manager(self):
        checkpoint = read_json(CHECKPOINT_FILE)
        pid = checkpoint.get("process_id")
        if checkpoint.get("status") == "RUNNING" and process_alive(pid):
            messagebox.showinfo("이미 실행 중", f"현재 실험 PID {pid}가 실행 중입니다.\n중복 실행하지 않습니다.")
            return
        if self.manager_process is not None and self.manager_process.poll() is None:
            messagebox.showinfo("이미 실행 중", "Research Manager가 이미 실행 중입니다.")
            return
        python_exe = ROOT / ".venv" / "Scripts" / "python.exe"
        manager = ROOT / "research_manager.py"
        try:
            self.manager_process = subprocess.Popen(
                [str(python_exe), str(manager), "monitor"],
                cwd=str(ROOT),
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            )
            self.footer.set(f"Research Manager started (PID {self.manager_process.pid})")
        except Exception as exc:
            messagebox.showerror("시작 실패", str(exc))
        self.refresh()

    def stop_manager(self):
        if self.manager_process is None or self.manager_process.poll() is not None:
            messagebox.showinfo("매니저 상태", "이 GUI가 직접 실행한 Research Manager가 없습니다.")
            return
        if not messagebox.askyesno("매니저 중단", "Research Manager만 중단합니다.\n현재 실험 Python 프로세스 자체는 종료하지 않습니다.\n\n계속할까요?"):
            return
        try:
            self.manager_process.terminate()
            self.footer.set("Research Manager stopped")
        except Exception as exc:
            messagebox.showerror("중단 실패", str(exc))

    def show_results(self):
        state = read_json(STATE_FILE)
        checkpoint = read_json(CHECKPOINT_FILE)
        heartbeat = read_json(HEARTBEAT_FILE)
        rows = last_experiments(50)
        champion = state.get("champion", {})
        lines = [
            "===== RESEARCH STATUS =====",
            f"Champion : {champion.get('lb_score', '-')}",
            f"Artifact : {champion.get('artifact', '-')}",
            f"Experiment : {state.get('current_experiment', '-')}",
            f"Hypothesis : {state.get('current_hypothesis', '-')}",
            f"Research level : {state.get('research_level', '-')}",
            f"Checkpoint : {checkpoint.get('status', '-')}",
            f"PID : {checkpoint.get('process_id', '-')}",
            f"Heartbeat : {heartbeat.get('last_heartbeat', '-')}",
            "",
            "===== EXPERIMENT HISTORY =====",
        ]
        for row in rows:
            lines.append(f"{row.get('experiment_id', '-')} | {row.get('hypothesis_id', '-')} | {row.get('decision', '-')} | prod={row.get('production_result', '-')}")
        win = tk.Toplevel(self)
        win.title("전체 연구 결과")
        win.geometry("900x650")
        text = tk.Text(win, wrap="word", font=("Consolas", 10))
        text.pack(fill="both", expand=True, padx=10, pady=10)
        text.insert("1.0", "\n".join(lines))
        text.configure(state="disabled")

    def refresh(self):
        state = read_json(STATE_FILE)
        checkpoint = read_json(CHECKPOINT_FILE)
        heartbeat = read_json(HEARTBEAT_FILE)
        champion = state.get("champion", {})
        self.card_values["champion"].set(str(champion.get("lb_score", "-")))
        self.card_values["experiment"].set(str(state.get("current_experiment") or "-"))
        self.card_values["hypothesis"].set(str(state.get("current_hypothesis") or "-"))
        self.card_values["level"].set(f"{state.get('research_level', '-')} {state.get('level_name', '')}".strip())
        pid = checkpoint.get("process_id")
        self.card_values["pid"].set(str(pid) if pid else "-")
        hb = heartbeat.get("last_heartbeat") or "-"
        self.card_values["heartbeat"].set(hb.split("T")[-1] if "T" in hb else hb)
        ck_status = checkpoint.get("status", "UNKNOWN")
        self.status_var.set({"RUNNING": "● RUNNING", "COMPLETED": "● WAITING", "FAILED": "● NEEDS RECOVERY", "INTERRUPTED": "● NEEDS RECOVERY"}.get(ck_status, f"● {ck_status}"))

        result_lines = [f"{r.get('experiment_id', '-')} | {r.get('hypothesis_id', '-')} | {r.get('decision', '-')} | prod={r.get('production_result', '-')}" for r in last_experiments(8)]
        log_lines = tail_log(18)

        self.results_text.configure(state="normal")
        self.results_text.delete("1.0", "end")
        self.results_text.insert("1.0", "\n".join(result_lines) if result_lines else "No experiment log yet.")
        self.results_text.configure(state="disabled")

        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.insert("1.0", "\n".join(log_lines) if log_lines else "No manager log yet.")
        self.log_text.configure(state="disabled")
        self.footer.set(f"Experiments logged: {count_log()} | Checkpoint: {ck_status}")
        self.after(POLL_MS, self.refresh)

    def on_close(self):
        if self.manager_process is not None and self.manager_process.poll() is None:
            if not messagebox.askyesno("종료", "GUI만 종료할까요? Research Manager는 계속 실행됩니다."):
                return
        self.destroy()


if __name__ == "__main__":
    ResearchMachine().mainloop()