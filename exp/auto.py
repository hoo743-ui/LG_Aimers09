r"""실험 큐를 무인으로 돌리고 판정까지 기록한다.

## 무엇을 하고 무엇을 하지 않는가

**한다** — `plans/queue/*.json` 을 이름 순으로 집어 샤드로 학습하고, 폴드 안
짝비교로 델타·표준오차·부호일치를 계산해 `exp/AUTO_LEDGER.csv` 와
`exp/AUTO_REPORT.md` 에 남긴다. 캐시된 칸은 건너뛰므로 몇 번이고 다시 돌려도 된다.

**하지 않는다** — 베이스라인·제출물·`BEST_CONFIG.json` 을 **자동으로 바꾸지
않는다.** 오늘 lr 축이 실험 경로에서 +5.83(유의)이었는데 제출 경로에서 −2.88 로
뒤집혔다 (4-17). 실험 경로 숫자만으로 채택하면 그때 잘못 채택했을 것이다.
자동화는 **측정과 기록까지**이고, 채택은 제출 경로 확인 뒤 사람이 한다.

## 판정 규칙 (4-6, 4-17 의 규칙 ⑤⑥)

| 판정 | 조건 | 다음 행동 |
|---|---|---|
| `PATH_CHECK` | 평균 > 2×SE, 폴드 부호일치, 평균 ≥ 10 | 제출 경로에서 재라 |
| `WEAK` | 평균 > 2×SE, 부호일치, 평균 < 10 | 경로 편향(±13) 안이다. 보류 |
| `SPLIT` | 부호갈림 | 변동폭만큼 할인. 기각 쪽 |
| `REJECT` | 평균 ≤ 2×SE | 기각 |
| `NEGATIVE` | 평균 < 0 이고 유의 | 닫는다 |

10점 경계는 임의 값이 아니다 — 실험 경로와 제출 경로가 같은 설정에서 13점
어긋나므로(4-17), 그보다 작은 이득은 경로를 건널 수 없다.

    .\.venv\Scripts\python.exe exp\auto.py --queue plans\queue --shards 2
    echo. > exp\AUTO_STOP     # 진행 중인 플랜을 마치고 멈춘다
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

import analyze as A
import runner as R

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPDIR = os.path.join(ROOT, "exp")
PREDS = os.path.join(EXPDIR, "preds")
LEDGER = os.path.join(EXPDIR, "AUTO_LEDGER.csv")
REPORT = os.path.join(EXPDIR, "AUTO_REPORT.md")
STOP = os.path.join(EXPDIR, "AUTO_STOP")

FIELDS = ["ts", "plan", "exp", "tag", "ref", "mean", "se", "n", "pos",
          "same_sign", "fold_means", "verdict", "cells_missing", "note"]
PATH_BIAS = 10.0          # 4-17: 이 밑은 제출 경로를 못 건넌다


def verdict_of(s):
    """analyze.report 의 요약 한 건 -> 판정 문자열."""
    mean, se, n = s["mean"], s["se"], s["n"]
    sig = (se == se) and abs(mean) > 2 * se        # se==se 는 NaN 배제
    if not s["same_sign"]:
        return "SPLIT"
    if mean < 0:
        return "NEGATIVE" if sig else "REJECT"
    if not sig:
        return "REJECT"
    return "PATH_CHECK" if mean >= PATH_BIAS else "WEAK"


def run_plan(path, shards, py):
    """runner 를 샤드로 띄운다. 하나라도 실패하면 (False, 로그) 를 준다."""
    plan = json.load(open(path, encoding="utf-8"))
    exp = plan["exp"]
    procs, logs = [], []
    for k in range(shards):
        log = os.path.join(EXPDIR, f"auto_{exp}_{k}.log")
        logs.append(log)
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        procs.append((subprocess.Popen(
            [py, os.path.join(EXPDIR, "runner.py"), "--plan", path,
             "--shard", str(k), "--nshard", str(shards)],
            stdout=open(log, "w", encoding="utf-8"), stderr=subprocess.STDOUT,
            cwd=ROOT, env=env), log))
    ok = True
    for p, log in procs:
        if p.wait() != 0:
            ok = False
            print(f"  [실패] 샤드 종료코드 {p.returncode} — {log}", flush=True)
    return plan, ok, logs


def summarize(plan):
    """**로그가 아니라 `exp/preds/*.npz` 에서 읽는다.**

    샤드 여러 개가 `experiment_log.jsonl` 에 동시에 append 하면 줄이 찢어진다.
    그러면 짝비교에서 그 칸이 조용히 빠져 n 이 줄고, 판정이 5칸으로 내려간
    것을 아무도 모른다 (exp020 에서 실제로 일어났다). npz 는 키마다 파일이
    하나라 경쟁이 없다. 플랜에서 설정 키를 직접 계산해 그 파일을 읽는다.
    """
    ref = plan.get("_ref", "cat_tuned")
    rows, missing = [], 0
    for cfg in R.expand(plan):
        path = os.path.join(PREDS, f"{R.cfg_key(cfg)}.npz")
        if not os.path.exists(path):
            missing += 1
            continue
        d = np.load(path, allow_pickle=True)
        m = json.loads(str(d["meta"]))
        rows.append({**m, "tag": cfg["tag"], "exp": plan["exp"]})
    if missing:
        print(f"  [경고] 예측 파일 {missing}개 없음 — 그만큼 판정에서 빠진다",
              flush=True)
    if not rows:
        return ref, {}, missing
    return ref, A.report(rows, "score_fixed", ref), missing


def append_ledger(plan_path, plan, ref, summary, missing=0):
    exp = plan["exp"]
    new = not os.path.exists(LEDGER)
    with open(LEDGER, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()
        for tag, s in sorted(summary.items(),
                             key=lambda kv: -kv[1]["mean"]):
            w.writerow({
                "ts": time.strftime("%Y-%m-%d %H:%M"),
                "plan": os.path.basename(plan_path), "exp": exp,
                "tag": tag, "ref": ref,
                "mean": f"{s['mean']:.2f}", "se": f"{s['se']:.2f}",
                "n": s["n"], "pos": f"{s['pos']}/{s['n']}",
                "same_sign": "yes" if s["same_sign"] else "no",
                "fold_means": " ".join(f"{k}:{v:+.1f}"
                                       for k, v in s["fold_means"].items()),
                "verdict": verdict_of(s),
                "cells_missing": missing,
                "note": plan.get("_why", "").replace("\n", " ")[:200],
            })


def write_report():
    """사람이 읽는 요약. 판정이 센 것부터."""
    if not os.path.exists(LEDGER):
        return
    rows = list(csv.DictReader(open(LEDGER, encoding="utf-8")))
    order = {"PATH_CHECK": 0, "WEAK": 1, "SPLIT": 2, "REJECT": 3,
             "NEGATIVE": 4}
    rows.sort(key=lambda r: (order.get(r["verdict"], 9), -float(r["mean"])))
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("# 자동 실험 결과\n\n"
                "`exp/auto.py` 가 쓴다. 판정 규칙은 그 파일 docstring 참고.\n"
                "**PATH_CHECK 만이 다음 행동(제출 경로 확인)을 요구한다.**\n\n")
        f.write("| 판정 | exp | tag | 평균 | SE | n | 부호 | 폴드별 |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| **{r['verdict']}** | {r['exp']} | `{r['tag']}` | "
                    f"{float(r['mean']):+.2f} | {r['se']} | {r['n']} | "
                    f"{r['same_sign']} | {r['fold_means']} |\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default=os.path.join(ROOT, "plans", "queue"))
    ap.add_argument("--shards", type=int, default=2)
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()

    if os.path.exists(STOP):
        os.remove(STOP)
        print("이전 STOP 표시를 지웠다")

    plans = sorted(glob.glob(os.path.join(a.queue, "*.json")))
    if not plans:
        print(f"큐가 비었다: {a.queue}")
        return
    print(f"큐 {len(plans)}개 | 샤드 {a.shards}\n")

    for i, path in enumerate(plans, 1):
        if os.path.exists(STOP):
            print("STOP 표시 발견 — 여기서 멈춘다")
            break
        name = os.path.basename(path)
        print(f"[{i}/{len(plans)}] {name} 시작", flush=True)
        t0 = time.time()
        plan, ok, _ = run_plan(path, a.shards, a.python)
        ref, summary, missing = summarize(plan)
        append_ledger(path, plan, ref, summary, missing)
        write_report()
        mins = (time.time() - t0) / 60
        verdicts = {t: verdict_of(s) for t, s in summary.items()}
        print(f"[{i}/{len(plans)}] {name} 완료 {mins:.1f}분 "
              f"{'(샤드 오류 있음)' if not ok else ''}", flush=True)
        for t, v in verdicts.items():
            s = summary[t]
            print(f"    {v:11s} {t:16s} {s['mean']:+7.2f} SE {s['se']:5.2f} "
                  f"{s['pos']}/{s['n']}+", flush=True)
        done = os.path.join(a.queue, "done")
        os.makedirs(done, exist_ok=True)
        os.replace(path, os.path.join(done, name))

    print(f"\n큐 종료. 요약: {REPORT}")


if __name__ == "__main__":
    main()
