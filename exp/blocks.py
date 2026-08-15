r"""파생 블록 디스크 캐시 — `raw -> feature -> model` 반복을 없앤다.

## 왜 필요한가

`d_decomp.py` 는 실행할 때마다 `build_state_lag` 를 **세 번**(lag 0/1/2) 돌리고
등가성 검증용 `build_state` 를 한 번 더 돌린 뒤, `eb_shrunk` 5개와 상황 기준선을
다시 만든다. `--only` 로 구성 하나만 재도 이 앞단이 통째로 붙는다 —
실측 2~4분이고, 후보 하나당 학습이 3폴드 15분인 것에 비하면 20%가 넘는다.

블록은 **후보와 무관하게 값이 같다.** 한 번 만들어 `exp/cache/blocks/` 에 두고
`mmap` 으로 읽으면 시간도 RAM 도 아낀다 (여러 실행이 같은 파일을 공유한다).

## 오염 방지

캐시가 조용히 틀린 값을 주는 것이 가장 위험하다. 그래서 파일명에 **정의 태그**를
박는다. 정의가 바뀌면 태그를 바꿔야 하고, 태그가 다르면 파일이 달라서 자동으로
다시 만들어진다. 태그를 안 바꾸고 정의만 바꾸는 실수를 막으려면
`--fresh` 로 강제 재생성한다.

    from blocks import Blocks
    B = Blocks()
    D = B.get("D", "asof13_v1", lambda: build_D())

## 스레드/메모리

`threads()` 는 현재 여유 RAM 을 보고 학습 스레드 수를 정한다. 모델 정의는
바꾸지 않는다 — CatBoost 의 `thread_count` 는 결과가 아니라 속도만 바꾼다.
"""
import io
import json
import os
import subprocess
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BDIR = os.path.join(ROOT, "exp", "cache", "blocks")


def free_gb():
    """여유 물리 메모리(GB). psutil 없이 wmic/PowerShell 로 읽는다."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"],
            capture_output=True, text=True, timeout=20).stdout.strip()
        return float(out) / 1024 / 1024
    except Exception:
        return 4.0                      # 못 읽으면 보수적으로


def threads(cap=12):
    """여유 RAM 에 맞춘 학습 스레드 수. 8번 지시의 구간을 그대로 쓴다."""
    g = free_gb()
    if g >= 4.0:
        return cap
    if g >= 2.0:
        return max(cap // 2, 2)
    return 0                            # 0 = 새 학습을 시작하면 안 된다


class Blocks:
    def __init__(self, fresh=False, quiet=False):
        os.makedirs(BDIR, exist_ok=True)
        self.fresh = fresh
        self.quiet = quiet
        self.mem = {}
        self.stats = {}

    def path(self, name, tag):
        return os.path.join(BDIR, f"{name}__{tag}.npy")

    def get(self, name, tag, builder):
        """`name` 블록을 준다. 없으면 만들어 저장하고, 있으면 mmap 으로 연다.

        `tag` 는 **정의의 지문**이다. 정의를 바꾸면 반드시 같이 바꾼다.
        """
        key = (name, tag)
        if key in self.mem:
            return self.mem[key]
        p = self.path(name, tag)
        if os.path.exists(p) and not self.fresh:
            a = np.load(p, mmap_mode="r")
            self.stats[name] = ("cache", 0.0, a.shape)
            if not self.quiet:
                print(f"  [블록] {name:<12} 캐시 {a.shape}", flush=True)
        else:
            t = time.time()
            a = np.ascontiguousarray(builder())
            np.save(p, a)
            dt = time.time() - t
            self.stats[name] = ("build", dt, a.shape)
            if not self.quiet:
                print(f"  [블록] {name:<12} 생성 {a.shape} {dt:.0f}s", flush=True)
            a = np.load(p, mmap_mode="r")
        self.mem[key] = a
        return a

    def report(self):
        built = sum(v[1] for v in self.stats.values())
        hit = sum(1 for v in self.stats.values() if v[0] == "cache")
        return (f"블록 {len(self.stats)}개 (캐시 적중 {hit}), "
                f"생성 시간 {built:.0f}s, 여유 RAM {free_gb():.1f}GB")


def log_run(path, row):
    """실험 한 건의 자원 사용을 한 줄로 남긴다 (12번 지시)."""
    row = dict(row)
    line = json.dumps(row, ensure_ascii=False)
    with io.open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
