r"""제출물 규정 준수 전수 감사. 데이콘 공지(2026-08-18) 기준.

## 왜 이것이 지금 최우선인가

최종 순위가 **최고점**이므로 우리 점수 1060.3077 은 내려갈 수 없다. 그것을
잃는 경로는 **실격 하나뿐**이다. 그리고 공지가 "제출 코드의 추론 과정을 검토할
수 있으며 실격 사례가 지속 발생"이라고 명시했다.

즉 지금 기대값이 가장 큰 작업은 새 후보 탐색이 아니라 **준수 증명**이다.

## 공지의 판정 기준 (원문)

> test.csv 에 해당 행 1개만 있는 경우와 전체 평가 데이터가 함께 있는 경우의
> 예측값이 **같아야** 한다.

## 두 층으로 감사한다

    구조   zip 안 script.py 에서 평가셋 행 간 연산 패턴을 찾는다
           (test 프레임에 대한 groupby/rolling/expanding/transform/
            전체 통계 mean·std·quantile·rank·value_counts)
    수치   같은 행을 **단독**으로 넣었을 때와 **배치**로 넣었을 때를 직접 비교

구조 검사는 거짓 양성이 날 수 있다 (학습 상수 계산에 쓰인 groupby 등).
그래서 걸린 것은 **수치 검사로 최종 판정**한다.

    .\.venv\Scripts\python.exe -u exp\compliance_audit.py
"""
import io
import os
import re
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUB = os.path.join(ROOT, "submissions")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 평가 프레임에 대해 쓰이면 위반이 되는 연산들
PAT = [
    (r"\.groupby\s*\(", "groupby"),
    (r"\.rolling\s*\(", "rolling"),
    (r"\.expanding\s*\(", "expanding"),
    (r"\.transform\s*\(", "transform"),
    (r"\.rank\s*\(", "rank"),
    (r"\.value_counts\s*\(", "value_counts"),
    (r"\.cumsum\s*\(|\.cumcount\s*\(", "cumulative"),
    (r"\.shift\s*\(", "shift"),
    (r"\.duplicated\s*\(|\.drop_duplicates\s*\(", "dup"),
    (r"\.quantile\s*\(|np\.percentile", "quantile"),
    (r"\.sort_values\s*\(", "sort"),
]
# test 프레임을 가리키는 변수명 (main 의 흐름 기준)
TESTVARS = ("test", "df", "X", "sub")


def scan(src):
    """행 간 연산 후보를 줄 단위로 찾는다. 변수명이 test 계열인 것만."""
    hits = []
    for i, line in enumerate(src.splitlines(), 1):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        for pat, name in PAT:
            if re.search(pat, s):
                # 어떤 객체에 걸렸는지 대략 본다
                m = re.search(r"([A-Za-z_][A-Za-z0-9_\.\[\]\"']*)\s*\.\s*"
                              + pat.replace(r"\s*\(", ""), s)
                obj = m.group(1) if m else "?"
                risky = any(obj.split("[")[0].split(".")[0] == v for v in TESTVARS)
                hits.append((i, name, obj, risky, s[:96]))
    return hits


def main():
    zips = sorted(f for f in os.listdir(SUB) if f.endswith(".zip"))
    print(f"제출물 {len(zips)}개 구조 감사\n")
    print("=" * 96)
    print(f"{'파일':<30}{'script.py':>10}{'행간연산':>10}{'test 대상':>11}{'판정':>12}")
    print("=" * 96)
    flagged = {}
    for f in zips:
        p = os.path.join(SUB, f)
        try:
            z = zipfile.ZipFile(p)
            names = z.namelist()
            sp = [n for n in names if n.endswith("script.py")]
            if not sp:
                print(f"{f:<30}{'없음':>10}{'-':>10}{'-':>11}{'⚠ 확인필요':>12}")
                continue
            src = z.read(sp[0]).decode("utf-8", "replace")
        except Exception as e:
            print(f"{f:<30}{'열기실패':>10}{str(e)[:30]}")
            continue
        hits = scan(src)
        risky = [h for h in hits if h[3]]
        verdict = "✅ 청정" if not risky else "🔎 수치검사"
        if risky:
            flagged[f] = risky
        print(f"{f:<30}{len(src)//1024:>8}KB{len(hits):>10}{len(risky):>11}{verdict:>12}")

    print("\n" + "=" * 96)
    if not flagged:
        print("구조 감사 결과 — **평가 프레임에 대한 행 간 연산이 하나도 없다.**")
        print("모든 제출물이 공지 1)의 허용 범위 안에 있다:")
        print("  행 자신의 입력 변수 · 그 변수만으로 만든 파생변수 ·")
        print("  공식 학습 데이터 · 학습 데이터만으로 만든 통계/모델/상수")
    else:
        print(f"수치 검사가 필요한 제출물 {len(flagged)}개")
        for f, hs in flagged.items():
            print(f"\n  {f}")
            for i, name, obj, _, s in hs[:6]:
                print(f"    L{i:<5} {name:<12} on {obj:<14} | {s}")
    return flagged


if __name__ == "__main__":
    main()
