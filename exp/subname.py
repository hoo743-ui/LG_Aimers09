r"""제출물 파일명 규칙 (2026-08-24 확정). 빌더가 전부 이걸 통과해야 파일을 쓴다.

## 형식

    YYMMDD_순번_의미        예) 260825_1_kp6      = 2026-08-25 그날 1번째 제출
    cand_의미               예) cand_kp1          = 아직 슬롯이 없는 예비/보관본

## 왜 이 형식인가

날짜가 앞이라 `ls` 정렬이 곧 제출 순서다. 회차 번호(LB_LEDGER.csv 의 행 번호)는
원장이 이미 갖고 있으므로 이름에 중복으로 넣지 않는다 — 순서가 바뀌면 이름을
고쳐야 하는 형식은 결국 틀린 이름을 남긴다.

옛 이름(`cand_kb45` 등 55회차까지)은 **그대로 둔다.** 원장·커밋 메시지·
COMPLIANCE 기록이 전부 그 이름을 가리키고 있어 바꾸면 추적이 끊긴다.
"""
import re

PAT_SLOT = re.compile(r"^\d{6}_\d{1,2}_[A-Za-z0-9]+$")
PAT_HOLD = re.compile(r"^cand_[A-Za-z0-9_]+$")


def check(name):
    """제출물 이름을 검사한다. 어긋나면 AssertionError."""
    assert not name.lower().endswith(".zip"), f"확장자는 빼고 준다: {name}"
    ok = PAT_SLOT.match(name) or PAT_HOLD.match(name)
    assert ok, (
        f"\n제출물 이름 규칙 위반: {name!r}\n"
        f"  슬롯 배정됨 : YYMMDD_순번_의미   예) 260825_1_kp6\n"
        f"  슬롯 없음   : cand_의미          예) cand_kp1\n"
        f"  (규칙은 CLAUDE.md 8-b · exp/subname.py)")
    if PAT_HOLD.match(name):
        print(f"  ℹ️ '{name}' 는 슬롯 미배정 이름이다. 낼 날이 정해지면 "
              f"git mv 로 YYMMDD_순번_의미 로 바꾼다")
    return name
