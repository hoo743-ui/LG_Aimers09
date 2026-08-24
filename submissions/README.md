# submissions/ 보존 규칙 (2026-08-24 정리)

356 MB / 102개 -> **25 MB / 9개**. 지운 93개는 **git 이력에 그대로 있다.**

## 보존하는 것

| 파일 | 이유 |
|---|---|
| `cand_kb45.zip` | 🚩 **현 Champion 1090.0094882798** (55회차). 절대 덮어쓰지 않는다 |
| `cand_mir.zip` | 48회차 Champion. `exp/build_lvlk.py` 의 `BASE_ZIP` **코드 의존** |
| `cand_rob2.zip` | 45회차 Champion. `exp/build_aux*.py` 의 기반 |
| `260825_1_kp6.zip` · `260825_2_kp3.zip` | 8/25 제출 (투수 수준축 모양 t=−6 / −3) |
| `260825_4_kbw06.zip` · `260825_5_kbw14.zip` | 8/25 제출 (타자 수준축 크기 w x0.6 / x1.4) |
| `cand_kp1.zip` | 예비 (t=−1). 슬롯이 정해지면 `git mv` 로 개명한다 |

**규칙 — 남기는 것은 ① 현 Champion ② Champion 계보 중 코드가 참조하는 것
③ 아직 안 낸 후보. 그 외는 지운다.**

## 이름 규칙 (2026-08-24 확정)

```
YYMMDD_순번_의미   260825_1_kp6   8/25 그날 1번째 제출   <- 날짜가 앞이라 ls 정렬 = 제출 순서
cand_의미          cand_kp1       아직 슬롯이 없는 예비본
```

회차 번호는 `LB_LEDGER.csv` 가 갖고 있으니 이름에 중복으로 넣지 않는다. 슬롯이
정해지면 `git mv` 한다 — **rename 은 sha256 을 바꾸지 않으므로** 원장에 적어둔
해시가 그대로 유효하다. 빌더가 `exp/subname.py` 로 강제하고, 55회차까지의 옛
이름(`cand_*`)은 추적 유지를 위해 **바꾸지 않는다.**

## 왜 지워도 되는가

측정값은 파일이 아니라 **원장**에 있다 — `LB_LEDGER.csv` 에 회차·구성·LB 점수·
전이율·해석이 전부 있고, `DEVIATION_LEDGER.md` 에 빌드 방법과 sha256 이 있다.
zip 은 그 기록의 사본일 뿐이고, 필요하면 이력에서 꺼내거나 빌더로 재생성한다.

## 되살리는 법

```powershell
# 하나만
git checkout HEAD -- submissions/cand_bw25.zip

# 정리 직전 상태로 전부 (커밋 전이면)
git checkout HEAD -- submissions/

# 커밋 후라면 정리 커밋의 부모에서
git checkout <정리커밋>^ -- submissions/
```

## 주의

`.git` 은 393 MB 그대로다. **작업 디렉터리 352 MB 만 회수됐고 저장소 크기는
줄지 않는다.** 이력까지 줄이려면 history rewrite 가 필요한데, 제출 아티팩트의
추적 가능성을 잃으므로 **하지 않는다** (규정 검토 대응에 쓸 수 있다).

`model_cand/` 235 MB 는 이번 정리 대상이 아니다. 손대기 전에 `build_w.py` 등이
참조하는 pkl 을 먼저 확인할 것.
