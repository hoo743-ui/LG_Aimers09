# 규정 준수 증거 — 데이콘 공지(2026-08-18) 기준

감사 시각 2026-08-18. 재현: `.\.venv\Scripts\python.exe -u exp\compliance_audit.py`

## 공지가 요구하는 것

> 행 A 의 예측값은 **행 A 의 입력 변수 · 그 변수만으로 만든 파생변수 · 공식 학습
> 데이터 · 학습 데이터만으로 만든 통계/모델/파생변수** 만으로 생성되어야 한다.
>
> 판정 — test.csv 에 그 행 1개만 있을 때와 전체가 함께 있을 때의 예측값이
> **같아야** 한다.

## 1. 구조 감사 — 제출물 37개 전수

zip 안 `script.py` 에서 평가 프레임에 대한 행 간 연산(groupby · rolling ·
expanding · transform · rank · value_counts · cumsum · shift · quantile · sort)을
정규식으로 탐색했다.

```
제출물 37개 · 평가 프레임 대상 행 간 연산 **0건**
```

## 2. 수치 감사 — 핵심 9개, 각 zip **자신의** script.py 로 실행

2024 대역 200행을 배치로 예측한 값과, 같은 행을 **1행짜리 프레임**으로 넣어
다시 예측한 값을 비교했다.

| 제출물 | sha256 | 배치 평균 | 최대 절대오차 |
|---|---|---:|---:|
| cand_submit_1 | 40cdd6ba16 | 0.4749 | **0.00e+00** |
| cand_submit_2 | 3e3951cb9d | 0.4741 | **0.00e+00** |
| cand_submit_3 | 82799a5cd6 | 0.4749 | **0.00e+00** |
| **cand_wd060 (현 Champion)** | **bd798b9a6e** | 0.4749 | **0.00e+00** |
| cand_wd035 | 306e6a495f | 0.4749 | **0.00e+00** |
| cand_dev075 | fd77227b20 | 0.4749 | **0.00e+00** |
| cand_mono10 | 686386bbaa | 0.4896 | **0.00e+00** |
| cand_final | 5152edc733 | 0.4751 | **0.00e+00** |
| cand_asof_xl | 5e7fd0a372 | 0.4749 | **0.00e+00** |

## 3. 우리 추론이 무엇을 쓰는가

```
행 자신의 공식 컬럼        47열 (asof_* 포함 — data_description L182 사용 허가)
행 자신의 파생             D 분해 · X/H1 곱 · TrackMan ctx
학습 데이터만의 상수        asof_prior[선수] · 후처리 편차 4축 표 · 잔차 차등 3축 표
                          (전부 학습 구간에서만 만들어 모델 파일에 담긴 상수)
조회 키                   pitcher_id · pitcher_hand · batter_hand ·
                          strikes_before · num_runners_on  (전부 그 행의 컬럼)
```

**평가셋의 다른 행을 읽는 코드 경로가 존재하지 않는다.**

## 4. 우리가 하지 않은 것 (측정은 했고 제출은 안 했다)

연구 과정에서 **전이적 접근의 값이 +163.6** 임을 측정했다
(`research/exp015_access.py`, 폴드 2024). `asof_pitcher_n` 이 매 투구 1씩 오르는
통산 카운터라, 평가셋에서 한 투수의 `asof_n` 최대 행을 집으면 그 시즌 실현
성공률이 상관 **0.9986** 으로 복원된다 — 라벨 없이도.

**그 경로는 감사 전용이며 어떤 제출물에도 들어가 있지 않다.** 위 1·2 절이 그것을
증명한다. 해당 실험 코드는 `research/` 아래에 있고 `submissions/` 어디에도
포함되지 않는다.


---

## 5. 8/18 이후 제출물 (37~63회차) — 회차별 관문 기록 (2026-08-26 재검사)

위 1·2 절은 8/18 의 37개 시점이다. 그 뒤 낸 것들을 한 표로 모은다.
**"재검사" 열이 있는 행은 2026-08-26 00:30 에 현재 파일로 관문 1·2 를 다시 돌린 결과다**
(`exp/compliance_audit.py` + `exp/rowindep_zip.py --rows 200`, 로그 scratchpad `gate_all.log`).
`submissions/` 에서 지운 zip 은 원장(`DEVIATION_LEDGER.md`)의 당시 통과 기록과
git 추가 커밋(`git show <commit>:submissions/<이름>.zip`) 으로 추적한다. sha256 은 앞 16자.

모든 제출물의 조회 키는 그 행 자신의 컬럼(`pitcher_id · pitcher_hand · batter_hand ·
strikes_before · num_runners_on · batter_id`)뿐이고, 표는 학습 데이터(잔차·출장수)로만
만든 상수다. 39회차 이후는 **모델 재학습 없음**(46·47·51 제외) — 후처리 가중·표 모양만 바뀌었다.

| 회차 | 파일 | sha256 | 파일 소재 | 구조 | 단독 vs 배치 | 근거 |
|---:|---|---|---|---|---:|---|
| 37 | cand_full2 | 8937e4b6503cc2ce | git bbf6a18 | 청정 | 0.00e+00 | 원장 당시 기록 |
| 38 | cand_dcfull | 4028e18cb4f20765 | git 28d1db9 | 청정 | 0.00e+00 | 원장 당시 기록 |
| 39 | cand_dv080 | 03e2274881056742 | git fac5897 | 청정 | 0.00e+00 | 원장 (스칼라만 변경) |
| 40 | cand_dv060 | 124d5741a493e928 | git df1a559 | 청정 | 0.00e+00 | 원장 (스칼라만 변경) |
| 41 | cand_h1 | 4e89063b3b4a58cd | git 9155a4e | 청정 | 0.00e+00 | 원장 (스칼라만 변경) |
| 42 | cand_ch040 | ce4ca6cca70ac600 | git 9155a4e | 청정 | 0.00e+00 | 원장 |
| 43 | cand_opt1 | 86fe2b9d6b7958fd | git d73fed6 | 청정 | 0.00e+00 | 원장 |
| 44 | cand_rob1 | af34bb0f864cd61a | git b96c213 | 청정 | 0.00e+00 | 원장 |
| 45 | cand_rob2 | c9b3c4b3ec4bcfa4 | submissions/ | 청정 | **0.000e+00** | **재검사 8/26** |
| 46 | cand_aux8 | 97c7632d0a302fa2 | git 4952946 | 청정 | 0.00e+00 | 원장 (관문 4개 통과, 재학습) |
| 47 | cand_aux1 | b3b454895ff17713 | git aba7ae6 | 청정 | 0.00e+00 | 원장 (재학습) |
| 48 | cand_mir | fdfb0af1bac1ff67 | submissions/ | 청정 | **0.000e+00** | **재검사 8/26** |
| 49 | cand_mir2 | a12b189302110a2c | git 826d60d | 청정 | 0.00e+00 | 원장 |
| 50 | (47 재기재, 별도 파일 없음) | — | — | — | — | LB_LEDGER 참조 |
| 51 | cand_s15 | 9da14385ff6f7634 | git 7bfc7fb | 청정 | 0.00e+00 | 원장 (시드 15, 재학습) |
| 52 | cand_kbf | a1de87836b0cb2de | git 2e8e1f0 | 청정 | 0.00e+00 | 원장 |
| 53 | cand_kbr | 1cb7fb1ab07cdb9f | git 2e8e1f0 | 청정 | 0.00e+00 | 원장 |
| 54 | cand_kb3 | df68ddda2a5d1731 | git 2e8e1f0 | 청정 | 0.00e+00 | 원장 |
| 55 | cand_kb45 | 8c8c624c3837a8d7 | submissions/ | 청정 | **0.000e+00** | **재검사 8/26** |
| 56 | 260825_1_kp6 | 0d5ca51b02aeccd1 | submissions/ | 청정 | **0.000e+00** | **재검사 8/26** |
| 57 | 260825_2_kp3 | 57cd8f32e6b45db6 | submissions/ | 청정 | **0.000e+00** | **재검사 8/26** |
| 58 | 260825_3_cnt1 | 58ce734d7dea5196 | submissions/ | 청정 | **0.000e+00** | **재검사 8/26** |
| 59 | 260825_4_kbhi | 3bbef6117ee6dbe1 | submissions/ | 청정 | **0.000e+00** | **재검사 8/26** |
| **60** | **260825_5_kbn (Champion)** | **6fb46a03a62b17e2** | submissions/ | 청정 | **0.000e+00** | **재검사 8/26** |
| 61 | 260826_1_kbn08 | 15a39d9e883cc32f | submissions/ | 청정 | **0.000e+00** | **재검사 8/26** |
| 62 | 260826_2_kpn | befe6f125b39fd7f | submissions/ | 청정 | **0.000e+00** | **재검사 8/26** |
| 63 | 260826_3_kpn08 | 490b616104147264 | submissions/ | 청정 | **0.000e+00** | 빌드 직후 8/26 00:50, verify 1~4 |
| 64 | 260826_4_kbln | 859c06f66bc380bd | submissions/ | 청정 | **0.000e+00** | **재검사 8/26** |
| 예비(5번 후보) | cand_kbln08 (log n s=0.8) | 7e40b5236ebac5d3 | submissions/ | 청정 | **0.000e+00** | 빌드 직후 8/26 01:35, verify 1~4 |
| (예정 5) | 260826_5_kblnm (구 cand_kblnm, log n s=−0.4 미러) | 3ae7bc61d6ed6744 | submissions/ | 청정 | **0.000e+00** | 빌드 직후 8/26 01:35, verify 1~4 |
| 예비 | cand_cntm / kbw06 / kbw14 / kp1 | b9ef1bbc… / 3194bb71… / 00dd27d2… / 25beb534… | submissions/ | 청정 | **0.000e+00** | **재검사 8/26** (미제출) |

```
2026-08-26 재검사 요약   현존 zip 15개   구조 청정 15/15   단독 vs 배치 최대 절대오차 0.000e+00  15/15
```

### 앞으로의 규칙

새 제출물을 빌드할 때마다 이 표에 한 줄을 **제출 전에** 추가한다 (sha256 · 구조 · 오차).
행이 없는 zip 은 제출하지 않는다.
