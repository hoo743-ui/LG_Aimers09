# 다른 PC 에서 이어받기

이 저장소는 **코드·기록·제출본만** 담는다. 데이터(700MB)·가상환경·실험 캐시는
`.gitignore` 로 빠져 있으므로 아래 순서로 복원한다.

---

## 1. 클론

```powershell
git clone https://github.com/hoo743-ui/LG_Aimers09.git open
cd open
```

## 2. 대회 데이터 내려받기 (필수, 저장소에 없음)

데이콘 대회 페이지에서 받아 `data/` 에 그대로 둔다.

```
data/train.csv                 368 MB
data/trackman_history.csv      354 MB
data/test.csv                  (샘플 5행. 실제 245,789행은 평가 서버에만 있다)
data/sample_submission.csv
```

> `test.csv` 가 5행뿐인 것은 정상이다. 코드 제출 대회라 평가셋은 서버에만 있다.

## 3. 가상환경 3개

셋 다 필요하다. **섞어 쓰면 사고가 난다** (`DEVIATION_LEDGER.md` §13-e — `.venv`
의 sklearn 1.6.1 로 만든 pkl 을 제출 환경 1.7.2 가 못 읽어 검증 4단계에서
걸렸다. 로컬 추론은 멀쩡히 돌았으므로 검증이 없었으면 슬롯을 날렸다).

| 환경 | 용도 | 핀 |
|---|---|---|
| `venv_submit` | **제출본 빌드 전용** | `requirements_cat.txt` |
| `oldnp` | 낮은 numpy 로드 검사 (verify 4단계) | numpy 2.1.3 |
| `.venv` | 실험용 (torch·tabpfn 등 자유) | 제약 없음 |

```powershell
py -3.13 -m venv venv_submit
.\venv_submit\Scripts\python.exe -m pip install -r requirements_cat.txt

py -3.13 -m venv oldnp
.\oldnp\Scripts\python.exe -m pip install scikit-learn==1.7.2 joblib==1.5.3 pandas==2.3.3 numpy==2.1.3 catboost==1.2.10

py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements_cat.txt scipy
```

실험을 이어서 하려면 `.venv` 에 추가로:

```powershell
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install "tabpfn==2.0.9" psutil pypdf
```

> `tabpfn` 은 8.x 를 쓰지 말 것 — 가중치 다운로드에 브라우저 로그인과 라이선스
> 동의를 요구해 규정 1)("누구에게나 공개")과 부딪힐 소지가 있다 (§10).

## 4. 실험 캐시 재생성

`exp/cache/` 는 `.gitignore` 대상이다. 한 번 만들면 이후 실험이 파싱 없이 돈다.

**두 단계다.** `prep.py` 만 돌리면 `cols.json` 에 `prod` 키가 없어서
`asof_state*.py` · `caafe*.py` · `mlp_*.py` 등 대부분의 실험이 `KeyError` 로 죽는다.

```powershell
.\.venv\Scripts\python.exe exp\prep.py         # X 57컬럼
.\.venv\Scripts\python.exe exp\prep_tmx.py     # +6컬럼, prod 55 고정
```

만들어지는 것: `exp/cache/X.npy` (1,475,092 x 63, float32, 372MB),
`y.npy`, `season.npy`, `cols.json`.

`prep_tmx.py` 는 **덧붙이는** 스크립트라 두 번 돌리면 컬럼이 중복된다. 다시
만들려면 `exp/cache/` 를 지우고 처음부터 돌릴 것.

## 5. 동작 확인

```powershell
.\.venv\Scripts\python.exe verify_submit.py --zip submissions\cand_asof.zip
```

`점검 통과 — 1,2,3,4` 가 나오면 환경이 맞다.

---

## 지금 상태 (2026-08-15)

| | |
|---|---|
| 최고 실측 LB | **1040.8656** (22회차, `cand_asof.zip`) |
| 대기 후보 | 🚩 `cand_asof_f.zip` (23회차, 빌드·검증 완료 — 아래) |
| 남은 제출 | **2장** |

### 22회차가 무엇이었나 (§17) — 현 챔피언

`asof_*` 가 시즌 리셋이 아니라 **통산 누적**이라는 사실에서 출발한다. 학습
데이터에서 그 선수의 통산을 빼면 **2025 시즌 현재 상태**가 대수적으로 복원된다.

```
cur_n    = asof_n(행) - prior_n[선수]
cur_rate = ((asof_n * asof_rate)(행) - prior_events[선수]) / cur_n
```

모델이 보는 통산 rate 에는 이력과 현재 폼이 섞여 있고 모델은 그것을 **원리적으로
못 가른다.** 이 분해가 갈라 준다. 2024 폴드 배수 1.1287 중 **78~87% 가 평가셋으로
전이**됐다 (CAAFE 는 0% 였다). **아핀도 클린이라 규정 4) 회색지대 밖이다.**

### 제출본

| 파일 | 상태 |
|---|---|
| `submissions/cand_asof.zip` | ✅ 22회차 — **1040.8656** (현 챔피언) |
| 🚩 `submissions/cand_asof_f.zip` | **미제출.** 23회차 후보 (아래) |
| `submissions/cand_nest_clean.zip` | 미제출. 954.3 기대 — 열위라 쓸 이유 없음 |
| `submissions/cand_caafe*.zip` | 제출 완료 (952.4983 / 924.3786) |

### 23회차 후보 — AS-OF 축 확장 F (§18)

SETUP 이 남긴 세 갈래를 `exp/asof_state2.py` 로 한 번에 쟀고 **F 만 살았다.**

| 갈래 | vs D min | 판정 |
|---|---:|---|
| **F 최근 경기 vs 시즌내 누적** | **1.0026** (3/3) | 🚩 채택 |
| E `cur − prior` | 0.9679 | 기각 — D 가 이미 `cur` 을 갖고 `prior` 는 §17-d 에서 이미 0 |
| B 타자 확장 (격차 포함) | 0.9713 | 기각 — §5-c 와 같은 자리 |

**판정 분모를 A(55p)가 아니라 D 로 바꾼 것이 핵심이다.** D 는 이미 22회차
제출본에 들어가 있으므로 질문은 "기준보다 나은가"가 아니라 "D 에 더해서 오르는가"다.

F 가 CAAFE 와 갈리는 지점 — CAAFE 는 `prev1 − 통산rate` 로 **척도가 다른 두 값을
뺐고** 평가셋 전이가 0 이었다(§15). 이제 `cur_*`(시즌내 누적)이 있으니 같은 창
안에서 뺀다: "최근 k 경기가 **이번 시즌 자기 평균보다** 좋은가".

```
form_succ{1,3,5} = asof_pitcher_prev{k}_game_success_rate − cur_succ
form_mid{1,3,5}  = asof_pitcher_prev{k}_game_middle_rate  − cur_mid
```

빌드·검증 (`exp/build_asof.py --form --build`, `venv_submit`):

```
피처       74 = 기본47 + ctx8 + AS-OF 13(D) + form 6(F)
2024 홀드아웃  대조(D) 899.4 -> 신규 911.6   배수 1.0136
              (대조 899.4 는 §17-d 생산 경로 값과 정확히 일치 — 재현됨)
아핀       alpha 1.090000  center 0.578117   <- 클린 (평가셋 정보 미사용)
verify     4단계 통과, 추론 6.7초/600초
기대 LB    1040.8656 x 1.0136 = 1055.0,  전이 78~87% 가정 1051.9~1053.2
하한       min gain 기준 1043.6
```

시드 안정성(게이트 E) — 마진이 노이즈의 2배라 독립 시드쌍으로 재측정했다.
(42,43) 에서 +6.4, (44,45) 에서 +4.4 로 **부호가 유지된다** (D 수준 자체는
시드쌍 사이에서 +3.3 움직인다).

### 다음에 할 것

1. **23회차 제출 판단** — `cand_asof_f.zip`. 제출하면 슬롯 1장 남는다.
2. **아핀은 여전히 건드리지 말 것** (§17-i) — 최적 구간 1.020~1.160 한가운데다.
3. AS-OF 축에서 남은 것은 F 의 변형뿐인데, **폴드를 보고 변형을 고르면 CAAFE 와
   같은 함정**이다. 사전 등록한 형태를 그대로 쓴다.

### 검증 시 주의 — 3단계 확률 범위가 좁게 나오는 것은 정상이다

`cand_asof*.zip` 은 3단계에서 0.36~0.56 처럼 좁게 나온다. 3단계가 2024 를 test
대역으로 쓰는데 번들 `asof_prior` 는 train 전체에서 만들어져 그 구간에서
`cur_n`=0 으로 클립되기 때문이다 (§17-h). **진짜 경로는 실제 `test.csv` 로 값을
찍어 확인한다.**

### 읽을 순서

1. **`DEVIATION_LEDGER.md`** — 전체 실험 기록. 최신 발견은 §15~§17.
   - §15 CAAFE 전이 0, 3/3 게이트가 뚫린 건
   - §16 클린 아핀 (회색지대의 실제 가격은 28 이 아니라 1)
   - §17 AS-OF 분해 (오늘 유일하게 크기가 다른 축)
2. **`LB_LEDGER.csv`** — 제출 21회 전부의 예측·실측·오차
3. `STATUS.md` — 08-12 기준이라 **오래됐다**. 위 둘을 우선한다
4. `BEST_CONFIG.json` — 모델 설정과 닫힌 축 목록

### 판정 규칙 (오늘 갱신)

`min(폴드별 배수)` 를 **반드시** 함께 본다. CAAFE 가 3/3 을 통과하고도 평가셋
이득이 0 이었던 이유가 `min = 1.0001` 이었기 때문이다 (§15-c). 부호 개수는
크기를 무시하고, 기하평균은 퇴화 폴드(2023, `rho^2` 76.5)에 지배된다.

### 함정 두 개

- **`verify_submit.py` 3단계는 2024 를 test 대역으로 쓴다.** "시즌 내 상태"를
  쓰는 피처는 거기서 죽으므로 검증되지 않는다. 실제 `test.csv` 로 값을 찍어
  따로 확인할 것 (§17-h).
- **제출본은 반드시 `venv_submit` 으로 빌드할 것.** §13-e 참조.
