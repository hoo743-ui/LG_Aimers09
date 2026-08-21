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

## 지금 상태 (2026-08-21)

| | |
|---|---|
| 최고 실측 LB | 🚩 **1075.4602** (41회차, `cand_h1.zip`) |
| 구성 | 82피처 + 후처리 9가중 `w=[0.095671, 0.394645, 0.13394, 0.215261, 0.65, 0.65, 0.65, 2.0, 2.10]` |
| 이력 | 1071.8146(36) → 1073.8237(39) → 1074.8798(40) → **1075.4602(41)** |
| 제출 예산 | **하루 5회** (2026-08-21 은 1장 사용, 4장 남음) |
| 대기 후보 | `cand_rob1.zip` — EXP044 강건최적, 앙상블 78 최악 증분 +0.502 |

### 🚩 43회차가 확정한 것 — **미측정 방향으로 외삽하지 않는다**

`cand_opt1`(EXP042 해석적 최적) = **1074.0991**. 예상 1076.76, 어긋남 −2.66.

```
LB 11점은 전부 5차원 블록 부분공간 안에 있었다
  측정된 방향  편차 전역배수 · c_hand · (c_2S+c_runner) · L_pitcher · L_batter
  미측정       편차 내부 3 dof + 2S/주자 분리 1 dof = **4 dof**
opt1 은 정확히 그 4 dof 로 갔고 g 의 **블록 내부 모양**은 로컬에서 빌린 값이었다
```

12점 재적합에서 블록을 **어느 것이든** 하나 쪼개면 RMS 가 전부 0.2887 로 같다 —
관측 하나로는 어느 dof 가 틀렸는지 **원리적으로 구별되지 않는다.**

### 🚩 후처리 축 상관 구조 (EXP043 실측)

```
dev0_platoon x c_hand     0.547      <- 같은 정보의 두 표현 (투수 x 타자손)
dev3_runner  x c_runner   0.510      <- 〃 (투수 x 주자유무)
dev2_count   x c_2strike  0.271
dev1_adv     x c_2strike  0.154
나머지 31 쌍               |r| < 0.04
```

`c_hand`·`c_2strike`·`c_runner`·`L_pitcher`·`L_batter` 다섯은 서로 거의 직교다.
**그 다섯 사이의 좌표 조정은 안전하고, 편차-대비 짝을 반대로 미는 것이 위험하다.**

### 🚩 현행 판정 기준 (2026-08-21, EXP044)

```
후보의 점수는 단일 적합의 점추정이 아니라
  min_e [ S_e(w) - S_e(h1) ]     e = 블록구조 6 x LOO 13 = 78 개 적합
로 판정한다. **최악 증분이 양수가 아니면 제출하지 않는다.**
증분은 반드시 적합마다 짝지어 뺀다 (min 끼리 빼면 최악값이 아니다)
편차 4축의 내부 모양은 h1 고정 — 전역 배수만 자유
```

이 기준을 어제 적용했으면 opt1(최악 −4.35)을 막았다.

### 라운드 루프 (세 명령)

```powershell
.\.venv\Scripts\python.exe expdd_obs.py --zip submissions\cand_rob1.zip --lb <점수> --round 44
.\.venv\Scripts\python.exe -u research\exp044_robust.py
.env_submit\Scripts\python.exe -u expeweight.py --src submissions\cand_h1.zip --out <이름> --w <9개> --tag "..."
```

**제출 전 관문 4개 (예외 없음)**

```powershell
.\.venv\Scripts\python.exe -u exp\compliance_audit.py --zip submissions\<후보>.zip
.\.venv\Scripts\python.exe -u expowindep.py       --zip submissions\<후보>.zip   # 0.000e+00
$env:PYTHONIOENCODING="utf-8"; .\.venv\Scripts\python.exe -u verify_submit.py --zip submissions\<후보>.zip
# + TYPE 판정 — 가중 조정만이면 안전 등급. 표/모델/아핀을 건드리면 TYPE B
```

> `verify_submit.py` 는 `PYTHONIOENCODING=utf-8` 없이 돌리면 cp949 로 죽는다.

### 🚩 지금 유효한 전이 위계 (38회차로 확정, 43회차로 보강)

```
안전  이미 LB 로 **측정된 방향**의 가중 조정      39/40/41 전부 양수
위험  측정된 적 없는 방향으로의 외삽             43 opt1 -2.66, 41 b차용 -0.42
위험  새 후처리 축                              37 볼우위 -7.07
금지  표 제작 방식 (원천 창 · 감쇠 · k · 평활)    30 -9.95 / 38 -4.93
금지  모델 적합 절차                             32 단조 -65.73
```

### 후보 판정은 5-b 로 (2026-08-18)

```
1  생산 2024 < 0        -> 제출 금지
2  0 ~ +3.8%           -> 승격 금지, LB 사용 금지 (기록만)
3  >= +3.8%            -> 시드 4~6 -> 생산 검증 -> LB 후보
4  새 family           -> 의사미래 전이 부호를 먼저 (기각용)
```

단일 전이 배수는 폐기됐다. family 전이 조회표는 `CLAUDE.md` 5-b.

**08-16~17 에 닫은 축** — `game_type` 양방향, 조건 변수 감사 전체, PN 투구수
역산, prev 창 분해, 경로 배분 12조합, TMX/TMR, 시드 3→7, 아핀/center,
그리고 **현행 TrackMan 8열 제거**(2024 −11.7 — 죽은 무게가 아니었다).
자세한 것은 `PROGRESS.md` 최신 절과 `DEVIATION_LEDGER.md` 끝.

### ⚠️ 전이식 — **폐기됨 (2026-08-18).** 아래는 이력이다

> 2026-08-17 전이 연구에서 단일 배수가 부분집합에 따라 `b=+0.145 ~ +0.512` 로
> 3배 흔들리는 것이 확인됐다. **점추정에 쓰지 말 것.** 현행 규칙은 위 5-b
> 게이트(음수 금지 / 3.8% 미만 승격 금지)다. 이 표는 앵커 기록으로만 남긴다.

```
LB 이득 = 0.830 x (생산 경로 2024 홀드아웃 이득 - 2.07%)      잔차 RMS 2.9점
```

| 축 | 생산 2024 | LB 이득 | 회차 |
|---|---:|---:|---|
| D 현재상태 | +12.87% | +8.97% | 22 |
| F 최근vs시즌 | +1.36% | −0.82% | 23 |
| X D×맥락 | +2.78% | +0.375% | 24 |
| H1 수준확장 | +2.12% | +0.494% | 25 |

**X 와 H1 이 역전한다** (로컬이 작은 쪽이 LB 는 크다). 단조 모형이 아니고,
예측 오차가 ±3점이다.

**후보 판정 (옛 규칙 — 5-b 로 대체됨)**
```
로컬 2024 < 2.1%     기대값 음수 — 다른 강한 근거 없으면 버린다
2.1 ~ 3%             산포가 ±3점이라 3/3 이고 worst 양수면 낼 만하다
> 3%                 명확한 후보
```
> 현행 문턱은 **+3.8%** 다 (전이 연구의 2시그마 요구). 위 2.1% 는 옛 값이다.

> 25회차에서 나는 2.12% 를 "손익분기 아래"로만 읽고 보류를 권고했다가 +5.16 을
> 놓칠 뻔했다. **±3점 산포 모형에서 −2점 예측은 기각 근거가 못 된다.**

### 1100 까지

```
1049.9226 -> 1100   재적합 계수로 로컬 2024 이득 8.1% 필요
   D 12.87% / X 2.78% / H1 2.12%
```

D×맥락 계열은 포화했다 (§22-i — H3 12열이 2024 에서 H1 6열보다 낮다).

### 22회차가 무엇이었나 (§17) — 현 챔피언

`asof_*` 가 시즌 리셋이 아니라 **통산 누적**이라는 사실에서 출발한다. 학습
데이터에서 그 선수의 통산을 빼면 **2025 시즌 현재 상태**가 대수적으로 복원된다.

```
cur_n    = asof_n(행) - prior_n[선수]
cur_rate = ((asof_n * asof_rate)(행) - prior_events[선수]) / cur_n
```

모델이 보는 통산 rate 에는 이력과 현재 폼이 섞여 있고 모델은 그것을 **원리적으로
못 가른다.** 이 분해가 갈라 준다. 2024 폴드 배수 1.1287 중 **78~87% 가 평가셋으로
전이**됐다. **아핀도 클린이라 규정 4) 회색지대 밖이다.**

### 23회차가 무엇이었나 (§18~19) — F 는 전이가 음수였다

`prev{k}_game_rate − cur_rate`(F). 로컬 2024 +1.36%, **평가셋 −0.82% (전이 −60%)**.
아핀이 아니라 전부 `rho` 였다 (`alpha` 동일, `center` 차이는 절편에 0.1점 미만).

### 24회차 (§20~21) — D×맥락, **1044.7656 로 성공**

```
dx_{succ,mid}_{adv,onb,sh,bs} = cur_{succ,mid} x {카운트우위, 주자유무, 같은손, 볼-스트라이크}
```

트리는 축평행 분할이라 두 열의 곱을 스스로 못 만든다. 재료는 전부 그 행 자신의
값이라 규칙 4)에 안전하다. 생산 경로 2024 홀드아웃 899.4 → 924.4 (배수 1.0278),
아핀 `alpha` 1.090000 `center` 0.584040 (클린).

`min gain` 1.0049 로 §19-b 게이트에 미달했지만 **손익분기(2.34%)는 넘었고**,
법칙이 예측한 +0.375% 가 정확히 실현됐다.

### 🚩 판정 규칙 (2026-08-15 갱신) — 크기가 게이트다

| 축 | min gain | 로컬 2024 | 평가셋 전이 |
|---|---:|---:|---:|
| CAAFE | 1.0001 | +2.56% | 0% |
| **F** | **1.0026** | +1.36% | **−60%** |
| D | 1.0322 | +12.87% | +78~87% |

**축 하나가 슬롯 하나를 받으려면 `min gain` ≥ 1.01 이고 시드 노이즈의 5배
이상이어야 한다.** F 는 마진이 노이즈의 2배였고 그 사실을 게이트 E 에서 이미
손에 쥐고도 "부호가 두 시드쌍에서 유지된다"를 통과 근거로 삼았다. 공통 성분이
큰 두 측정에서 부호 유지는 **약한 증거**다.

### 제출본

| 파일 | 상태 |
|---|---|
| `submissions/cand_asof.zip` | ✅ 22회차 — **1040.8656** (최고) |
| `submissions/cand_asof_f.zip` | ✅ 23회차 — 1032.3182 |
| `submissions/cand_nest_clean.zip` | 미제출. 954.3 기대 — 열위 |
| `submissions/cand_caafe*.zip` | 제출 완료 (952.4983 / 924.3786) |

### 아핀은 건드리지 말 것 (§17-i)

적률을 풀면 최적이 1041.1~1049.5 이고 현 손실이 0.2~8.6 뿐이다. `alpha` 1.09 가
최적 구간(1.020~1.160) 한가운데에 있다.

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
