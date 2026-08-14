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

```powershell
.\.venv\Scripts\python.exe exp\prep.py
```

만들어지는 것: `exp/cache/X.npy` (1,475,092 x 123, float32, 726MB),
`y.npy`, `season.npy`, `cols.json`.

## 5. 동작 확인

```powershell
.\.venv\Scripts\python.exe verify_submit.py --zip submissions\cand_asof.zip
```

`점검 통과 — 1,2,3,4` 가 나오면 환경이 맞다.

---

## 지금 상태 (2026-08-14)

| | |
|---|---|
| 최고 실측 LB | **955.2193** (19회차) |
| 남은 제출 | **3장** |
| 미제출 후보 | 아래 3종 (전부 verify 4단계 통과) |

### 제출 후보

| 파일 | 기대 | 규정 4) | 성격 |
|---|---|---|---|
| `submissions/cand_asof.zip` | **986~1078** | **완전 회피** | AS-OF 현재상태 분해 (§17) |
| `submissions/cand_nest_clean.zip` | 954.3 | 완전 회피 | 챔피언 + 클린 아핀 (§16) |
| `submissions/cand_caafe.zip` | *(제출됨 952.4983)* | 회색지대 | — |

**권고 순서: `cand_asof.zip` 먼저.** 하방이 986 이라 챔피언(955)을 넘고, 규정
회색지대를 벗어나며, 실측 하나로 AS-OF 축의 전이율이 확정된다.

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
