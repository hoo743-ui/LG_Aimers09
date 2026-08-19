# LG Aimers 9기 — Tabular Machine Learning

> **Trackman 데이터를 이용해 투구 단위 제구 성공 확률을 예측한 머신러닝 프로젝트**

**2026.06 — 현재 · 개인 · 진행 중**  
**Current Champion: 1071.8146**

2019–2024년 KBO Trackman 데이터를 학습하고, 학습 구간 밖의 2025년 시즌을 평가 대상으로 사용했습니다.

평가 지표는 `1e5 × corr(prediction, target)^2`인 상관 기반 지표입니다. 따라서 단순한 스케일·절편 보정보다 **예측의 순위 구조 자체를 개선하는 것**이 중요했습니다.

## Result

| Stage | Score |
|---|---:|
| RandomForest baseline | 549.64 |
| CatBoost + Calibration | 942.46 |
| Previous Champion | 1057.34 |
| Pitcher main effect | 1061.50 |
| Batter main effect | 1071.31 |
| **Current Champion** | **1071.8146** |

## Core Analysis

### `asof_*`의 의미를 다시 확인

`asof_*`가 시즌별 누적값이 아니라 **통산 누적값**임을 확인하고 직전 시즌까지의 누적값을 분리해 현재 시즌 상태를 복원했습니다.

```text
cur_n = asof_n - prior_n
```

복원한 `cur_n`은 실제 시즌 내 순번과 **100% 일치**했습니다.

### Validation Audit

`game_type` 하나만으로 일부 fold의 점수를 상당 부분 설명할 수 있음을 확인했습니다. 특히 `game_type=F` 성공률이 **2022년 0.709 → 2023년 0.473**으로 변했고, 양 시즌 공통 투수 92명 중 91명이 같은 방향으로 이동했습니다.

이를 통해 높은 local fold score가 실제 일반화 성능을 의미하는지 별도로 검증했습니다.

### Feature Screening

학습 1회에 15–30분이 걸렸기 때문에 후보 피처와 기존 모델 잔차의 관계를 이용해 이미 회수된 정보와 유사한 후보를 학습 전에 기각했습니다.

이 방법은 **채택 기준이 아니라 기각 기준**으로 사용했습니다.

### Current Champion

투수 주효과와 타자 주효과를 추가해 기준선을 재구성한 뒤, 새로운 기준선에서 **투수 × 볼우위 (`balls > strikes`)** 축을 확인했습니다.

현재 Champion은 **1071.8146**입니다.

## Ongoing Work

현재도 새로운 표현 방식과 정보의 상한을 확인하면서 추가 실험을 진행하고 있습니다. 목표는 단순히 점수를 높이는 것이 아니라 **현재 모델이 이미 설명하고 있는 정보와 아직 남아 있는 정보의 경계**를 찾는 것입니다.

## Experiment Principles

1. 데이터의 의미를 먼저 확인한다.
2. Validation score가 실제 전이를 의미하는지 확인한다.
3. 가설을 세우고 한 번에 하나씩 검증한다.
4. 실패한 실험도 원인을 분해해 기록한다.
5. 새로운 점수보다 재현 가능한 근거를 우선한다.

## Repository Guide

- `DEVIATION_LEDGER.md` — 실험 / 편차 기록
- `LB_LEDGER.csv` — 리더보드 실험 기록
- `PHASE3_PLAYBOOK.md` — 실험 전략
- `PROGRESS.md` — 진행 로그
- `STATUS.md` — 현재 상태
- `SETUP.md` — 실행 환경
- `SUBMISSION_PLAN.md` — 제출 계획

## Tech Stack

`Python` `CatBoost` `scikit-learn` `pandas` `NumPy` `Jupyter`

## Key Takeaway

이 프로젝트에서 가장 크게 배운 것은 특정 모델이 아니라 **데이터 생성 과정과 validation 설계를 이해한 뒤, 실험 결과를 근거로 다음 가설을 정하는 방법**입니다.
