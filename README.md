# LG Aimers 9기 — Tabular Machine Learning

> Trackman 투구 데이터를 활용해 **투구 단위 제구 성공 확률을 예측**한 머신러닝 프로젝트입니다.

## Overview

2019~2024년 Trackman 데이터를 기반으로 투구가 이루어지기 전에 확인할 수 있는 정보를 활용해 `control_success` 확률을 예측했습니다.

핵심은 **데이터 leakage를 방지하면서 연도·시즌 변화에도 일반화되는 모델을 만드는 것**이었습니다.

## My Work

- EDA 및 연도별 데이터 분포 분석
- 데이터 전처리 및 Feature Engineering
- Leakage 가능 변수 검토
- 모델 및 설정별 반복 실험
- OOF / validation 기반 성능 검증
- 실험 결과 기록 및 가설 수립
- 추론 비용과 성능을 함께 고려한 모델 선택

## Problem Definition

```text
Trackman data
     ↓
EDA / Validation
     ↓
Feature Engineering
     ↓
Leakage Check
     ↓
Model Training
     ↓
OOF / Validation
     ↓
Probability Prediction
```

약 245,789개 샘플을 제한된 시간 내 추론해야 했기 때문에 정확도뿐 아니라 **추론 비용과 안정적인 일반화**도 함께 고려했습니다.

## Engineering / Analysis Focus

### Leakage Prevention

예측 시점 이후에 알 수 있는 정보를 학습에 사용하지 않도록 변수의 생성 시점과 의미를 확인했습니다.

### Feature Engineering

투수·타자 특성, 볼카운트, 주자 상황, 과거 투구 이력 등 예측 시점에 이용 가능한 정보를 조합해 피처를 설계했습니다.

### Experiment-driven Development

한 번에 여러 요소를 변경하기보다 실험 결과를 기록하고 다음 가설을 설정하는 방식으로 모델을 개선했습니다.

## Repository Guide

실험 기록이 많기 때문에 README에는 프로젝트의 핵심만 남기고 세부 실험 자료는 다음 파일에서 확인할 수 있습니다.

- `DEVIATION_LEDGER.md` — 실험 / 편차 기록
- `LB_LEDGER.csv` — 리더보드 실험 기록
- `PHASE3_PLAYBOOK.md` — 실험 전략
- `SETUP.md` — 실행 환경
- `STATUS.md` — 프로젝트 상태
- `SUBMISSION_PLAN.md` — 제출 계획

## Tech Stack

Python · pandas · NumPy · PyTorch · scikit-learn · Jupyter Notebook

## Key Takeaway

이 프로젝트를 통해 모델 구조 자체보다 **데이터의 생성 과정과 검증 설계가 일반화 성능에 더 큰 영향을 줄 수 있다는 점**을 경험했습니다.
