# script.py
import os

import joblib
import pandas as pd

ID_COL = "row_id"
TARGET_COL = "control_success"


# =======================
# 데이터 로드 유틸
# =======================

def load_test(path):
    """평가 데이터(csv) 로드. 한 행이 투구 하나."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    if ID_COL not in df.columns:
        raise ValueError(f"test 데이터에 {ID_COL} 컬럼이 없음: {list(df.columns)[:5]}")
    return df


def load_sample_submission(path):
    """sample_submission.csv 로드 — 제출 파일의 row_id 순서/컬럼 기준."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    if list(df.columns[:2]) != [ID_COL, TARGET_COL]:
        raise ValueError(
            f"sample_submission 컬럼이 ({ID_COL}, {TARGET_COL})이 아님: "
            f"{list(df.columns)}")
    return df


# =======================
# 학습 때 사용한 전처리 (그대로)
# =======================

def build_features(df, bundle):
    """모델 입력 추출.

    모델 파일에 features 목록이 있으면 그대로 골라낸다. 학습 때 일부 컬럼을
    제외했다면 추론에서도 똑같이 빼야 하고, 열이 하나라도 다르면
    ColumnTransformer 가 이름 불일치로 실패하기 때문이다.
    목록이 없으면 예전처럼 row_id 만 뺀다.

    범주형 인코딩(top_bottom, game_type, base_state)과 결측 처리는
    모델 파일 안의 파이프라인이 함께 수행하므로 여기서는 컬럼만 고른다.
    """
    if isinstance(bundle, dict) and bundle.get("features"):
        missing = [c for c in bundle["features"] if c not in df.columns]
        if missing:
            raise ValueError(f"test 데이터에 없는 컬럼: {missing}")
        return df[bundle["features"]]
    return df.drop(columns=[ID_COL])


# =======================
# 예측
# =======================

def predict_proba(bundle, X):
    """제구 성공 확률 예측.

    모델 파일은 두 형식을 허용한다.
      - dict  : {"models": [...], "alpha": float, "center": float}
                여러 시드의 예측을 평균한 뒤 중심값 쪽으로 축소한다.
                축소는 과신을 줄여 Brier 를 낮추기 위한 것이다.
      - 그 외 : 단일 estimator (베이스라인 호환)
    """
    if not isinstance(bundle, dict):
        return bundle.predict_proba(X)[:, 1]

    models = bundle["models"]
    acc = None
    for m in models:
        p = m.predict_proba(X)[:, 1]
        acc = p if acc is None else acc + p
    preds = acc / len(models)

    alpha = float(bundle.get("alpha", 1.0))
    center = float(bundle.get("center", 0.5))
    preds = center + alpha * (preds - center)
    return preds.clip(0.0, 1.0)


# =======================
# 제출 파일 생성 유틸
# =======================

def merge_predictions(sub, ids, preds):
    """sample_submission의 row_id 순서에 맞춰 예측 확률 병합.

    예측에 없는 row_id는 sample_submission의 기존 값(placeholder)을 유지한다.
    """
    pred_map = dict(zip(ids, preds))
    values, n_missing = [], 0
    for rid, cur in zip(sub[ID_COL], sub[TARGET_COL]):
        p = pred_map.get(rid)
        if p is None:
            n_missing += 1
            values.append(cur)
        else:
            values.append(p)
    if n_missing:
        print(f" 경고: 예측이 없어 placeholder를 유지한 row_id {n_missing}건")
    sub[TARGET_COL] = values
    return sub


def save_submission(path, sub):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sub.to_csv(path, index=False, encoding="utf-8")


# =======================
# main
# =======================

def main():
    # ---- 경로 변수 (필요에 따라 수정) ----
    TEST_DIR = "./data"            # test.csv, sample_submission.csv 위치
    MODEL_DIR = "./model"          # rf.pkl 위치
    OUT_DIR = "./output"
    TEST_PATH = os.path.join(TEST_DIR, "test.csv")
    SAMPLE_SUB_PATH = os.path.join(TEST_DIR, "sample_submission.csv")
    MODEL_PATH = os.path.join(MODEL_DIR, "rf.pkl")
    OUT_PATH = os.path.join(OUT_DIR, "submission.csv")

    # ---- 모델 로드 ----
    print("Load model...")
    model = joblib.load(MODEL_PATH)
    if isinstance(model, dict):
        print(f" OK. 앙상블 {len(model['models'])}개 "
              f"alpha={model.get('alpha', 1.0):.4f} "
              f"center={model.get('center', 0.5):.4f}")
    else:
        print(f" OK. n_features={getattr(model, 'n_features_in_', '?')}")

    # ---- 테스트 데이터 로드 ----
    print("Load test data...")
    test = load_test(TEST_PATH)
    sub = load_sample_submission(SAMPLE_SUB_PATH)
    print(f" test={len(test)}  submission={len(sub)}")

    # ---- 전처리 (학습과 동일) ----
    print("Build features...")
    ids = test[ID_COL].tolist()
    X = build_features(test, model)
    print(f" features={X.shape[1]}")

    # ---- 예측 (제구 성공 확률) ----
    print("Inference model...")
    preds = predict_proba(model, X) if len(X) else []
    print(f" preds={len(preds)}")

    # ---- sample_submission 기반 결과 생성 ----
    print("Build submission...")
    sub = merge_predictions(sub, ids, preds)
    save_submission(OUT_PATH, sub)
    print(f"✅ Saved: {OUT_PATH} (rows={len(sub)})")


if __name__ == "__main__":
    main()
