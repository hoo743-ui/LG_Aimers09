# script.py
import os

import joblib
import pandas as pd

ID_COL = "row_id"
TARGET_COL = "control_success"

# 이 스크립트는 각 행을 독립적으로 예측한다. 평가셋의 다른 행에서 얻은 통계
# (평균, 빈도, 분포, rolling, target encoding)를 만들지 않는다 —
# data_description.md 5) 평가 데이터 예측 원칙이 이를 금지한다.
# 과거 이 자리에 평가셋 전체의 prev1 평균으로 예측 중심을 옮기는 보정이 있었고,
# 그건 "평가 데이터 전체를 보고 만든 사후 보정값"에 해당해 제거했다.


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

def add_derived(df):
    """학습과 동일한 파생 컬럼. final_train.py 의 같은 이름 함수와 반드시 일치해야 한다.

    same_hand — 투수와 타자의 좌우가 같은가.

    동일 손 매치업은 투수에게 유리하다. 그런데 그 효과는 pitcher_hand 단독으로도
    batter_hand 단독으로도 주변부 평균이 상쇄돼 0 이다.

              좌타   우타
        좌완    +      -
        우완    -      +

    탐욕적 부스팅은 루트에서 주변부 이득으로 분할을 고르므로 이 상호작용을
    영원히 선택하지 않는다. 트리를 1100 그루 돌려도 못 찾는다 — 구조적
    사각지대다. 그래서 컬럼으로 미리 만들어 준다 (3폴드 x 2시드 +24.66).

    양쪽 다 int {1,2} 라 산술로 결정적이다. pd.factorize 를 쓰면 등장 순서로
    코드가 매겨져 train 과 test 에서 값이 갈리므로 절대 쓰지 말 것.

    이 파생은 그 행 자신의 두 컬럼만 본다. 평가셋의 다른 행을 참조하지 않는다.
    """
    df = df.copy()
    df["same_hand"] = (df["pitcher_hand"] == df["batter_hand"]).astype(int)
    return df


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

    각 행은 자기 자신의 피처만으로 예측된다. 평가셋의 다른 행을 참조하는
    연산은 하지 않는다 — 아래 세 단계 어디에도 평가셋 집계가 없다.

    모델 파일은 두 형식을 허용한다.
      - dict  : {"models": [...], "alpha": float, "center": float,
                 "blend": {"model": ..., "weight": float} | None,
                 "center_shift": {"feature": str, "lam": float, "c": float} | None}
      - 그 외 : 단일 estimator (베이스라인 호환)

    순서는 학습 때 측정한 것과 같아야 한다.
      1) 앙상블 평균
      2) 계열 혼합 — 로지스틱을 소량 섞는다. 단독으로는 크게 지지만 실패 방식이
         달라 오차가 상쇄된다
      3) 과신 교정 — center 쪽으로 alpha 배 축소 (alpha 와 center 는 상수)
      4) 중심 보정 — p + lam*(그 행의 anchor - c). lam 과 c 는 학습 시점 상수이고
         anchor 는 그 행 자신의 asof_* 공식 입력이다. **평가셋 전체의 anchor
         평균을 내는 것은 규칙 위반이며, 여기서는 하지 않는다.**
    """
    if not isinstance(bundle, dict):
        return bundle.predict_proba(X)[:, 1]

    models = bundle["models"]
    acc = None
    for m in models:
        p = m.predict_proba(X)[:, 1]
        acc = p if acc is None else acc + p
    preds = acc / len(models)

    blend = bundle.get("blend")
    if blend:
        w = float(blend["weight"])
        preds = (1.0 - w) * preds + w * blend["model"].predict_proba(X)[:, 1]

    alpha = float(bundle.get("alpha", 1.0))
    center = float(bundle.get("center", 0.5))
    preds = center + alpha * (preds - center)

    shift = bundle.get("center_shift")
    if shift:
        col = shift["feature"]
        if col not in X.columns:
            raise ValueError(f"중심 보정 anchor 컬럼이 없다: {col}")
        # 결측은 c 로 메운다 — 그 행에서는 보정이 0 이 된다
        a = X[col].fillna(shift["c"]).to_numpy(dtype=float)
        preds = preds + float(shift["lam"]) * (a - float(shift["c"]))

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
        b, s = model.get("blend"), model.get("center_shift")
        print(f" OK. 앙상블 {len(model['models'])}개 "
              f"alpha={model.get('alpha', 1.0):.4f} "
              f"center={model.get('center', 0.5):.4f} "
              f"features={len(model.get('features', []))}")
        blend_txt = "없음" if not b else "lr w=%.2f" % b["weight"]
        shift_txt = "없음" if not s else "%s lam=%.3f" % (s["feature"], s["lam"])
        print(f"     혼합={blend_txt} | 중심보정={shift_txt}")
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
    # 파생 컬럼은 항상 만든다. 실제로 쓸지는 bundle["features"] 가 정하므로,
    # 파생을 모르는 옛 pkl 이 와도 그냥 무시된다.
    X = build_features(add_derived(test), model)
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
