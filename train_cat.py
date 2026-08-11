r"""CatBoost 제출 후보 학습. 피처 정의는 final_train.py 를 그대로 쓴다.

## 왜 CatBoost 인가 (exp006/exp008)

3폴드 x 2시드에서 HGB 대비 **평균 +64.84, 표준오차 5.72, 6/6 양수**다.
폴드별로 2021 +55.9 / 2022 +79.9 / 2024 +58.8 로 부호가 완전히 일치하고,
시드 안정성도 압도적이다 (fold2024 두 시드 차이 HGB 12.49 vs Cat **0.08**).

HGB 와 섞는 것은 **기각**했다. 예측 상관은 0.9688 로 다양성 자체는 있지만,
세 폴드 모두 순수 CatBoost 가 가장 높았고 워크포워드 가중치 최적화도
hgb=0.10 / cat=0.90 으로 밀었다 (규칙 7·8: 재보고 결정했다).

## 배포 위험 — 반드시 읽을 것

`requirements.txt` 에 catboost 를 추가해야 한다. 이 프로젝트에서 실패한
제출 2건이 전부 **환경 문제**였다 (6-1 zip 경로, 6-4 numpy 핀). 새 의존성은
같은 종류의 위험이다. 제출 전에 `verify_submit.py` 4단계를 반드시 통과시킬 것.

시드는 3개면 충분하다 — 위 측정에서 시드 분산이 HGB 의 1/10 이하였다.

    .\venv_submit\Scripts\python.exe train_cat.py --seeds 3 --save
"""
import argparse
import importlib.util
import os
import time

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

spec = importlib.util.spec_from_file_location("ft", "final_train.py")
ft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ft)

DATA_DIR = ft.DATA_DIR
TARGET = ft.TARGET
CAT_COLS = ft.CAT_COLS
VAL_SEASON = ft.VAL_SEASON

# 최근 형태 편차 (EXP019). prev{1,3,5} 게임 비율에서 통산 비율을 뺀다.
# 같은 행 안의 뺄셈이라 5) 원칙에 안전하고, 두 컬럼 모두 test.csv 의 공식
# asof_* 컬럼이다. 4-9 에서 통한 것도 수준값이 아니라 편차였다.
DEV_SPEC = [(f"dev_p{k}_{kind[:3]}",
             f"asof_pitcher_prev{k}_game_{kind}_rate",
             f"asof_pitcher_{kind}_rate")
            for kind in ("success", "middle") for k in (1, 3, 5)]


def add_dev(df):
    """행 내 뺄셈으로 편차 컬럼을 만든다. 원본 컬럼은 그대로 둔다."""
    for name, recent, ref in DEV_SPEC:
        df[name] = df[recent].astype("float64") - df[ref].astype("float64")
    return df


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--min-leaf", type=int, default=1000)
    p.add_argument("--l2", type=float, default=3.0)
    p.add_argument("--n-iter", type=int, default=1200)
    p.add_argument("--border-count", type=int, default=128,
                   help="exp006/008 이 검증한 값은 128 이다. CatBoost 기본값은 "
                        "254 이고, 그대로 두면 fold2024 에서 717.97 -> 695.85 로 "
                        "떨어졌다. 경계가 적을수록 규제가 세다 (4-1 과 같은 방향).")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--dev-feats", action="store_true",
                   help="최근 형태 편차 6개를 추가한다 (EXP019). 실험 경로에서 "
                        "3폴드 +10.39 이지만 부호가 갈렸다 — 제출 경로 확인용.")
    p.add_argument("--save-val-pred", default=None,
                   help="검증 시즌 예측을 npz 로 저장한다. 캘리브레이션(alpha) "
                        "을 재학습 없이 재기 위한 것 — 제출 경로 예측의 퍼짐은 "
                        "실험 경로와 다르므로 여기서 다시 재야 한다 (4-17).")
    p.add_argument("--drop-feats", default=None)
    p.add_argument("--model-out", default="./model_cand/cat.pkl")
    p.add_argument("--save", action="store_true")
    return p.parse_args()


def make_pipeline(args, features, seed):
    num_cols = [c for c in features if c not in CAT_COLS]
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                               unknown_value=-1), CAT_COLS),
        ("num", "passthrough", num_cols),
    ])
    clf = CatBoostClassifier(
        iterations=args.n_iter, learning_rate=args.lr, depth=args.depth,
        min_data_in_leaf=args.min_leaf, l2_leaf_reg=args.l2,
        border_count=args.border_count,
        random_seed=seed, verbose=0, allow_writing_files=False)
    return Pipeline([("pre", pre), ("clf", clf)])


def main():
    args = parse_args()

    if args.dev_feats and args.save:
        # bundle 의 features 에 dev_* 가 들어가면 script.py 의 build_features 가
        # "컬럼 없음" 으로 죽는다. 추론 경로 지원을 먼저 넣을 것.
        raise SystemExit(
            "--dev-feats 와 --save 를 같이 쓰려면 script.py 가 dev 컬럼을 "
            "만들 수 있어야 한다. 먼저 검증만(--save 없이) 재고, 수치가 "
            "살아남으면 script.py 에 지원을 넣을 것.")

    test_cols = pd.read_csv(os.path.join(DATA_DIR, "test.csv"),
                            encoding="utf-8-sig", nrows=0).columns
    all_features = [c for c in test_cols if c != ft.ID]
    features = list(all_features)
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"),
                        encoding="utf-8-sig", usecols=all_features + [TARGET])

    # 상황 조건부 Trackman 편차 8개 (4-9). 시계 컬럼은 뺀다.
    tm = ft.load_trackman()
    train = ft.attach_ctx_train(train, tm)
    ctx_feats = [c for c in ft.COUNT_FEATS + ft.HAND_FEATS
                 if c not in ("tmc_n", "tmh_n")]
    features += ctx_feats
    c_all, h_all = ft.ctx_tables(tm, 9999)
    ctx_pack = {"count": ft.pack_table(c_all, ft.COUNT_KEY),
                "hand": ft.pack_table(h_all, ft.HAND_KEY),
                "hand_map": {str(k): v for k, v in ft.HAND.items()},
                "count_key": ft.COUNT_KEY, "hand_key": ft.HAND_KEY}

    if args.dev_feats:
        train = add_dev(train)
        features += [n for n, _, _ in DEV_SPEC]
        print(f"최근 형태 편차 {len(DEV_SPEC)}개 추가")

    if args.drop_feats:
        gone = [c.strip() for c in args.drop_feats.split(",") if c.strip()]
        features = [c for c in features if c not in gone]
        print(f"추가 제거: {gone}")
    print(f"피처 {len(features)}개")

    is_val = train["season"] == VAL_SEASON
    X_tr, y_tr = train.loc[~is_val, features], train.loc[~is_val, TARGET]
    X_va = train.loc[is_val, features]
    y_va = train.loc[is_val, TARGET].to_numpy()
    r = y_va.mean()
    base = r * (1 - r)
    print(f"학습 {len(X_tr):,} | 검증 {len(X_va):,} ({VAL_SEASON})")
    print(f"설정: lr={args.lr} depth={args.depth} min_leaf={args.min_leaf} "
          f"l2={args.l2} n_iter={args.n_iter} seeds={args.seeds}\n")

    acc = np.zeros(len(X_va))
    for k, seed in enumerate(range(42, 42 + args.seeds), start=1):
        m = make_pipeline(args, features, seed)
        t = time.time()
        m.fit(X_tr, y_tr)
        p = m.predict_proba(X_va)[:, 1]
        acc += p
        print(f"  s{seed} 단일 {ft.score_of(y_va, p, base):8.2f}  "
              f"누적 {ft.score_of(y_va, acc / k, base):8.2f}  "
              f"[{time.time()-t:.0f}s]")

    p_ens = acc / args.seeds
    print(f"\n=== 검증 {VAL_SEASON} ===")
    print(f"  앙상블 {ft.score_of(y_va, p_ens, base):8.2f}   "
          f"범위 {p_ens.min():.4f}~{p_ens.max():.4f}")
    print(f"  중심 편차 {p_ens.mean() - r:+.4f} "
          f"(예측 {p_ens.mean():.4f} vs 실제 {r:.4f})")

    if args.save_val_pred:
        os.makedirs(os.path.dirname(args.save_val_pred) or ".", exist_ok=True)
        np.savez_compressed(
            args.save_val_pred, p=p_ens.astype(np.float32),
            y=y_va.astype(np.int8),
            center_train=float(y_tr.mean()), base=float(base),
            config=f"lr={args.lr} i={args.n_iter} l2={args.l2} "
                   f"border={args.border_count} seeds={args.seeds}")
        print(f"  검증 예측 저장: {args.save_val_pred}")

    if not args.save:
        print("\n(저장하려면 --save)")
        return

    print(f"\n전체 데이터 재학습 (반복수 +10%) ...")
    models = []
    for seed in range(42, 42 + args.seeds):
        a2 = argparse.Namespace(**vars(args))
        a2.n_iter = int(args.n_iter * 1.1)
        m = make_pipeline(a2, features, seed)
        t = time.time()
        m.fit(train[features], train[TARGET])
        models.append(m)
        print(f"  s{seed} (n_iter={a2.n_iter}) 완료 [{time.time()-t:.0f}s]")

    bundle = {
        "models": models, "alpha": 1.0,
        "center": float(train[TARGET].mean()),
        "features": list(features),
        "spec": [f"cat-s{s}" for s in range(42, 42 + args.seeds)],
        "shift": None, "detrend": None, "ctx": ctx_pack,
        "note": "catboost ensemble; predict: mean(proba) -> clip(0,1)",
    }
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    joblib.dump(bundle, args.model_out, compress=3)
    print(f"\n저장 완료: {args.model_out} "
          f"({os.path.getsize(args.model_out)/1e6:.1f} MB, 모델 {len(models)}개)")


if __name__ == "__main__":
    main()
