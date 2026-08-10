r"""실험 러너 — temporal fold x seed 로 재고, 예측을 캐시한다.

## 설계 이유

1. **예측을 캐시한다.** 점수만 캐시하면 앙상블 가중치를 다시 최적화할 때
   재학습해야 한다. 검증 예측(float32 25만 = 1MB)을 저장하면 앙상블 조합,
   상관(다양성), 캘리브레이션을 **학습 없이** 다시 계산할 수 있다.
2. **키는 설정 전체의 해시다.** (모델, 피처집합, 하이퍼파라미터, 시드, 폴드)
   가 같으면 절대 재학습하지 않는다.
3. **점수는 두 가지로 남긴다.**
   - `score_fixed` : 지정한 반복수에서의 점수. **판단 기준.** 제출물과 같은 조건
   - `score_best` / `iter_best` : 곡선 최적. 진단용 — 검증셋에 맞춘 낙관 편향이
     있어 절대값은 부풀려져 있다 (4-6). README 의 과거 수치는 이쪽이다

## 폴드

fold Y = "Y 미만 시즌으로 학습, Y 로 검증". 2023 은 어떤 구성으로도 0점이라
비교 정보가 없어 기본에서 뺀다 (4-6). 기본 폴드는 2021, 2022, 2024.

    .\.venv\Scripts\python.exe exp\runner.py --plan plans\exp001.json
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "exp", "cache")
PREDS = os.path.join(ROOT, "exp", "preds")
LOG = os.path.join(ROOT, "experiment_log.jsonl")

DEFAULT_FOLDS = [2021, 2022, 2024]

# ---- 시계 컬럼. 6회차에서 LB 29점을 잃힌 것들 (4-9) ----
CLOCK_FEATS = ["tmc_n", "tmh_n"]


def load_cache():
    X = np.load(f"{CACHE}/X.npy", mmap_mode="r")
    y = np.load(f"{CACHE}/y.npy")
    season = np.load(f"{CACHE}/season.npy")
    meta = json.load(open(f"{CACHE}/cols.json"))
    return X, y, season, meta


def resolve_features(meta, spec):
    """피처 집합 이름/명세 -> 컬럼 이름 목록.

    spec = {"base": "prod"|"raw47"|"all", "drop": [...], "add": [...]}
    """
    cols = meta["cols"]
    ctx = set(meta["ctx"])
    base = spec.get("base", "prod")
    if base == "raw47":                       # 상황 피처 없음 (ctx_feat 의 base)
        sel = [c for c in cols if c not in ctx]
    elif base == "all":                       # 시계 포함 전량 (6회차 구성)
        sel = list(cols)
    elif base == "prod":                      # 7회차 확정 구성: 47 + 편차 8
        # prep_tmx.py 가 컬럼을 덧붙인 뒤에는 명시 목록을 쓴다. 안 그러면 새
        # 컬럼이 자동으로 딸려 들어가 이미 캐시된 실험의 'prod' 의미가 바뀐다.
        sel = list(meta["prod"]) if "prod" in meta else [
            c for c in cols if c not in CLOCK_FEATS]
    else:
        raise ValueError(base)
    sel = [c for c in sel if c not in set(spec.get("drop", []))]
    for c in spec.get("add", []):
        if c not in sel:
            sel.append(c)
    missing = [c for c in sel if c not in cols]
    if missing:
        raise ValueError(f"캐시에 없는 컬럼: {missing}")
    return sel


def cfg_key(cfg):
    """모델을 결정하는 것만 해시한다.

    `exp`/`tag` 는 라벨일 뿐이라 넣으면 안 된다 — 넣으면 실험 이름이 다르다는
    이유만으로 똑같은 모델을 다시 학습하게 된다 (규칙 14).
    """
    core = {k: cfg[k] for k in ("model", "features", "hparams", "fold", "seed")}
    payload = json.dumps(core, sort_keys=True, ensure_ascii=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def score_of(y, p, base):
    return max(0.0, 100000 * (1 - ((p - y) ** 2).mean() / base))


def build_model(cfg, n_features, cat_idx):
    m = cfg["model"]
    h = cfg["hparams"]
    if m == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_iter=h["n_iter"], learning_rate=h.get("lr", 0.02),
            max_leaf_nodes=h.get("leaves", 10),
            min_samples_leaf=h.get("min_leaf", 1000),
            l2_regularization=h.get("l2", 1.0),
            max_features=h.get("max_features", 1.0),
            max_bins=h.get("max_bins", 255),
            early_stopping=False, random_state=cfg["seed"])
    if m == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=h["n_iter"], learning_rate=h.get("lr", 0.02),
            num_leaves=h.get("leaves", 10),
            min_child_samples=h.get("min_leaf", 1000),
            reg_lambda=h.get("l2", 1.0),
            colsample_bytree=h.get("max_features", 1.0),
            subsample=h.get("subsample", 1.0),
            subsample_freq=1 if h.get("subsample", 1.0) < 1 else 0,
            random_state=cfg["seed"], n_jobs=-1, verbose=-1)
    if m == "cat":
        from catboost import CatBoostClassifier
        extra = {}
        # grow_policy 를 바꾸면 트리 모양이 달라진다. 기본 SymmetricTree 는
        # min_data_in_leaf 를 조용히 무시하지만 Depthwise/Lossguide 는 쓴다.
        if h.get("grow_policy"):
            extra["grow_policy"] = h["grow_policy"]
        if h.get("max_leaves"):
            extra["max_leaves"] = h["max_leaves"]
        if h.get("subsample"):
            extra["bootstrap_type"] = h.get("bootstrap_type", "Bernoulli")
            extra["subsample"] = h["subsample"]
        return CatBoostClassifier(
            iterations=h["n_iter"], learning_rate=h.get("lr", 0.02),
            depth=h.get("depth", 6),
            min_data_in_leaf=h.get("min_leaf", 1000),
            l2_leaf_reg=h.get("l2", 3.0),
            rsm=h.get("rsm", 1.0),
            border_count=h.get("max_bins", 128),
            **extra,
            random_seed=cfg["seed"], verbose=0, allow_writing_files=False,
            # 스레드 수는 반드시 설정에 박는다. CatBoost 는 스레드 수에 따라
            # 결과가 달라진다 — fold2024 seed42 에서 6스레드 717.97 vs
            # 11스레드 712.29. 환경변수로 두면 샤드 구성이 바뀔 때마다
            # 비교가 오염되고, 캐시 키에도 안 들어가 조용히 섞인다.
            thread_count=h["threads"])
    raise ValueError(m)


def staged_curve(cfg, model, Xv, y, base, n_iter):
    """반복수별 검증 점수 곡선 + 최종 반복수에서의 예측."""
    m = cfg["model"]
    if m == "hgb":
        curve = []
        p_last = None
        for proba in model.staged_predict_proba(Xv):
            p_last = proba[:, 1]
            curve.append(score_of(y, p_last, base))
        return np.array(curve, dtype=np.float32), p_last.astype(np.float32)
    if m == "lgbm":
        # 곡선은 100 간격으로만 (매 반복 예측은 너무 비싸다)
        curve = []
        step = max(1, n_iter // 20)
        for k in range(step, n_iter + 1, step):
            curve.append(score_of(y, model.predict_proba(
                Xv, num_iteration=k)[:, 1], base))
        p = model.predict_proba(Xv)[:, 1].astype(np.float32)
        return np.array(curve, dtype=np.float32), p
    if m == "cat":
        curve = []
        step = max(1, n_iter // 20)
        for k in range(step, n_iter + 1, step):
            curve.append(score_of(y, model.predict_proba(
                Xv, ntree_end=k)[:, 1], base))
        p = model.predict_proba(Xv)[:, 1].astype(np.float32)
        return np.array(curve, dtype=np.float32), p
    raise ValueError(m)


def run_one(cfg, X, y, season, meta, force=False):
    key = cfg_key(cfg)
    pred_path = os.path.join(PREDS, f"{key}.npz")
    if os.path.exists(pred_path) and not force:
        d = np.load(pred_path, allow_pickle=True)
        out = json.loads(str(d["meta"]))
        # 같은 모델을 다른 실험에서 재사용하는 경우다. 라벨만 현재 실험 것으로
        # 갈아끼우고 로그에 한 줄 더 남긴다 — analyze 가 이 exp 로 찾을 수 있게.
        if (out.get("exp"), out.get("tag")) != (cfg.get("exp"), cfg.get("tag")):
            out["exp"], out["tag"] = cfg.get("exp", "?"), cfg.get("tag", "")
            with open(LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
        return {**out, "cached": True}

    feats = resolve_features(meta, cfg["features"])
    idx = [meta["cols"].index(c) for c in feats]
    cat_idx = [feats.index(c) for c in meta["cat"] if c in feats]
    Y = cfg["fold"]

    tr = season < Y
    va = season == Y
    Xtr = np.ascontiguousarray(X[np.where(tr)[0]][:, idx])
    Xva = np.ascontiguousarray(X[np.where(va)[0]][:, idx])
    ytr, yva = y[tr].astype(np.int8), y[va].astype(np.int8)
    r = float(yva.mean())
    base = r * (1 - r)

    t0 = time.time()
    model = build_model(cfg, len(feats), cat_idx)
    model.fit(Xtr, ytr)
    fit_s = time.time() - t0

    n_iter = cfg["hparams"]["n_iter"]
    curve, p = staged_curve(cfg, model, Xva, yva, base, n_iter)
    step = 1 if cfg["model"] == "hgb" else max(1, n_iter // 20)
    b = int(np.argmax(curve))

    out = {
        "key": key, "exp": cfg.get("exp", "?"), "tag": cfg.get("tag", ""),
        "model": cfg["model"], "fold": Y, "seed": cfg["seed"],
        "hparams": cfg["hparams"], "features": cfg["features"],
        "n_features": len(feats),
        "score_fixed": float(score_of(yva, p, base)),
        "score_best": float(curve[b]), "iter_best": int((b + 1) * step),
        "pred_mean": float(p.mean()), "pred_std": float(p.std()),
        "pred_min": float(p.min()), "pred_max": float(p.max()),
        "actual_rate": r, "center_dev": float(p.mean() - r),
        "n_train": int(tr.sum()), "n_val": int(va.sum()),
        "fit_sec": round(fit_s, 1), "cached": False,
    }
    os.makedirs(PREDS, exist_ok=True)
    np.savez_compressed(pred_path, p=p, curve=curve,
                        meta=json.dumps({k: v for k, v in out.items()
                                         if k != "cached"}))
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(out, ensure_ascii=False) + "\n")
    return out


def load_pred(cfg):
    d = np.load(os.path.join(PREDS, f"{cfg_key(cfg)}.npz"), allow_pickle=True)
    return d["p"], d["curve"]


def expand(plan):
    """plan -> 개별 config 목록. variants x folds x seeds 의 곱."""
    out = []
    for v in plan["variants"]:
        for fold in plan.get("folds", DEFAULT_FOLDS):
            for seed in plan["seeds"]:
                out.append({
                    "exp": plan["exp"], "tag": v["tag"],
                    "model": v.get("model", "hgb"),
                    "features": v.get("features", {"base": "prod"}),
                    "hparams": {**plan.get("hparams", {}),
                                **v.get("hparams", {})},
                    "fold": fold, "seed": seed,
                })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    plan = json.load(open(a.plan, encoding="utf-8"))
    cfgs = expand(plan)
    mine = [c for i, c in enumerate(cfgs) if i % a.nshard == a.shard]
    X, y, season, meta = load_cache()
    for i, c in enumerate(mine, 1):
        r = run_one(c, X, y, season, meta, force=a.force)
        flag = "cache" if r["cached"] else f"{r['fit_sec']:.0f}s"
        print(f"[{a.shard}] {i}/{len(mine)} {r['tag']:14s} f{r['fold']} "
              f"s{r['seed']} fixed {r['score_fixed']:8.2f} "
              f"best {r['score_best']:8.2f}@{r['iter_best']:<5d} [{flag}]",
              flush=True)


if __name__ == "__main__":
    main()
