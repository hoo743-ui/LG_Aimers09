r"""Trackman 물리 프로필을 붙여 이득을 잰다.

trackman_link.py 가 만든 대응표로 투수별 요약값을 계산해 피처로 더한다.
지금 모델은 투수의 제구력을 '과거 성공률'이라는 결과 지표로만 안다. 그 결과를
만들어내는 원인 — 릴리스 포인트의 일관성, 구위, 레퍼토리 — 을 직접 준다.

제구의 물리적 실체는 릴리스 반복성이다. 매번 같은 지점에서 같은 팔 궤적으로
던지는 투수가 제구가 좋다. rel_height / rel_side / extension 의 산포가 그것을
거의 직접 측정한다.

시점 규칙을 지킨다. fold Y 를 평가할 때 프로필은 **Y 이전 시즌의 trackman** 만
쓴다. 전 기간을 쓰면 미래 정보가 새어 검증 점수가 부풀려진다. 실제 제출에서는
평가가 2025 이고 trackman 이 2024 까지이므로 전량이 과거다.

행 독립도 지킨다 — 프로필은 투수 단위 상수이고 평가셋의 다른 행을 보지 않는다.

사용법:
    .\.venv\Scripts\python.exe trackman_feat.py --folds 2021,2022,2024
"""
import argparse
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

DATA_DIR = "./data"
ID = "row_id"
TARGET = "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state"]
LR, LEAVES, MIN_LEAF, L2, N_ITER = 0.02, 10, 1000, 1.0, 1400

# 산포를 재는 물리량. 평균과 표준편차를 모두 만든다 — 평균은 구위, 표준편차는
# 반복성이다. 제구에는 후자가 더 직접적이다.
PHYS = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break",
        "extension", "rel_height", "rel_side"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--folds", default="2021,2022,2024")
    p.add_argument("--map", default="pitcher_id_map.csv")
    p.add_argument("--min-conf", type=float, default=0.9)
    p.add_argument("--keep", default="all",
                   help="all | release (릴리스 반복성 4개만) | 쉼표로 직접 지정")
    p.add_argument("--skip-base", action="store_true",
                   help="base 는 시드·설정이 같으면 값이 재현되므로 건너뛸 수 있다")
    return p.parse_args()


# 제구와 가장 직결되는 것만. 26개를 한꺼번에 넣으면 유효한 몇 개가 나머지에
# 희석된다 — delta 가 9개를 한꺼번에 넣어 실패한 것과 같은 구조를 피한다.
RELEASE = ["tm_fb_rel_height_std", "tm_fb_rel_side_std",
           "tm_fb_extension_std", "tm_fb_rel_speed_std"]


def select(prof, keep):
    if keep == "all":
        return prof
    names = RELEASE if keep == "release" else keep.split(",")
    missing = [c for c in names if c not in prof.columns]
    if missing:
        raise SystemExit(f"없는 피처: {missing}")
    return prof[names]


def build_profile(tm, upto_season, id_map):
    """upto_season 미만 시즌의 trackman 으로 투수별 프로필을 만든다."""
    sub = tm[tm["season"] < upto_season]
    if not len(sub):
        return None

    g = sub.groupby("pitcher_trackman_id")
    agg = {f"tm_{c}_{s}": g[c].agg(s) for c in PHYS for s in ("mean", "std")}
    prof = pd.DataFrame(agg)
    prof["tm_n"] = g.size()

    # 패스트볼만 골라 다시 잰다. 구종이 섞이면 산포에 레퍼토리 차이가 섞여
    # 반복성을 제대로 못 잰다. 패스트볼은 표본이 가장 크고 비교도 공정하다.
    fb = sub[sub["pitch_type_group"] == "fastball"]
    if len(fb):
        gf = fb.groupby("pitcher_trackman_id")
        for c in ("rel_height", "rel_side", "extension", "rel_speed"):
            prof[f"tm_fb_{c}_std"] = gf[c].std()
        prof["tm_fb_n"] = gf.size()

    # 레퍼토리 — 구종 수와 비율의 엔트로피(고를수록 크다)
    mix = (sub.groupby(["pitcher_trackman_id", "pitch_type_group"]).size()
           .unstack(fill_value=0))
    share = mix.div(mix.sum(axis=1), axis=0)
    prof["tm_n_types"] = (mix > 0).sum(axis=1)
    prof["tm_mix_entropy"] = -(share * np.log(share.where(share > 0, 1))).sum(1)
    for c in share.columns:
        prof[f"tm_share_{c}"] = share[c]

    # trackman id -> train pitcher_id 로 옮긴다
    prof = prof.join(id_map.set_index("pitcher_trackman_id")["pitcher_id"],
                     how="inner").set_index("pitcher_id")
    return prof


def make_model(cols, seed=42):
    num = [c for c in cols if c not in CAT_COLS]
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                               unknown_value=-1), CAT_COLS),
        ("num", "passthrough", num),
    ])
    clf = HistGradientBoostingClassifier(
        max_iter=N_ITER, learning_rate=LR, max_leaf_nodes=LEAVES,
        min_samples_leaf=MIN_LEAF, l2_regularization=L2,
        early_stopping=False, random_state=seed)
    return Pipeline([("pre", pre), ("clf", clf)])


def best_curve(clf, X_t, y, base):
    best = (0, 0.0)
    for i, proba in enumerate(clf.staged_predict_proba(X_t), start=1):
        p = proba[:, 1]
        s = max(0.0, 100000 * (1 - ((p - y) ** 2).mean() / base))
        if s > best[1]:
            best = (i, s)
    return best


def main():
    args = parse_args()
    folds = [int(f) for f in args.folds.split(",")]

    id_map = pd.read_csv(args.map)
    id_map = id_map[id_map["conf"] >= args.min_conf]
    print(f"대응표 {len(id_map)}명 (신뢰도 {args.min_conf} 이상)")

    tm = pd.read_csv(f"{DATA_DIR}/trackman_history.csv", encoding="utf-8-sig",
                     usecols=["season", "pitcher_trackman_id",
                              "pitch_type_group"] + PHYS)
    test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig",
                            nrows=0).columns
    cols = [c for c in test_cols if c != ID]
    train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                        usecols=cols + [TARGET])

    res = {}
    for Y in folds:
        prof = build_profile(tm, Y, id_map)
        if prof is None:
            print(f"fold {Y}: 이전 시즌 trackman 없음, 건너뜀")
            continue
        prof = select(prof, args.keep)

        tr = train[train["season"] < Y]
        va = train[train["season"] == Y]
        y = va[TARGET].to_numpy()
        base = y.mean() * (1 - y.mean())
        cov_tr = tr["pitcher_id"].isin(prof.index).mean()
        cov_va = va["pitcher_id"].isin(prof.index).mean()
        print(f"\n--- fold {Y} (프로필 {len(prof)}명, {prof.shape[1]}개 피처) ---")
        print(f"  투구 커버리지: 학습 {cov_tr:.1%}  검증 {cov_va:.1%}")

        joined = {}
        for name, df in (("tr", tr), ("va", va)):
            joined[name] = df.join(prof, on="pitcher_id")
        new_cols = cols + list(prof.columns)

        variants = [("base", cols), ("trackman", new_cols)]
        if args.skip_base:
            variants = variants[1:]
        for label, use in variants:
            t = time.time()
            m = make_model(use)
            m.fit(joined["tr"][use], joined["tr"][TARGET])
            X_t = m.named_steps["pre"].transform(joined["va"][use])
            it, sc = best_curve(m.named_steps["clf"], X_t, y, base)
            res.setdefault(label, {})[Y] = sc
            print(f"  {label:9s} {sc:9.2f} (iter {it:4d})  [{time.time()-t:.0f}s]")

    # base 를 건너뛴 경우를 위한 기준값 (같은 시드·설정에서 측정된 값)
    KNOWN_BASE = {2021: 1330.01, 2022: 2250.11, 2024: 640.01}

    print("\n=== 요약 ===")
    done = sorted(res.get("trackman", {}))
    base_vals = {Y: res.get("base", {}).get(Y, KNOWN_BASE.get(Y)) for Y in done}
    if any(base_vals[Y] is None for Y in done):
        return
    print(f"{'변형':10s}" + "".join(f"{f'fold{Y}':>11s}" for Y in done)
          + f"{'평균':>10s}")
    src = "" if "base" in res else "  (기존 측정값)"
    print(f"{'base':10s}" + "".join(f"{base_vals[Y]:11.2f}" for Y in done)
          + f"{np.mean([base_vals[Y] for Y in done]):10.2f}{src}")
    vals = [res["trackman"][Y] for Y in done]
    print(f"{'trackman':10s}" + "".join(f"{v:11.2f}" for v in vals)
          + f"{np.mean(vals):10.2f}")
    d = [res["trackman"][Y] - base_vals[Y] for Y in done]
    print(f"\n차이" + "".join(f"{x:+11.2f}" for x in d)
          + f"{np.mean(d):+10.2f}")


if __name__ == "__main__":
    main()
