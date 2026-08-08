r"""trackman 에 현재 모델이 모르는 것이 있는가 — 학습 없이 점수로 반증한다.

왜 다시 보는가. trackman_history.csv 는 354MB, 179만 행으로 train.csv(368MB,
147만 행)와 거의 같은 크기인데 기여가 **-5.95** 다 (4-4). 실패 방식에 문제가
있었다는 정황이 셋이다.

  1) 투구 단위 데이터를 투수 단위로 뭉갰다. 179만 행에서 뽑은 게 투수당 요약
     26개다 — 해상도를 스스로 버렸다
  2) 실패 원인으로 적은 "pitcher_id 와 중복"은 근거가 약해졌다. pitcher_id 는
     데뷔 순 프록시일 뿐이다 (4-7)
  3) 26개를 한꺼번에 넣었다. 4-4 자신이 "유효한 몇 개가 나머지에 희석된다"고
     적어놓고 그렇게 했다

trackman 에만 있고 train.csv 어디에도 없는 것은 **구종**이다. train.csv 는
`asof_pitcher_*_rate` 라는 통산 비율만 준다 — 또 주변부 통계다. trackman 은
`(투수, 카운트, 타자좌우) -> 구종 분포`를 만들 수 있다.

규칙. 금지되는 것은 "현재 투구의 **실제** 구종"이다. 사전 정보(그 행의
pitcher_id / 카운트 / 타자좌우)와 과거 로그로 만든 **예측 분포**는 `asof_*` 와
같은 성격이라 허용된다. 시점도 지킨다 — fold Y 의 피처는 Y 이전 trackman 만 쓴다.

무엇을 재는가. 상관이 아니라 **점수**다. 그리고 기준선을 실제 모델 잔차로 둔다.

    gain = E[(A-Ā)(y-p)]^2 / E[(A-Ā)^2]  x  100000/r(1-r)

질문이 "이 정보에 신호가 있나"가 아니라 **"현재 모델이 이미 알고 있나"** 가
된다. 이미 알고 있으면 잔차와 무관해져 0 이 나온다.

**자체 검증.** trackman 의 통산 구종 비율은 `asof_pitcher_breaking_rate` 로
모델에 이미 들어 있다. 그러므로 `career_bb` 항목은 0 근처여야 한다. 안 나오면
도구를 믿을 수 없다.

정직하게: 이력 폴드에서 계수를 맞춰 평가 폴드에 옮긴 값만 실전값이다.

예측 캐시는 blend_test.py 가 만든다 (`.blendcache/`).

    .\.venv\Scripts\python.exe probe_trackman.py
"""
import os

import numpy as np
import pandas as pd

DATA = "./data"
CACHE = "./.blendcache"
TARGET = "control_success"
MAP = "pitcher_id_map.csv"
FOLDS = [2021, 2022, 2024]
MIN_CONF = 0.9
K = 300          # 셀 표본 축소


def load_preds(Y, seeds=1):
    acc, n = None, 0
    for s in range(42, 42 + seeds):
        p = os.path.join(CACHE, f"{Y}_hgb_seed{s}.npy")
        if os.path.exists(p):
            v = np.load(p)
            acc = v if acc is None else acc + v
            n += 1
    if acc is None:
        raise SystemExit(f"{Y} 캐시 없음 — blend_test.py 를 먼저 돌릴 것")
    return acc / n


def cell_mean(keys, vals, k, prior):
    """키별 평균을 표본 축소해서 반환. 딕셔너리 대신 groupby 로."""
    df = pd.DataFrame({"k": keys, "v": vals})
    g = df.groupby("k", observed=True)["v"].agg(["mean", "size"])
    n = g["size"].to_numpy()
    return pd.Series(prior + (g["mean"].to_numpy() - prior) * (n / (n + k)),
                     index=g.index)


def main():
    if not os.path.exists(MAP):
        raise SystemExit(f"{MAP} 없음 — trackman_link.py 로 대응표를 먼저 만들 것")

    id_map = pd.read_csv(MAP)
    id_map = id_map[id_map["conf"] >= MIN_CONF]
    t2p = dict(zip(id_map["pitcher_trackman_id"], id_map["pitcher_id"]))
    print(f"대응표 {len(t2p)}명 (신뢰도 {MIN_CONF} 이상)")

    print("trackman 로드 중 ...", flush=True)
    tm = pd.read_csv(f"{DATA}/trackman_history.csv", encoding="utf-8-sig",
                     usecols=["season", "pitcher_trackman_id", "pitch_type_group",
                              "balls_before", "strikes_before", "batter_hand",
                              "rel_speed", "rel_height", "rel_side"])
    tm["pitcher_id"] = tm["pitcher_trackman_id"].map(t2p)
    tm = tm[tm["pitcher_id"].notna()].copy()
    tm["pitcher_id"] = tm["pitcher_id"].astype(int)
    tm["count_state"] = tm["balls_before"] * 3 + tm["strikes_before"]
    tm["is_bb"] = (tm["pitch_type_group"] == "breaking").astype(float)
    tm["is_fb"] = (tm["pitch_type_group"] == "fastball").astype(float)
    print(f"  {len(tm):,} 행 매핑됨 | 구종 {tm['pitch_type_group'].unique().tolist()}")
    print(f"  trackman batter_hand 값 {sorted(tm['batter_hand'].dropna().unique().tolist())[:6]}")

    # 릴리스 반복성 — 4-4 가 "제구의 물리적 실체"로 지목한 것. 평균이 아니라
    # 산포다. 패스트볼만 골라야 구종 차이가 산포에 섞이지 않는다.
    fb = tm[tm["pitch_type_group"] == "fastball"]
    rep = fb.groupby(["season", "pitcher_id"])[["rel_height", "rel_side"]].std()
    rep = rep.rename(columns={"rel_height": "relh_std", "rel_side": "rels_std"})
    rep = rep.reset_index()
    # 투수당 시즌별 반복성을 다시 trackman 행에 붙여 아래 공용 경로를 그대로 쓴다
    tm = tm.merge(rep, on=["season", "pitcher_id"], how="left")

    print("train 로드 중 ...", flush=True)
    tr = pd.read_csv(f"{DATA}/train.csv", encoding="utf-8-sig",
                     usecols=["season", "pitcher_id", "balls_before",
                              "strikes_before", "batter_hand", TARGET])
    tr["count_state"] = tr["balls_before"] * 3 + tr["strikes_before"]
    season = tr["season"].to_numpy()
    y = tr[TARGET].to_numpy(dtype=float)

    # 후보. (이름, 키 컬럼들, trackman 값 컬럼, 설명)
    #   career_bb 는 asof_pitcher_breaking_rate 로 모델에 이미 있다 -> 0 이어야 한다
    CANDS = [
        ("career_bb", ["pitcher_id"], "is_bb",
         "통산 브레이킹 비율 (모델에 이미 있다 — 자체 검증용)"),
        ("bb_by_count", ["pitcher_id", "count_state"], "is_bb",
         "투수 x 카운트 브레이킹 비율"),
        ("relh_std", ["pitcher_id"], "relh_std",
         "패스트볼 릴리스 높이 산포 = 반복성 (4-4 가 지목한 물리적 실체)"),
        ("rels_std", ["pitcher_id"], "rels_std",
         "패스트볼 릴리스 좌우 산포 = 반복성"),
        ("fb_by_count", ["pitcher_id", "count_state"], "is_fb",
         "투수 x 카운트 패스트볼 비율"),
        ("velo_by_count", ["pitcher_id", "count_state"], "rel_speed",
         "투수 x 카운트 평균 구속"),
        ("relh_career", ["pitcher_id"], "rel_height",
         "통산 릴리스 높이 (물리 프로필 — 4-4 가 실패한 계열)"),
    ]

    print(f"\n{'후보':>16}{'폴드':>7}{'커버':>8}{'sd':>9}{'계수':>10}"
          f"{'상한':>9}{'이식이득':>10}")
    print("-" * 72)

    for name, keys, val, desc in CANDS:
        per = {}
        for Y in FOLDS:
            # 시점 규칙 — fold Y 는 Y 이전 trackman 만 본다
            src = tm[(tm["season"] < Y) & tm[val].notna()]
            if not len(src):
                continue
            prior = float(src[val].mean())
            key_src = (src[keys[0]].astype(str) if len(keys) == 1 else
                       src[keys[0]].astype(str) + "|" + src[keys[1]].astype(str))
            table = cell_mean(key_src.to_numpy(), src[val].to_numpy(), K, prior)

            m = season == Y
            sub = tr.loc[m]
            key_tr = (sub[keys[0]].astype(str) if len(keys) == 1 else
                      sub[keys[0]].astype(str) + "|" + sub[keys[1]].astype(str))
            a = table.reindex(key_tr.to_numpy()).to_numpy()
            cov = np.isfinite(a).mean()
            a = np.where(np.isfinite(a), a, prior)
            a = a - a.mean()

            p = load_preds(Y)
            yy = y[m]
            resid = p - yy
            denom = yy.mean() * (1 - yy.mean())
            num, den = float((a * resid).mean()), float((a ** 2).mean())
            # 조인이 실패하면 모든 값이 prior 로 채워져 분산이 0 이 된다.
            # 그때 -num/den 은 발산해 말도 안 되는 계수를 낸다 — 반드시 막는다.
            if cov < 0.3 or den < 1e-12:
                print(f"{name:>16}{Y:>7}{cov:8.1%}   조인 실패 또는 분산 0 — 건너뜀")
                continue
            lam = -num / den
            cap = 100000 * (num ** 2 / den) / denom
            per[Y] = (cov, float(np.sqrt(den)), lam, cap, a, resid, denom, num, den)

        if len(per) < 2:
            continue
        hist = [Y for Y in FOLDS[:-1] if Y in per]
        ev = FOLDS[-1]
        num_h = sum(per[Y][7] for Y in hist)
        den_h = sum(per[Y][8] for Y in hist)
        lam_t = 0.0 if den_h <= 0 else -num_h / den_h
        cov, sd, lam, cap, a, resid, denom, _, _ = per[ev]
        nr = resid + lam_t * a
        base = max(0.0, 100000 * (1 - (resid ** 2).mean() / denom))
        gain = max(0.0, 100000 * (1 - (nr ** 2).mean() / denom)) - base

        for Y in FOLDS:
            if Y not in per:
                continue
            cov, sd, lam, cap, *_ = per[Y]
            extra = f"{gain:+10.2f}" if Y == ev else ""
            print(f"{name if Y == FOLDS[0] else '':>16}{Y:>7}{cov:8.1%}"
                  f"{sd:9.4f}{lam:+10.3f}{cap:9.2f}{extra}")
        print(f"{'':>16}{'이식계수':>7}{'':>17}{lam_t:+10.3f}    {desc}")
        print()

    print("""읽는 법.
  상한     그 폴드에서 최적 계수를 썼을 때의 이득. 자기 정답을 봤으므로 낙관적
  이식이득 이력 폴드에서 맞춘 계수를 평가 폴드에 적용한 실제 이득. **이 열만 실전값**
  career_bb 가 0 근처가 아니면 도구를 믿을 수 없다 (모델에 이미 있는 정보다)

  이 도구는 후보를 죽이는 데만 쓴다. 살아남은 것은 반드시 3폴드 x 2시드 학습으로
  확인할 것 — park 이 안정성 상관 +0.819 였는데 실제로는 -23.09 였다 (4-6).""")


if __name__ == "__main__":
    main()
