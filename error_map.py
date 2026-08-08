r"""손실이 어디에 있는가 — 기법이 아니라 진단.

왜 분석의 성격을 바꾸는가. 지금까지 찾은 것은 전부 +8~16 짜리인데(4-7/4-8/4-9)
1등과의 격차는 로컬 환산 160 이다. 기법을 더 시도하는 방식으로는 자릿수가 안
맞는다. 무엇을 시도할지 정하기 전에 **손실이 어디 있는지**를 먼저 재야 한다.

Brier 를 분해하면 질문이 좁혀진다.

    Brier = 불확실성 - 해상도(resolution) + 신뢰도오차(reliability)
    BSS   = (해상도 - 신뢰도오차) / 불확실성

우리 2024 BSS 는 0.0066 이고 중심 편차가 먹는 게 0.0006 이다. 실제 판별력은
약 0.0072. LB 1000 이면 BSS 0.0100 이므로, 완벽히 캘리브레이션돼도 **해상도가
39% 높아야 한다.** 캘리브레이션 손질로는 못 메운다. 그 39% 가 어느 구간에
있는지를 찾는 것이 이 도구의 목적이다.

무엇을 보는가.

  1) 전역 분해 — 해상도 / 신뢰도오차. 캘리브레이션으로 되찾을 수 있는 총량
  2) 구간별 캘리브레이션 격차 — 전체 중심이 맞아도 구간별로 갈릴 수 있다.
     구간 오프셋은 학습 폴드에서 배워 행 단위로 적용하면 **합법**이다
     (4-9 와 같은 구조: 상수 + 그 행 자신의 피처)
  3) 구간별 해상도 — 어디서 우리가 판별을 못 하고 있는지. 여기가 진짜 격차다
  4) 콜드스타트 — 2025 투구의 약 20% 가 신인이다. 그 행들에서 asof_* 가 얇다
  5) 시즌 내 표류 — 4-2 는 시즌 **간** 이동만 봤다. 시즌 **안**은 미측정이다

정직하게 재는 법. 구간 오프셋을 그 폴드 정답으로 맞추면 당연히 이득이라 상한일
뿐이다. 이력 폴드에서 맞춰 평가 폴드에 옮긴 값을 함께 낸다 (4-9 와 같은 규약).

예측 캐시는 blend_test.py 가 만든다 (`.blendcache/`).

    .\.venv\Scripts\python.exe error_map.py
"""
import argparse
import os

import numpy as np
import pandas as pd

DATA = "./data/train.csv"
TARGET = "control_success"
CACHE = "./.blendcache"
HIST = [2021, 2022]
EVAL = 2024
NBIN = 20          # 캘리브레이션 곡선 구간 수


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="hgb")
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--min-n", type=int, default=2000,
                   help="구간 최소 표본. 작으면 격차가 노이즈다")
    return p.parse_args()


def load_preds(Y, model, seeds):
    acc, n = None, 0
    for s in range(42, 42 + seeds):
        path = os.path.join(CACHE, f"{Y}_{model}_seed{s}.npy")
        if os.path.exists(path):
            v = np.load(path)
            acc = v if acc is None else acc + v
            n += 1
    if acc is None:
        raise SystemExit(f"{Y} 캐시 없음 — blend_test.py 를 먼저 돌릴 것")
    return acc / n


def decompose(y, p, nbin=NBIN):
    """Murphy 분해. 예측값을 분위수 구간으로 묶어 잰다."""
    q = pd.qcut(pd.Series(p), nbin, labels=False, duplicates="drop").to_numpy()
    ybar = y.mean()
    unc = ybar * (1 - ybar)
    rel = res = 0.0
    for k in np.unique(q):
        m = q == k
        nk = m.sum() / len(y)
        rel += nk * (p[m].mean() - y[m].mean()) ** 2
        res += nk * (y[m].mean() - ybar) ** 2
    return unc, res, rel


def seg_table(df, key, y, p, denom, min_n):
    """구간별 캘리브레이션 격차와 해상도 기여."""
    g = pd.DataFrame({"k": df[key].to_numpy(), "y": y, "p": p})
    out = []
    ybar = y.mean()
    for k, sub in g.groupby("k", observed=True):
        n = len(sub)
        if n < min_n:
            continue
        gap = sub["p"].mean() - sub["y"].mean()
        # 이 구간의 격차가 전체 Brier 에서 먹는 점수
        lost = 100000 * (n / len(g)) * gap ** 2 / denom
        # 이 구간 안에서의 판별력 (구간 평균 대비)
        loc = ((sub["p"] - sub["p"].mean()) * (sub["y"] - sub["y"].mean())).mean()
        out.append((k, n, n / len(g), sub["y"].mean(), sub["p"].mean(),
                    gap, lost, loc))
    return out


def main():
    args = parse_args()
    cols = ["season", "game_month", "balls_before", "strikes_before", "inning",
            "pitcher_hand", "batter_hand", "asof_pitcher_n", "asof_batter_n",
            "asof_pitcher_success_rate", "pitcher_team_id", TARGET]
    df = pd.read_csv(DATA, encoding="utf-8-sig", usecols=cols)
    season = df["season"].to_numpy()

    df["count_state"] = df["balls_before"] * 3 + df["strikes_before"]
    df["same_hand"] = (df["pitcher_hand"] == df["batter_hand"]).astype(int)
    df["p_exp"] = pd.cut(df["asof_pitcher_n"].fillna(-1),
                         [-2, -0.5, 100, 500, 2000, 6000, 10 ** 9],
                         labels=["없음", "~100", "~500", "~2k", "~6k", "6k+"])
    df["b_exp"] = pd.cut(df["asof_batter_n"].fillna(-1),
                         [-2, -0.5, 100, 500, 2000, 10 ** 9],
                         labels=["없음", "~100", "~500", "~2k", "2k+"])
    df["inn"] = np.clip(df["inning"], 1, 10)

    ev = season == EVAL
    y = df.loc[ev, TARGET].to_numpy(dtype=float)
    p = load_preds(EVAL, args.model, args.seeds)
    if len(p) != ev.sum():
        raise SystemExit(f"캐시 {len(p)} != 행 {ev.sum()}")
    sub = df.loc[ev].reset_index(drop=True)
    denom = y.mean() * (1 - y.mean())
    base = max(0.0, 100000 * (1 - ((p - y) ** 2).mean() / denom))

    # ---- 1) 전역 분해 ----
    unc, res, rel = decompose(y, p)
    print(f"=== {EVAL} 전역 분해 (점수 {base:.2f}) ===")
    print(f"  불확실성     {unc:.6f}")
    print(f"  해상도       {res:.6f}   ({100000*res/denom:8.1f} 점 상당)")
    print(f"  신뢰도오차   {rel:.6f}   ({100000*rel/denom:8.1f} 점 상당) "
          f"<- 캘리브레이션으로 되찾을 수 있는 총량")
    print(f"  BSS = (해상도-신뢰도오차)/불확실성 = {(res-rel)/unc:.5f}")
    print(f"  예측 범위 {p.min():.4f}~{p.max():.4f}  표준편차 {p.std():.4f}")
    print(f"  중심 편차 {p.mean()-y.mean():+.5f}")
    print(f"\n  LB 1000 = BSS 0.0100 이 필요하다. 신뢰도오차를 0 으로 만들어도")
    print(f"  BSS 는 {res/unc:.5f} 이므로, **해상도가 "
          f"{100*(0.0100*unc/res - 1):.0f}% 더 있어야 한다.**")

    # ---- 2) 구간별 ----
    segs = [("p_exp", "투수 경험(누적 투구수)"), ("b_exp", "타자 경험"),
            ("count_state", "볼카운트"), ("inn", "이닝"),
            ("game_month", "월 (시즌 내 표류)"), ("same_hand", "좌우 매치업"),
            ("pitcher_team_id", "투수 팀")]
    print(f"\n=== 구간별 캘리브레이션 격차 (최소 {args.min_n} 표본) ===")
    print("  'lost' = 그 구간의 격차가 전체 점수에서 먹는 몫")
    total_recover = {}
    for key, label in segs:
        rows = seg_table(sub, key, y, p, denom, args.min_n)
        if not rows:
            continue
        tot = sum(r[6] for r in rows)
        total_recover[label] = tot
        print(f"\n  --- {label} --- (구간 격차 합계 {tot:6.1f} 점)")
        print(f"    {'구간':>8}{'표본':>9}{'비중':>7}{'실제':>8}{'예측':>8}"
              f"{'격차':>9}{'lost':>8}")
        for k, n, frac, ya, pa, gap, lost, loc in rows:
            print(f"    {str(k):>8}{n:9d}{frac:7.1%}{ya:8.4f}{pa:8.4f}"
                  f"{gap:+9.4f}{lost:8.1f}")

    print("\n=== 구간별 격차 합계 순위 (되찾기 상한) ===")
    for label, tot in sorted(total_recover.items(), key=lambda x: -x[1]):
        print(f"  {label:24s} {tot:7.1f} 점")

    print(f"""
읽는 법.
  전역 분해가 이 도구의 핵심이다. 신뢰도오차가 작으면 캘리브레이션 계열
  (4-9 중심 보정, 구간 오프셋)로 되찾을 것이 별로 없다는 뜻이고, 남은 격차는
  전부 해상도 — 즉 **판별력 자체**의 문제다. 그건 새 정보가 있어야 한다.

  구간별 'lost' 는 그 폴드 정답으로 잰 **상한**이다. 실제로 쓰려면 이력 폴드에서
  오프셋을 배워 옮겨야 하고, 4-9 에서 봤듯 그 과정에서 대부분 사라진다.""")


if __name__ == "__main__":
    main()
