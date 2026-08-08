r"""중심 편차를 **합법적으로** 줄일 수 있는지 잰다 — 학습 없이 캐시로.

왜 이게 지금 최우선인가. probe_cross.py 를 모델 잔차 기준선으로 바꿔 훑으니
2차 상호작용은 상한이 +8 로 고갈됐고(4-7), 대신 `main` 열이 전부 음수인 원인이
드러났다 — 중심 편차다.

    2024 잔차 평균 -0.01186  ->  0.0119^2 / 0.24981 x 100000 = 56.7 점

지금 남은 손실 중 가장 큰 단일 항목이다. 규칙 위반이었던 국면 보정(4-3)이
만든 +53.17 이 정확히 이 자리다.

무엇이 합법인가. 평가셋 **다른 행**을 평균내는 것은 금지다(4-3 이 그래서 걸렸다).
각 행 **자신의** `asof_*` 는 운영이 제공한 공식 입력이라 쓸 수 있다. 그리고
`prev1/prev3/prev5_game_success_rate` 는 그 투수의 직전 경기들에서 계산되므로
평가 시즌(2025)의 국면 수준을 담고 있다 — 학습 데이터에는 없는 정보다.

    p' = p + lambda * (anchor_row - c)      c, lambda 는 학습 시점 상수

lambda 가 상수이고 anchor 가 그 행 자신의 피처이므로 행 독립이 유지된다.

4-5 는 이 축에서 이미 한 번 실패했다 ("행 단위 기준선 잔차 학습" -23.8). 이유는
기준선의 표본이 작아 노이즈가 1:1 로 실렸다는 것이었다. 그 진단은 맞고, 해법은
**표본이 큰 anchor 를 고르고 lambda 를 최적화하는 것**이다. 닫힌 해가 있다.

    Brier 변화 = 2*lambda*E[(A-c)(p-y)] + lambda^2*E[(A-c)^2]
    lambda*    = -E[(A-c)(p-y)] / E[(A-c)^2]
    최대 이득  =  E[(A-c)(p-y)]^2 / E[(A-c)^2]

분모가 anchor 의 분산이다. **노이즈가 큰 anchor 는 그 자체로 상한을 깎는다** —
prev1(1경기)은 분산 ~0.01 로 상한이 한 자리이고, prev5(5경기)는 ~0.002 로
수십 점이 된다. 4-5 의 실패는 anchor 선택의 문제였다.

정직하게 재는 법. 자기 폴드의 정답으로 맞춘 lambda 는 당연히 이득이라 의미가
없다. **이력 폴드에서 맞춘 lambda 를 평가 폴드에 옮겨** 확인한다 — 4-5 가
기각한 지점(연도별 부호 반전)이 바로 이것이므로 반드시 이 열을 볼 것.

예측 캐시는 blend_test.py 가 만든다 (`.blendcache/`).

    .\.venv\Scripts\python.exe center_test.py
"""
import argparse
import os

import numpy as np
import pandas as pd

DATA = "./data/train.csv"
TARGET = "control_success"
CACHE = "./.blendcache"
FOLDS = [2021, 2022, 2024]

PREV1 = "asof_pitcher_prev1_game_success_rate"
PREV3 = "asof_pitcher_prev3_game_success_rate"
PREV5 = "asof_pitcher_prev5_game_success_rate"
CUM = "asof_pitcher_success_rate"
BCUM = "asof_batter_success_rate"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="hgb")
    p.add_argument("--seeds", type=int, default=1)
    return p.parse_args()


def load_preds(Y, model, seeds):
    acc, n = None, 0
    for s in range(42, 42 + seeds):
        path = os.path.join(CACHE, f"{Y}_{model}_seed{s}.npy")
        if not os.path.exists(path):
            continue
        v = np.load(path)
        acc = v if acc is None else acc + v
        n += 1
    if acc is None:
        raise SystemExit(f"{Y} 예측 캐시 없음 — blend_test.py 를 먼저 돌릴 것")
    return acc / n, n


def main():
    args = parse_args()
    df = pd.read_csv(DATA, encoding="utf-8-sig",
                     usecols=["season", TARGET, PREV1, PREV3, PREV5, CUM, BCUM])
    season = df["season"].to_numpy()
    y = df[TARGET].to_numpy(dtype=np.float64)

    # anchor 후보. 표본이 클수록 노이즈가 작고 상한이 높지만, 창이 넓을수록
    # 국면 반영이 늦다 (4-2: 누적형은 최근 2년에 0.025 뒤처진다). 그 절충을 본다.
    def col(c):
        return df[c].to_numpy(dtype=np.float64)

    p1, p3, p5 = col(PREV1), col(PREV3), col(PREV5)
    anchors = {
        "prev1": p1,
        "prev3": p3,
        "prev5": p5,
        "prev135": np.nanmean(np.vstack([p1, p3, p5]), axis=0),
        "prev35": np.nanmean(np.vstack([p3, p5]), axis=0),
        "cumul": col(CUM),
        "batter_cumul": col(BCUM),
    }

    print(f"모델 {args.model} | 폴드 {FOLDS}\n")

    # ---- 폴드별로 lambda* 와 상한 ----
    per_fold = {}
    for Y in FOLDS:
        p, nseed = load_preds(Y, args.model, args.seeds)
        m = season == Y
        if len(p) != m.sum():
            raise SystemExit(f"{Y}: 캐시 {len(p)} != 행 {m.sum()}")
        yy = y[m]
        c = float(y[season < Y].mean())        # 학습 데이터 성공률 (학습 시점 상수)
        r = yy.mean()
        denom = r * (1 - r)
        resid = p - yy                          # 양수면 과대예측
        base = max(0.0, 100000 * (1 - (resid ** 2).mean() / denom))
        print(f"fold {Y} (시드 {nseed})  기준 {base:8.2f}  "
              f"중심편차 {resid.mean():+.5f}  c={c:.4f}  실제={r:.4f}")
        per_fold[Y] = dict(p=p, y=yy, c=c, denom=denom, base=base, mask=m)

    print(f"\n{'anchor':>14}{'폴드':>7}{'sd(A-c)':>9}{'lambda*':>9}"
          f"{'상한':>9}{'이식이득':>9}{'이식후편차':>11}")
    print("-" * 70)

    hist, ev = FOLDS[:-1], FOLDS[-1]
    for name, A in anchors.items():
        # 결측은 c 로 메운다 — 신호 없음으로 취급 (그 행에서 shift 가 0 이 된다)
        num_h, den_h = 0.0, 0.0
        lam_by_fold = {}
        for Y in FOLDS:
            d = per_fold[Y]
            a = A[d["mask"]]
            a = np.where(np.isnan(a), d["c"], a) - d["c"]
            resid = d["p"] - d["y"]
            num, den = float((a * resid).mean()), float((a ** 2).mean())
            lam = 0.0 if den <= 0 else -num / den
            cap = 0.0 if den <= 0 else 100000 * (num ** 2 / den) / d["denom"]
            lam_by_fold[Y] = (lam, cap, float(np.sqrt(den)), a, resid)
            if Y in hist:
                num_h += num
                den_h += den

        # **leave-one-out 이식.** 각 폴드에 나머지 폴드들에서 맞춘 lambda 를
        # 적용한다. 한 폴드만 보고 채택하면 그 폴드에 맞춘 셈이다 (4-6).
        gains, devs = {}, {}
        for Y in FOLDS:
            # 풀링 최적해는 분모 가중 평균이다: lambda = sum(lam_Z*den_Z)/sum(den_Z)
            num_o = sum(lam_by_fold[Z][0] * (lam_by_fold[Z][3] ** 2).mean()
                        for Z in FOLDS if Z != Y)
            den_o = sum((lam_by_fold[Z][3] ** 2).mean() for Z in FOLDS if Z != Y)
            lam_o = 0.0 if den_o <= 0 else num_o / den_o
            d = per_fold[Y]
            _, _, _, a, resid = lam_by_fold[Y]
            nr = resid + lam_o * a
            gains[Y] = (max(0.0, 100000 * (1 - (nr ** 2).mean() / d["denom"]))
                        - d["base"])
            devs[Y] = (float(resid.mean()), float(nr.mean()), lam_o)

        for Y in FOLDS:
            lam, cap, sd, _, _ = lam_by_fold[Y]
            d0, d1, lam_o = devs[Y]
            print(f"{name if Y == FOLDS[0] else '':>14}{Y:>7}{sd:9.4f}"
                  f"{lam:+9.3f}{cap:9.2f}{gains[Y]:+9.2f}"
                  f"   {d0:+.5f}->{d1:+.5f}  (λ_LOO {lam_o:+.3f})")
        mean_g = float(np.mean([gains[Y] for Y in FOLDS]))
        ok = all(gains[Y] > 0 for Y in FOLDS)
        print(f"{'':>14}{'평균':>7}{'':>27}{mean_g:+9.2f}"
              f"   부호 {'일치' if ok else '★엇갈림'}")
        print()

    # ---- 고정 lambda 훑기 ----
    # LOO 로 lambda 를 맞추게 두면 cap 이 큰 폴드가 lambda 를 끌어올려, 이미
    # 중심이 맞은 폴드(2022)를 크게 망친다. 이득 곡선이
    #     gain(lambda) = cap * (1 - (1 - lambda/lambda*)^2)
    # 이므로 **작게 고정하면** cap 이 큰 폴드에서 이득을 상당히 가져오면서
    # cap 이 작은 폴드의 손실은 작게 묶인다. 그 절충점을 직접 찾는다.
    print("\n=== 고정 lambda (LOO 로 맞추지 않고 상수로 박는다) ===")
    lams = [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.09]
    for name, A in anchors.items():
        cells = []
        for Y in FOLDS:
            d = per_fold[Y]
            a = A[d["mask"]]
            a = np.where(np.isnan(a), d["c"], a) - d["c"]
            resid = d["p"] - d["y"]
            row = []
            for lam in lams:
                nr = resid + lam * a
                row.append(max(0.0, 100000 * (1 - (nr ** 2).mean() / d["denom"]))
                           - d["base"])
            cells.append(row)
        arr = np.array(cells)                      # 폴드 x lambda
        print(f"\n  {name}")
        print(f"{'lambda':>10}" + "".join(f"{l:9.2f}" for l in lams))
        for i, Y in enumerate(FOLDS):
            print(f"{Y:>10}" + "".join(f"{x:+9.2f}" for x in arr[i]))
        best = int(np.argmax([arr[:, j].mean() if (arr[:, j] > 0).all()
                              else -1e9 for j in range(len(lams))]))
        allpos = (arr[:, best] > 0).all()
        print(f"{'평균':>10}" + "".join(f"{arr[:, j].mean():+9.2f}"
                                        for j in range(len(lams))))
        print(f"{'최악':>10}" + "".join(f"{arr[:, j].min():+9.2f}"
                                        for j in range(len(lams))))
        if allpos:
            print(f"      → 세 폴드 모두 양수인 최고 lambda = {lams[best]:.2f}"
                  f"  평균 {arr[:, best].mean():+.2f}"
                  f"  최악 {arr[:, best].min():+.2f}")
        else:
            print("      → 세 폴드 모두 양수인 lambda 가 없다")

    # ---- 중심 보정 x 계열 혼합 — 겹치는가 ----
    # 혼합(4-8)의 이득은 오차 상쇄였고(중심 편차가 거의 안 변했다) 이쪽은 수준
    # 보정이다. 기전이 다르므로 더해질 수 있다. 캐시가 있으니 바로 확인한다.
    print("\n=== 중심 보정 x 계열 혼합 (겹침 확인) ===")
    W_LR, LAM, ANC = 0.10, 0.03, "prev1"
    A = anchors[ANC]
    print(f"혼합 hgb {1-W_LR:.2f} / lr {W_LR:.2f}  |  중심 {ANC} λ={LAM}")
    print(f"{'폴드':>7}{'기준':>10}{'+혼합':>9}{'+중심':>9}{'+둘다':>9}"
          f"{'합-단순합':>11}")
    print("-" * 56)
    tot = np.zeros(4)
    for Y in FOLDS:
        d = per_fold[Y]
        a = A[d["mask"]]
        a = np.where(np.isnan(a), d["c"], a) - d["c"]
        p_lr = np.load(os.path.join(CACHE, f"{Y}_lr_seed42.npy"))
        variants = {
            "base": d["p"],
            "blend": (1 - W_LR) * d["p"] + W_LR * p_lr,
            "center": d["p"] + LAM * a,
            "both": (1 - W_LR) * d["p"] + W_LR * p_lr + LAM * a,
        }
        s = {}
        for k, pv in variants.items():
            pv = np.clip(pv, 0.0, 1.0)
            s[k] = max(0.0, 100000 * (1 - ((pv - d["y"]) ** 2).mean() / d["denom"]))
        add = (s["blend"] - s["base"]) + (s["center"] - s["base"])
        print(f"{Y:>7}{s['base']:10.2f}{s['blend']-s['base']:+9.2f}"
              f"{s['center']-s['base']:+9.2f}{s['both']-s['base']:+9.2f}"
              f"{s['both']-s['base']-add:+11.2f}")
        tot += [s["base"], s["blend"] - s["base"],
                s["center"] - s["base"], s["both"] - s["base"]]
    tot /= len(FOLDS)
    print(f"{'평균':>7}{tot[0]:10.2f}{tot[1]:+9.2f}{tot[2]:+9.2f}{tot[3]:+9.2f}")
    print("  '합-단순합' 이 0 근처면 두 효과가 독립이다. 음수면 겹친다")

    print(f"""
읽는 법.
  sd(A-c)  anchor 의 산포. 클수록 노이즈가 커서 상한이 깎인다
  lambda*  그 폴드에서 Brier 를 최소화하는 계수 (자기 정답을 봤으므로 낙관적)
  상한     그 폴드에서 최적 lambda 를 썼을 때의 이득. **달성 불가능한 상한이다**
  이식이득 {hist} 에서 맞춘 lambda 를 {ev} 에 적용한 실제 이득. **이 열만 실전값이다**
  이식후편차 적용 후 중심 편차. 0 에 가까워야 성공이다

폴드별 lambda 부호가 갈리면 이식은 실패한다 — 4-5 가 기각한 지점이 그것이다.
이식이득이 양수인 anchor 만 interact_feat.py 로 정식 확인할 것.""")


if __name__ == "__main__":
    main()
