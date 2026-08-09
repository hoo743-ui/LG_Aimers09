r"""10회차 실패 원인 분리 — CatBoost 설정 변경만 따로 잰다.

배경. 10회차(LB 848.77, 9회차 대비 -26.89)는 일곱 가지를 한 번에 바꿨다.

    depth 6->7 | l2 10->300 | 그루 1100->600 | 시드 2->4
    cb 0.60->1.00 | hgb 0.30->0 | lr 0.10->0

앞의 네 개(CatBoost 설정)와 뒤의 세 개(앙상블 구조)가 뒤섞여 있어 LB 하락이
어느 쪽 탓인지 알 수 없다. 이 스크립트는 **앙상블 구조를 9회차 그대로 고정
하고 CatBoost 만 교체**해서 그 축의 기여를 단독으로 잰다.

    A (기준)  hgb .30 / cb .60 (d6 · 1100 · l2 10) / lr .10 / lam .03  — LB 875.66
    B (실험)  hgb .30 / cb .60 (d7 ·  600 · l2 300) / lr .10 / lam .03

가중치·lam·피처·전처리·HGB·로지스틱은 A 와 B 가 완전히 동일하다. 다른 것은
cb 슬롯에 들어가는 예측 배열 하나뿐이다.

**시드는 별도 축으로 분리해서 보고한다.** 1시드끼리 비교해야 하이퍼파라미터
효과가 순수하게 나오고, 4시드는 거기에 시드 평균 효과가 얹힌 값이다. 둘을
합쳐서 보면 10회차와 똑같은 실수를 반복하는 셈이다.

예측 캐시 생산자:
    A 쪽 cb : catboost_test.py   -> {Y}_cb_d6_l210_it1100_noid_seed{S}.npy
    B 쪽 cb : cb_feat.py --only base --l2 300 --seed S
              -> {Y}_cbf_base_d7_it600_l2300_s{S}.npy
    hgb/lr  : blend_test.py      -> {Y}_hgb_seed42.npy, {Y}_lr_seed42.npy

    .\.venv\Scripts\python.exe iso_cb.py
"""
import os

import numpy as np
import pandas as pd

DATA = "./data/train.csv"
CACHE = "./.blendcache"
TARGET = "control_success"
PREV1 = "asof_pitcher_prev1_game_success_rate"
FOLDS = [2021, 2022, 2024]

# 9회차 앙상블 — A 와 B 가 공유한다. 이 값은 이 실험에서 절대 건드리지 않는다.
W_CB, W_LR, LAM = 0.60, 0.10, 0.03

SEEDS = [42, 43, 44, 45]
TAG_A = "cb_d6_l210_it1100_noid_seed{s}"      # 9회차 CatBoost
TAG_B = "cbf_base_d7_it600_l2300_s{s}"        # 10회차 CatBoost


def load(Y, name):
    p = os.path.join(CACHE, f"{Y}_{name}.npy")
    return np.load(p) if os.path.exists(p) else None


def load_seeds(Y, tag):
    """있는 시드만 모아 평균 낸다. (평균배열, 쓴 시드 목록)"""
    got = [(s, load(Y, tag.format(s=s))) for s in SEEDS]
    got = [(s, p) for s, p in got if p is not None]
    if not got:
        return None, []
    return np.mean([p for _, p in got], axis=0), [s for s, _ in got]


def main():
    df = pd.read_csv(DATA, encoding="utf-8-sig",
                     usecols=["season", TARGET, PREV1])
    season = df["season"].to_numpy()
    y_all = df[TARGET].to_numpy(dtype=float)

    fold = {}
    for Y in FOLDS:
        m = season == Y
        y = y_all[m]
        c = float(y_all[season < Y].mean())
        fold[Y] = dict(
            y=y, denom=y.mean() * (1 - y.mean()),
            hgb=load(Y, "hgb_seed42"), lr=load(Y, "lr_seed42"),
            anc=df.loc[m, PREV1].fillna(c).to_numpy(dtype=float) - c)

    def sc(Y, cb, w_cb=W_CB, w_lr=W_LR):
        """9회차 혼합식에 cb 만 갈아 끼운다."""
        d = fold[Y]
        p = (1 - w_cb - w_lr) * d["hgb"] + w_cb * cb + w_lr * d["lr"]
        p = np.clip(p + LAM * d["anc"], 0, 1)
        return max(0.0, 100000 * (1 - ((p - d["y"]) ** 2).mean() / d["denom"]))

    def solo(Y, p):
        d = fold[Y]
        return max(0.0, 100000 * (1 - ((np.clip(p, 0, 1) - d["y"]) ** 2).mean()
                                  / d["denom"]))

    print("[EXPERIMENT]")
    print("9회차 앙상블 고정 (hgb .30 / cb .60 / lr .10 / lam .03)")
    print("CatBoost 만 d6·1100·l2 10 -> d7·600·l2 300 교체\n")

    # (표시명, 태그, 시드, 가중치) — 마지막 두 항목은 10회차 제출본 재현이라
    # 가중치가 다르다. 이 실험의 후보가 아니라 **분해용 측정**이다.
    SPECS = (
        ("A 기준  1시드", TAG_A, [42], (W_CB, W_LR)),
        ("B 실험  1시드", TAG_B, [42], (W_CB, W_LR)),
        ("A 기준  전시드", TAG_A, SEEDS, (W_CB, W_LR)),
        ("B 실험  전시드", TAG_B, SEEDS, (W_CB, W_LR)),
        ("10회차 1시드", TAG_B, [42], (1.0, 0.0)),
        ("10회차 전시드", TAG_B, SEEDS, (1.0, 0.0)),
    )
    rows = []
    for label, tag, seeds, (w_cb, w_lr) in SPECS:
        vals, used, cbsolo = {}, {}, {}
        for Y in FOLDS:
            ps = [(s, load(Y, tag.format(s=s))) for s in seeds]
            ps = [(s, p) for s, p in ps if p is not None]
            if not ps:
                vals[Y], used[Y], cbsolo[Y] = None, [], None
                continue
            cb = np.mean([p for _, p in ps], axis=0)
            vals[Y] = sc(Y, cb, w_cb, w_lr)
            cbsolo[Y] = solo(Y, cb)
            used[Y] = [s for s, _ in ps]
        rows.append((label, vals, used, cbsolo))

    def line(label, vals):
        cells = "".join(f"{vals[Y]:>11.2f}" if vals[Y] is not None
                        else f"{'-':>11}" for Y in FOLDS)
        got = [vals[Y] for Y in FOLDS if vals[Y] is not None]
        avg = f"{np.mean(got):>11.2f}" if len(got) == len(FOLDS) else f"{'-':>11}"
        return f"{label:>16}{cells}{avg}"

    def dline(label, a, b):
        cells, got = "", []
        for Y in FOLDS:
            if a[Y] is None or b[Y] is None:
                cells += f"{'-':>11}"
            else:
                cells += f"{b[Y] - a[Y]:>+11.2f}"
                got.append(b[Y] - a[Y])
        avg = f"{np.mean(got):>+11.2f}" if len(got) == len(FOLDS) else f"{'-':>11}"
        return f"{label:>16}{cells}{avg}"

    hdr = f"{'':>16}" + "".join(f"{Y:>11}" for Y in FOLDS) + f"{'평균':>10}"

    print("[RESULT] 혼합 점수 (9회차 앙상블에 cb 만 교체)")
    print(hdr)
    print("-" * (16 + 11 * (len(FOLDS) + 1)))
    for label, vals, _, _ in rows[:4]:
        print(line(label, vals))
    print()
    print(dline("d 1시드", rows[0][1], rows[1][1]))
    print(dline("d 전시드", rows[2][1], rows[3][1]))
    print(dline("d 시드효과 A", rows[0][1], rows[2][1]))
    print(dline("d 시드효과 B", rows[1][1], rows[3][1]))

    # ---- 10회차 분해. 후보 탐색이 아니라 이미 제출한 것의 원인 배분이다 ----
    print("\n[분해] 10회차가 가져간 로컬 이득을 두 축으로 나눈다")
    print(hdr)
    print("-" * (16 + 11 * (len(FOLDS) + 1)))
    for label, vals, _, _ in rows[4:]:
        print(line(label, vals))
    print()
    print("  -- 1시드 기준 --")
    print(dline("(1) 하이퍼", rows[0][1], rows[1][1]))
    print(dline("(2) 앙상블구조", rows[1][1], rows[4][1]))
    print(dline("합계 = 10-A", rows[0][1], rows[4][1]))
    # 4시드 기준. (2) 축은 A 캐시가 필요 없으므로 세 폴드 모두 시드가 맞는다.
    # (1) 축과 합계는 A 쪽 2021·2022 가 아직 1시드라 B 에 유리하게 기운다.
    print("  -- 4시드 기준 (A 캐시가 1시드인 폴드는 * 로 기운다) --")
    print(dline("(1) 하이퍼*", rows[2][1], rows[3][1]))
    print(dline("(2) 앙상블구조", rows[3][1], rows[5][1]))
    print(dline("합계 = 10-A*", rows[2][1], rows[5][1]))

    print("\n[참고] CatBoost 단독 (혼합 전, 중심보정 전)")
    print(hdr)
    print("-" * (16 + 11 * (len(FOLDS) + 1)))
    for label, _, _, cbsolo in rows:
        print(line(label, cbsolo))

    print("\n[쓴 시드]")
    for label, _, used, _ in rows:
        print(f"  {label}: " + "  ".join(f"{Y}={used[Y]}" for Y in FOLDS))

    # ---- 판정 (지시받은 기준 그대로, 임의 변경 금지) ----
    #   Case A : 평균 +5 이상 AND 2024 +25 이상 AND 부호 대부분 일치
    #   Case B : 평균 +5 미만 (노이즈 가능)  -> 확정하지 않음
    #   Case C : 평균 음수 또는 부호 불안정  -> 9회차 유지
    # 셋 중 어디에도 안 들어가는 구간(평균은 크고 2024 만 미달)이 실제로
    # 존재한다. 그걸 Case C 로 흘려보내면 "악화"로 잘못 기록되므로 따로 찍는다.
    def verdict(tag_txt, a, b):
        d = [b[Y] - a[Y] for Y in FOLDS
             if a[Y] is not None and b[Y] is not None]
        if len(d) != len(FOLDS):
            print(f"\n[DIAGNOSIS] {tag_txt}: 캐시 부족으로 판정 보류")
            return
        sign = sum(1 for x in d if x > 0)
        mean = float(np.mean(d))
        print(f"\n[DIAGNOSIS] {tag_txt}")
        print(f"  평균 d {mean:+.2f} | 부호 일치 {sign}/{len(FOLDS)} "
              f"| 2024 d {d[-1]:+.2f}")
        # 부호 기준은 mix_cb.py 와 같다 — **3폴드 전부 양수**. 하나라도 음수면
        # 불안정으로 본다 (4-8 의 RF 실패가 이 완화를 허용했다가 난 사고다).
        if mean < 0 or sign < len(FOLDS):
            print(f"  -> Case C (악화 또는 불안정). 음수 폴드 "
                  f"{len(FOLDS)-sign}개. 9회차 구성 유지.")
        elif mean < 5:
            print("  -> Case B (판정 불가). 노이즈 범위. 확정하지 않는다.")
        elif d[-1] >= 25:
            print("  -> Case A (개선). 단, 순수 CatBoost 제출은 금지. "
                  "다음은 가중치만 따로 검증한다.")
        else:
            print(f"  -> Case A 미달. 평균({mean:+.2f})과 부호({sign}/"
                  f"{len(FOLDS)})는 통과했으나 2024 가 +25 미만이다. "
                  "채택 보류.")

    verdict("1시드 기준 (하이퍼파라미터 순수 효과)", rows[0][1], rows[1][1])
    verdict("전시드 기준 (하이퍼 + 시드 합계)", rows[2][1], rows[3][1])
    verdict("앙상블 구조 축 1시드 (참고, 이번 라운드 후보 아님)",
            rows[1][1], rows[4][1])
    verdict("앙상블 구조 축 4시드 (참고, 이번 라운드 후보 아님)",
            rows[3][1], rows[5][1])

    print("""
읽는 법. 판정은 **1시드 행**으로 한다 - 하이퍼파라미터 효과를 시드 평균
효과와 섞지 않기 위해서다. 전시드 행은 10회차가 CatBoost 쪽에서 실제로
가져간 것의 총량이고, 그중 시드 몫은 'd 시드효과' 두 줄로 분리된다.
분해 표의 (2) 앙상블구조 행은 이미 제출한 10회차의 원인 배분일 뿐,
이번 라운드에서 가중치를 건드리라는 뜻이 아니다.""")


if __name__ == "__main__":
    main()
