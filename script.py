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

def attach_ctx(df, bundle):
    """상황 조건부 Trackman 피처를 붙인다.

    모델 파일에 담긴 조회표를 (pitcher_id, 볼카운트) 와 (pitcher_id, 타자좌우)
    로 붙인다. 표는 학습 시점에 2019~2024 trackman 으로 만들어 pkl 에 들어
    있으므로, 평가 서버에 trackman 파일이 없어도 동작한다.

    **행 독립이다.** 각 행은 자기 자신의 투수/카운트/타자좌우로만 조회하며
    평가셋의 다른 행을 일절 참조하지 않는다. 표의 값도 평가 데이터가 아니라
    2019~2024 과거 로그에서 나왔다.

    표에 없는 조합(표본 부족, 미지 투수)은 결측으로 둔다 — HGB 는 결측을 분기
    조건으로 직접 학습하므로 채워 넣는 것보다 낫다.
    """
    ctx = bundle.get("ctx") if isinstance(bundle, dict) else None
    if not ctx:
        return df

    out = df
    for part, keycols in (("count", ctx["count_key"]), ("hand", ctx["hand_key"])):
        tab = ctx[part]
        frame = pd.DataFrame(tab["vals"], columns=tab["cols"])
        keys = pd.DataFrame(tab["keys"], columns=keycols).astype(str)
        frame = pd.concat([keys, frame], axis=1).set_index(keycols)

        # 조인 키를 문자열로 맞춘다. 표는 학습 때 문자열로 굳혀 담았고,
        # test 쪽 정수/실수 표기가 환경에 따라 달라질 수 있어서다.
        left = pd.DataFrame(index=out.index)
        for c in keycols:
            if c == "batter_hand":
                hand_map = {int(k): v for k, v in ctx["hand_map"].items()}
                left[c] = out[c].map(hand_map).astype(str)
            else:
                left[c] = out[c].astype("int64").astype(str)

        vals = left.join(frame, on=keycols)[tab["cols"]]
        out = pd.concat([out, vals], axis=1)
    return out


CAAFE_COLS = ["cf_same_hand", "cf_form5", "cf_share_reverse", "cf_share_ball",
              "cf_mix_entropy", "cf_log_pn", "cf_trend13", "cf_trend35",
              "cf_midform1", "cf_midtrend13", "cf_ball_minus_strike"]


def attach_caafe(df):
    """야구 도메인 파생변수 11개.

    전부 **그 행 안의 값만** 쓰는 산술이다. 평가셋의 다른 행도, 학습 데이터의
    집계도 참조하지 않으므로 5) 평가 데이터 예측 원칙에 안전하다 — test.csv 에
    이 행 하나만 있어도 값이 같다.

    선정 근거: 도메인 후보 40개를 만들어 3폴드(2022/2023/2024)에서 재고,
    **세 폴드 모두에서 중요도가 반복적으로 높았던 것만** 남겼다. 폴드 점수로
    고르면 폴드에 과적합되므로 중요도 일관성을 기준으로 삼았다.

    갈래
      같은손        투수손 == 타자손. 플래툰 효과는 투수의 가장 안정적인 특성이다
                    (시즌 간 지속성 0.42~0.45, 전체 성공률은 0.20~0.67).
      폼 편차/추세   prev{1,3,5} 게임과 시즌 비율의 차. 트리가 축평행 분할로
                    한 번에 못 만드는 형태다.
      실패 구성     실패가 가운데/볼/반대로 쪼개지는 비율. 투수의 지문.
      구종 엔트로피  구사 분포의 다양성.
    """
    import numpy as np

    g = lambda c: df[c].astype("float64")
    eps = 1e-6
    ps = g("asof_pitcher_success_rate")
    pm, pb = g("asof_pitcher_middle_rate"), g("asof_pitcher_ball_rate")
    pr, pst = g("asof_pitcher_reverse_rate"), g("asof_pitcher_strike_rate")
    fail = (1.0 - ps).clip(lower=eps)
    mix = [g(f"asof_pitcher_{k}_rate").clip(lower=eps, upper=1.0)
           for k in ("fastball", "breaking", "offspeed")]
    tot = sum(mix)
    ent = -sum((m / tot) * np.log(m / tot) for m in mix)

    df["cf_same_hand"] = (g("pitcher_hand") == g("batter_hand")).astype("float64")
    df["cf_form5"] = g("asof_pitcher_prev5_game_success_rate") - ps
    df["cf_share_reverse"] = pr / fail
    df["cf_share_ball"] = pb / fail
    df["cf_mix_entropy"] = ent
    df["cf_log_pn"] = np.log1p(g("asof_pitcher_n"))
    df["cf_trend13"] = (g("asof_pitcher_prev1_game_success_rate")
                        - g("asof_pitcher_prev3_game_success_rate"))
    df["cf_trend35"] = (g("asof_pitcher_prev3_game_success_rate")
                        - g("asof_pitcher_prev5_game_success_rate"))
    df["cf_midform1"] = g("asof_pitcher_prev1_game_middle_rate") - pm
    df["cf_midtrend13"] = (g("asof_pitcher_prev1_game_middle_rate")
                           - g("asof_pitcher_prev3_game_middle_rate"))
    df["cf_ball_minus_strike"] = pb - pst
    return df


# (as-of 비율 컬럼, 그 비율의 분모 n 컬럼, 라벨, 어느 선수 id 로 묶이는가)
ASOF_SPEC = [("asof_pitcher_success_rate", "asof_pitcher_n", "succ", "pitch"),
             ("asof_pitcher_middle_rate", "asof_pitcher_n", "mid", "pitch"),
             ("asof_pitcher_ball_rate", "asof_pitcher_n", "ball", "pitch"),
             ("asof_pitcher_reverse_rate", "asof_pitcher_n", "rev", "pitch"),
             ("asof_pitcher_strike_rate", "asof_pitcher_n", "str", "pitch"),
             ("asof_pitcher_fastball_rate", "asof_pitcher_pitchmix_n", "fb", "mix"),
             ("asof_pitcher_breaking_rate", "asof_pitcher_pitchmix_n", "bb", "mix"),
             ("asof_pitcher_offspeed_rate", "asof_pitcher_pitchmix_n", "os", "mix"),
             ("asof_batter_success_rate", "asof_batter_n", "bsucc", "bat"),
             ("asof_batter_middle_rate", "asof_batter_n", "bmid", "bat")]
ASOF_NCOL = {"pitch": ("asof_pitcher_n", "pitcher_id"),
             "mix": ("asof_pitcher_pitchmix_n", "pitcher_id"),
             "bat": ("asof_batter_n", "batter_id")}
ASOF_COLS = [f"cur_{lb}" for _, _, lb, _ in ASOF_SPEC] + \
            [f"cur_logn_{k}" for k in ("pitch", "mix", "bat")]

# 최근 경기 vs **시즌내 누적** (F). 공식 prev{1,3,5} 게임 컬럼에서 그 행의
# `cur_*` 를 뺀다. CAAFE 는 `prev1 - 통산rate` 였는데 통산에는 이력이 섞여
# 있어 척도가 다른 두 값을 뺀 것이었다. `cur_*` 는 같은 시즌 안의 누적이라
# 뺄셈이 성립한다 — 같은 창 안에서 "최근이 시즌평균보다 좋은가"가 된다.
FORM_SPEC = [("succ", "asof_pitcher_prev{}_game_success_rate"),
             ("mid", "asof_pitcher_prev{}_game_middle_rate")]
FORM_WIN = (1, 3, 5)
FORM_COLS = [f"form_{lb}{k}" for lb, _ in FORM_SPEC for k in FORM_WIN]

# D x 맥락 (X). "D 가 특별히 강하거나 약해지는 상황이 있는가"를 트리에 직접
# 준다. 트리는 두 열의 곱을 분할로 근사하기 어렵고(축평행), 그래서 이 8개는
# 모델이 스스로 못 만드는 형태다. 재료는 전부 그 행 자신의 값이다.
CTX_MUL = [("adv", "카운트 우위 (strikes > balls)"),
           ("onb", "주자 유무"),
           ("sh", "같은 손 (투수손 == 타자손)"),
           ("bs", "볼 - 스트라이크")]
CTX_COLS = [f"dx_{lb}_{tag}" for lb in ("succ", "mid") for tag, _ in CTX_MUL]

# 수준 확장 (H1). X 는 cur_succ / cur_mid 두 수준에만 맥락을 곱했다. 투수의
# 볼 비율 · 반대투구 비율 · 스트라이크 비율에도 같은 국면 의존성이 있다 —
# 상대 손과 카운트 국면에 따라 다르게 읽혀야 하는 값들이다.
# 곱하는 맥락은 2024 기여가 가장 컸던 둘만 쓴다 (같은손 +9.6, 볼-스트라이크 +7.0).
LVL_RATES = ("ball", "rev", "str")
LVL_MUL = ("sh", "bs")
LVL_COLS = [f"lx_{r}_{t}" for r in LVL_RATES for t in LVL_MUL]

# 비 (RX). X/H1 은 **곱**을 줬다. 비는 한 번도 주지 않았다.
#
# 자료 생성 감사(2026-08-18)에서 투수 결과 비율들의 상관 구조가 나왔다.
#
#     corr(success, reverse) = -0.865      corr(strike, ball) = -0.779
#
# 즉 success 와 reverse 는 독립 지표가 아니라 **제구 방향 한 축의 양끝**이다.
# 그 축 위의 위치는 두 값의 **비**인데, 모델에는 두 비율이 따로 들어가 있을
# 뿐이고 축평행 트리는 비를 만들 수 없다 (곱을 못 만드는 것과 같은 이유).
#
# 다섯 비율은 합이 1 이 아니다 — success+reverse+middle = 0.893, strike+ball
# = 0.813 으로 이름 없는 계급이 둘 있다. 그래서 단순 차가 아니라 로그비를 쓴다.
RX_EPS = 1e-3
RX_PAIR = [("cmd", "succ", "rev"),      # 제구 방향 축
           ("zone", "str", "ball"),     # 존 축
           ("mist", "mid", "succ")]     # 실투 성향 대비 커맨드
RX_COLS = ([f"rx_{tag}" for tag, _, _ in RX_PAIR]
           + [f"rx_{tag}_{m}" for tag, _, _ in RX_PAIR for m in ("sh", "bs")])

# 관측되지 않은 결과 계급 (UC). 2026-08-18 자료 생성 감사에서 나왔다.
#
#     success + reverse + middle = 0.893 (sd 0.028)   -> 이름 없는 위치 계급 ~10.7%
#     ball    + strike           = 0.813 (sd 0.020)   -> 이름 없는 접촉 계급 ~18.7%
#     fastball + breaking + offspeed = 1.000          -> 구종은 완전한 분할
#
# 앞의 둘은 **관측되지 않은 계급의 비율을 우리가 계산할 수 있다**는 뜻이다.
# 정보는 이미 모델 안에 있지만(기존 5열의 선형결합, R^2 = 1.000000) 축평행
# 트리는 `1 - a - b` 를 만들 수 없다 — X/H1(곱)이 통한 것과 같은 구조다.
#
# D 틀이 이것을 **시즌 내 값**으로 복원해 주므로 current-state family 에 속한다.
# 실측 전이율이 가장 좋은 계열이다 (+0.697).
UC_SPEC = [("other", ("succ", "rev", "mid")),      # 위치 4번째 계급
           ("cont", ("ball", "str"))]              # 접촉 계급
UC_CLIP = (-0.5, 1.5)                              # D 복원 잡음의 꼬리를 자른다
UC_COLS = ([f"uc_{t}" for t, _ in UC_SPEC] + [f"uc_{t}_car" for t, _ in UC_SPEC]
           + [f"uc_{t}_{m}" for t, _ in UC_SPEC for m in ("sh", "bs")])
UC_CAREER = {"other": ("asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
                       "asof_pitcher_middle_rate"),
             "cont": ("asof_pitcher_ball_rate", "asof_pitcher_strike_rate")}

# 위약 열 (NZ). **정보가 0 인 결정론적 잡음** 8열.
# RX(-14.6)와 UC(-13.5)가 거의 같은 값을 낸 것이 수상해서 만든 통제군이다.
# 열을 추가하는 것 자체의 비용을 재기 위한 것이지 후보가 아니다.
# 행 자신의 값만으로 만들어지므로 행 독립이다 (규정 4 안전).
NZ_COLS = [f"nz_{j}" for j in range(8)]

# 구종 믹스 x 맥락 (MX). **실제로 통한 패턴의 직접 연장**이다.
#
#     X   cur_{succ,mid}       x {adv, onb, sh, bs}   -> LB +3.90
#     H1  cur_{ball,rev,str}   x {sh, bs}             -> LB +5.16
#     MX  cur_{fb,bb,os}       x {sh, bs}             -> 미검증
#
# `cur_fb/bb/os` 는 시즌 내 구종 비율이고 모델에 **원시값으로만** 들어가 있다.
# 맥락과의 곱은 한 번도 준 적이 없다. "이 투수는 같은손 상대로 변화구를 더
# 던진다" 같은 조건부 성향은 축평행 트리가 만들기 어렵다.
# current-state family 라 전이율이 가장 좋은 계열(+0.697)에 속한다.
MX_RATES = ("fb", "bb", "os")
MX_MUL = ("sh", "bs")
MX_COLS = [f"mx_{r}_{m}" for r in MX_RATES for m in MX_MUL]

# 2스트라이크 국면 (K2). 2026-08-16 진단에서 나왔다 — 폴드 2024 를 상황별로
# 쪼개니 `rho^2` 가 **볼카운트에서만** 5.7배 흔들렸다 (아웃 0.90~1.12,
# 주자수 0.91~1.47, 이닝 0.78~1.25, 손 0.89~0.92 는 전부 평평).
#
#   2스트라이크  0-2 338 / 1-2 637 / 2-2 778 / 3-2 699   <- 전체 916 대비 무너짐
#   3볼         3-0 1928 / 3-1 1504                     <- 오히려 높음
#
# 해석 — 3볼은 스트라이크를 던져야 하므로 행동이 강제돼 예측이 쉽다.
# 2스트라이크는 유인구냐 승부냐가 갈리는데 **그 의도가 데이터에 없다.**
# 게다가 타깃 정의상 "존에서 크게 벗어난 공"이 실패라, 의도적으로 뺀 공도
# 실패로 라벨링된다. 의도는 못 보지만 **그 투수가 얼마나 자주 빼는가**는
# 볼 수 있다 — `cur_ball`(볼 비율)과 `cur_rev`(반대투구 비율)이다.
#
# `bs`(볼−스트라이크)와 다르다. `bs=0` 이 (0-0)·(1-1)·(2-2)를 전부 포함하듯
# `1{strikes==2}` 는 `bs` 의 함수가 아니다. 기존 82열 전체로 선형 회귀해도
# R^2 가 0.16~0.48 에 그친다.
K2_RATES = ("ball", "rev")
K2_COLS = [f"k2_{r}_{t}" for r in K2_RATES for t in ("2s", "2slow")]

# 경험적 베이즈 축소 (EB). `cur_rate` 는 표본이 적으면 분산이 크다 —
# `cur_n < 100` 인 행이 13.5% 다. 그 선수의 이력(`prior_rate`)으로 당겨
# **수준은 유지하면서 잡음만 깎는다.**
#
#     eb_r = (cur_events_r + k_r * parent_r) / (cur_n + k_r)
#     parent_r = prior_events_r / prior_n   (이력이 없으면 리그평균)
#
# `k_r` 은 학습 데이터에서 적률법으로 뽑아 번들에 담는다 —
# `k = mu(1-mu) / sigma^2_between`. 고정 500 이 아니다 (20-c 의 S500R 은 그
# 고정값으로 교체를 시도했다가 min 0.5633 으로 무너졌다).
#
# **원시 `cur_rate` 는 그대로 둔다.** 교체가 아니라 추가다.
EB_RATES = ("succ", "mid", "ball", "rev", "str")
EB_COLS = [f"eb_{r}" for r in EB_RATES]


def attach_asof_state(df, bundle):
    """AS-OF 분해 — 통산 누적에서 **현재 시즌 상태**를 복원한다.

    `data_description.md` L86 이 정의하듯 `asof_*` 는 "해당 행의 투구 직전까지"의
    누적이고, 시즌마다 리셋되지 않는 **통산값**이다 (실측 확인: 한 투수의
    `asof_pitcher_n` 이 2019 2,757 -> 2024 15,449 로 시즌을 넘어 이어진다).

    따라서 그 선수의 **학습 구간 통산**을 빼면 현재 시즌 상태가 나온다.

        cur_n    = asof_n(행) - prior_n[선수]
        cur_rate = (asof_n * asof_rate)(행) - prior_events[선수]) / cur_n

    `prior_*` 는 학습 데이터만으로 만들어 모델 파일에 담은 상수다.

    **왜 새 정보인가** — 모델이 보는 `asof_pitcher_success_rate` 는 통산이라
    이력과 현재 폼이 섞여 있고, 모델은 그 선수의 직전 시즌말 통산을 모르므로
    원리적으로 못 가른다. 이 분해가 그것을 갈라 준다.

    **규정 5)** — 행 자신의 공식 `asof_*` 컬럼(L182 에서 사용 허가 명시)과
    학습 데이터 상수만 쓴다. 평가셋의 다른 행을 보지 않으므로 test.csv 에 이 행
    하나만 있어도 값이 같다.

    검증 — 2024 폴드에서 `cur_n` 이 실제 시즌내 순번과 100.0000% 일치,
    `cur_rate` 복원 평균절대오차 3.1e-6.
    """
    import numpy as np

    pri = bundle.get("asof_prior") or {}
    N = {}
    for kind, (ncol, idcol) in ASOF_NCOL.items():
        tab = pri.get(kind, {})
        ids = df[idcol].astype("int64")
        n_now = df[ncol].astype("float64").fillna(0.0)
        p_n = pd.Series([tab.get(i, (0.0,))[0] for i in ids],
                        index=df.index, dtype="float64")
        cur = (n_now - p_n).clip(lower=0.0)
        N[kind] = (n_now, cur, ids, tab)
        df[f"cur_logn_{kind}"] = np.log1p(cur)
    eb = (bundle.get("eb") or {}) if isinstance(bundle, dict) else {}
    for j, (rc, nc, lb, kind) in enumerate(ASOF_SPEC):
        n_now, cur, ids, tab = N[kind]
        # 같은 kind 안에서 몇 번째 비율인지 (prior 벡터의 위치)
        pos = [k for k, sp in enumerate(ASOF_SPEC) if sp[3] == kind].index(j) + 1
        p_e = pd.Series([tab.get(i, (0.0,))[pos] if len(tab.get(i, (0.0,))) > pos
                         else 0.0 for i in ids], index=df.index, dtype="float64")
        tot = n_now * df[rc].astype("float64").fillna(0.0)
        df[f"cur_{lb}"] = ((tot - p_e) / cur).where(cur > 0)
        if eb and lb in EB_RATES:
            k = float(eb["k"][lb])
            mu = float(eb["mu"][lb])
            p_n = (n_now - cur).clip(lower=0.0)     # 이력 표본수
            parent = (p_e / p_n).where(p_n > 0, mu)
            df[f"eb_{lb}"] = (tot - p_e + k * parent) / (cur + k)
    for lb, pat in FORM_SPEC:
        for k in FORM_WIN:
            src = pat.format(k)
            df[f"form_{lb}{k}"] = (df[src].astype("float64")
                                   - df[f"cur_{lb}"]) if src in df else np.nan
    mul = {"adv": (df["strikes_before"].astype("float64")
                   > df["balls_before"].astype("float64")).astype("float64"),
           "onb": (df["num_runners_on"].astype("float64") > 0).astype("float64"),
           "sh": (df["pitcher_hand"].astype("float64")
                  == df["batter_hand"].astype("float64")).astype("float64"),
           "bs": (df["balls_before"].astype("float64")
                  - df["strikes_before"].astype("float64"))}
    for lb in ("succ", "mid"):
        for tag, _ in CTX_MUL:
            df[f"dx_{lb}_{tag}"] = df[f"cur_{lb}"] * mul[tag]
    for r in LVL_RATES:
        for t in LVL_MUL:
            df[f"lx_{r}_{t}"] = df[f"cur_{r}"] * mul[t]
    # 학습 경로는 bundle 에 "features" 가 없다 (asof_prior 만 넘어온다).
    # 그래서 features 가 없으면 **계산한다** — 이 가드를 잘못 써서 UC/NZ 실험
    # 두 건이 무효가 됐다 (열이 한 번도 만들어지지 않았다).
    _feats = bundle.get("features")
    def _want(cols):
        return _feats is None or any(c in _feats for c in cols)

    if _want(MX_COLS):
        for r in MX_RATES:
            for m in MX_MUL:
                df[f"mx_{r}_{m}"] = df[f"cur_{r}"] * mul[m]
    if _want(NZ_COLS):
        seed = (df["asof_pitcher_n"].astype("float64") * 0.7392
                + df["balls_before"].astype("float64") * 0.3711
                + df["strikes_before"].astype("float64") * 0.1137
                + df["outs_before"].astype("float64") * 0.0531)
        for j in range(8):
            df[f"nz_{j}"] = np.sin(seed * (13.0 + 7.0 * j) + 2.3 * j)
    if _want(UC_COLS):
        for tag, parts in UC_SPEC:
            v = 1.0
            for q in parts:
                v = v - df[f"cur_{q}"].astype("float64")
            v = v.clip(*UC_CLIP)
            df[f"uc_{tag}"] = v
            for m in ("sh", "bs"):
                df[f"uc_{tag}_{m}"] = v * mul[m]
            c = 1.0
            for q in UC_CAREER[tag]:
                c = c - df[q].astype("float64")
            df[f"uc_{tag}_car"] = c.clip(*UC_CLIP)
    # RX(로그비)는 **번들이 요구할 때만** 만든다. 2026-08-18 폴드 실측에서 −14.6
    # 으로 기각됐고, 추론 경로에 불필요한 연산을 남기지 않는 편이 안전하다.
    if _want(RX_COLS):
        for tag, hi, lo in RX_PAIR:
            a = df[f"cur_{hi}"].astype("float64").clip(lower=0.0) + RX_EPS
            b = df[f"cur_{lo}"].astype("float64").clip(lower=0.0) + RX_EPS
            v = np.log(a) - np.log(b)
            df[f"rx_{tag}"] = v
            for m in ("sh", "bs"):
                df[f"rx_{tag}_{m}"] = v * mul[m]
    st = df["strikes_before"].astype("float64")
    bl = df["balls_before"].astype("float64")
    k2m = {"2s": (st == 2).astype("float64"),
           "2slow": ((st == 2) & (bl <= 1)).astype("float64")}
    for r in K2_RATES:
        for t in ("2s", "2slow"):
            df[f"k2_{r}_{t}"] = df[f"cur_{r}"] * k2m[t]
    return df


def attach_aux(df, bundle):
    """보조 라벨 확률 P̂(결과|그 행) 을 피처로 붙인다 (EXP049).

    학습셋에서 `asof_pitcher_n` 증분으로 복원한 투구 단위 결과
    (`middle · reverse · ball · strike · fastball · breaking · offspeed` 와
    그 합성 `H = middle ∨ reverse`) 를 타깃으로 학습한 모델들이다.

    **규정 4 안전** — 각 보조 모델은 학습 데이터만으로 만들어졌고, 입력은 그
    행 자신의 피처뿐이다. 다른 행을 보지 않으므로 행 하나만 있어도 같은 값이
    나온다. 라벨 복원은 **학습 단계에서만** 일어난다.

    번들에 "aux" 키가 있을 때만 동작하므로 기존 제출본은 영향받지 않는다.
    """
    aux = bundle.get("aux") if isinstance(bundle, dict) else None
    if not aux:
        return df
    base = aux["base"]
    missing = [c for c in base if c not in df.columns]
    if missing:
        raise ValueError(f"보조 모델 입력 컬럼 없음: {missing}")
    Xb = df[base]
    for name, models in aux["models"].items():
        acc = None
        for m in models:                      # 2겹의 평균 (학습 때와 동일)
            p = m.predict_proba(Xb)[:, 1]
            acc = p if acc is None else acc + p
        df[f"aux_{name}"] = acc / len(models)
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
    연산은 하지 않는다.

    모델 파일은 두 형식을 허용한다.
      - dict  : {"models": [...], "alpha": float, "center": float}
                여러 모델의 예측을 평균한 뒤 중심값 쪽으로 축소한다.
                축소는 과신을 줄여 Brier 를 낮추기 위한 것이고, alpha 와
                center 는 학습 시점에 정해져 모델 파일에 담긴 상수다.
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

    preds = preds + platoon_adjust(bundle, X)

    alpha = float(bundle.get("alpha", 1.0))
    center = float(bundle.get("center", 0.5))
    preds = center + alpha * (preds - center)
    return preds.clip(0.0, 1.0)


def platoon_adjust(bundle, X):
    """조건부 편차 항들을 더한다. 없으면 0.

    표는 **학습 구간에서만** 만들어져 모델 파일에 담긴 상수다 (4-30). 조회 키는
    그 행 자신의 컬럼뿐이라 행 독립이고, 평가셋의 다른 행을 보지 않는다 —
    5) 원칙에 안전하다.

    담긴 값은 수준값이 아니라 **부모 집단 대비 편차**다 (투수 x 타자손이면 그
    투수 자신의 전체 성공률 대비). 리그 수준의 연도 이동은 뺄셈에서 소거된다.
    표에 없는 조합은 0 — 중립값이다.

    형식은 둘 다 받는다.
      - dict : {"w": float, "table": {(pid, hand): dev}}          (14~15회차)
      - list : [{"w": float, "cols": [...], "table": {...}}, ...]  (일반형)
    """
    pl = bundle.get("platoon") if isinstance(bundle, dict) else None
    if not pl:
        return 0.0
    specs = pl if isinstance(pl, list) else [
        {"w": pl["w"], "cols": ["pitcher_id", "batter_hand"],
         "table": pl["table"]}]
    total = 0.0
    for sp in specs:
        tab = sp["table"]
        # numpy 를 새로 import 하지 않는다 — 이 프로젝트에서 실패한 제출 2건이
        # 전부 추론 환경 문제였다 (6-1, 6-4). pandas 만으로 끝낸다.
        cols = [X[c].astype("int64") for c in sp["cols"]]
        keys = zip(*cols)
        v = pd.Series([tab.get(k, 0.0) for k in keys],
                      dtype="float64").to_numpy()
        total = total + float(sp["w"]) * v
    return total


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
              f"center={model.get('center', 0.5):.4f} "
              f"features={len(model.get('features', []))}")
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
    test = attach_ctx(test, model)
    # 번들이 요구할 때만 만든다 — 기존 제출본은 영향받지 않는다.
    if isinstance(model, dict) and any(
            c in (model.get("features") or []) for c in CAAFE_COLS):
        test = attach_caafe(test)
    if isinstance(model, dict) and any(
            c in (model.get("features") or []) for c in ASOF_COLS):
        test = attach_asof_state(test, model)
    test = attach_aux(test, model)          # 번들에 "aux" 가 있을 때만
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
    # 순수 ASCII 로 찍는다. 평가 컨테이너에 LANG 이 없으면 파이썬이 stdout 을
    # ASCII 로 잡아서, 비ASCII 문자 하나 때문에 CSV 를 다 쓰고도 예외로 끝난다.
    print(f"[OK] Saved: {OUT_PATH} (rows={len(sub)})")


if __name__ == "__main__":
    main()
