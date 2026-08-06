r"""연도 누설 스크리닝 — 피처가 신호인지 **시계**인지 가른다.

6회차(796.24)를 잃고 얻은 검사다. `tmc_n` 은 trackman 이 쌓일수록 커져 시즌에
따라 단조 증가했고, 성공률이 해마다 떨어지므로(4-2) **학습 기간 안에서는 점수를
올려줬다.** 2025 에서는 학습 범위를 벗어나 포화된다. in-era 검증은 이걸 못 잡는다
— 폴드가 전부 학습 기간 안에 있기 때문이다.

여기서는 모델을 학습하지 않는다. 피처 **하나만으로** "이 행이 Y년인가 Y−1년인가"
를 가르는 AUC 를 순위통계로 계산한다 (Mann-Whitney U / (n1*n2)).

  AUC 0.50  두 해가 구분 안 됨 → 시계 성분 없음
  AUC 0.70  뚜렷한 연도 성분
  AUC 0.90+ 사실상 시계. 2025 에서 외삽 불가

**주의 — 높은 AUC 가 곧 나쁨은 아니다.** `asof_*` 누적 비율은 리그 국면이
실제로 이동해서 연도와 상관된다(4-2). 그건 진짜 신호다. 위험한 것은 값의
**의미가 아니라 척도가** 시간에 끌려가는 것 — 표본 수, 누적 카운트처럼
"데이터가 쌓여서" 커지는 양이다.

그래서 판단은 AUC 단독이 아니라 **AUC + 단조성**으로 한다. 연도별 중앙값이
단조 증가/감소하면 시계일 가능성이 높다.

사용법:
    .\.venv\Scripts\python.exe era_screen.py
    .\.venv\Scripts\python.exe era_screen.py --only tmc_,tmh_
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import rankdata

DATA_DIR = "./data"
TARGET = "control_success"
SAMPLE = 400_000
SEED = 42
PAIRS = [(2020, 2021), (2021, 2022), (2022, 2023), (2023, 2024)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--only", default=None,
                   help="쉼표로 구분한 접두사. 예: tmc_,tmh_")
    p.add_argument("--top", type=int, default=25)
    return p.parse_args()


def auc_by_rank(x, g):
    """g(0/1) 두 집단을 x 로 가르는 AUC. 결측은 제외한다."""
    ok = ~np.isnan(x)
    x, g = x[ok], g[ok]
    n1, n0 = int(g.sum()), int((1 - g).sum())
    if n1 < 50 or n0 < 50:
        return np.nan
    r = rankdata(x)
    return (r[g == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    args = parse_args()
    import importlib.util
    spec = importlib.util.spec_from_file_location("ft", "final_train.py")
    ft = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ft)

    test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig",
                            nrows=0).columns
    cols = [c for c in test_cols if c != "row_id"]
    train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                        usecols=cols + [TARGET])
    df = ft.attach_ctx_train(train, ft.load_trackman())

    skip = {"season", TARGET, "top_bottom", "game_type", "base_state"}
    feats = [c for c in df.columns
             if c not in skip and pd.api.types.is_numeric_dtype(df[c])]
    if args.only:
        pre = tuple(args.only.split(","))
        feats = [c for c in feats if c.startswith(pre)]
    print(f"피처 {len(feats)}개 | 연도쌍 {len(PAIRS)}개\n")

    df = df.sample(min(SAMPLE, len(df)), random_state=SEED)
    rows = []
    for f in feats:
        aucs = []
        for a, b in PAIRS:
            sub = df[df["season"].isin([a, b])]
            aucs.append(auc_by_rank(sub[f].to_numpy(dtype=float),
                                    (sub["season"] == b).to_numpy().astype(int)))
        aucs = [x for x in aucs if not np.isnan(x)]
        if not aucs:
            continue
        # 0.5 에서 떨어진 정도 (방향 무시)
        dev = float(np.mean([abs(x - 0.5) for x in aucs]))
        # 방향 일관성 — 매 쌍에서 같은 방향이면 단조 시계
        signs = [np.sign(x - 0.5) for x in aucs]
        mono = abs(sum(signs)) == len(signs)
        med = df.groupby("season")[f].median()
        rows.append((f, 0.5 + dev, mono, dev, med))

    rows.sort(key=lambda r: -r[3])
    print(f"{'피처':34s} {'평균|AUC|':>9} {'단조':>5}   시즌별 중앙값")
    print("-" * 100)
    for f, auc, mono, _, med in rows[:args.top]:
        flag = "예" if mono else "-"
        mark = " 🚩" if (auc > 0.75 and mono) else ""
        vals = " ".join(f"{v:>8.3g}" for v in med.values)
        print(f"{f:34s} {auc:9.3f} {flag:>5}   {vals}{mark}")

    print("\n🚩 = 평균|AUC| 0.75 초과 + 매 연도쌍에서 같은 방향.")
    print("   값이 '쌓여서' 커지는 양이면 시계다 — 2025 에서 학습 범위를 벗어난다.")
    print("   리그 국면이 실제로 이동한 것(asof_* 비율)은 진짜 신호일 수 있으니")
    print("   AUC 만 보지 말고 그 양이 무엇인지 함께 판단할 것.")


if __name__ == "__main__":
    main()
