r"""LV-002 — LV-001 의 1위 '카운트(12) 수준'을 정밀 검증한다.

LV-001 에서 카운트 12셀 수준 표가 **LV0 +7.18 / FULL +7.21 / 2023 +10.99**,
현행 후처리와 중복 0.006 으로 스윕 전체 1위였다. 그런데 원장 §4 는 오라클
진단에서 "카운트 −4.1, 회수 완료"로 이 축을 닫았다. **모순이므로 직접 본다.**

여기서는 이득을 최적화하지 않는다 — 셀 평균을 그대로 찍고, 가중은 w=1 고정으로
walk-forward 만 잰다.
"""
import io, os, sys, zipfile
import joblib, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
from lv001_level_sweep import champion_pred, r2, level_table, lookup, gain_curve  # noqa
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

df = pd.read_csv(os.path.join(ROOT, "data", "train.csv"),
                 usecols=["season","control_success","balls_before","strikes_before",
                          "pitcher_id","batter_id","pitcher_hand","batter_hand",
                          "num_runners_on","strikes_before"])
df = df[df.season >= 2022].reset_index(drop=True)
season = df.season.to_numpy(); y = df.control_success.to_numpy(float)
oof = np.load(os.path.join(ROOT, "exp", "champ_oof.npz"))
pv = champion_pred(df, oof, season)
cnt = df.balls_before.to_numpy(np.int64) * 4 + df.strikes_before.to_numpy(np.int64)
M = {s: season == s for s in (2022, 2023, 2024)}

print("카운트 12셀 — Champion 잔차 평균 (y − 예측). 시즌별 독립 추정\n")
print(f"{'B-S':>5}{'n(22)':>9}{'잔차22':>9}{'n(23)':>9}{'잔차23':>9}"
      f"{'n(24)':>9}{'잔차24':>9}   부호")
tab = {}
for c in sorted(np.unique(cnt)):
    row = []
    for s in (2022, 2023, 2024):
        m = M[s] & (cnt == c)
        r = (y[m] - pv[s][cnt[M[s]] == c]).mean()
        row.append((int(m.sum()), r))
    tab[c] = row
    sg = "".join("+" if r > 0 else "-" for _, r in row)
    print(f"{c//4}-{c%4:<3}" + "".join(f"{n:>9,}{r:>+9.4f}" for n, r in row)
          + f"   {sg}{'  ✅ 3시즌 동부호' if sg in ('+++','---') else ''}")

same = sum(1 for c in tab if len({r > 0 for _, r in tab[c]}) == 1)
print(f"\n3시즌 부호 일치 {same}/12 셀")
r2223 = np.array([tab[c][0][1] * tab[c][0][0] + tab[c][1][1] * tab[c][1][0] for c in sorted(tab)])
n2223 = np.array([tab[c][0][0] + tab[c][1][0] for c in sorted(tab)])
r24 = np.array([tab[c][2][1] for c in sorted(tab)])
print(f"corr(2022+23 셀평균, 2024 셀평균) = {np.corrcoef(r2223/n2223, r24)[0,1]:+.4f}")

print("\n--- walk-forward, 가중 w=1.0 고정 (최적화 없음) ---")
for tgt, src in ((2023, [2022]), (2024, [2022, 2023])):
    ks = np.concatenate([cnt[M[s]] for s in src])
    rs = np.concatenate([y[M[s]] - pv[s] for s in src])
    u, t, _ = level_table(ks, rs, 0)                 # 12셀 x 35k행 — 축소 불필요
    v = lookup(u, t, cnt[M[tgt]])
    base = r2(pv[tgt], y[M[tgt]])
    g = gain_curve(pv[tgt], v, y[M[tgt]])
    got = r2(pv[tgt] + v, y[M[tgt]])
    print(f"  목표 {tgt}  표={src}   기준선 {base:8.2f} -> w=1 {got:8.2f} "
          f"({got-base:+6.2f})   w* {g['w_opt']:.2f} 이득 {g['gain_opt']:+.2f}")

print("\n--- 이미 모델이 보는 것인가: 카운트를 in-model 피처로 갖고 있는데도 남는가 ---")
for s in (2022, 2023, 2024):
    m = M[s]
    o = pd.DataFrame({"c": cnt[m], "y": y[m], "p": pv[s]}).groupby("c").mean()
    print(f"  {s}  |평균잔차| 최대 {np.abs(o.y-o.p).max():.4f}  "
          f"성공률 범위 {o.y.min():.3f}~{o.y.max():.3f}  "
          f"예측 범위 {o.p.min():.3f}~{o.p.max():.3f}")
