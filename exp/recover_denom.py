r"""반올림된 비율 컬럼에서 **숨은 분모**를 역산한다.

## 왜 되는가

원본 CSV 의 비율은 소수 6자리로 저장돼 있다. 분모가 작은 기약분수는 이
정밀도에서 유일하게 특정된다 — 분모 400 이하 기약분수의 평균 간격이 2e-5 인데
반올림 허용오차는 5e-7 이라, 무작위 실수가 우연히 맞을 확률은 약 5% 다.

    n=1501, rate=0.576949  ->  정확히 866/1501

## 한 컬럼만 쓰면 안 된다

`round(x*m)/m == x` 를 만족하는 최소 m 은 **기약분수의 분모**, 즉 참값의
약수다. 45/90 은 1/2 로 줄어 m=2 가 나온다. 그래서 **같은 창을 공유하는 두
컬럼**(예: 같은 경기 구간의 success_rate 와 middle_rate)을 동시에 정수로
만드는 최소 m 을 찾는다. 단독 복원의 완전일치는 51% 였고 결합 복원은 100% 다.

## 검증 방법

역산값이 맞는지는 **물리적 필연**으로 확인한다. 창이 포개져 있으면 단조성
(p1 <= p3 <= p5), 창 길이가 다르면 정규화 후 같은 대역에 오는지
(p1 / (p3/3) / (p5/5) 의 중앙값 41.0 / 37.3 / 35.2).

## 규정 4

그 행의 자기 값만 쓴다. 다른 행도 전체 분포도 참조하지 않으므로 행 하나만
줘도 같은 값이 나온다.

## 이 프로젝트에서의 결과

`prev{1,3,5}_game` 의 투구 수를 100% 복원했으나 **기각**됐다 (2024 −1.4).
모델이 비율의 granularity 로 이미 간접 추정하고 있었다. 자세한 것은
DEVIATION_LEDGER.md 의 "PN" 절.
"""
import numpy as np

TOL = 5.1e-7          # 소수 6자리 반올림의 최대 오차


def recover(cols, maxd):
    """`cols` 를 **동시에** k/m 으로 만드는 최소 m. 0 = 복원 실패.

    cols  : 같은 창을 공유하는 비율 배열들 (길이가 같아야 한다)
    maxd  : 분모 상한. 창이 길수록 크게 준다 (prev1 200, prev3 480, prev5 760)
    """
    n = len(cols[0])
    out = np.zeros(n, dtype=np.int32)
    todo = np.ones(n, bool)
    for c in cols:
        todo &= np.isfinite(c)
    idx = np.flatnonzero(todo)
    cur = [c[idx] for c in cols]
    live = np.ones(len(idx), bool)
    for m in range(1, maxd + 1):
        if not live.any():
            break
        sel = np.flatnonzero(live)
        ok = np.ones(len(sel), bool)
        for c in cur:
            v = c[sel]
            ok &= np.abs(v - np.round(v * m) / m) <= TOL
        hit = sel[ok]
        out[idx[hit]] = m
        live[hit] = False
    return out
