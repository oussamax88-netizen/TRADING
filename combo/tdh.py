"""TDH (Trend Day Hunter) on the shared engine. Rules (NY time):
20-day trend = prev RTH close vs 20-day average (cache trend_d). ATR = 14-day mean 09:30-16:00 range (cache atr).
Skip flat opens: |09:30 open - prev close| < flat*ATR. Opening range 09:30-09:40. After 09:40 the FIRST 1m close
beyond the range that is also >= trig*ATR from the 09:30 open decides the day; trade only if it agrees with the trend.
Entry next 1m open, stop 0.2*ATR, no target, exit 15:59."""
import sys
sys.path.insert(0, '/home/user/TRADING/scalp')
import numpy as np
from engine import load, signals_frame, simulate, full_report

def signals(mk, trig=0.15, stopk=0.2, flat=0.15, sides=(1, -1)):
    D = load(mk); mod, dayi, o, h, l, c, atr, tr, pdc = (D[x] for x in ('mod', 'dayi', 'o', 'h', 'l', 'c', 'atr', 'trend_d', 'pdc'))
    starts = np.r_[0, np.where(np.diff(dayi) != 0)[0] + 1]; ends = np.r_[starts[1:], len(dayi)]
    ks, ds, st = [], [], []
    for a, b in zip(starts, ends):
        m = mod[a:b]; i0 = np.where(m == 570)[0]
        if not len(i0) or np.isnan(atr[a]) or np.isnan(pdc[a]): continue
        k0 = a + i0[0]; op = o[k0]; A = atr[k0]
        if abs(op - pdc[k0]) < flat * A: continue
        orr = np.where((m >= 570) & (m < 580))[0]
        hi, lo = h[a + orr].max(), l[a + orr].min()
        for j in np.where((m >= 580) & (m < 930))[0]:
            k = a + j
            up = c[k] > hi and c[k] - op >= trig * A; dn = c[k] < lo and op - c[k] >= trig * A
            if up or dn:
                d = 1 if up else -1
                if d == tr[k] and d in sides: ks.append(k); ds.append(d); st.append(stopk * A)
                break
    return signals_frame(ks, ds, stop_pts=np.array(st))

if __name__ == '__main__':
    import json
    for mk in ('NQ', 'ES'):
        r = full_report(mk, signals(mk)); print(mk, json.dumps({k: r[k] for k in ('n','per_week','win','pf','pf_IS','pf_OOS','pf_2x_cost','maxDD_R','by_year','by_side','pf_opposite')}))
