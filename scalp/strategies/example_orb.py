"""Example / template: 5-minute opening-range breakout (09:30-09:35 NY), first 1m close outside the range,
stop at the other side of the range (min 0.12 ATR), target 2R, max hold 60 minutes. Both markets."""
import sys, json
sys.path.insert(0, '/home/user/TRADING/scalp')
import numpy as np
from engine import load, signals_frame, full_report

def signals(mk, or_min=5, rr=2.0, maxbars=60):
    D = load(mk)
    mod, dayi, h, l, c, atr = D['mod'], D['dayi'], D['h'], D['l'], D['c'], D['atr']
    ks, ds, stops = [], [], []
    # group bar indices by day
    starts = np.r_[0, np.where(np.diff(dayi) != 0)[0] + 1]; ends = np.r_[starts[1:], len(dayi)]
    for a, b in zip(starts, ends):
        m = mod[a:b]
        orr = np.where((m >= 570) & (m < 570 + or_min))[0]
        if len(orr) < or_min - 1 or np.isnan(atr[a]):
            continue
        hi, lo = h[a + orr].max(), l[a + orr].min()
        after = np.where((m >= 570 + or_min) & (m < 660))[0]
        for j in after:                          # first close outside the range (known at that bar's close)
            k = a + j
            if c[k] > hi or c[k] < lo:
                d = 1 if c[k] > hi else -1
                stop = max(c[k] - lo if d == 1 else hi - c[k], 0.12 * atr[k])
                ks.append(k); ds.append(d); stops.append(stop); break
    stops = np.array(stops)
    return signals_frame(ks, ds, stop_pts=stops, tgt_pts=rr * stops, maxbars=maxbars)

if __name__ == '__main__':
    res = {}
    for mk in ('NQ', 'ES'):
        rep = full_report(mk, signals(mk))
        res[mk] = rep
        print(mk, json.dumps(rep, default=str))
    json.dump(res, open('/home/user/TRADING/scalp/results/example_orb.json', 'w'), default=str, indent=1)
