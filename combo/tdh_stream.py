"""Bar-by-bar TDH simulation with the exact structure of the MQL5 EA (TDH_NQ.mq5). Compare with combo/tdh.py."""
import sys; sys.path.insert(0, '/home/user/TRADING/scalp'); sys.path.insert(0, '/home/user/TRADING/combo')
import numpy as np, pandas as pd
from engine import load, simulate, pf
from tdh import signals
D = load('NQ'); t, o, h, l, c = D['t'], D['o'], D['h'], D['l'], D['c']; n = len(t)
cur = None; rth = None; rngs, cls = [], []; atr = np.nan; trend = 0; pdc = np.nan
op = np.nan; orh = orl = None; decided = False; flat_ok = False
sig = []
for k in range(n):
    mod = t[k] % 1440; date = t[k] // 1440
    if date != cur:
        if rth is not None: rngs.append(rth[0] - rth[1]); cls.append(rth[2])
        rth = None; cur = date
        atr = np.mean(rngs[-14:]) if len(rngs) >= 14 else np.nan
        trend = int(np.sign(cls[-1] - np.mean(cls[-20:]))) if len(cls) >= 20 else 0
        pdc = cls[-1] if cls else np.nan
        op = np.nan; orh = orl = None; decided = False; flat_ok = False
    if 570 <= mod < 960:
        rth = [h[k], l[k], c[k]] if rth is None else [max(rth[0], h[k]), min(rth[1], l[k]), c[k]]
    if mod == 570:
        op = o[k]; flat_ok = (not np.isnan(atr)) and (not np.isnan(pdc)) and abs(op - pdc) >= 0.15 * atr
    if 570 <= mod < 580 and not np.isnan(op):
        orh = h[k] if orh is None else max(orh, h[k]); orl = l[k] if orl is None else min(orl, l[k])
    if 580 <= mod < 930 and not decided and not np.isnan(op) and orh is not None:
        up = c[k] > orh and c[k] - op >= 0.15 * atr; dn = c[k] < orl and op - c[k] >= 0.15 * atr
        if up or dn:
            decided = True; d = 1 if up else -1
            if flat_ok and d == trend: sig.append((k, d, 0.2 * atr))
s_ref = signals('NQ')
ks = np.array([x[0] for x in sig])
pass
ref = set(s_ref.k.to_numpy()); mine = set(ks)
miss = sorted(ref - mine); extra = sorted(mine - ref)
print('only research', len(miss), 'only stream', len(extra))
for k in miss[:6] + extra[:6]:
    dd = t[k] // 1440; i0 = np.where((t // 1440 == dd) & (t % 1440 == 570))[0]
    print(pd.Timestamp(t[k] * 60 * 10**9), 'research' if k in ref else 'stream', 'cache atr', round(D['atr'][k], 1), 'trend', D['trend_d'][k], 'pdc', D['pdc'][k])
