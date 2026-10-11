"""Legit 'no CHOCH' version: limit order at the zone edge on the first touch (not conditioned on any later bounce)."""
import sys, pickle, numpy as np, pandas as pd
from numba import njit
sys.path.insert(0, 'obchoch')
from research import sim_exit, NS
from evalr import pf, IS_END, COST, summ

@njit(cache=True)
def find_touch(start, zlo, zhi, exp_day, mod, dayi, l, is5, f5c, w0, w1):
    k = start
    while k < len(l):
        if dayi[k] > exp_day:
            return -1
        if l[k] <= zhi:
            if mod[k] < w0 or mod[k] >= w1:
                return -1
            return k
        if is5[k] and f5c[k] < zlo:
            return -1
        k += 1
    return -1

def build(S, mk, w1=570):
    a, z = pickle.load(open(f'{S}/{mk}.pkl', 'rb'))
    rows = []
    for zr in z.itertuples(index=False):
        s = zr.dir
        if s == 1: o, h, l, c, f5c = a['o'], a['h'], a['l'], a['c'], a['f5c']; zlo, zhi = zr.zlo, zr.zhi
        else:      o, h, l, c, f5c = -a['o'], -a['l'], -a['h'], -a['c'], -a['f5c']; zlo, zhi = -zr.zhi, -zr.zlo
        life = {'30m': 5, '1h': 5, '4h': 10, '1d': 20}[zr.tf]
        k = find_touch(zr.start, zlo, zhi, a['dayi'][zr.start] + life, a['mod'], a['dayi'], l, a['is5'], f5c, 180, w1)
        if k < 0:
            continue
        cm = pd.Timestamp(zr.created); r = dict(side=s, tf=zr.tf, cmod=cm.hour * 60 + cm.minute, entry_k=k,
                                               entry_t=pd.Timestamp(a['tmin'][k] * NS), mod=a['mod'][k], trend_d=s * a['trend_d'][k])
        for en, px in (('edge', min(o[k], zhi)), ('mid', min(o[k], (zlo + zhi) / 2))):
            if en == 'mid' and l[k] > (zlo + zhi) / 2:
                r['R_mid'] = np.nan; continue
            risk = px - zlo
            if risk <= 0:
                r[f'R_{en}'] = np.nan; continue
            xk, xp, _ = sim_exit(k, px, zlo, px + 1e9, 0.0, a['dayi'], a['mod'], o, h, l, c)
            r[f'R_{en}'] = (xp - px - COST[mk]) / risk; r[f'x_{en}'] = xk
        rows.append(r)
    return pd.DataFrame(rows).sort_values('entry_k')

S = sys.argv[1]
for mk in ('NQ', 'ES'):
    t = build(S, mk)
    t = t[(t.entry_t >= '2023-10-01') & (t.entry_t < '2026-10-01')]
    asia = (t.cmod >= 1080) | (t.cmod <= 180)
    for nm, m in (('Asia zones', asia), ('all zones', t.side != 0)):
        for en in ('edge', 'mid'):
            s = t[m].dropna(subset=[f'R_{en}']); s = s.assign(R=s[f'R_{en}'])
            keep, last = [], -1
            for i, e, x in zip(s.index, s.entry_k, s[f'x_{en}']):
                if e > last: keep.append(i); last = x
            print(mk, nm, 'limit at', en, summ(s.loc[keep]))
