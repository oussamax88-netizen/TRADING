"""H4: trade the zone break. Demand zone broken by a 5m close below it -> short at next 1m open (mirror for supply)."""
import sys, pickle, numpy as np, pandas as pd
from numba import njit
sys.path.insert(0, 'obchoch')
from engine import WIN_START, WIN_END, FORCE_CLOSE
from research import sim_exit, NS
from evalr import summ, COST, pf

@njit(cache=True)
def find_break(start, zlo, exp_day, dayi, mod, is5, f5c):
    k = start
    n = len(dayi)
    while k < n:
        if dayi[k] > exp_day:
            return -1
        if is5[k] and f5c[k] < zlo:
            return k
        k += 1
    return -1

S = sys.argv[1]
rows = []
for mk in ('NQ', 'ES'):
    a, z = pickle.load(open(f'{S}/{mk}.pkl', 'rb'))
    dayi, mod = a['dayi'], a['mod']
    for zr in z.itertuples(index=False):
        s = zr.dir
        f5c = a['f5c'] if s == 1 else -a['f5c']
        zlo, zhi = (zr.zlo, zr.zhi) if s == 1 else (-zr.zhi, -zr.zlo)
        k = find_break(zr.start, zlo, dayi[zr.start] + {'30m': 5, '1h': 5, '4h': 10, '1d': 20}[zr.tf], dayi, mod, a['is5'], f5c)
        if k < 0 or mod[k] < WIN_START or mod[k] >= WIN_END or k + 1 >= len(dayi) or dayi[k + 1] != dayi[k]:
            continue
        atr = a['atr'][k]
        if np.isnan(atr):
            continue
        td = -s   # trade direction
        o, h, l, c = (a['o'], a['h'], a['l'], a['c']) if td == 1 else (-a['o'], -a['l'], -a['h'], -a['c'])
        e = k + 1; px = o[e]
        r = dict(mk=mk, tf=zr.tf, td=td, entry_k=e, entry_t=pd.Timestamp(a['tmin'][e] * NS), trend=td * a['trend_d'][k], mod=mod[k])
        # stops: other side of the zone (zone top for a short after demand break), or 0.2 ATR
        far = -zhi if td == 1 else zhi   # in trade space: for td=-1 (orig short) stop at orig zhi -> negated -zhi ... handle below
        stop_zone = (-zr.zhi if td == -1 else zr.zlo)  # trade-space: short -> -(zone high); long (supply broken) -> zone low
        for sn, st in (('zone', stop_zone), ('atr02', px - 0.2 * atr)):
            risk = px - st
            if risk <= 0:
                continue
            for tn, tg in (('2R', 2), ('EOD', 1e9)):
                xk, xp, why = sim_exit(e, px, st, px + tg * risk, 0.0, dayi, mod, o, h, l, c)
                r[f'R_{sn}_{tn}'] = (xp - px - COST[mk]) / risk
                r[f'x_{sn}_{tn}'] = xk
        rows.append(r)
t = pd.DataFrame(rows).sort_values('entry_k')
t = t[(t.entry_t >= '2023-10-01') & (t.entry_t < '2026-10-01')]
out = []
for mk in ('NQ', 'ES'):
    for filt, fm in (('all', lambda d: d.td != 0), ('with trend', lambda d: d.trend > 0), ('30m+1h', lambda d: d.tf.isin(['30m', '1h'])), ('4h+1d', lambda d: d.tf.isin(['4h', '1d']))):
        for ex in ('zone_2R', 'zone_EOD', 'atr02_2R', 'atr02_EOD'):
            s = t[(t.mk == mk) & fm(t)].dropna(subset=[f'R_{ex}']).copy(); s['R'] = s[f'R_{ex}']
            keep, last = [], -1
            for i, e, x in zip(s.index, s.entry_k, s[f'x_{ex}']):
                if e > last: keep.append(i); last = x
            r = summ(s.loc[keep]); r.update(mk=mk, f=filt, ex=ex); out.append(r)
print(pd.DataFrame(out).pivot_table(index=['f', 'ex'], columns='mk', values=['pf_IS', 'pf_OOS', 'wk'], sort=False).round(2).to_string())
