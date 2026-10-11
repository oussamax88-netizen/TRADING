import sys, numpy as np, pandas as pd
sys.path.insert(0, 'obchoch')
from evalr import trades, summ
S = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 5
pd.set_option('display.width', 250)
T = {mk: pd.read_pickle(f'{S}/{mk}_table_N{N}.pkl') for mk in ('NQ', 'ES')}
H = {
 'base 30m+1h':        lambda t: t.tf.isin(['30m', '1h']),
 'H1 4h':              lambda t: t.tf == '4h',
 'H1 1d':              lambda t: t.tf == '1d',
 'H1 4h+1d':           lambda t: t.tf.isin(['4h', '1d']),
 'H2 sweep PDH/PDL':   lambda t: t.sweep_pd,
 'H2 zone beyond PD':  lambda t: t.zone_beyond_pd,
 'H3 FVG':             lambda t: t.fvg & t.tf.isin(['30m', '1h']),
 'H3 disp>=3':         lambda t: (t.disp >= 3) & t.tf.isin(['30m', '1h']),
 'H3 FVG+disp>=3':     lambda t: t.fvg & (t.disp >= 3),
 'H5 killzones':       lambda t: ((t['mod'] >= 180) & (t['mod'] < 300)) | ((t['mod'] >= 570) & (t['mod'] < 660)),
 'H5 NY 9:30-11':      lambda t: (t['mod'] >= 570) & (t['mod'] < 660),
 'H5 London 3-5':      lambda t: (t['mod'] >= 180) & (t['mod'] < 300),
 'with daily trend':   lambda t: t.trend_d > 0,
 'H2+trend':           lambda t: t.sweep_pd & (t.trend_d > 0),
 'H3 FVG+trend':       lambda t: t.fvg & (t.trend_d > 0),
}
rows = []
for name, f in H.items():
    for ex in ('zone_TR', 'zone_2R', 'zone_EOD', 'atr02_EOD'):
        st, tg = ex.split('_')
        for mk, t in T.items():
            r = summ(trades(t, mk, f(t), st, tg)); r.update(h=name, ex=ex, mk=mk); rows.append(r)
d = pd.DataFrame(rows)
p = d.pivot_table(index=['h', 'ex'], columns='mk', values=['pf_IS', 'pf_OOS', 'n', 'wk'], sort=False)
print(p.round(2).to_string())
