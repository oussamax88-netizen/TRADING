import sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, 'obchoch')
from evalr import trades, summ, pf, IS_END
from validate_helpers import full
S = sys.argv[1]; pd.set_option('display.width', 250)
rng = np.random.default_rng(3)
def asia(t, c0=1080, c1=180, e1=570): return ((t.cmod >= c0) | (t.cmod <= c1)) & (t['mod'] < e1)
P = []
for N in (5, 3):
    for mk in ('NQ', 'ES'):
        t = pd.read_pickle(f'{S}/{mk}_table_N{N}.pkl')
        for tn, tm in (('all', t.side != 0), ('with trend', t.trend_d > 0), ('with 1h trend', t.trend_h > 0)):
            for st in ('zone', 'atr02'):
                for tg in ('EOD', 'EODbe', '3R'):
                    s = trades(t, mk, asia(t) & tm, st, tg); r = summ(s); r.update(N=N, mk=mk, trend=tn, ex=f'{st}/{tg}'); P.append(r)
d = pd.DataFrame(P)
print(d.pivot_table(index=['trend', 'ex'], columns=['N', 'mk'], values=['pf_IS', 'pf_OOS'], sort=False).round(2).to_string())
print(d.pivot_table(index=['trend'], columns=['N', 'mk'], values='wk', aggfunc='mean', sort=False).round(2).to_string())
# neighbour family for with-trend
fam = []
for mk in ('NQ', 'ES'):
    for N in (3, 5):
        t = pd.read_pickle(f'{S}/{mk}_table_N{N}.pkl')
        for c0, c1, e1 in ((1080, 180, 570), (1020, 180, 570), (1140, 180, 570), (1080, 120, 570), (1080, 240, 570), (1080, 180, 480), (1080, 180, 660)):
            for st in ('zone', 'atr02', 'swing'):
                for tg in ('EOD', 'EODbe'):
                    s = trades(t, mk, asia(t, c0, c1, e1) & (t.trend_d > 0), st, tg); fam.append(dict(mk=mk, pf=pf(s.R), n=len(s)))
fam = pd.DataFrame(fam); print('with-trend family: variants', len(fam), 'median PF by market', fam.groupby('mk').pf.median().round(2).to_dict(), 'min', fam.groupby('mk').pf.min().round(2).to_dict())
