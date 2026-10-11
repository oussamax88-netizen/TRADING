import sys, numpy as np, pandas as pd
sys.path.insert(0, 'obchoch')
from evalr import trades, summ
S = sys.argv[1]; pd.set_option('display.width', 250)
T = {mk: pd.read_pickle(f'{S}/{mk}_table_N5.pkl') for mk in ('NQ', 'ES')}
rows = []
for lo, hi in ((240, 600), (180, 600), (300, 600), (240, 720), (240, 480), (180, 720), (120, 900), (360, 900)):
    for ex in ('zone_TR', 'zone_2R', 'zone_EOD', 'atr02_EOD'):
        st, tg = ex.split('_')
        for mk, t in T.items():
            r = summ(trades(t, mk, (t.age_min > lo) & (t.age_min <= hi), st, tg)); r.update(age=f'{lo/60:.0f}-{hi/60:.0f}h', ex=ex, mk=mk); rows.append(r)
d = pd.DataFrame(rows)
print(d.pivot_table(index=['age', 'ex'], columns='mk', values=['pf_IS', 'pf_OOS', 'wk'], sort=False).round(2).to_string())
