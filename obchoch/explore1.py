import sys, numpy as np, pandas as pd
sys.path.insert(0, 'obchoch')
from evalr import trades, summ, pf, IS_END
S = sys.argv[1]
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
T = {mk: pd.read_pickle(f'{S}/{mk}_table_N5.pkl') for mk in ('NQ', 'ES')}
# 1. exit styles, no filter
print('=== exit style x stop, all signals (one position at a time). PF all | IS | OOS')
rows = []
for mk, t in T.items():
    for st in ('zone', 'swing', 'atr02'):
        for tg in ('1R', '2R', '3R', 'EOD', 'EODbe', 'TR'):
            s = trades(t, mk, t.side != 0, st, tg); r = summ(s); r.update(mk=mk, stop=st, tgt=tg); rows.append(r)
print(pd.DataFrame(rows).pivot_table(index=['stop', 'tgt'], columns='mk', values=['pf', 'pf_IS', 'pf_OOS', 'wk']).round(2).to_string())
