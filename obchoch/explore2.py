import sys, numpy as np, pandas as pd
sys.path.insert(0, 'obchoch')
from evalr import trades, pf, IS_END, COST
S = sys.argv[1]; EX = sys.argv[2] if len(sys.argv) > 2 else 'zone_EOD'
st, tg = EX.split('_')
pd.set_option('display.width', 250)
T = {mk: pd.read_pickle(f'{S}/{mk}_table_N5.pkl') for mk in ('NQ', 'ES')}
FEAT = ['pcm', 'zone_h', 'age_min', 'away', 'pen', 'choch_min', 'mod', 'ext', 'loc_pd', 'gap', 'day_move', 'risk_atr']
CAT = ['side', 'tf', 'trend_d', 'trend_h', 'tdh', 'dow']
out = []
for mk, t in T.items():
    t = t.copy(); t['risk_atr'] = t[f'risk_{st}'] / t.atr
    s = trades(t, mk, t.side != 0, st, tg, overlap=False)
    s['half'] = np.where(s.entry_t < IS_END, 'IS', 'OOS')
    for f in FEAT + CAT:
        if f in CAT:
            s['b'] = s[f].astype(str)
        else:
            qs = s.loc[s.half == 'IS', f].quantile([1/3, 2/3]).to_numpy()
            s['b'] = np.select([s[f] <= qs[0], s[f] <= qs[1], s[f] > qs[1]], ['lo', 'mid', 'hi'], 'nan')
        for (b, h), g in s.groupby(['b', 'half']):
            out.append(dict(mk=mk, f=f, b=b, half=h, n=len(g), pf=pf(g.R)))
o = pd.DataFrame(out).pivot_table(index=['f', 'b'], columns=['mk', 'half'], values='pf').round(2)
n = pd.DataFrame(out).pivot_table(index=['f', 'b'], columns=['mk', 'half'], values='n', aggfunc='sum')
o['minN'] = n.min(axis=1)
o['allgt1.1'] = (o[[c for c in o.columns if c[0] in ('NQ', 'ES')]] > 1.1).all(axis=1)
print(f'Exit {EX}. PF per bucket (signals not overlap-filtered). IS buckets fixed on IS terciles.')
print(o.to_string())
