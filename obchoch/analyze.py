import sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, 'obchoch')
from engine import run_variant
S = sys.argv[1]
pd.set_option('display.width', 200)
def pf(x): return round(x[x > 0].sum() / -x[x <= 0].sum(), 2) if (x <= 0).any() else np.inf
def tab(t, by):
    return t.groupby(by).agg(n=('R', 'size'), win=('R', lambda x: round((x > 0).mean()*100)), pf=('R', pf), netR=('R', lambda x: round(x.sum(), 1)))
for mk, cost in (('NQ', 1.5), ('ES', 0.6)):
    g = pd.read_pickle(f'{S}/{mk}_grid.pkl')
    print(f'\n######## {mk}')
    for f in ('choch', 'entry', 'trend', 'big', 'life', 'buf'):
        print(f, g.groupby(f).pf.median().round(3).to_dict())
    print(g.sort_values('pf', ascending=False).head(5)[['choch','entry','trend','big','life','buf','n','perwk','pf','pf_IS','pf_OOS','pf_2xcost']].round(2).to_string())
    a, z = pickle.load(open(f'{S}/{mk}.pkl', 'rb'))
    p = dict(N=3, deadline=0, entry=0, trend='D', big=1, life='A', bufk=0.0)
    t = run_variant(a, z, p, cost); t = t[(t.entry_t >= '2023-10-01') & (t.entry_t < '2026-10-01')].copy()
    t['gross'] = (t.pts + cost) / t.risk
    t['hour'] = t.entry_t.dt.hour; t['year'] = t.entry_t.dt.year
    t['with_trend'] = t.rr == 2
    t['wd'] = t.entry_t.dt.day_name().str[:3]
    print('default: PF net', pf(t.R), ' PF before costs', pf(t.gross), ' median risk pts', t.risk.median(), ' cost as % of risk', round(cost / t.risk.median() * 100, 1))
    for by in ('year', 'side', 'with_trend', 'tf', 'hour', 'wd'):
        print(tab(t, by).T.to_string()); print()
    t.to_pickle(f'{S}/{mk}_default.pkl')
