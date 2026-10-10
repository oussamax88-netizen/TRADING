import sys, pickle, time, numpy as np, pandas as pd
sys.path.insert(0, 'obchoch')
from engine import run_variant
S = sys.argv[1]
COST = {'NQ': 1.5, 'ES': 0.6}
def stats(t):
    w = t.R[t.R > 0]; ls = t.R[t.R <= 0]
    pf = w.sum() / -ls.sum() if len(ls) else np.inf
    weeks = (t.entry_t.max() - t.entry_t.min()).days / 7
    return dict(n=len(t), perwk=round(len(t)/weeks, 2), win=round((t.R > 0).mean()*100, 1), pf=round(pf, 2), netR=round(t.R.sum(), 1),
                avgW=round(w.mean(), 2), avgL=round(ls.mean(), 2), medrisk=round(t.risk.median(), 1))
for mk in ('NQ', 'ES'):
    a, z = pickle.load(open(f'{S}/{mk}.pkl', 'rb'))
    p = dict(N=3, deadline=0, entry=0, trend='D', big=1, life='A', bufk=0.0)
    t0 = time.time(); t = run_variant(a, z, p, COST[mk]); 
    t = t[(t.entry_t >= '2023-10-01') & (t.entry_t < '2026-10-01')]
    print(mk, round(time.time()-t0, 1), 's', stats(t))
    print(t.reason.value_counts().to_dict(), 'rr', t.rr.value_counts().to_dict())
    print(t[['entry_t','side','tf','zlo','zhi','entry','stop','target','exit','R','reason']].head(8).to_string())
    t.to_pickle(f'{S}/{mk}_default.pkl')
