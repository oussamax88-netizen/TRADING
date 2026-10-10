import sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, 'obchoch')
from engine import run_variant
S = sys.argv[1]
def pf(x): return round(x[x > 0].sum() / -x[x <= 0].sum(), 2)
for mk, cost in (('NQ', 1.5), ('ES', 0.6)):
    a, z = pickle.load(open(f'{S}/{mk}.pkl', 'rb'))
    for dep in (0.0, 1.0, 2.0, 3.0):
        res = []
        for N, dl, ent, tr, big in [(3,0,0,'D',1),(5,0,1,'D',2),(5,0,1,'D',0),(3,0,1,'H',1),(3,30,0,'D',0)]:
            t = run_variant(a, z, dict(N=N, deadline=dl, entry=ent, trend=tr, big=big, life='A', bufk=0.0, depk=dep), cost)
            t = t[(t.entry_t >= '2023-10-01') & (t.entry_t < '2026-10-01')]
            res.append((len(t), pf(t.R), pf((t.pts + cost) / t.risk)))
        print(mk, 'depart', dep, 'x zone | (trades, PF net, PF gross) for 5 variants:', res)
