import sys, pickle, itertools, numpy as np, pandas as pd
sys.path.insert(0, 'obchoch')
from engine import run_variant
S, mk = sys.argv[1], sys.argv[2]
COST = {'NQ': 1.5, 'ES': 0.6}[mk]
a, z = pickle.load(open(f'{S}/{mk}.pkl', 'rb'))
CHOCH = {'A3': (3, 0), 'B5': (5, 0), 'C3_30m': (3, 30)}
rows = []
for ch, ent, tr, big, life, buf in itertools.product(CHOCH, (0, 1), ('H', 'D'), (0, 1, 2), ('A', 'B'), (0.0, 0.1)):
    N, dl = CHOCH[ch]
    p = dict(N=N, deadline=dl, entry=ent, trend=tr, big=big, life=life, bufk=buf)
    t = run_variant(a, z, p, COST)
    t = t[(t.entry_t >= '2023-10-01') & (t.entry_t < '2026-10-01')]
    t2 = run_variant(a, z, p, 2 * COST); t2 = t2[(t2.entry_t >= '2023-10-01') & (t2.entry_t < '2026-10-01')]
    def pf(x): return x[x > 0].sum() / -x[x <= 0].sum()
    ins = t[t.entry_t < '2025-04-01']; oos = t[t.entry_t >= '2025-04-01']
    rows.append(dict(choch=ch, entry='next_open' if ent == 0 else 'retest', trend=tr, big=['none', 'A', 'B'][big], life=life, buf=buf,
                     n=len(t), perwk=len(t) / 156.4, win=(t.R > 0).mean(), pf=pf(t.R), netR=t.R.sum(),
                     pf_IS=pf(ins.R), pf_OOS=pf(oos.R), pf_2xcost=pf(t2.R)))
g = pd.DataFrame(rows)
g.to_pickle(f'{S}/{mk}_grid.pkl')
print(mk, 'variants', len(g), 'median PF', round(g.pf.median(), 3), 'max PF', round(g.pf.max(), 3))
