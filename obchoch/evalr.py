import numpy as np, pandas as pd
COST = {'NQ': 1.5, 'ES': 0.6}
IS_END = pd.Timestamp('2025-04-01'); START = pd.Timestamp('2023-10-01'); END = pd.Timestamp('2026-10-01')

def pf(x):
    x = np.asarray(x); l = -x[x <= 0].sum()
    return x[x > 0].sum() / l if l > 0 else np.nan

def trades(t, mk, mask, stop='zone', tgt='2R', overlap=True, costmult=1.0):
    """Apply mask, compute net R, one position at a time."""
    s = t[mask & (t.entry_t >= START) & (t.entry_t < END)].copy()
    if tgt == 'TR':   # 2R with daily trend, 1R against (user's rule)
        s['g'] = np.where(s.trend_d > 0, s[f'g_{stop}_2R'], s[f'g_{stop}_1R'])
        s['x'] = np.where(s.trend_d > 0, s[f'x_{stop}_2R'], s[f'x_{stop}_1R'])
    else:
        s['g'] = s[f'g_{stop}_{tgt}']; s['x'] = s[f'x_{stop}_{tgt}']
    s = s[s[f'risk_{stop}'].notna()]
    s['R'] = s.g - costmult * COST[mk] / s[f'risk_{stop}']
    if overlap and len(s):
        keep, last = [], -1
        for i, e, x in zip(s.index, s.entry_k, s.x):
            if e > last:
                keep.append(i); last = x
        s = s.loc[keep]
    return s

def summ(s):
    if len(s) == 0:
        return dict(n=0)
    i, o = s[s.entry_t < IS_END], s[s.entry_t >= IS_END]
    return dict(n=len(s), wk=round(len(s) / 156.4, 2), pf=round(pf(s.R), 2), pf_IS=round(pf(i.R), 2), n_IS=len(i),
                pf_OOS=round(pf(o.R), 2), n_OOS=len(o), netR=round(s.R.sum(), 1), win=round((s.R > 0).mean() * 100))
