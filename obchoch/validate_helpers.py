import numpy as np, pandas as pd
from evalr import pf
def full(s):
    s = s.sort_values('entry_t'); R = s.R
    eq = R.cumsum(); dd = (eq - eq.cummax()).min()
    los = (R <= 0).astype(int); streak = los.groupby((los != los.shift()).cumsum()).cumsum().max()
    m = s.groupby(s.entry_t.dt.to_period('M')).R.sum()
    return dict(n=len(s), wk=round(len(s) / 156.4, 2), win=round((R > 0).mean() * 100), avgW=round(R[R > 0].mean(), 2), avgL=round(R[R <= 0].mean(), 2),
                pf=round(pf(R), 2), netR=round(R.sum(), 1), maxDD=round(dd, 1), streak=int(streak), losing_months=round((m < 0).mean() * 100))
