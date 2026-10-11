"""Independent re-implementation + robustness checks of trend_gap_fill (skeptic pass). Built from the rule text only.

Rules: trading days = dates with a 09:30 bar. R = close of previous trading date's 16:14 bar (else cache pdc).
g = o[09:30]-R; gap filter |g| >= gmin*ATR (cache atr at 09:30). d = -sign(g); keep only d == trend (cache trend_d
at 09:30, or own SMA-n trend for the neighbour test). Signal bar 09:59, market entry 10:00 open. Stop A = 09:30..09:59
extreme opposite to d measured from c[09:59] + buf*ATR, floored at max(fl*ATR, MIN). Exit: hold to 15:59 force close.
"""
import sys, json
sys.path.insert(0, '/home/user/TRADING/scalp')
import numpy as np
import pandas as pd
from engine import load, signals_frame, simulate, pf, IS_END

MIN = {'NQ': 20.0, 'ES': 4.0}
BASE = dict(gmin=0.1, buf=0.02, fl=0.12, minmul=1.0, sigmod=599, exmod=None, trend_n=20, ref='1614')
_IX = {}


def ix(mk):
    """(dayi*1440+mod) -> bar index, plus per-date RTH close series for own trend."""
    if mk not in _IX:
        D = load(mk)
        key = D['dayi'] * 1440 + D['mod']
        s = pd.Series(np.arange(len(key)), index=key)
        s = s[~s.index.duplicated()]
        rth = (D['mod'] >= 570) & (D['mod'] < 960)
        cl = pd.Series(D['c'][rth]).groupby(D['dayi'][rth]).last()
        _IX[mk] = (s, cl)
    return _IX[mk]


def table(mk, p):
    D = load(mk); s, cl = ix(mk)
    o, h, l, c = D['o'], D['h'], D['l'], D['c']
    days = np.unique(D['dayi'][D['mod'] == 570])
    own = p['trend_n'] != 20 or p.get('own', False)
    if own:
        tr_own = np.sign(cl - cl.rolling(p['trend_n']).mean()).shift(1)   # previous RTH days only
    rows = []
    for j in range(1, len(days)):
        dy, pv = int(days[j]), int(days[j - 1])
        k0 = int(s[dy * 1440 + 570]); a = D['atr'][k0]
        if np.isnan(a):
            continue
        r_key = pv * 1440 + 974
        R = c[int(s[r_key])] if (p['ref'] == '1614' and r_key in s.index) else D['pdc'][k0]
        if np.isnan(R):
            continue
        g = o[k0] - R
        if g == 0 or abs(g) < p['gmin'] * a:
            continue
        kk = dy * 1440 + p['sigmod']
        if kk not in s.index:
            continue
        k = int(s[kk])
        trd = tr_own.get(dy, np.nan) if own else D['trend_d'][k0]
        if np.isnan(trd) or trd == 0:
            continue
        d = -int(np.sign(g))
        if d != int(trd):
            continue
        ext = (c[k] - l[k0:k + 1].min()) if d == 1 else (h[k0:k + 1].max() - c[k])
        st = max(ext + p['buf'] * a, p['fl'] * a, MIN[mk] * p['minmul'])
        mb = 0
        if p['exmod'] is not None:
            ke = s.get(dy * 1440 + p['exmod'], -1)
            mb = int(ke) - (k + 1) if ke > k + 1 else 0
        rows.append((dy, k, d, st, mb, a))
    return pd.DataFrame(rows, columns=['dayi', 'k', 'd', 'st', 'mb', 'atr'])


def run(mk, t, cost=1.0):
    return simulate(mk, signals_frame(t.k.to_numpy(), t.d.to_numpy(), stop_pts=t.st.to_numpy(), maxbars=t.mb.to_numpy()), costmult=cost)


def split(tr):
    i = tr[tr.entry_t < IS_END]; o = tr[tr.entry_t >= IS_END]
    return dict(n=len(tr), pf=round(pf(tr.R), 3), nIS=len(i), IS=round(pf(i.R), 3), nOOS=len(o), OOS=round(pf(o.R), 3))


def rand_time(mk, t, lo, hi, reps=20, seed=1):
    """Same days, same direction and stop pts, random signal minute in [lo, hi] (entry next bar), hold to close."""
    D = load(mk); rng = np.random.default_rng(seed); out = []
    starts = {}
    for dy in t.dayi:
        w = np.where((D['dayi'] == dy) & (D['mod'] >= lo) & (D['mod'] <= hi))[0] if dy not in starts else starts[dy]
        starts[dy] = w
    for _ in range(reps):
        ks = np.array([rng.choice(starts[dy]) for dy in t.dayi])
        x = t.assign(k=ks, mb=0)
        out.append(pf(run(mk, x).R))
    return np.array(out)


def rand_days(mk, n_target, reps=20, seed=2):
    """Same count of days drawn from ALL valid-trend days (no gap filter), trend-direction 10:00 entry, stop A, hold to close."""
    p = dict(BASE, gmin=0.0)
    D = load(mk)
    allt = table_trend_all(mk)
    rng = np.random.default_rng(seed); out = []
    for _ in range(reps):
        x = allt.iloc[np.sort(rng.choice(len(allt), n_target, replace=False))]
        out.append(pf(run(mk, x).R))
    return np.array(out), split(run(mk, allt))


def table_trend_all(mk):
    """All valid days, direction = trend (gap ignored), stop A for that direction."""
    D = load(mk); s, _ = ix(mk); o, h, l, c = D['o'], D['h'], D['l'], D['c']
    rows = []
    for dy in np.unique(D['dayi'][D['mod'] == 570]):
        dy = int(dy); k0 = int(s[dy * 1440 + 570]); a = D['atr'][k0]; trd = D['trend_d'][k0]
        if np.isnan(a) or np.isnan(trd) or trd == 0 or (dy * 1440 + 599) not in s.index:
            continue
        k = int(s[dy * 1440 + 599]); d = int(trd)
        ext = (c[k] - l[k0:k + 1].min()) if d == 1 else (h[k0:k + 1].max() - c[k])
        rows.append((dy, k, d, max(ext + 0.02 * a, 0.12 * a, MIN[mk]), 0, a))
    return pd.DataFrame(rows, columns=['dayi', 'k', 'd', 'st', 'mb', 'atr'])


NEIGH = {'gmin0.05': dict(gmin=0.05), 'gmin0.15': dict(gmin=0.15), 'gmin0.2': dict(gmin=0.2),
         'buf0.00': dict(buf=0.0), 'buf0.05': dict(buf=0.05), 'floor0.08': dict(fl=0.08), 'floor0.16': dict(fl=0.16),
         'min0.75x': dict(minmul=0.75), 'min1.25x': dict(minmul=1.25), 'sig09:44': dict(sigmod=584), 'sig10:14': dict(sigmod=614),
         'exit15:00': dict(exmod=900), 'exit15:30': dict(exmod=930), 'trendSMA10': dict(trend_n=10), 'trendSMA40': dict(trend_n=40),
         'ref_pdc': dict(ref='pdc')}

if __name__ == '__main__':
    res = {}
    for mk in ('NQ', 'ES'):
        r = {}
        t = table(mk, BASE); tr = run(mk, t)
        r['base'] = split(tr); r['base_2x'] = round(pf(run(mk, t, 2.0).R), 3)
        tr['yr'] = tr.entry_t.dt.year
        r['by_year'] = {int(y): (len(g), round(pf(g.R), 2)) for y, g in tr.groupby('yr')}
        # sanity: own SMA-20 trend vs cache trend_d
        r['own_sma20_check'] = split(run(mk, table(mk, dict(BASE, own=True))))  # own SMA20 path; should match
        nb = {}
        for nm, ch in NEIGH.items():
            nb[nm] = split(run(mk, table(mk, dict(BASE, **ch))))
        r['neigh'] = nb
        r['neigh_median_pf'] = round(float(np.median([v['pf'] for v in nb.values()])), 3)
        r['neigh_share_gt1.1'] = round(float(np.mean([v['pf'] > 1.1 for v in nb.values()])), 2)
        tt = tr[['k']].merge(t, on='k')           # traded rows only
        for nm, (lo, hi) in {'rt_0931_1530': (570, 929), 'rt_0945_1100': (584, 659)}.items():
            a = rand_time(mk, tt, lo, hi)
            r[nm] = dict(med=round(float(np.median(a)), 3), p95=round(float(np.percentile(a, 95)), 3), beat=bool(pf(tr.R) > np.percentile(a, 95)))
        a, allr = rand_days(mk, len(tt))
        r['rand_days_trend_dir'] = dict(med=round(float(np.median(a)), 3), p95=round(float(np.percentile(a, 95)), 3), beat=bool(pf(tr.R) > np.percentile(a, 95)), all_days=allr)
        res[mk] = r
        print(mk, json.dumps(r, default=str))
    json.dump(res, open('/home/user/TRADING/scalp/results/trend_gap_fill_verify.json', 'w'), default=str, indent=1)
