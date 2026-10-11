"""Shared scalping backtest engine (1-minute, New York time). ALL strategy tests must use this.

Usage in a strategy file:
    import sys; sys.path.insert(0, '/home/user/TRADING/scalp')
    from engine import load, simulate, full_report, signals_frame
    D = load('NQ')                       # dict of numpy arrays, see build_cache.py
    sig = signals_frame(k=..., d=...)    # k = index of the bar whose CLOSE triggers the signal
    tr = simulate('NQ', sig)             # trades, one position at a time
    rep = full_report('NQ', sig)         # headline + IS/OOS + years + 2x cost + bootstrap + controls

Engine rules (non-negotiable, already implemented):
  * A signal at bar k may only use data up to the CLOSE of bar k. Entry happens on bar k+1 or later.
  * entry types: 0 = market at open of k+1; 1 = limit at eprice valid ebars bars; 2 = stop-entry at eprice valid ebars bars.
    Limit/stop fills: if the bar opens beyond the price -> fill at the open, else at the price when the bar touches it.
  * Stop and target may be given in points from the fill (stop_pts / tgt_pts) or as absolute prices (stop_px / tgt_px).
  * Fill bar: only the stop can be hit (no target fill on the entry bar).
  * Later bars: day change -> exit previous close; NY minute >= close_mod (15:59) -> exit at open;
    hold >= maxbars bars -> exit at open; open beyond stop -> exit at open (gap); stop touched -> stop;
    target touched -> target. Stop and target in the same bar = STOP. be_pts > 0: after a bar closes with
    favourable excursion >= be_pts, the stop moves to entry.
  * Cost per round trip in points (default NQ 1.5, ES 0.6) is subtracted from every trade.
  * One position at a time per market: a signal is skipped if its entry bar is not after the previous exit bar.
  * Results reported only for entries 2023-10-01 .. 2026-09-30. In-sample = before 2025-04-01, out-of-sample = after.
"""
import numpy as np
import pandas as pd
from numba import njit

S = '/tmp/claude-0/-home-user-TRADING/b9cb08f9-c0e5-5011-839c-6894039a978b/scratchpad'
COST = {'NQ': 1.5, 'ES': 0.6}
START = np.datetime64('2023-10-01'); END = np.datetime64('2026-10-01'); IS_END = np.datetime64('2025-04-01')
WEEKS = 156.4
_CACHE = {}


def load(mk):
    if mk not in _CACHE:
        z = np.load(f'{S}/scalp_{mk}.npz')
        _CACHE[mk] = {k: z[k] for k in z.files}
    return _CACHE[mk]


def signals_frame(k, d, etype=0, eprice=np.nan, ebars=0, stop_pts=np.nan, tgt_pts=np.inf, stop_px=np.nan, tgt_px=np.nan,
                  maxbars=0, be_pts=0.0):
    """Build a signal table. Scalars broadcast. k must be int bar indices, d = +1 long / -1 short."""
    k = np.asarray(k, dtype=np.int64)
    n = len(k)
    f = lambda x, dt=np.float64: np.broadcast_to(np.asarray(x, dtype=dt), (n,)).copy()
    sig = pd.DataFrame({'k': k, 'd': f(d, np.int64), 'etype': f(etype, np.int64), 'eprice': f(eprice), 'ebars': f(ebars, np.int64),
                        'stop_pts': f(stop_pts), 'tgt_pts': f(tgt_pts), 'stop_px': f(stop_px), 'tgt_px': f(tgt_px),
                        'maxbars': f(maxbars, np.int64), 'be_pts': f(be_pts)})
    return sig.sort_values('k', kind='stable').reset_index(drop=True)


@njit(cache=True)
def _sim(o, h, l, c, mod, dayi, sk, sd, et, ep, eb, sp, tp, spx, tpx, mb, be, cost, close_mod, last_entry_mod):
    n = len(o); ns = len(sk)
    out = np.full((ns, 7), np.nan)   # entry_k, entry_px, stop, target, exit_k, exit_px, reason
    last_exit = -1
    for i in range(ns):
        k = sk[i]; d = sd[i]
        if k + 1 >= n or k + 1 <= last_exit or dayi[k + 1] != dayi[k]:
            continue
        e = -1; px = 0.0
        if et[i] == 0:
            e = k + 1; px = o[e]
        else:
            m = k + 1
            while m < n and m <= k + eb[i] and dayi[m] == dayi[k] and mod[m] < last_entry_mod:
                if et[i] == 1:   # limit
                    if (d == 1 and o[m] <= ep[i]) or (d == -1 and o[m] >= ep[i]):
                        e = m; px = o[m]; break
                    if (d == 1 and l[m] <= ep[i]) or (d == -1 and h[m] >= ep[i]):
                        e = m; px = ep[i]; break
                else:            # stop entry
                    if (d == 1 and o[m] >= ep[i]) or (d == -1 and o[m] <= ep[i]):
                        e = m; px = o[m]; break
                    if (d == 1 and h[m] >= ep[i]) or (d == -1 and l[m] <= ep[i]):
                        e = m; px = ep[i]; break
                m += 1
        if e < 0 or mod[e] >= last_entry_mod:
            continue
        st = spx[i] if not np.isnan(spx[i]) else px - d * sp[i]
        tg = tpx[i] if not np.isnan(tpx[i]) else px + d * tp[i]
        risk = d * (px - st)
        if not (risk > 0) or not (d * (tg - px) > 0):
            continue
        # fill bar: only the stop
        if (d == 1 and l[e] <= st) or (d == -1 and h[e] >= st):
            out[i, 0] = e; out[i, 1] = px; out[i, 2] = st; out[i, 3] = tg; out[i, 4] = e; out[i, 5] = st; out[i, 6] = 1
            last_exit = e
            continue
        cur = st
        if be[i] > 0 and d * ((h[e] if d == 1 else l[e]) - px) >= be[i]:
            cur = px
        m = e + 1
        xk = -1; xp = 0.0; why = 0
        while m < n:
            if dayi[m] != dayi[e]:
                xk = m - 1; xp = c[m - 1]; why = 3; break
            if mod[m] >= close_mod:
                xk = m; xp = o[m]; why = 3; break
            if mb[i] > 0 and m - e >= mb[i]:
                xk = m; xp = o[m]; why = 4; break
            if (d == 1 and o[m] <= cur) or (d == -1 and o[m] >= cur):
                xk = m; xp = o[m]; why = 1; break
            if (d == 1 and l[m] <= cur) or (d == -1 and h[m] >= cur):
                xk = m; xp = cur; why = 1; break
            if (d == 1 and h[m] >= tg) or (d == -1 and l[m] <= tg):
                xk = m; xp = tg; why = 2; break
            if be[i] > 0 and d * ((h[m] if d == 1 else l[m]) - px) >= be[i]:
                if (d == 1 and px > cur) or (d == -1 and px < cur):
                    cur = px
            m += 1
        if xk < 0:
            xk = n - 1; xp = c[n - 1]; why = 3
        out[i, 0] = e; out[i, 1] = px; out[i, 2] = st; out[i, 3] = tg; out[i, 4] = xk; out[i, 5] = xp; out[i, 6] = why
        last_exit = xk
    return out


def simulate(mk, sig, cost=None, costmult=1.0, close_mod=959, last_entry_mod=955, period=True):
    D = load(mk)
    cost = (COST[mk] if cost is None else cost) * costmult
    s = sig.sort_values('k', kind='stable').reset_index(drop=True)
    out = _sim(D['o'], D['h'], D['l'], D['c'], D['mod'], D['dayi'], s.k.to_numpy(np.int64), s.d.to_numpy(np.int64),
               s.etype.to_numpy(np.int64), s.eprice.to_numpy(np.float64), s.ebars.to_numpy(np.int64), s.stop_pts.to_numpy(np.float64),
               s.tgt_pts.to_numpy(np.float64), s.stop_px.to_numpy(np.float64), s.tgt_px.to_numpy(np.float64),
               s.maxbars.to_numpy(np.int64), s.be_pts.to_numpy(np.float64), float(cost), int(close_mod), int(last_entry_mod))
    tr = pd.DataFrame(out, columns=['entry_k', 'entry', 'stop', 'target', 'exit_k', 'exit', 'reason'])
    tr['d'] = s.d.to_numpy(); tr['k'] = s.k.to_numpy()
    tr = tr.dropna(subset=['entry_k']).copy()
    for col in ('entry_k', 'exit_k', 'reason'):
        tr[col] = tr[col].astype(np.int64)
    tr['entry_t'] = pd.to_datetime(D['t'][tr.entry_k] * 60_000_000_000)
    tr['risk'] = tr.d * (tr.entry - tr.stop)
    tr['pts'] = tr.d * (tr.exit - tr.entry) - cost
    tr['R'] = tr.pts / tr.risk
    tr['hold'] = tr.exit_k - tr.entry_k
    if period:
        tr = tr[(tr.entry_t >= START) & (tr.entry_t < END)]
    return tr.reset_index(drop=True)


def pf(x):
    x = np.asarray(x, dtype=float)
    lo = -x[x <= 0].sum()
    return float(x[x > 0].sum() / lo) if lo > 0 else float('nan')


def headline(tr):
    if len(tr) == 0:
        return dict(n=0)
    R = tr.R.to_numpy(); eq = np.cumsum(R); dd = float((eq - np.maximum.accumulate(eq)).min())
    los = (R <= 0).astype(int); streak = 0; cur = 0
    for v in los:
        cur = cur + 1 if v else 0; streak = max(streak, cur)
    i = tr[tr.entry_t < IS_END]; o = tr[tr.entry_t >= IS_END]
    return dict(n=len(tr), per_week=round(len(tr) / WEEKS, 2), win=round(float((R > 0).mean() * 100), 1),
                avgW=round(float(R[R > 0].mean()), 2) if (R > 0).any() else None, avgL=round(float(R[R <= 0].mean()), 2) if (R <= 0).any() else None,
                pf=round(pf(R), 3), netR=round(float(R.sum()), 1), maxDD_R=round(dd, 1), lose_streak=int(streak),
                pf_IS=round(pf(i.R), 3), n_IS=len(i), pf_OOS=round(pf(o.R), 3), n_OOS=len(o),
                avg_pts=round(float(tr.pts.mean()), 2), med_risk_pts=round(float(tr.risk.median()), 2), med_hold_min=float(tr.hold.median()))


def full_report(mk, sig, n_random=30, seed=0, **kw):
    """Headline + by year/side + 2x cost + bootstrap + controls (opposite direction, random direction)."""
    rng = np.random.default_rng(seed)
    tr = simulate(mk, sig, **kw)
    rep = headline(tr)
    if len(tr) == 0:
        return rep
    rep['by_year'] = {int(y): dict(n=len(g), pf=round(pf(g.R), 2)) for y, g in tr.groupby(tr.entry_t.dt.year)}
    rep['by_side'] = {('long' if s == 1 else 'short'): dict(n=len(g), pf=round(pf(g.R), 2)) for s, g in tr.groupby('d')}
    rep['pf_2x_cost'] = round(pf(simulate(mk, sig, costmult=2.0, **kw).R), 3)
    rep['pf_no_cost'] = round(pf(simulate(mk, sig, cost=0.0, **kw).R), 3)
    R = tr.R.to_numpy()
    bs = np.array([pf(rng.choice(R, len(R))) for _ in range(2000)])
    rep['boot_pf_5_50_95'] = [round(float(x), 2) for x in np.nanpercentile(bs, [5, 50, 95])]
    rep['p_pf_below_1'] = round(float(np.mean(bs < 1)), 3)
    opp = sig.copy(); opp['d'] = -opp.d
    # mirrored absolute levels make no sense for the opposite trade: convert to points around the signal close
    if opp.stop_px.notna().any() or opp.tgt_px.notna().any():
        rep['pf_opposite'] = None
    else:
        rep['pf_opposite'] = round(pf(simulate(mk, opp, **kw).R), 3)
    if sig.stop_px.isna().all() and sig.tgt_px.isna().all() and (sig.etype == 0).all():
        rr = []
        for _ in range(n_random):
            x = sig.copy(); x['d'] = np.where(rng.random(len(x)) < 0.5, 1, -1); rr.append(pf(simulate(mk, x, **kw).R))
        rep['pf_random_dir_median'] = round(float(np.nanmedian(rr)), 3); rep['pf_random_dir_95'] = round(float(np.nanpercentile(rr, 95)), 3)
    return rep
