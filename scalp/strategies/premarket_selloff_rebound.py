"""premarket_selloff_rebound -- Pre-market inventory rebound after a large prior-RTH sell-off (03:00 -> cash open).

Hypothesis (Boyarchenko, Larsen & Whelan, RFS 2023 "The overnight drift"): index-futures returns accrue overnight
around the European open as liquidity providers who absorbed end-of-day sell imbalances unwind inventory when
overseas buyers arrive. The effect is asymmetric: large sell-offs are followed by robust rebounds. The classic
02:00-03:00 NY window is outside the trader's hours (and has decayed since 2021), so we test whether the CONDITIONAL
rebound still pays 03:00 -> 09:30/10:00 NY.

Mechanical rules (as implemented):
  * RTH = bars with NY minute-of-day in [570, 960) (09:30..15:59). For every calendar date with RTH bars:
      op = open of its first RTH bar (09:30), cl = close of its last RTH bar (15:59 bar close; 12:59 on half days).
  * P(date) = cl - op of the PREVIOUS RTH date (completed session; Friday is the previous RTH date for Monday,
    holidays are skipped). Consistency check: the previous close equals the cache 'pdc' at the signal bar.
  * ATR = cache 'atr' at the signal bar (14-day mean RTH range of previous RTH days only).
  * Signal bar k = the bar with minute 179 (02:59) on a date that has an RTH session (trading calendar is known in
    advance); the entry bar k+1 must be on the same date. Everything used at k is <= the 02:59 close.
  * LONG if P <= -thr*ATR.  Both-sides variant: also SHORT if P >= +thr*ATR.
  * Entry: market at the open of bar k+1 (03:00), etype 0. stop_pts = max(stopk*ATR, 20 NQ / 4 ES). No target.
  * Time exit at the open of the first bar of the same date with minute >= T: maxbars = idx(T) - (k+1),
    T in {570 (09:30), 600 (10:00)}. (Engine exits at the open of bar e+maxbars, i.e. exactly the T bar open.)
  * One trade per day by construction.

Grid (24 variants, the selection budget): thr {0.4, 0.6, 0.8} x T {570, 600} x stopk {0.3, 0.5} x side {long, both}.
Selection rule (fixed BEFORE running, IS only): score = min(IS PF NQ, IS PF ES); plateau = mean score over the cell and
its thr-neighbours (same T, stopk, side); pick max plateau, tie-break by the cell's own min IS PF.

Mandatory controls (NOT counted toward the budget):
  * drift baseline: long EVERY RTH date 03:00 -> T with the same stop (does conditioning add beyond drift?)
  * reversed: same rules with d reversed (full_report pf_opposite; also per grid cell)
  * random-day control: PF of the chosen rule vs PF of random same-size subsets of the drift-baseline trades
  * random direction (full_report)
Diagnostics (not used for selection): sub-windows 03:00-06:00 / 06:00-09:30 / 09:30-10:00, sub-condition splits.
"""
import sys, json
sys.path.insert(0, '/home/user/TRADING/scalp')
import numpy as np
import pandas as pd
from engine import load, signals_frame, simulate, headline, full_report, pf, IS_END

MINSTOP = {'NQ': 20.0, 'ES': 4.0}
THRS = (0.4, 0.6, 0.8)
TS = (570, 600)
STOPKS = (0.3, 0.5)
SIDES = ('long', 'both')
OUT = '/home/user/TRADING/scalp/results/premarket_selloff_rebound.json'
IS_WEEKS = (np.datetime64('2025-04-01') - np.datetime64('2023-10-01')).astype(int) / 7.0
_F = {}


def features(mk, kmod=179):
    """Per RTH date: signal bar k (minute kmod), P = previous RTH day's close-open, ATR, bar indices of later minutes."""
    key = (mk, kmod)
    if key in _F:
        return _F[key]
    D = load(mk)
    mod, date, dayi = D['mod'], D['date'], D['dayi']
    rth = (mod >= 570) & (mod < 960)
    ri = np.where(rth)[0]
    r = pd.DataFrame({'date': date[ri], 'o': D['o'][ri], 'c': D['c'][ri]}).groupby('date').agg(op=('o', 'first'), cl=('c', 'last'))
    r['P'] = (r.cl - r.op).shift(1)          # previous RTH date only (completed)
    r['pcl'] = r.cl.shift(1)
    k = np.where(mod == kmod)[0]
    k = k[np.isin(date[k], r.index.to_numpy())]
    k = k[k + 1 < len(mod)]
    k = k[dayi[k + 1] == dayi[k]]
    P = r.P.reindex(date[k]).to_numpy()
    pcl = r.pcl.reindex(date[k]).to_numpy()
    atr = D['atr'][k]
    ok = np.isfinite(P) & np.isfinite(atr)
    chk = np.isfinite(pcl) & np.isfinite(D['pdc'][k])
    assert np.allclose(pcl[chk], D['pdc'][k][chk]), 'previous close mismatch with cache pdc'
    # bar index of the first bar on the same date with minute >= m (for time exits / sub-windows)
    first_ge = {}
    for m in (360, 570, 600):
        idx = np.full(len(k), -1, dtype=np.int64)
        for j, kk in enumerate(k):
            b = kk + 1
            while b < len(mod) and dayi[b] == dayi[kk] and mod[b] < m:
                b += 1
            if b < len(mod) and dayi[b] == dayi[kk]:
                idx[j] = b
        first_ge[m] = idx
    out = dict(D=D, k=k[ok], P=P[ok], atr=atr[ok], idx={m: v[ok] for m, v in first_ge.items()})
    _F[key] = out
    return out


def signals(mk, thr=0.6, T=570, stopk=0.3, side='long', mode='rule', kmod=179, mask_fn=None):
    """mode 'rule' = the hypothesis; 'drift' = long every RTH day (control)."""
    F = features(mk, kmod)
    k, P, atr, iT = F['k'], F['P'], F['atr'], F['idx'][T]
    mb = iT - (k + 1)
    ok = (iT > 0) & (mb > 0)
    if mode == 'drift':
        d = np.ones(len(k), dtype=np.int64)
    else:
        d = np.where(P <= -thr * atr, 1, 0)
        if side == 'both':
            d = np.where(P >= thr * atr, -1, d)
    ok &= d != 0
    if mask_fn is not None:
        ok &= mask_fn(F, k)
    stop = np.maximum(stopk * atr, MINSTOP[mk])
    return signals_frame(k[ok], d[ok], etype=0, stop_pts=stop[ok], maxbars=mb[ok])


def grid():
    rows = []
    for side in SIDES:
        for T in TS:
            for stopk in STOPKS:
                for thr in THRS:
                    r = dict(side=side, T=T, stopk=stopk, thr=thr)
                    for mk in ('NQ', 'ES'):
                        sig = signals(mk, thr, T, stopk, side)
                        h = headline(simulate(mk, sig))
                        o = sig.copy(); o['d'] = -o.d; ho = headline(simulate(mk, o))
                        r[f'{mk}_n'] = h['n']; r[f'{mk}_pw'] = h['per_week']
                        r[f'{mk}_IS'] = h['pf_IS']; r[f'{mk}_OOS'] = h['pf_OOS']; r[f'{mk}_pf'] = h['pf']
                        r[f'{mk}_revIS'] = ho['pf_IS']; r[f'{mk}_revOOS'] = ho['pf_OOS']
                    rows.append(r)
    return pd.DataFrame(rows)


def select(g):
    m = g.copy()
    m['minIS'] = m[['NQ_IS', 'ES_IS']].min(axis=1)
    score = []
    for _, r in m.iterrows():
        ti = THRS.index(r.thr)
        nb = [THRS[j] for j in (ti - 1, ti, ti + 1) if 0 <= j < len(THRS)]
        score.append(m[(m.side == r.side) & (m['T'] == r['T']) & (m.stopk == r.stopk) & (m.thr.isin(nb))].minIS.mean())
    m['plateau'] = score
    b = m.sort_values(['plateau', 'minIS'], ascending=False).iloc[0]
    return m, dict(thr=float(b.thr), T=int(b['T']), stopk=float(b.stopk), side=str(b.side))


def drift_baseline():
    out = []
    for mk in ('NQ', 'ES'):
        for T in TS:
            for stopk in STOPKS:
                h = headline(simulate(mk, signals(mk, T=T, stopk=stopk, mode='drift')))
                out.append(dict(mk=mk, T=T, stopk=stopk, n=h['n'], pf_IS=h['pf_IS'], pf_OOS=h['pf_OOS'], pf=h['pf']))
    return pd.DataFrame(out)


def random_day_control(mk, ch, n=4000, seed=7):
    """Does conditioning beat random same-size subsets of the long-drift days? Long side only, per IS/OOS segment.
    (All trades are one per day 03:00 -> T with the same stop, so sampling baseline trades is exact.)"""
    rng = np.random.default_rng(seed)
    base = simulate(mk, signals(mk, T=ch['T'], stopk=ch['stopk'], mode='drift'))
    rule = simulate(mk, signals(mk, ch['thr'], ch['T'], ch['stopk'], 'long'))
    out = {}
    for seg, fn in (('IS', lambda x: x.entry_t < IS_END), ('OOS', lambda x: x.entry_t >= IS_END)):
        b = base[fn(base)].R.to_numpy(); rr = rule[fn(rule)]
        if len(rr) == 0:
            continue
        p0 = pf(rr.R)
        sims = np.array([pf(rng.choice(b, len(rr), replace=False)) for _ in range(n)])
        out[seg] = dict(n_rule=len(rr), pf_rule_long=round(p0, 3), pf_all_days=round(pf(b), 3),
                        rand_subset_median=round(float(np.nanmedian(sims)), 3),
                        p_value_rand_ge_rule=round(float(np.mean(sims >= p0)), 3))
    return out


def subwindows(mk, ch):
    """Diagnostic: where in the morning does the conditioned return accrue? long-after-selloff vs all-day drift."""
    out = {}
    F179 = features(mk, 179); F359 = features(mk, 359); F569 = features(mk, 569)
    specs = [('03:00-06:00', 179, 360), ('06:00-09:30', 359, 570), ('09:30-10:00', 569, 600), ('03:00-09:30', 179, 570),
             ('03:00-10:00', 179, 600)]
    for name, kmod, T in specs:
        F = {179: F179, 359: F359, 569: F569}[kmod]
        # the condition uses the same P/ATR as at 02:59 (P is fixed per date, ATR per date) -> identical day set
        for mode in ('rule', 'drift'):
            k, P, atr = F['k'], F['P'], F['atr']
            iT = F['idx'][T] if T in F['idx'] else None
            mb = iT - (k + 1)
            ok = (iT > 0) & (mb > 0)
            d = np.ones(len(k), dtype=np.int64) if mode == 'drift' else np.where(P <= -ch['thr'] * atr, 1, 0)
            ok &= d != 0
            stop = np.maximum(ch['stopk'] * atr, MINSTOP[mk])
            h = headline(simulate(mk, signals_frame(k[ok], d[ok], stop_pts=stop[ok], maxbars=mb[ok])))
            out[f'{name}|{"long_after_selloff" if mode == "rule" else "long_all_days"}'] = dict(
                n=h.get('n'), pf_IS=h.get('pf_IS'), pf_OOS=h.get('pf_OOS'))
    return out


def split_pf(tr, key):
    out = {}
    tr = tr.copy(); tr['seg'] = np.where(tr.entry_t < IS_END, 'IS', 'OOS')
    for v, g in tr.groupby(key, observed=True):
        gi = g[g.seg == 'IS']; go = g[g.seg == 'OOS']
        out[str(v)] = dict(n_IS=len(gi), pf_IS=round(pf(gi.R), 2) if len(gi) else None,
                           n_OOS=len(go), pf_OOS=round(pf(go.R), 2) if len(go) else None)
    return out


def partial_edges(mk, ch, side='both'):
    """Sub-condition splits of the chosen thr/T/stopk. side='both' shows the side split; side='long' = chosen rule."""
    F = features(mk); D = F['D']
    tr = simulate(mk, signals(mk, ch['thr'], ch['T'], ch['stopk'], side))
    kk = tr.k.to_numpy()
    pm = dict(zip(F['k'], F['P'] / F['atr']))
    x = np.array([pm[q] for q in kk])
    tr['side'] = np.where(tr.d == 1, 'long(after selloff)', 'short(after rally)')
    tr['absP'] = pd.cut(np.abs(x), [0, 1.0, 1.5, np.inf], right=False, labels=['<1.0', '1.0-1.5', '>=1.5'])
    td = D['trend_d'][kk]
    tr['vs_trend'] = np.where(tr.d == td, 'with_20d_trend', np.where(tr.d == -td, 'against_20d_trend', 'flat'))
    tr['dow'] = np.array(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])[D['dow'][kk]]
    # overnight move already made by 02:59 (c[k] - previous close), in trade direction
    onm = tr.d.to_numpy() * (D['c'][kk] - D['pdc'][kk]) / D['atr'][kk]
    tr['overnight_by_0259'] = np.where(onm > 0, 'already_moved_our_way', 'not_yet_or_against')
    rel = D['atr'][kk] / D['c'][kk]
    med = np.median(rel[tr.entry_t < IS_END])
    tr['vol'] = np.where(rel >= med, 'high_vol', 'low_vol')
    # where did the prior day close in its own range (close-location value), info
    clv = (D['pdc'][kk] - D['pdl'][kk]) / np.maximum(D['pdh'][kk] - D['pdl'][kk], 1e-9)
    tr['prior_close_loc'] = pd.cut(clv, [-0.01, 0.25, 0.75, 1.01], labels=['near_low', 'mid', 'near_high'])
    return {key: split_pf(tr, key) for key in ('side', 'absP', 'vs_trend', 'dow', 'overnight_by_0259', 'vol', 'prior_close_loc')}


def grid_splits(g):
    """Median IS/OOS PF across grid cells by structural dimension (robustness view; info only)."""
    rows = {}
    for mk in ('NQ', 'ES'):
        for dim in ('side', 'T', 'stopk', 'thr'):
            for v, gg in g.groupby(dim):
                rows[f'{mk}|{dim}={v}'] = dict(IS=round(float(gg[f'{mk}_IS'].median()), 3), OOS=round(float(gg[f'{mk}_OOS'].median()), 3))
    return rows


def short(rep):
    keys = ('n', 'per_week', 'win', 'pf', 'pf_IS', 'n_IS', 'pf_OOS', 'n_OOS', 'avg_pts', 'med_risk_pts', 'med_hold_min',
            'pf_2x_cost', 'pf_no_cost', 'p_pf_below_1', 'pf_opposite', 'pf_random_dir_median')
    return {k: rep.get(k) for k in keys}


if __name__ == '__main__':
    pd.set_option('display.width', 220)
    g = grid()
    print('=== GRID (24 variants; selection uses IS columns only; rev* = d reversed control) ===')
    print(g.to_string(index=False))
    med = {mk: float(np.median(g[f'{mk}_IS'])) for mk in ('NQ', 'ES')}
    med_oos = {mk: float(np.median(g[f'{mk}_OOS'])) for mk in ('NQ', 'ES')}
    print('median IS PF all 24 variants:', med, '| median OOS (info only):', med_oos)
    m, ch = select(g)
    print('\n=== IS plateau scores ===')
    print(m[['side', 'T', 'stopk', 'thr', 'NQ_IS', 'ES_IS', 'minIS', 'plateau']].to_string(index=False))
    print('CHOSEN (IS only):', ch)
    db = drift_baseline(); print('\n=== drift baseline: long every RTH day 03:00 -> T ===\n', db.to_string(index=False))
    gs = grid_splits(g); print('\n=== median PF across grid by dimension ===')
    for kx, v in gs.items():
        print('  ', kx, v)
    res = dict(slug='premarket_selloff_rebound', chosen=ch, variants_tested=len(g), median_IS_pf=med, median_OOS_pf_info=med_oos,
               drift_baseline=db.to_dict(orient='records'), grid_splits=gs, grid=g.to_dict(orient='records'))
    for mk in ('NQ', 'ES'):
        sig = signals(mk, **ch)
        rep = full_report(mk, sig)
        print(f'\n=== {mk} full_report chosen ===\n', json.dumps(rep, default=str))
        diag = dict(cost_3x_pf=round(pf(simulate(mk, sig, costmult=3.0).R), 3),
                    random_day_control=random_day_control(mk, ch), subwindows=subwindows(mk, ch),
                    partial_edges_both_sides=partial_edges(mk, ch, 'both'),
                    partial_edges_chosen=partial_edges(mk, ch, ch['side']))
        print(f'--- {mk} 3x cost PF:', diag['cost_3x_pf'])
        print(f'--- {mk} random-day control (long side):', json.dumps(diag['random_day_control']))
        print(f'--- {mk} sub-windows:')
        for kx, v in diag['subwindows'].items():
            print('   ', kx, v)
        for nm in ('partial_edges_both_sides', 'partial_edges_chosen'):
            print(f'--- {mk} {nm}:')
            for kx, v in diag[nm].items():
                print('   ', kx, json.dumps(v))
        res[mk] = rep; res[mk + '_diag'] = diag
    def verdict(r):
        a, b = r['NQ'], r['ES']
        def ok_pass(x, y):
            return (x['pf_IS'] >= 1.3 and x['pf_OOS'] >= 1.3 and y['pf_OOS'] >= 1.1 and x['per_week'] >= 3
                    and x['pf_2x_cost'] >= 1.1)
        if ok_pass(a, b) or ok_pass(b, a):
            return 'PASS'
        if any(x['pf_IS'] >= 1.15 and x['pf_OOS'] >= 1.15 for x in (a, b)):
            return 'BORDERLINE'
        return 'FAIL'
    res['verdict'] = verdict(res)
    res['verdict_note'] = ('IS and OOS PF >= 1.3 on both markets and 2x-cost PF >= 1.29, but only ~1.1-1.2 trades/week '
                           '(< 3/week floor) -> not PASS; hold is 6.5 h (03:00 -> 09:30), not a scalp. Short side fails OOS.')
    print('VERDICT:', res['verdict'])
    json.dump(res, open(OUT, 'w'), default=str, indent=1)
    print('saved', OUT)
