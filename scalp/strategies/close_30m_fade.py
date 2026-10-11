"""close_30m_fade -- Last-30-minute fade of the day's move (0DTE-era flip of market intraday momentum).

Hypothesis: Gao et al. (JFE 2018) / Baltussen et al. (JFE 2021) document last-half-hour MOMENTUM (the day's
return predicts the 15:30-16:00 return) driven by short-gamma hedging. Since daily SPX 0DTE expiries, dealers are
often net LONG same-day gamma, so hedging leans against the day's move into the close (pinning), and MOC /
leveraged-ETF rebalancing flows are front-run. Prediction on 2023-2026 data: the sign flips -> FADE the day's move.

Mechanical rules (as implemented):
  * Signal bar k = the 1-minute bar with NY minute-of-day == 929 (15:29; close known at 15:30:00).
    Robustness variant: k = bar 914 (15:14) -> entry 15:15.  Days without that bar (half days) are skipped.
  * x = (c[k] - REF) / ATR, ATR = 14-day mean RTH range of PREVIOUS days (cache 'atr').
      REF 'pdc'      = previous RTH day's 15:59 close  -> "rest-of-day" move incl. overnight,
      REF 'rth_open' = today's 09:30 open               -> RTH move so far,
      REF 'vwap'     = running RTH VWAP at bar k (bars 09:30..k inclusive).
  * If |x| >= thr (and x != 0): direction d = -sign(x) (fade). Market entry at the open of bar k+1 (etype 0).
  * Stop = max(s * ATR, 20 pts NQ / 4 pts ES) from the fill. No target. Exit = engine force-close at the 15:59 open.
  * One trade per day by construction (one bar 929 per date).
  * Control: momentum direction (pf_opposite in full_report) = the published Gao/Baltussen rule.

Grid (24 variants, counted toward the selection budget):
  main      : REF {pdc, rth_open, vwap} x thr {0, 0.25, 0.5} x s {0.15, 0.30}, signal 15:29 -> 18
  robustness: REF {pdc, rth_open, vwap} x s {0.15, 0.30}, thr 0.25, signal 15:14       ->  6
Selection rule (fixed before running, IS only): among the 18 main cells, score = mean over the cell and its
thr-neighbours (same REF, adjacent thr, both s) of min(IS PF NQ, IS PF ES); pick the max (plateau, both markets).
Diagnostics NOT counted toward selection: maxbars=15 for the chosen cell, same rule at earlier signal times
(time-specificity control), sub-condition splits (partial edges).
RESULT NOTE: the pre-registered rule picked REF vwap / thr 0.5 / s 0.15, which is also the single best IS cell on both
markets and sits at the grid edge (0.47 NQ / 0.73 ES trades per week). It is kept as the reported pick (rule not changed
after seeing results); the frequency-floored alternative is printed for information only.
NOTE: the proposer had already viewed OOS for ~24 points-PF cells of this idea -> OOS is partly viewed.
"""
import sys, json
sys.path.insert(0, '/home/user/TRADING/scalp')
import numpy as np
import pandas as pd
from engine import load, signals_frame, simulate, headline, full_report, pf, IS_END

MINSTOP = {'NQ': 20.0, 'ES': 4.0}
REFS = ('pdc', 'rth_open', 'vwap')
THRS = (0.0, 0.25, 0.5)
SS = (0.15, 0.30)
OUT = '/home/user/TRADING/scalp/results/close_30m_fade.json'
IS_WEEKS = (np.datetime64('2025-04-01') - np.datetime64('2023-10-01')).astype(int) / 7.0


def features(mk, kmod=929, ref='pdc'):
    """Per-day signal bars and the move statistic x. Everything uses data <= close of bar k."""
    D = load(mk)
    k = np.where(D['mod'] == kmod)[0]
    k = k[k + 1 < len(D['mod'])]
    k = k[D['dayi'][k + 1] == D['dayi'][k]]           # the entry bar must exist on the same day
    atr = D['atr'][k]
    refv = D[ref][k]                                    # pdc: prev day; rth_open: 09:30 today; vwap: running to k
    x = (D['c'][k] - refv) / atr
    return D, k, x, atr


def signals(mk, ref='pdc', thr=0.25, s=0.30, kmod=929, maxbars=0, mask_fn=None):
    D, k, x, atr = features(mk, kmod, ref)
    ok = np.isfinite(x) & np.isfinite(atr) & (np.abs(x) >= thr) & (x != 0)
    if mask_fn is not None:
        ok &= mask_fn(D, k, x)
    k, x, atr = k[ok], x[ok], atr[ok]
    d = -np.sign(x).astype(np.int64)
    stop = np.maximum(s * atr, MINSTOP[mk])
    return signals_frame(k, d, etype=0, stop_pts=stop, maxbars=maxbars)


def grid():
    rows = []
    for kmod, refs, thrs in ((929, REFS, THRS), (914, REFS, (0.25,))):
        for ref in refs:
            for thr in thrs:
                for s in SS:
                    r = dict(kmod=kmod, ref=ref, thr=thr, s=s)
                    for mk in ('NQ', 'ES'):
                        h = headline(simulate(mk, signals(mk, ref, thr, s, kmod)))
                        r[f'{mk}_n'] = h['n']; r[f'{mk}_pw'] = h['per_week']
                        r[f'{mk}_IS'] = h['pf_IS']; r[f'{mk}_OOS'] = h['pf_OOS']; r[f'{mk}_pf'] = h['pf']
                        r[f'{mk}_pwIS'] = round(h['n_IS'] / IS_WEEKS, 2)
                    rows.append(r)
    return pd.DataFrame(rows)


def select(g):
    """IS-only plateau selection among the 18 main (15:29-signal) cells."""
    m = g[g.kmod == 929].copy()
    m['minIS'] = m[['NQ_IS', 'ES_IS']].min(axis=1)
    score = []
    for _, r in m.iterrows():
        ti = THRS.index(r.thr)
        nb = [THRS[j] for j in (ti - 1, ti, ti + 1) if 0 <= j < len(THRS)]
        score.append(m[(m.ref == r.ref) & (m.thr.isin(nb))].minIS.mean())
    m['plateau'] = score
    best = m.sort_values(['plateau', 'minIS'], ascending=False).iloc[0]
    return m, dict(ref=best.ref, thr=float(best.thr), s=float(best.s), kmod=929)


def select_freq_floor(g, floor=3.0):
    """INFO ONLY (not the reported pick): same IS plateau score, candidates restricted to >= floor trades/week IS on both
    markets (the verdict's frequency floor). Neighbours unchanged."""
    m, _ = select(g)
    m = m[(m.NQ_pwIS >= floor) & (m.ES_pwIS >= floor)]
    b = m.sort_values(['plateau', 'minIS'], ascending=False).iloc[0]
    return dict(ref=b.ref, thr=float(b.thr), s=float(b.s), kmod=929)


def controls(rng_seed=1):
    """Momentum (opposite) and random-direction controls for representative cells, IS and OOS (diagnostic)."""
    rng = np.random.default_rng(rng_seed); out = []
    for mk in ('NQ', 'ES'):
        for ref, thr, s in (('pdc', 0, 0.15), ('pdc', 0.25, 0.30), ('rth_open', 0.25, 0.15), ('vwap', 0.25, 0.30), ('vwap', 0.5, 0.15)):
            sig = signals(mk, ref, thr, s); h = headline(simulate(mk, sig))
            o = sig.copy(); o['d'] = -o.d; ho = headline(simulate(mk, o))
            ri, ro = [], []
            for _ in range(40):
                x = sig.copy(); x['d'] = np.where(rng.random(len(x)) < .5, 1, -1); hr = headline(simulate(mk, x))
                ri.append(hr['pf_IS']); ro.append(hr['pf_OOS'])
            out.append(dict(mk=mk, ref=ref, thr=thr, s=s, fade_IS=h['pf_IS'], fade_OOS=h['pf_OOS'], mom_IS=ho['pf_IS'],
                            mom_OOS=ho['pf_OOS'], rand_IS_med=round(float(np.median(ri)), 3), rand_OOS_med=round(float(np.median(ro)), 3)))
    return pd.DataFrame(out)


def side_baseline():
    """Is the long-side fade more than the unconditional long drift into the close? (diagnostic, thr 0.25)"""
    out = []
    for mk in ('NQ', 'ES'):
        for ref in REFS:
            D, k, x, atr = features(mk, 929, ref)
            ok = np.isfinite(x) & np.isfinite(atr)
            for s in SS:
                for name, m, d in (('long_all_days', ok, 1), ('long_fade_down_day', ok & (x <= -0.25), 1),
                                   ('short_all_days', ok, -1), ('short_fade_up_day', ok & (x >= 0.25), -1)):
                    h = headline(simulate(mk, signals_frame(k[m], d, stop_pts=np.maximum(s * atr[m], MINSTOP[mk]))))
                    out.append(dict(mk=mk, ref=ref, s=s, rule=name, n=h['n'], pf_IS=h['pf_IS'], pf_OOS=h['pf_OOS']))
    return pd.DataFrame(out)


def grid_splits():
    """Median PF across all 24 cells by sub-condition, IS vs OOS (robustness of partial edges)."""
    rows = []
    for mk in ('NQ', 'ES'):
        D = load(mk)
        for kmod, thrs in ((929, THRS), (914, (0.25,))):
            for ref in REFS:
                for thr in thrs:
                    for s in SS:
                        tr = simulate(mk, signals(mk, ref, thr, s, kmod)); kk = tr.k.to_numpy(); td = D['trend_d'][kk]
                        seg = np.where(tr.entry_t < IS_END, 'IS', 'OOS')
                        rth_mv = D['c'][kk] - D['rth_open'][kk]; on_mv = D['rth_open'][kk] - D['pdc'][kk]
                        lab = {'side': np.where(tr.d == 1, 'long', 'short'),
                               'day_vs_20d_trend': np.where(tr.d == -td, 'fade_trendwise_day', 'fade_countertrend_day'),
                               'gap': np.where(np.sign(rth_mv) == np.sign(on_mv), 'rth_continues_gap', 'rth_reverses_gap')}
                        for key, L in lab.items():
                            for v in np.unique(L):
                                for sg in ('IS', 'OOS'):
                                    mm = (L == v) & (seg == sg)
                                    rows.append(dict(mk=mk, key=key, val=v, seg=sg, pf=pf(tr.R[mm])))
    df = pd.DataFrame(rows)
    return df.pivot_table(index=['mk', 'key', 'val'], columns='seg', values='pf', aggfunc='median').round(3)


def split_pf(tr, key):
    """PF by sub-condition for IS and OOS."""
    out = {}
    tr = tr.copy(); tr['seg'] = np.where(tr.entry_t < IS_END, 'IS', 'OOS')
    for v, g in tr.groupby(key):
        gi = g[g.seg == 'IS']; go = g[g.seg == 'OOS']
        out[str(v)] = dict(n_IS=len(gi), pf_IS=round(pf(gi.R), 2) if len(gi) else None,
                           n_OOS=len(go), pf_OOS=round(pf(go.R), 2) if len(go) else None)
    return out


def partial_edges(mk, ch):
    D, k, x, atr = features(mk, ch['kmod'], ch['ref'])
    sig = signals(mk, ch['ref'], ch['thr'], ch['s'], ch['kmod'])
    tr = simulate(mk, sig)
    kk = tr.k.to_numpy()
    xm = dict(zip(k, x))
    xs = np.array([xm[q] for q in kk])
    tr['side'] = np.where(tr.d == 1, 'long(fade down day)', 'short(fade up day)')
    tr['absx'] = pd.cut(np.abs(xs), [0, 0.5, 1.0, np.inf], right=False, labels=['<0.5', '0.5-1', '>=1'])
    td = D['trend_d'][kk]
    tr['vs_trend'] = np.where(tr.d == td, 'fade_with_20d_trend', np.where(tr.d == -td, 'fade_against_20d_trend', 'flat'))
    dow = D['dow'][kk]
    tr['dow'] = np.array(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])[dow]
    # monthly-opex (3rd Friday) vs other days
    dates = pd.to_datetime(D['date'][kk] * 86400, unit='s')
    third_fri = (dates.dayofweek == 4) & (dates.day >= 15) & (dates.day <= 21)
    tr['opex'] = np.where(third_fri, 'monthly_opex', 'other')
    # volatility regime: ATR/price vs its IS median (threshold from IS only)
    vol = atr_rel = D['atr'][kk] / D['c'][kk]
    med = np.median(vol[tr.entry_t < IS_END])
    tr['vol'] = np.where(atr_rel >= med, 'high_vol', 'low_vol')
    # did the day's RTH move agree with the overnight move? (pdc-move vs rth_open-move sign)
    rth_mv = D['c'][kk] - D['rth_open'][kk]; on_mv = D['rth_open'][kk] - D['pdc'][kk]
    tr['on_vs_rth'] = np.where(np.sign(rth_mv) == np.sign(on_mv), 'rth_continues_gap', 'rth_reverses_gap')
    return {key: split_pf(tr, key) for key in ('side', 'absx', 'vs_trend', 'dow', 'opex', 'vol', 'on_vs_rth')}


def time_control(mk, ch):
    """Same fade rule (hold to 15:59) from earlier signal minutes -- is the effect specific to the close?"""
    out = {}
    for kmod in (689, 779, 809, 839, 869, 899, 914, 929, 944):
        h = headline(simulate(mk, signals(mk, ch['ref'], ch['thr'], ch['s'], kmod)))
        out[f'{kmod // 60:02d}:{kmod % 60:02d}'] = dict(n=h.get('n'), pf_IS=h.get('pf_IS'), pf_OOS=h.get('pf_OOS'))
    return out


def short(rep):
    keys = ('n', 'per_week', 'win', 'pf', 'pf_IS', 'n_IS', 'pf_OOS', 'n_OOS', 'avg_pts', 'med_risk_pts', 'med_hold_min',
            'pf_2x_cost', 'pf_no_cost', 'p_pf_below_1', 'pf_opposite', 'pf_random_dir_median')
    return {k: rep.get(k) for k in keys}


if __name__ == '__main__':
    pd.set_option('display.width', 200)
    g = grid()
    print('=== GRID (24 variants; selection uses IS columns only) ===')
    print(g.to_string(index=False))
    med = {mk: float(np.median(g[f'{mk}_IS'])) for mk in ('NQ', 'ES')}
    print('median IS PF all 24 variants:', med)
    m, ch = select(g)
    print('\n=== IS plateau scores (main 18) ===')
    print(m[['ref', 'thr', 's', 'NQ_IS', 'ES_IS', 'minIS', 'plateau']].to_string(index=False))
    print('CHOSEN (IS only):', ch)

    alt = select_freq_floor(g)
    print('INFO: same rule with >=3 trades/week IS floor would pick:', alt)
    med_oos = {mk: float(np.median(g[f'{mk}_OOS'])) for mk in ('NQ', 'ES')}
    print('median OOS PF all 24 variants (info only):', med_oos)
    ct = controls(); print('\n=== controls: fade vs momentum vs random direction ===\n', ct.to_string(index=False))
    sb = side_baseline(); print('\n=== side vs unconditional drift into close ===\n', sb.to_string(index=False))
    gs = grid_splits(); print('\n=== median PF across 24 cells by sub-condition ===\n', gs.to_string())
    res = dict(slug='close_30m_fade', chosen=ch, variants_tested=len(g), median_IS_pf=med, median_OOS_pf_info=med_oos,
               alt_pick_freq_floor_info=alt, controls=ct.to_dict(orient='records'), side_baseline=sb.to_dict(orient='records'),
               grid_splits={f'{a}|{b}|{c}': v for (a, b, c), v in gs.to_dict(orient='index').items()},
               oos_note='OOS partly viewed by the proposer (~24 points-PF cells) before this test',
               grid=g.to_dict(orient='records'))
    for mk in ('NQ', 'ES'):
        sig = signals(mk, **ch)
        rep = full_report(mk, sig)
        print(f'\n=== {mk} full_report chosen ===\n', json.dumps(rep, default=str))
        diag = {}
        sig15 = signals(mk, **ch, maxbars=15)
        h15 = headline(simulate(mk, sig15)); h15['pf_2x_cost'] = round(pf(simulate(mk, sig15, costmult=2.0).R), 3)
        diag['maxbars15'] = h15
        diag['cost_3x_pf'] = round(pf(simulate(mk, sig, costmult=3.0).R), 3)
        diag['time_control'] = time_control(mk, ch)
        diag['partial_edges'] = partial_edges(mk, ch)
        print(f'--- {mk} maxbars=15 (diagnostic):', json.dumps(h15, default=str))
        print(f'--- {mk} 3x cost PF:', diag['cost_3x_pf'])
        print(f'--- {mk} time control (same rule, earlier signal minute):', json.dumps(diag['time_control']))
        print(f'--- {mk} partial edges:')
        for kk, v in diag['partial_edges'].items():
            print('   ', kk, json.dumps(v))
        res[mk] = rep; res[mk + '_diag'] = diag
    json.dump(res, open(OUT, 'w'), default=str, indent=1)
    print('saved', OUT)
