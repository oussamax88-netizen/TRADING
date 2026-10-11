"""Gamma-regime switch: VWAP 2-sigma FADE in a calm regime, VWAP-band FOLLOW in a stressed regime.
Slug: gamma_regime_vwap.  Both markets, same rules (stops scale with ATR / VWAP sigma).

Hypothesis (Dim-Eraker-Vilkov 0DTE / dealer gamma; Baltussen et al. hedging demand & intraday momentum; Park-Zhao):
dealers long gamma (calm, rising markets) sell rallies / buy dips -> stretched VWAP excursions mean-revert; dealers
short gamma (falling, volatile markets) chase -> intraday momentum.  The same VWAP-sigma signal should flip sign with
the regime, which would explain why unconditional fades and breakouts tested ~1.0 on this data.

Mechanical rules as implemented
-------------------------------
VWAP / sigma (causal): RTH = NY minutes 09:30..15:59 of the calendar date.  tp = (h+l+c)/3, weights = tick volume.
   vwap[k] = sum(vol*tp)/sum(vol) over RTH bars 09:30..k (identical to the cache 'vwap', asserted).
   sd[k]   = sqrt( sum vol*(tp - vwap[k])^2 / sum vol ) over the same bars (computed as E[tp^2]-vwap^2 on prices
             shifted by the day's 09:30 open for numerical precision; a constant shift does not change sd).
   Bands U[k] = vwap[k] + K*sd[k], L[k] = vwap[k] - K*sd[k].
Regime per calendar day (known before the open, prior RTH days only):
   ATR14 = cache 'atr' (mean RTH range of the 14 previous RTH days).  med60 = median of ATR14 over the 60 previous
   RTH dates (strictly earlier; needs 60 values).
   'ATR'      : CALM if ATR14 <= med60 else STRESS.
   'ATRTREND' : CALM if ATR14 <= med60 AND trend_d == +1 (cache: sign(prev close - SMA20 of prev closes)), else STRESS.
   Days with NaN ATR14/med60 (or NaN trend_d for ATRTREND) are not traded.
Trigger (signal bar k, decision at the close of k): 10:30 <= NY minute of k <= 15:00 (mod 630..900), bar k-1 is an
   RTH bar of the same date, and a FRESH band crossing:  upper: c[k] > U[k] and c[k-1] <= U[k-1];
   lower: c[k] < L[k] and c[k-1] >= L[k-1].  (= first close beyond a band after a close back inside it.)
   s = +1 for an upper-band break, -1 for a lower-band break.
CALM FADE  : d = -s, market entry at the open of k+1 (etype 0), tgt_pts = |c[k] - vwap[k]| (back to VWAP),
             stop_pts = max(0.12*ATR14, MINSTOP, 1.0*sd[k]), maxbars 60.  MINSTOP = 20 pts NQ / 4 pts ES.
STRESS FOLLOW: d = +s, etype 0, stop_pts = max(0.12*ATR14, MINSTOP, |c[k] - vwap[k]|), no target.
             Exit '1559': engine force-close at the 15:59 open.
             Exit 'TRAIL': first later bar j (same day, RTH) whose 1m close is back across VWAP
             (long: c[j] < vwap[j], short: c[j] > vwap[j]) -> maxbars = j - k -> exit at the open of j+1.
Engine: one position at a time (overlapping triggers are skipped, the crossing is consumed).
At most 2 TAKEN trades per day: the candidate list is simulated once, and every candidate after the 2nd taken trade of
   a day is dropped (later candidates cannot change earlier trades, so the first two trades are unchanged).
Grid (24): K {2.0, 2.5} x regime {ATR, ATRTREND} x mode {fadeC, folS1559, folStrail, both1559, bothTrail} = 20
           + wrong-arm controls at K 2.0 for both regime defs {folC1559 (follow on CALM days), fadeS (fade on STRESS
           days)} = 4.
Selection (IS only, fixed before running): among the 20 hypothesis variants with >= 50 IS trades on both markets,
   score = min(IS PF NQ, IS PF ES); plateau = mean score of the variant and its K-neighbour (same regime/mode);
   max plateau wins, tie -> own score.  Controls are never selectable.
"""
import sys, json
sys.path.insert(0, '/home/user/TRADING/scalp')
import numpy as np
import pandas as pd
from engine import load, signals_frame, simulate, full_report, headline, pf, IS_END

MINSTOP = {'NQ': 20.0, 'ES': 4.0}
WIN = (630, 900)
OUT = '/home/user/TRADING/scalp/results/gamma_regime_vwap.json'
IS_WEEKS = (np.datetime64('2025-04-01') - np.datetime64('2023-10-01')).astype(int) / 7.0
OOS_WEEKS = (np.datetime64('2026-10-01') - np.datetime64('2025-04-01')).astype(int) / 7.0
_P = {}


def prep(mk):
    if mk in _P:
        return _P[mk]
    D = load(mk)
    h, l, c, vol, mod, date = D['h'], D['l'], D['c'], D['vol'], D['mod'], D['date']
    rth = (mod >= 570) & (mod < 960)
    base = np.where(rth, D['rth_open'], 0.0)
    tpx = np.where(rth, (h + l + c) / 3 - base, 0.0)
    w = np.where(rth, vol, 0.0)
    g = pd.DataFrame({'w': w, 'pv': w * tpx, 'p2': w * tpx * tpx}).groupby(date).cumsum()
    cw, cpv, cp2 = g.w.to_numpy(), g.pv.to_numpy(), g.p2.to_numpy()
    ok = rth & (cw > 0)
    vx = np.where(ok, cpv / np.maximum(cw, 1e-12), np.nan)
    var = np.where(ok, cp2 / np.maximum(cw, 1e-12) - vx * vx, np.nan)
    sd = np.sqrt(np.maximum(var, 0.0))
    vwap = vx + base
    m = np.isfinite(D['vwap']) | np.isfinite(vwap)
    assert np.allclose(vwap[m], D['vwap'][m], rtol=0.0, atol=1e-6, equal_nan=False), 'vwap mismatch'
    # ---- regime per date (prior RTH days only)
    rd = np.unique(date[rth])                                  # RTH dates (same set the cache ATR uses)
    first = np.searchsorted(date, rd)                          # first bar of each RTH date (date is sorted)
    atr_d = D['atr'][first]
    med60 = pd.Series(atr_d).rolling(60, min_periods=60).median().shift(1).to_numpy()
    pos = np.clip(np.searchsorted(rd, date), 0, len(rd) - 1)
    has = rd[pos] == date
    med_bar = np.where(has, med60[pos], np.nan)
    atr = D['atr']
    good = np.isfinite(atr) & np.isfinite(med_bar)
    calm_atr = good & (atr <= med_bar)
    calm_tr = calm_atr & (D['trend_d'] == 1)
    # regime code per bar: +1 CALM, -1 STRESS, 0 untradeable
    reg = {'ATR': np.where(good, np.where(calm_atr, 1, -1), 0),
           'ATRTREND': np.where(good & np.isfinite(D['trend_d']), np.where(calm_tr, 1, -1), 0)}
    ratio = atr / med_bar
    # trading-day boundaries (calendar date) for trail search
    day_end = np.r_[np.where(np.diff(D['dayi']) != 0)[0], len(date) - 1]
    end_of = day_end[D['dayi']]
    _P[mk] = dict(D=D, vwap=vwap, sd=sd, rth=rth, reg=reg, ratio=ratio, end_of=end_of)
    return _P[mk]


def triggers(mk, K):
    """Fresh band crossings in the signal window. Returns bar indices k and side s (+1 upper, -1 lower)."""
    P = prep(mk); D = P['D']
    c, mod, dayi = D['c'], D['mod'], D['dayi']
    vw, sd = P['vwap'], P['sd']
    U = vw + K * sd; L = vw - K * sd
    k = np.arange(1, len(c))
    km = k - 1
    ok = (mod[k] >= WIN[0]) & (mod[k] <= WIN[1]) & (dayi[km] == dayi[k]) & P['rth'][km] & np.isfinite(U[km]) & np.isfinite(U[k])
    up = ok & (c[k] > U[k]) & (c[km] <= U[km])
    dn = ok & (c[k] < L[k]) & (c[km] >= L[km])
    kk = np.r_[k[up], k[dn]]; ss = np.r_[np.ones(up.sum(), int), -np.ones(dn.sum(), int)]
    o = np.argsort(kk, kind='stable')
    return kk[o], ss[o]


def trail_bars(P, k, d):
    """First later bar j (same day, RTH) with the close back across VWAP -> maxbars = j - k (0 = none)."""
    D = P['D']; c = D['c']; vw = P['vwap']; rth = P['rth']
    e = P['end_of'][k]
    seg = slice(k + 1, e + 1)
    x = (d * (c[seg] - vw[seg]) < 0) & rth[seg]
    if not x.any():
        return 0
    return int(np.argmax(x) + 1)


def candidates(mk, K=2.0, regdef='ATR', fade_on=None, follow_on=None, fexit='1559'):
    """fade_on / follow_on: regime code (+1 CALM / -1 STRESS) on which that arm trades, or None."""
    P = prep(mk); D = P['D']
    c, atr = D['c'], D['atr']
    vw, sd, reg = P['vwap'], P['sd'], P['reg'][regdef]
    kk, ss = triggers(mk, K)
    rows = []
    for k, s in zip(kk, ss):
        r = reg[k]
        dist = abs(c[k] - vw[k])
        base = max(0.12 * atr[k], MINSTOP[mk])
        if fade_on is not None and r == fade_on:
            rows.append((k, -s, max(base, 1.0 * sd[k]), dist, 60, 'fade', r, s))
        elif follow_on is not None and r == follow_on:
            mb = trail_bars(P, k, s) if fexit == 'TRAIL' else 0
            rows.append((k, s, max(base, dist), np.inf, mb, 'follow', r, s))
    cand = pd.DataFrame(rows, columns=['k', 'd', 'stop', 'tgt', 'mb', 'arm', 'reg', 'side_band'])
    return cap2(mk, cand)


def to_sig(cand):
    return signals_frame(cand.k.to_numpy(), cand.d.to_numpy(), etype=0, stop_pts=cand.stop.to_numpy(),
                         tgt_pts=cand.tgt.to_numpy(), maxbars=cand.mb.to_numpy())


def cap2(mk, cand, cap=2):
    """Keep candidates up to (and including) the 2nd TAKEN trade of each day."""
    if len(cand) == 0:
        return cand
    D = load(mk)
    tr = simulate(mk, to_sig(cand), period=False)
    tr['day'] = D['dayi'][tr.k]
    tr['nth'] = tr.groupby('day').cumcount() + 1
    lastk = tr[tr.nth == cap].set_index('day').k
    day = D['dayi'][cand.k.to_numpy()]
    lim = lastk.reindex(day).to_numpy()
    keep = np.isnan(lim) | (cand.k.to_numpy() <= lim)
    return cand[keep].reset_index(drop=True)


MODES = {
    'fadeC':     dict(fade_on=1, follow_on=None, fexit='1559'),
    'folS1559':  dict(fade_on=None, follow_on=-1, fexit='1559'),
    'folStrail': dict(fade_on=None, follow_on=-1, fexit='TRAIL'),
    'both1559':  dict(fade_on=1, follow_on=-1, fexit='1559'),
    'bothTrail': dict(fade_on=1, follow_on=-1, fexit='TRAIL'),
}
CONTROLS = {
    'CTL_folC1559': dict(fade_on=None, follow_on=1, fexit='1559'),
    'CTL_fadeS':    dict(fade_on=-1, follow_on=None, fexit='1559'),
}
VARIANTS = [dict(K=K, regdef=r, mode=m, control=False) for K in (2.0, 2.5) for r in ('ATR', 'ATRTREND') for m in MODES] + \
           [dict(K=2.0, regdef=r, mode=m, control=True) for r in ('ATR', 'ATRTREND') for m in CONTROLS]


def vargs(v):
    a = dict(MODES.get(v['mode']) or CONTROLS[v['mode']])
    a.update(K=v['K'], regdef=v['regdef'])
    return a


def vname(v):
    return f"K{v['K']}_{v['regdef']}_{v['mode']}"


def run_grid():
    rows = []
    for v in VARIANTS:
        r = dict(name=vname(v), **v)
        for mk in ('NQ', 'ES'):
            tr = simulate(mk, to_sig(candidates(mk, **vargs(v))))
            i = tr[tr.entry_t < IS_END]; o = tr[tr.entry_t >= IS_END]
            r[f'{mk}_nIS'] = len(i); r[f'{mk}_pfIS'] = round(pf(i.R), 3); r[f'{mk}_wkIS'] = round(len(i) / IS_WEEKS, 2)
            r[f'{mk}_pfOOS'] = round(pf(o.R), 3); r[f'{mk}_nOOS'] = len(o)     # stored, NOT used for selection
        rows.append(r)
    g = pd.DataFrame(rows)
    g['score'] = g[['NQ_pfIS', 'ES_pfIS']].min(axis=1)
    pl = []
    for _, r in g.iterrows():
        nb = g[(g.regdef == r.regdef) & (g['mode'] == r['mode']) & (g.control == r.control)]
        pl.append(nb.score.mean())
    g['plateau'] = np.round(pl, 3)
    return g


def annotate(mk, v):
    P = prep(mk); D = P['D']
    cand = candidates(mk, **vargs(v))
    tr = simulate(mk, to_sig(cand))
    m = cand.set_index('k')
    tr['arm'] = m.loc[tr.k, 'arm'].to_numpy(); tr['reg'] = m.loc[tr.k, 'reg'].to_numpy()
    tr['mod'] = D['mod'][tr.k]; tr['trend'] = D['trend_d'][tr.k]; tr['ratio'] = P['ratio'][tr.k]
    tr['day'] = D['dayi'][tr.k]; tr['nth'] = tr.groupby('day').cumcount() + 1
    tr['IS'] = tr.entry_t < IS_END
    return tr


def split_table(tr, subs):
    out = {}
    for name, msk in subs.items():
        a = tr[msk & tr.IS]; b = tr[msk & ~tr.IS]
        out[name] = dict(n_IS=len(a), pf_IS=round(pf(a.R), 2), n_OOS=len(b), pf_OOS=round(pf(b.R), 2))
    return out


def partial_edges(mk, v):
    tr = annotate(mk, v)
    subs = {'long': tr.d == 1, 'short': tr.d == -1,
            'fade_arm': tr.arm == 'fade', 'follow_arm': tr.arm == 'follow',
            '10:30-12:00': tr['mod'] < 720, '12:00-13:30': (tr['mod'] >= 720) & (tr['mod'] < 810), '13:30-15:00': tr['mod'] >= 810,
            'first_trade_of_day': tr.nth == 1, 'second_trade_of_day': tr.nth == 2,
            'with_20d_trend': tr.d == tr.trend, 'against_20d_trend': tr.d == -tr.trend,
            'atr_ratio<0.9': tr.ratio < 0.9, 'atr_ratio_0.9-1.1': (tr.ratio >= 0.9) & (tr.ratio <= 1.1), 'atr_ratio>1.1': tr.ratio > 1.1,
            'exit_target': tr.reason == 2, 'exit_stop': tr.reason == 1, 'exit_time_or_trail': tr.reason >= 3}
    return split_table(tr, subs)


def arm_breakdown(mk):
    """Diagnostic: every arm x regime cell at K 2.0 (not used for selection). Uses the 'both' variants + controls."""
    out = {}
    for regdef in ('ATR', 'ATRTREND'):
        for mode in ('fadeC', 'folS1559', 'folStrail', 'CTL_folC1559', 'CTL_fadeS'):
            v = dict(K=2.0, regdef=regdef, mode=mode)
            tr = simulate(mk, to_sig(candidates(mk, **vargs(v))))
            tr['IS'] = tr.entry_t < IS_END
            out[f'{regdef}_{mode}'] = split_table(tr, {'all': tr.d != 0, 'long': tr.d == 1, 'short': tr.d == -1})
    return out


def random_time_control(mk, v, n=20, seed=1):
    """Same days, same arm/stop/target/exit rules, but trigger bars drawn at random inside the 10:30-15:00 window
    (side s random).  Median PF over n draws."""
    rng = np.random.default_rng(seed)
    P = prep(mk); D = P['D']
    a = vargs(v)
    real = candidates(mk, **a)
    days = np.unique(D['dayi'][real.k])
    c, atr, mod, dayi = D['c'], D['atr'], D['mod'], D['dayi']
    vw, sd, reg = P['vwap'], P['sd'], P['reg'][a['regdef']]
    win = np.where((mod >= WIN[0]) & (mod <= WIN[1]) & np.isin(dayi, days) & np.isfinite(vw))[0]
    wd = dayi[win]
    starts = np.searchsorted(wd, days); ends = np.searchsorted(wd, days, side='right')
    pfs = []
    for _ in range(n):
        rows = []
        for d0, s0, e0 in zip(days, starts, ends):
            if e0 - s0 < 3:
                continue
            for k in np.sort(rng.choice(win[s0:e0], 3, replace=False)):
                s = 1 if rng.random() < 0.5 else -1
                r = reg[k]; dist = max(abs(c[k] - vw[k]), MINSTOP[mk] / 4)
                base = max(0.12 * atr[k], MINSTOP[mk])
                if a['fade_on'] is not None and r == a['fade_on']:
                    rows.append((k, -s, max(base, sd[k]), dist, 60, 'fade', r, s))
                elif a['follow_on'] is not None and r == a['follow_on']:
                    mb = trail_bars(P, k, s) if a['fexit'] == 'TRAIL' else 0
                    rows.append((k, s, max(base, dist), np.inf, mb, 'follow', r, s))
        cand = cap2(mk, pd.DataFrame(rows, columns=['k', 'd', 'stop', 'tgt', 'mb', 'arm', 'reg', 'side_band']))
        pfs.append(pf(simulate(mk, to_sig(cand)).R))
    return round(float(np.nanmedian(pfs)), 3), round(float(np.nanpercentile(pfs, 95)), 3)


def lookahead_check(mk, v, ncheck=150, seed=3):
    """Recompute vwap/sd for random trigger bars from a TRUNCATED copy of the data (bars <= k only) and verify the
    trigger and the stop/target inputs are identical."""
    P = prep(mk); D = P['D']
    rng = np.random.default_rng(seed)
    kk, ss = triggers(mk, v['K'])
    for k in rng.choice(kk, min(ncheck, len(kk)), replace=False):
        dd = D['date'][k]
        idx = np.where((D['date'][:k + 1] == dd) & (D['mod'][:k + 1] >= 570))[0]
        tp = (D['h'][idx] + D['l'][idx] + D['c'][idx]) / 3; w = D['vol'][idx]
        vw = (w * tp).sum() / w.sum(); sd = np.sqrt((w * (tp - vw) ** 2).sum() / w.sum())
        assert abs(vw - P['vwap'][k]) < 1e-6 and abs(sd - P['sd'][k]) < 1e-5, (k, vw, P['vwap'][k], sd, P['sd'][k])
        idx2 = idx[:-1]
        tp2 = tp[:-1]; w2 = w[:-1]
        vw2 = (w2 * tp2).sum() / w2.sum(); sd2 = np.sqrt((w2 * (tp2 - vw2) ** 2).sum() / w2.sum())
        s = 1 if D['c'][k] > vw else -1
        assert s * (D['c'][k] - (vw + s * v['K'] * sd)) > 0
        assert s * (D['c'][k - 1] - (vw2 + s * v['K'] * sd2)) <= 1e-9
    return True


def by_year_str(rep):
    return ' '.join(f"{y}:{v['pf']}({v['n']})" for y, v in rep['by_year'].items())


if __name__ == '__main__':
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
    g = run_grid()
    print('=== GRID (selection uses IS columns only) ===')
    print(g[['name', 'NQ_nIS', 'NQ_wkIS', 'NQ_pfIS', 'ES_nIS', 'ES_wkIS', 'ES_pfIS', 'score', 'plateau']].to_string(index=False))
    med_all = {mk: round(float(g[f'{mk}_pfIS'].median()), 3) for mk in ('NQ', 'ES')}
    med_hyp = {mk: round(float(g[~g.control][f'{mk}_pfIS'].median()), 3) for mk in ('NQ', 'ES')}
    print('median IS PF all 24:', med_all, ' hypothesis 20:', med_hyp)
    elig = g[(~g.control) & (g.NQ_nIS >= 50) & (g.ES_nIS >= 50)]
    best = elig.sort_values(['plateau', 'score'], ascending=False).iloc[0]
    chosen = dict(K=float(best.K), regdef=best.regdef, mode=best['mode'], control=False)
    print('CHOSEN on IS:', vname(chosen), 'plateau', best.plateau, 'score', best.score)

    res = dict(slug='gamma_regime_vwap', chosen=vname(chosen), variants_tested=len(VARIANTS),
               median_IS_pf_all24=med_all, median_IS_pf_hyp20=med_hyp,
               grid_IS=g[['name', 'NQ_nIS', 'NQ_wkIS', 'NQ_pfIS', 'ES_nIS', 'ES_wkIS', 'ES_pfIS', 'score', 'plateau']].to_dict('records'))
    for mk in ('NQ', 'ES'):
        lookahead_check(mk, chosen)
        sig = to_sig(candidates(mk, **vargs(chosen)))
        rep = full_report(mk, sig)
        rep['by_year_str'] = by_year_str(rep)
        rep['random_time_control_med_95'] = random_time_control(mk, chosen)
        res[mk] = rep
        print(f'\n=== {mk} full_report ({vname(chosen)}) ===')
        print(json.dumps(rep, default=str))
    print('lookahead check ok (vwap/sd recomputed from truncated data at random trigger bars)')

    print('\n=== GRID with OOS (informational, NOT used for selection) ===')
    print(g[['name', 'NQ_pfIS', 'NQ_pfOOS', 'NQ_nOOS', 'ES_pfIS', 'ES_pfOOS', 'ES_nOOS']].to_string(index=False))
    res['grid_OOS_info'] = g[['name', 'NQ_pfIS', 'NQ_pfOOS', 'NQ_nOOS', 'ES_pfIS', 'ES_pfOOS', 'ES_nOOS']].to_dict('records')

    res['partial_edges'] = {}; res['arm_breakdown_K2'] = {}
    for mk in ('NQ', 'ES'):
        pe = partial_edges(mk, chosen); res['partial_edges'][mk] = pe
        print(f'\n=== {mk} partial edges (chosen variant) ===')
        for k_, v_ in pe.items():
            print(f'  {k_:22s} IS {v_["pf_IS"]}({v_["n_IS"]})  OOS {v_["pf_OOS"]}({v_["n_OOS"]})')
        ab = arm_breakdown(mk); res['arm_breakdown_K2'][mk] = ab
        print(f'--- {mk} arm x regime x side at K 2.0 ---')
        for k_, v_ in ab.items():
            print('  ', k_, ' | '.join(f"{s}: IS {x['pf_IS']}({x['n_IS']}) OOS {x['pf_OOS']}({x['n_OOS']})" for s, x in v_.items()))

    # post-selection diagnostics (NOT used for selection): fade arm reversed, and follow arm split by 20d trend
    def isoos(tr):
        i = tr[tr.entry_t < IS_END]; o = tr[tr.entry_t >= IS_END]
        return dict(n_IS=len(i), pf_IS=round(pf(i.R), 2), n_OOS=len(o), pf_OOS=round(pf(o.R), 2))
    diag = {}
    for mk in ('NQ', 'ES'):
        D = load(mk)
        for K in (2.0, 2.5):
            for name, on in (('fadeC', 1), ('fadeS', -1)):
                s_ = to_sig(candidates(mk, K=K, regdef='ATR', fade_on=on, follow_on=None)); s_['d'] = -s_.d
                diag[f'{mk}_K{K}_ATR_{name}_REVERSED'] = isoos(simulate(mk, s_))
            for fx in ('1559', 'TRAIL'):
                for on, rn in ((1, 'CALM'), (-1, 'STRESS')):
                    tr = simulate(mk, to_sig(candidates(mk, K=K, regdef='ATR', fade_on=None, follow_on=on, fexit=fx)))
                    tr['trend'] = D['trend_d'][tr.k]
                    diag[f'{mk}_K{K}_ATR_follow{rn}_{fx}_with20dTrend'] = isoos(tr[tr.d == tr.trend])
                    diag[f'{mk}_K{K}_ATR_follow{rn}_{fx}_against20dTrend'] = isoos(tr[tr.d == -tr.trend])
    res['diagnostics_post_selection'] = diag
    print('\n=== post-selection diagnostics ===')
    for k_, v_ in diag.items():
        print(f'  {k_:45s} IS {v_["pf_IS"]}({v_["n_IS"]})  OOS {v_["pf_OOS"]}({v_["n_OOS"]})')

    def ok13(r): return r['pf_IS'] >= 1.3 and r['pf_OOS'] >= 1.3
    passed = any(ok13(res[a]) and res[b]['pf_OOS'] >= 1.1 and res[a]['per_week'] >= 3 and res[a]['pf_2x_cost'] >= 1.1
                 for a, b in (('NQ', 'ES'), ('ES', 'NQ')))
    border = any(res[m]['pf_IS'] >= 1.15 and res[m]['pf_OOS'] >= 1.15 for m in ('NQ', 'ES'))
    res['verdict'] = 'PASS' if passed else ('BORDERLINE' if border else 'FAIL')
    print('VERDICT:', res['verdict'])
    json.dump(res, open(OUT, 'w'), default=str, indent=1)
    print('saved', OUT)
