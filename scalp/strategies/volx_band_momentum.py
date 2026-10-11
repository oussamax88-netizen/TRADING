"""Noise-band momentum only when volatility is NOT compressed (Zarattini 'atrF' filter inverted).
Slug: volx_band_momentum.  Both markets, same rules.

Mechanical rules as implemented
-------------------------------
Entries / exits: exactly zarattini_band_momentum.prep / bands / candidates (imported, unchanged, filt=False):
  full RTH days only (>= 380 RTH 1m bars), decision bars = closes of 09:59, 10:29, ..., 15:29;
  NOISE: UB = max(O,pdc)*(1+V*sigma_m), LB = min(O,pdc)*(1-V*sigma_m), sigma_m = mean |close_m/O-1| over 14 previous
         full days; VWAP: vwap*(1 +/- V*sigma_m) with sigma_m from |c-vwap|/vwap;
  c > UB -> long, c < LB -> short, market at next open; TRAIL/BAND exits checked at later decision bars and executed
  at the next open (engine maxbars); catastrophic stop max(0.35*ATR14, 20 NQ / 4 ES); otherwise exit at 15:59.
Regime filter (the ONLY difference): keep a day's candidates only if ratio >= lb, ratio = prep()['ratio'] =
  ATR14(day) / median(ATR14 of previous 100 full RTH days) (ATR14 = cache 'atr' = 14-day mean range of PREVIOUS days,
  median window shifted by one day -> known before 09:30). NaN ratio -> no trade.  Filtering whole days before the
  engine is identical to filtering rowok inside candidates() because positions never carry over a day.
Grid (24): REF {NOISE, VWAP} x V {1.0, 1.25} x exit {TRAIL, BAND} x lb {0.8, 0.9, 1.0}.
Selection (IS only): eligible = >= 60 IS trades on BOTH markets; score = min(NQ IS PF, ES IS PF);
  plateau = mean score over the cell and its lb neighbours (same REF/V/exit, |dlb| = 0.1); best plateau wins, tie -> score.
Controls: complement (ratio < lb) and unfiltered cell; engine opposite / random direction / 2x cost / bootstrap;
  info: with/against 20d trend, time of day; TDH-proxy overlap, daily-R correlation, PF on non-TDH days.
"""
import sys, json
sys.path.insert(0, '/home/user/TRADING/scalp'); sys.path.insert(0, '/home/user/TRADING/scalp/strategies')
import numpy as np
import pandas as pd
from engine import load, simulate, full_report, headline, pf, IS_END, START, END
import zarattini_band_momentum as Z
from gap_go_midnight import tdh_signals, daily

OUT = '/home/user/TRADING/scalp/results/volx_band_momentum.json'
VARIANTS = [dict(ref=r, V=v, exit=x, lb=b) for r in ('NOISE', 'VWAP') for v in (1.0, 1.25) for x in ('TRAIL', 'BAND')
            for b in (0.8, 0.9, 1.0)]
_C = {}


def vname(v):
    return f"{v['ref']}_V{v['V']}_{v['exit']}_lb{v['lb']}"


def base(mk, ref, V, exit):
    key = (mk, ref, V, exit)
    if key not in _C:
        c = Z.candidates(mk, ref=ref, V=V, exit=exit, filt=False)
        c['ratio'] = Z.prep(mk)['ratio'][c.row.to_numpy()]
        _C[key] = c
    return _C[key]


def cands(mk, ref, V, exit, lb=None, mode='ge'):
    c = base(mk, ref, V, exit)
    if mode == 'none':
        return c
    r = c.ratio.to_numpy()
    m = (r >= lb) if mode == 'ge' else (r < lb)          # NaN -> False in both
    return c[m].reset_index(drop=True)


def sig_of(c):
    return Z.to_sig(c)


def split(tr):
    i = tr[tr.entry_t < IS_END]; o = tr[tr.entry_t >= IS_END]
    return dict(n_IS=len(i), pf_IS=round(pf(i.R), 3) if len(i) else None, n_OOS=len(o), pf_OOS=round(pf(o.R), 3) if len(o) else None)


def run_grid():
    rows = []
    for v in VARIANTS:
        r = dict(name=vname(v), **v)
        for mk in ('NQ', 'ES'):
            tr = simulate(mk, sig_of(cands(mk, v['ref'], v['V'], v['exit'], v['lb'])))
            i = tr[tr.entry_t < IS_END]
            r[f'{mk}_nIS'] = len(i); r[f'{mk}_pfIS'] = round(pf(i.R), 3)
        rows.append(r)
    g = pd.DataFrame(rows)
    g['score'] = g[['NQ_pfIS', 'ES_pfIS']].min(axis=1)
    g['plateau'] = [round(g[(g.ref == r.ref) & (g.V == r.V) & (g.exit == r.exit) & ((g.lb - r.lb).abs() <= 0.11)].score.mean(), 3)
                    for _, r in g.iterrows()]
    g['eligible'] = (g.NQ_nIS >= 60) & (g.ES_nIS >= 60)
    return g


def annotate(mk, c, tr):
    D = load(mk); P = Z.prep(mk)
    m = c.set_index('k')
    tr = tr.copy()
    tr['row'] = m.loc[tr.k, 'row'].to_numpy(); tr['slot'] = m.loc[tr.k, 'slot'].to_numpy()
    tr['ratio'] = P['ratio'][tr.row]; tr['trend'] = D['trend_d'][tr.k]
    tr['gap'] = np.sign(P['O'][tr.row] - P['pdc'][tr.row]); tr['first'] = ~tr.duplicated('row')
    return tr


def partial_edges(mk, c, tr, lb):
    tr = annotate(mk, c, tr)
    subs = {  # fixed list, defined before looking at any result
        'long': tr.d == 1, 'short': tr.d == -1,
        '10:00-11:30': tr.slot <= 2, '11:30-13:30': (tr.slot >= 3) & (tr.slot <= 6), '13:30-15:30': tr.slot >= 7,
        'first_of_day': tr['first'], 'reentry': ~tr['first'],
        'with_20d_trend': tr.d == tr.trend, 'against_20d_trend': tr.d == -tr.trend,
        'with_gap': tr.d == tr.gap, 'against_gap': tr.d == -tr.gap,
        f'ratio_{lb}-1.1': tr.ratio < 1.1, 'ratio_1.1-1.3': (tr.ratio >= 1.1) & (tr.ratio < 1.3), 'ratio>=1.3': tr.ratio >= 1.3,
    }
    return {k: split(tr[m]) for k, m in subs.items()}


def tdh_overlap(mk, tr):
    tt = simulate(mk, tdh_signals(mk))
    g = daily(mk, tr); t = daily(mk, tt)
    both = g.index.intersection(t.index)
    D = load(mk)
    alld = np.unique(D['date'][D['mod'] == 600])
    alld = alld[(alld >= START.astype('datetime64[D]').astype(int)) & (alld < END.astype('datetime64[D]').astype(int))]
    ga = g.R.reindex(alld).fillna(0); ta = t.R.reindex(alld).fillna(0)
    trd = tr.copy(); trd['date'] = D['date'][trd.entry_k]
    non = trd[~trd.date.isin(t.index)]; on = trd[trd.date.isin(t.index)]
    return dict(strategy_days=len(g), tdh_days=len(t), overlap_days=len(both), overlap_share=round(len(both) / max(len(g), 1), 3),
                same_dir_on_overlap=int((g.loc[both, 'd'] == t.loc[both, 'd']).sum()),
                corr_daily_R_all_days=round(float(np.corrcoef(ga, ta)[0, 1]), 3),
                tdh_proxy=split(tt), on_tdh_days=split(on), non_tdh_days=split(non))


def by_year_str(rep):
    return ' '.join(f"{y}:{v['pf']}({v['n']})" for y, v in rep['by_year'].items())


def lookahead_checks(mk, c):
    D = load(mk); P = Z.prep(mk)
    # ATR used by ratio/stop must be constant within the day (i.e. built from previous days only)
    K = P['K']; a = D['atr'][K]
    ok = np.isfinite(a).all(axis=1)
    assert np.allclose(a[ok], P['atr'][ok][:, None]), 'atr changes intraday'
    # median window excludes the current day
    med = pd.Series(P['atr']).rolling(100).median().shift(1).to_numpy()
    j = 300; assert np.isclose(med[j], np.median(P['atr'][j - 100:j]))
    # decision bars at <= 15:29 close, entry at k+1 (engine), exits only after k
    assert (D['mod'][c.k] <= 929).all() and (c.mb >= 0).all()
    return True


if __name__ == '__main__':
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    g = run_grid()
    print('=== GRID (IN-SAMPLE ONLY, 24 variants) ===')
    print(g.drop(columns=['ref', 'V', 'exit', 'lb']).to_string(index=False))
    med_is = {mk: round(float(g[f'{mk}_pfIS'].median()), 3) for mk in ('NQ', 'ES')}
    print('median IS PF of all 24:', med_is)
    best = g[g.eligible].sort_values(['plateau', 'score'], ascending=False).iloc[0]
    ch = dict(ref=best.ref, V=float(best.V), exit=best.exit, lb=float(best.lb))
    print('CHOSEN on IS:', vname(ch), 'plateau', best.plateau, 'score', best.score)
    res = dict(slug='volx_band_momentum', chosen=vname(ch), variants_tested=len(VARIANTS), median_IS_pf=med_is,
               grid_IS=g.to_dict('records'))
    for mk in ('NQ', 'ES'):
        c = cands(mk, ch['ref'], ch['V'], ch['exit'], ch['lb'])
        lookahead_checks(mk, c)
        sig = sig_of(c)
        rep = full_report(mk, sig); rep['by_year_str'] = by_year_str(rep); res[mk] = rep
        print(f'\n=== {mk} full_report ({vname(ch)}) ===\n' + json.dumps(rep, default=str))
        tr = simulate(mk, sig)
        ctl = dict(chosen=split(tr),
                   complement_ratio_lt_lb=split(simulate(mk, sig_of(cands(mk, ch['ref'], ch['V'], ch['exit'], ch['lb'], 'lt')))),
                   unfiltered=split(simulate(mk, sig_of(cands(mk, ch['ref'], ch['V'], ch['exit'], mode='none')))),
                   zarattini_calm_le1=split(simulate(mk, Z.to_sig(Z.candidates(mk, ch['ref'], ch['V'], ch['exit'], filt=True)))))
        res[f'{mk}_controls'] = ctl
        print(mk, 'controls:', json.dumps(ctl))
        res[f'{mk}_partial_edges'] = pe = partial_edges(mk, c, tr, ch['lb'])
        for k_, v_ in pe.items():
            print(f'  {k_:18s} IS {v_["pf_IS"]}({v_["n_IS"]})  OOS {v_["pf_OOS"]}({v_["n_OOS"]})')
        res[f'{mk}_tdh'] = td = tdh_overlap(mk, tr)
        print(mk, 'TDH overlap:', json.dumps(td))
        # sanity: planned trail/band exits happen at the planned bar
        trf = simulate(mk, sig, period=False); m = c.set_index('k')
        mb = m.loc[trf.k, 'mb'].to_numpy(); r4 = (trf.reason == 4).to_numpy()
        assert (trf.exit_k.to_numpy()[r4] == (trf.k.to_numpy() + mb + 1)[r4]).all()
        print(mk, 'sanity ok: candidates', len(c), 'taken', len(trf), 'reasons', trf.reason.value_counts().to_dict())
    comp_weaker = {mk: (res[f'{mk}_controls']['complement_ratio_lt_lb']['pf_IS'] or 0) < res[mk]['pf_IS'] for mk in ('NQ', 'ES')}
    res['complement_weaker_IS'] = comp_weaker
    print('complement weaker in IS:', comp_weaker)
    # partials positive in BOTH halves (PF > 1, n >= 20 each half) - reported, not selected
    res['partials_both_halves'] = {mk: [k for k, v in res[f'{mk}_partial_edges'].items()
                                        if v['n_IS'] >= 20 and v['n_OOS'] >= 20 and v['pf_IS'] > 1 and v['pf_OOS'] > 1]
                                   for mk in ('NQ', 'ES')}
    print('partials PF>1 in both halves:', res['partials_both_halves'])
    # OOS of every grid cell: informational only, after the choice was fixed
    for mk in ('NQ', 'ES'):
        g[f'{mk}_pfOOS'] = [split(simulate(mk, sig_of(cands(mk, v['ref'], v['V'], v['exit'], v['lb']))))['pf_OOS'] for v in VARIANTS]
    print('\n=== GRID with OOS (informational, NOT used for selection) ===')
    print(g[['name', 'NQ_nIS', 'NQ_pfIS', 'NQ_pfOOS', 'ES_nIS', 'ES_pfIS', 'ES_pfOOS']].to_string(index=False))
    res['grid_OOS_info'] = g[['name', 'NQ_pfIS', 'NQ_pfOOS', 'ES_pfIS', 'ES_pfOOS']].to_dict('records')

    def ok13(r): return r['pf_IS'] >= 1.3 and r['pf_OOS'] >= 1.3
    passed = any(ok13(res[a]) and res[b]['pf_OOS'] >= 1.1 and res[a]['per_week'] >= 3 and res[a]['pf_2x_cost'] >= 1.1
                 for a, b in (('NQ', 'ES'), ('ES', 'NQ')))
    border = any(res[m]['pf_IS'] >= 1.15 and res[m]['pf_OOS'] >= 1.15 for m in ('NQ', 'ES'))
    res['verdict'] = 'PASS' if passed else ('BORDERLINE' if border else 'FAIL')
    print('VERDICT:', res['verdict'])
    json.dump(res, open(OUT, 'w'), default=str, indent=1)
    print('saved', OUT)
