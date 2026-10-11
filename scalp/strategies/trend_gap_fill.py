"""Gap against the 20-day trend, filled in the trend direction from 10:00 (slug: trend_gap_fill).

Rules as implemented (both markets, same rules, sizes scale by ATR):
  * Day table = org_gap_rejection.day_table: trading days = NY dates with a 09:30 bar; R = close of the previous
    trading date's 16:14 bar (fallback pdc); g = o[09:30] - R; ATR = cache atr at 09:30 (previous days only);
    trend = cache trend_d at 09:30 = sign(prev close - SMA20 of closes), previous days only.
  * Setup: |g| >= gmin*ATR, fill direction d = -sign(g). ONLY filter: d == trend (gap against the 20-day trend,
    trade in the trend direction). Days with trend 0/NaN are skipped. No traded-through-level condition.
  * Signal bar k = 09:59 bar (mod 599, must exist), market entry at the 10:00 open (etype 0). One trade per day.
  * Stop 'A' = org_gap_rejection.stopA: 09:30-09:59 extreme opposite to d minus/plus c[09:59], + 0.02*ATR,
    floored at max(0.12*ATR, 20 NQ / 4 ES) (points from the fill).  'F' = max(0.25*ATR, 20 NQ / 4 ES).
  * Exit 'close' = engine force-close 15:59; '2R' = tgt_pts 2*stop, maxbars 180; 'T13' = time exit at the open of
    the first bar >= 13:00 of that day (maxbars = idx(13:00) - (k+1)); on half days without a 13:00 bar the engine
    exits at the day's last close (no maxbars).
  * Grid 18: gmin {0.1,0.2,0.3} x stop {A,F} x exit {close,2R,T13}. Selection on IS only: score = min(NQ IS PF,
    ES IS PF); plateau = mean score over the cell and its gmin neighbours (same stop/exit); cell needs >= 60 IS trades
    on each market; highest plateau wins (tie -> higher score).
  * Controls at the chosen cell (non-selectable): (a) same entry against the trend (d == -trend);
    (b) drift: trend-direction 10:00 entry on all days with |g| < 0.05*ATR, same stop type and exit (stop A mirrored
    to the trend direction); plus info: trend-direction entry on ALL days; (c) engine opposite / random direction /
    bootstrap (full_report); (d) overlap with the TDH proxy (gap_go_midnight.tdh_signals); (e) by side, by year.
Lookahead check: R from a completed previous date, g uses the 09:30 open, ATR/trend from previous days only.
Stop A uses bars 09:30..09:59 only (known at the 09:59 close = k). Entry k+1 = 10:00 open. The T13 maxbars uses
only the clock position of the 13:00 bar (no price information).
"""
import sys, json
sys.path.insert(0, '/home/user/TRADING/scalp'); sys.path.insert(0, '/home/user/TRADING/scalp/strategies')
import numpy as np
import pandas as pd
from engine import load, signals_frame, simulate, full_report, headline, pf, IS_END, START, END
from org_gap_rejection import day_table, stopA, MINSTOP
from gap_go_midnight import tdh_signals, overlap_stats

SLUG = 'trend_gap_fill'
OUT = f'/home/user/TRADING/scalp/results/{SLUG}.json'
GMINS = (0.1, 0.2, 0.3)
GRID = [dict(gmin=g, stop=s, ex=e) for s in ('A', 'F') for e in ('close', '2R', 'T13') for g in GMINS]
_B = {}


def base(mk):
    """Day table with the 09:59 bar, valid trend and the index of the first bar >= 13:00 that day."""
    if mk in _B:
        return _B[mk]
    D = load(mk)
    t = day_table(mk)
    t = t[(t.k59 > t.k0) & t.trend.notna() & (t.trend != 0)].copy()
    t['trend'] = t.trend.astype(int)
    starts = np.r_[0, np.where(np.diff(D['dayi']) != 0)[0] + 1]; ends = np.r_[starts[1:], len(D['dayi'])]
    first = {}
    for a, b in zip(starts, ends):
        j = np.where(D['mod'][a:b] >= 780)[0]
        first[int(D['dayi'][a])] = a + int(j[0]) if len(j) else -1
    t['k13'] = [first.get(int(d), -1) for d in t.dayi]
    _B[mk] = t
    return t


def stop_for(t, mk, d, kind):
    if kind == 'F':
        return np.maximum(0.25 * t.atr.to_numpy(), MINSTOP[mk])
    x = t.copy(); x['d'] = d          # stopA uses the extreme opposite to x.d
    return np.asarray(stopA(x, mk), dtype=float)


def make_sig(mk, t, d, stop, ex):
    d = np.asarray(d, dtype=int)
    st = stop_for(t, mk, d, stop)
    k = t.k59.to_numpy()
    if ex == 'close':
        return signals_frame(k, d, stop_pts=st)
    if ex == '2R':
        return signals_frame(k, d, stop_pts=st, tgt_pts=2 * st, maxbars=180)
    mb = np.where(t.k13.to_numpy() > k + 1, t.k13.to_numpy() - (k + 1), 0)
    return signals_frame(k, d, stop_pts=st, maxbars=mb)


def days_for(mk, gmin, side='with'):
    t = base(mk)
    t = t[t.gA >= gmin]
    return t[(t.d == t.trend)] if side == 'with' else t[(t.d == -t.trend)]


def signals(mk, gmin=0.2, stop='A', ex='close', side='with', return_days=False):
    t = days_for(mk, gmin, side)
    s = make_sig(mk, t, t.d.to_numpy(), stop, ex)
    return (s, t) if return_days else s


def split_pf(tr, mask):
    x = tr[np.asarray(mask, dtype=bool)]
    i = x[x.entry_t < IS_END]; o = x[x.entry_t >= IS_END]
    return dict(n_IS=len(i), pf_IS=round(pf(i.R), 2) if len(i) else None, n_OOS=len(o), pf_OOS=round(pf(o.R), 2) if len(o) else None)


def summ(mk, sig):
    tr = simulate(mk, sig)
    return dict(n=len(tr), pf=round(pf(tr.R), 3), pf_2x=round(pf(simulate(mk, sig, costmult=2.0).R), 3)) | split_pf(tr, np.ones(len(tr), bool))


def run_grid():
    rows = []
    for p in GRID:
        r = dict(p)
        for mk in ('NQ', 'ES'):
            tr = simulate(mk, signals(mk, **p))
            i = tr[tr.entry_t < IS_END]; o = tr[tr.entry_t >= IS_END]
            r[f'{mk}_n'] = len(tr); r[f'{mk}_nIS'] = len(i)
            r[f'{mk}_IS'] = round(pf(i.R), 3); r[f'{mk}_OOS'] = round(pf(o.R), 3)
        r['score'] = min(r['NQ_IS'], r['ES_IS'])
        rows.append(r)
    g = pd.DataFrame(rows)
    pl = []
    for _, r in g.iterrows():
        same = g[(g.stop == r.stop) & (g.ex == r.ex)].sort_values('gmin').reset_index(drop=True)
        pos = int(np.where(np.isclose(same.gmin, r.gmin))[0][0])
        pl.append(float(same.iloc[max(pos - 1, 0): pos + 2].score.mean()))
    g['plateau'] = pl
    g['eligible'] = (g.NQ_nIS >= 60) & (g.ES_nIS >= 60)
    return g


# ------------------------------------------------------------------ controls
def ctrl_drift(mk, stop, ex, gmax=0.05):
    """KEY: trend-direction 10:00 entry on small-gap days (|g| < 0.05 ATR), same stop and exit; also all days."""
    t = base(mk)
    out = {}
    for name, m in (('small_gap_lt0.05', t.gA < gmax), ('all_days', np.ones(len(t), bool)),
                    ('gap_with_trend_ge0.1_trade_trend', (t.gA >= 0.1) & (t.d == -t.trend))):
        x = t[np.asarray(m)]
        out[name] = summ(mk, make_sig(mk, x, x.trend.to_numpy(), stop, ex))
    return out


def partials(mk, sig, days):
    tr = simulate(mk, sig).join(days.set_index('k59')[['gA', 'g', 'pdh', 'pdl', 'o930', 'dow', 'atr']], on='k')
    outside = np.where(tr.g > 0, tr.o930 > tr.pdh, tr.o930 < tr.pdl)
    return {'long(trend_up,gap_down)': split_pf(tr, tr.d == 1), 'short(trend_down,gap_up)': split_pf(tr, tr.d == -1),
            'gap<0.3ATR': split_pf(tr, tr.gA < 0.3), 'gap0.3-0.6ATR': split_pf(tr, (tr.gA >= 0.3) & (tr.gA < 0.6)),
            'gap>=0.6ATR': split_pf(tr, tr.gA >= 0.6),
            'open_outside_prior_range': split_pf(tr, outside), 'open_inside_prior_range': split_pf(tr, ~outside),
            'Mon-Tue': split_pf(tr, tr.dow <= 1), 'Wed-Fri': split_pf(tr, tr.dow >= 2),
            'exit_stop': split_pf(tr, tr.reason == 1), 'exit_time_or_tgt': split_pf(tr, tr.reason != 1)}


def tdh_before(mk, tr, tt):
    """Tradeable TDH split: did the TDH proxy already signal (same day) at or before the 09:59 bar?"""
    D = load(mk)
    first = tt.groupby(D['dayi'][tt.k])['k'].min()
    dk = pd.Series(D['dayi'][tr.k.to_numpy()]).map(first).to_numpy()
    fired = ~np.isnan(dk) & (dk <= tr.k.to_numpy())
    return dict(share_fired=round(float(fired.mean()), 3), fired=split_pf(tr, fired), not_fired=split_pf(tr, ~fired))


def by_year_str(rep):
    return ' '.join(f"{y}:{v['pf']}({v['n']})" for y, v in rep['by_year'].items())


if __name__ == '__main__':
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    grid = run_grid()
    print(grid.to_string())
    el = grid[grid.eligible]
    best = el.sort_values(['plateau', 'score'], ascending=False).iloc[0]
    P = dict(gmin=float(best.gmin), stop=best.stop, ex=best.ex)
    print('\nCHOSEN (IS plateau):', P)
    res = dict(slug=SLUG, chosen=P, variants_tested=len(grid),
               median_IS_pf={'NQ': round(float(grid.NQ_IS.median()), 3), 'ES': round(float(grid.ES_IS.median()), 3)},
               median_OOS_pf_info={'NQ': round(float(grid.NQ_OOS.median()), 3), 'ES': round(float(grid.ES_OOS.median()), 3)},
               grid=grid.to_dict('records'))
    for mk in ('NQ', 'ES'):
        sig, days = signals(mk, **P, return_days=True)
        rep = full_report(mk, sig)
        tr = simulate(mk, sig)
        rep['by_side_IS_OOS'] = {'long': split_pf(tr, tr.d == 1), 'short': split_pf(tr, tr.d == -1)}
        rep['ctrl_a_against_trend'] = summ(mk, signals(mk, **P, side='against'))
        rep['ctrl_b_drift'] = ctrl_drift(mk, P['stop'], P['ex'])
        tt = simulate(mk, tdh_signals(mk))
        rep['ctrl_d_tdh_overlap'] = overlap_stats(mk, tr, tt)
        rep['ctrl_d_tdh_overlap_note'] = ('non-TDH-day split is conditioned on the future (a TDH breakout can come after '
                                          '10:00); tradeable split below uses only TDH signals fired by 09:59')
        rep['ctrl_d_tdh_fired_by_0959'] = tdh_before(mk, tr, tt)
        rep['ctrl_b_drift_info_lt0.1'] = ctrl_drift(mk, P['stop'], P['ex'], gmax=0.1)['small_gap_lt0.05']
        rep['OOS_2025_vs_2026'] = {'2025-04..12': split_pf(tr, tr.entry_t.dt.year == 2025), '2026': split_pf(tr, tr.entry_t.dt.year == 2026)}
        rep['entry_minutes_check'] = sorted(set((tr.entry_t.dt.hour * 60 + tr.entry_t.dt.minute).tolist()))
        rep['partials'] = partials(mk, sig, days)
        rep['by_year_str'] = by_year_str(rep)
        res[mk] = rep
        print('\n' + mk, json.dumps({k: v for k, v in rep.items() if k != 'by_year'}, default=str))
    json.dump(res, open(OUT, 'w'), default=str, indent=1)
    print('saved', OUT)
