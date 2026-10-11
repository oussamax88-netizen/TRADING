"""ICT opening-range-gap (ORG) rejection: CE traded through by 10:00, then trade the fill direction (slug: org_gap_rejection).

Rules as implemented (both markets, same rules, sizes scale by ATR):
  * Trading days = NY dates that have a 09:30 bar (mod 570). 'Previous date' = previous such date (Friday for Monday).
  * R = close of the 16:14 bar (mod 974, i.e. the 16:15 print) of the previous date; if that bar is missing
    (half days, data holes) R = pdc (previous RTH close, cache value).
  * g = o[09:30 bar] - R;  ATR = cache atr at the 09:30 bar (14-day mean of previous RTH ranges).
    Setup if |g| >= gmin * ATR.  Fill direction d = -sign(g) (gap up -> short, gap down -> long).
  * Level LV = R + g/2 (CE = consequent encroachment, half the gap) or LV = R (full fill).
  * Entry A ('A'): the gap is 'traded through' if during the 09:30-09:59 bars (mod 570..599)
    gap up: min(low) <= LV, gap down: max(high) >= LV. Signal bar k = 09:59 bar (mod 599, must exist),
    market entry at the 10:00 open (etype 0).
    stop_pts = (short) max(high 09:30..09:59) - c[k] + 0.02*ATR, (long) c[k] - min(low 09:30..09:59) + 0.02*ATR,
    floored at max(0.12*ATR, 20 pts NQ / 4 pts ES). Stop is measured from the fill (engine stop_pts).
  * Entry B ('B'): k = 09:30 bar, stop-entry (etype 2) at LV in direction d, valid 60 bars (09:31..10:30),
    stop_pts = max(0.25*ATR, 0.12*ATR, 20/4).  (If the 09:31+ bar opens beyond LV the engine fills at the open.)
  * Exit: 'close' = engine force-close at the 15:59 open; 'tgt' = tgt_px = R - g (one gap beyond R in the fill
    direction) or 15:59. If at the time of entry the price is already beyond R - g the engine skips the trade
    (target behind the entry) -> the 'tgt' variants trade slightly fewer days. One trade per day.
  * Grid (24): gmin {0.1, 0.2, 0.3} x level {CE, FULL} x entry {A, B} x exit {close, tgt}.
  * Selection (IS only, fixed before looking): score = geometric mean of the IS PF on NQ and ES; plateau score =
    mean of that score over the variant and its gmin neighbours (same level/entry/exit); highest plateau wins.
  * Controls: opposite direction (engine for stop_pts-only variants, by hand otherwise); random direction for
    Entry A (engine) and by hand for Entry B (same fill bar and price: stop-entry for d, limit for -d);
    'no-fill' days: |g| >= gmin but LV NOT traded through by 09:59 -> same 10:00 entry in the fill direction
    (and in the gap direction).
  * Information printed after the chosen variant (NOT used for selection/verdict, partly post hoc): the original probe
    cell (0.2/CE/A/close), the Entry-B CE family, the unconditional gap fade (all |g| >= gmin days, 10:00 entry in the
    fill direction, Entry-A stop, hold to close) split by 20-day trend, and what IS-only rules with a minimum IS
    sample (30 / 50 trades per market) would have picked.

Lookahead check: R is from a completed previous date (16:14 bar). g uses the 09:30 open, ATR is from previous days.
Entry A uses highs/lows/closes of bars 09:30..09:59 only, all known at the close of the 09:59 bar (k).
Entry B uses only R, g, ATR (known at the 09:30 close) and the engine's stop-entry mechanics from k+1 on.
"""
import sys, json
sys.path.insert(0, '/home/user/TRADING/scalp')
import numpy as np
import pandas as pd
from engine import load, signals_frame, simulate, full_report, headline, pf, IS_END, START, END

MINSTOP = {'NQ': 20.0, 'ES': 4.0}
SLUG = 'org_gap_rejection'
OUT = f'/home/user/TRADING/scalp/results/{SLUG}.json'
_T = {}


def day_bars(D, minute):
    """dict dayi -> first bar index with that exact NY minute."""
    idx = np.where(D['mod'] == minute)[0]
    di = D['dayi'][idx]
    _, first = np.unique(di, return_index=True)
    return dict(zip(di[first].tolist(), idx[first].tolist()))


def day_table(mk):
    """One row per trading day with everything known at 09:30 / 09:59 (no direction decision)."""
    if mk in _T:
        return _T[mk]
    D = load(mk)
    o, h, l, c, mod, atr = D['o'], D['h'], D['l'], D['c'], D['mod'], D['atr']
    i570 = day_bars(D, 570); i599 = day_bars(D, 599); i974 = day_bars(D, 974)
    days = sorted(i570)
    rows = []
    for j in range(1, len(days)):
        di, prev = days[j], days[j - 1]
        k0 = i570[di]
        a = atr[k0]
        if np.isnan(a):
            continue
        if prev in i974:
            R = c[i974[prev]]; src = 1
        else:
            R = D['pdc'][k0]; src = 0
        if np.isnan(R):
            continue
        g = o[k0] - R
        if g == 0:
            continue
        k59 = i599.get(di, -1)
        if k59 > k0:
            seg = slice(k0, k59 + 1)            # bars 09:30 .. 09:59, all closed at k59's close
            hmax, lmin = h[seg].max(), l[seg].min()
            c59 = c[k59]
        else:
            hmax = lmin = c59 = np.nan
        rows.append(dict(dayi=di, k0=k0, k59=k59, date=D['date'][k0], dow=D['dow'][k0], R=R, src=src, g=g, atr=a,
                         gA=abs(g) / a, o930=o[k0], c930=c[k0], hmax=hmax, lmin=lmin, c59=c59,
                         trend=D['trend_d'][k0], pdh=D['pdh'][k0], pdl=D['pdl'][k0], pdc=D['pdc'][k0],
                         onh=D['on_high'][k0], onl=D['on_low'][k0]))
    t = pd.DataFrame(rows)
    t['d'] = -np.sign(t.g).astype(int)
    t['ent_t'] = pd.to_datetime(t.date.to_numpy() * 86400 * 10**9)
    _T[mk] = t
    return t


def level(t, lev):
    return t.R + t.g / 2 if lev == 'CE' else t.R


def filled(t, lev):
    LV = level(t, lev)
    return np.where(t.g > 0, t.lmin <= LV, t.hmax >= LV) & (t.k59 > t.k0)


def stopA(t, mk):
    raw = np.where(t.d == -1, t.hmax - t.c59, t.c59 - t.lmin) + 0.02 * t.atr
    return np.maximum.reduce([raw, 0.12 * t.atr, np.full(len(t), MINSTOP[mk])])


def signals(mk, gmin=0.2, lev='CE', entry='A', ex='close', return_days=False):
    t = day_table(mk)
    t = t[t.gA >= gmin].copy()
    t['LV'] = level(t, lev)
    t['tgtpx'] = t.R - t.g
    tg = np.nan if ex == 'close' else t.tgtpx.to_numpy()
    if entry == 'A':
        t = t[filled(t, lev)].copy()
        tg = np.nan if ex == 'close' else t.tgtpx.to_numpy()
        t['stop'] = stopA(t, mk)
        sig = signals_frame(t.k59.to_numpy(), t.d.to_numpy(), etype=0, stop_pts=t.stop.to_numpy(), tgt_px=tg)
    else:
        t['stop'] = np.maximum.reduce([0.25 * t.atr, 0.12 * t.atr, np.full(len(t), MINSTOP[mk])])
        sig = signals_frame(t.k0.to_numpy(), t.d.to_numpy(), etype=2, eprice=t.LV.to_numpy(), ebars=60,
                            stop_pts=t.stop.to_numpy(), tgt_px=tg)
    return (sig, t) if return_days else sig


def split_pf(tr, mask):
    x = tr[np.asarray(mask, dtype=bool)]
    i = x[x.entry_t < IS_END]; o = x[x.entry_t >= IS_END]
    return dict(n_IS=len(i), pf_IS=round(pf(i.R), 2) if len(i) else None, n_OOS=len(o), pf_OOS=round(pf(o.R), 2) if len(o) else None)


def is_oos(tr):
    return split_pf(tr, np.ones(len(tr), bool))


# ------------------------------------------------------------------ controls
def ctrl_opposite_manual(mk, sig, days, entry, ex):
    """Opposite direction for variants the engine cannot mirror (absolute target / stop entry)."""
    D = load(mk)
    days = days.set_index('k0' if entry == 'B' else 'k59')
    s = sig.copy()
    s['d'] = -s.d
    if ex == 'tgt':
        # mirrored target: same distance from the reference price (c[k] for A, LV for B), points from the fill
        ref = D['c'][s.k] if entry == 'A' else s.eprice.to_numpy()
        dist = np.abs(ref - s.tgt_px.to_numpy())
        s['tgt_px'] = np.nan; s['tgt_pts'] = dist
    if entry == 'B':
        s['etype'] = 1      # the mirror of a stop-entry at LV is a limit at LV in the other direction
    return is_oos(simulate(mk, s)) | dict(pf=round(pf(simulate(mk, s).R), 3))


def ctrl_random_B(mk, sig, n=30, seed=1):
    """Entry B random direction at the SAME fill bar and price: stop-entry for d, limit at the same price for -d."""
    rng = np.random.default_rng(seed)
    tr = simulate(mk, sig, period=False)
    base = sig.set_index('k').loc[tr.k]
    out = []
    for _ in range(n):
        dd = np.where(rng.random(len(tr)) < 0.5, 1, -1) * tr.d.to_numpy()
        same = dd == tr.d.to_numpy()
        tgt = base.tgt_px.to_numpy()
        tpts = np.where(np.isnan(tgt), np.inf, np.abs(tgt - tr.entry.to_numpy()))
        s = signals_frame(tr.entry_k.to_numpy() - 1, dd, etype=np.where(same, 2, 1), eprice=tr.entry.to_numpy(), ebars=1,
                          stop_pts=base.stop_pts.to_numpy(), tgt_pts=tpts)
        out.append(pf(simulate(mk, s).R))
    return dict(median=round(float(np.nanmedian(out)), 3), p95=round(float(np.nanpercentile(out, 95)), 3))


def ctrl_random_A_tgt(mk, sig, n=30, seed=1):
    """Entry A with absolute target: random direction, target as points (distance from c[k])."""
    D = load(mk)
    rng = np.random.default_rng(seed)
    dist = np.abs(D['c'][sig.k] - sig.tgt_px.to_numpy())
    out = []
    for _ in range(n):
        s = sig.copy(); s['d'] = np.where(rng.random(len(s)) < 0.5, 1, -1); s['tgt_px'] = np.nan; s['tgt_pts'] = dist
        out.append(pf(simulate(mk, s).R))
    return dict(median=round(float(np.nanmedian(out)), 3), p95=round(float(np.nanpercentile(out, 95)), 3))


def ctrl_nofill(mk, gmin, lev, ex):
    """Days with |g| >= gmin where LV was NOT traded through by 09:59: same 10:00 entry, fill dir and gap dir."""
    t = day_table(mk)
    t = t[(t.gA >= gmin) & (t.k59 > t.k0)].copy()
    t = t[~filled(t, lev)].copy()
    res = {}
    for name, sgn in (('fill_dir', 1), ('gap_dir', -1)):
        d = sgn * t.d.to_numpy()
        # stop on the far side of the 09:30-09:59 range (same formula, mirrored for the chosen direction)
        raw = np.where(d == -1, t.hmax - t.c59, t.c59 - t.lmin) + 0.02 * t.atr
        st = np.maximum.reduce([raw, 0.12 * t.atr, np.full(len(t), MINSTOP[mk])])
        tg = np.nan
        if ex == 'tgt':
            tg = np.where(d == t.d, t.R - t.g, t.R + 2 * t.g)     # gap dir: mirror (one more gap beyond the open)
        s = signals_frame(t.k59.to_numpy(), d, etype=0, stop_pts=st, tgt_px=tg)
        tr = simulate(mk, s)
        res[name] = dict(n=len(tr), pf=round(pf(tr.R), 3)) | is_oos(tr)
    return res


def ctrl_all_gap_days(mk, gmin, ex='close'):
    """Unconditional gap fade: ALL days with |g| >= gmin (filled or not), 10:00 market entry in the fill direction,
    Entry-A stop formula, hold to close. Tells whether the 'CE traded through' condition adds anything."""
    t = day_table(mk)
    t = t[(t.gA >= gmin) & (t.k59 > t.k0)].copy()
    s = signals_frame(t.k59.to_numpy(), t.d.to_numpy(), etype=0, stop_pts=stopA(t, mk))
    tr = simulate(mk, s)
    return dict(n=len(tr), pf=round(pf(tr.R), 3)) | is_oos(tr)


def move_stats(mk, gmin=0.2, lev='CE'):
    """Diagnostic: mean 10:00 open -> 15:59 open move in the fill direction (ATR units), filled vs not filled."""
    D = load(mk)
    i959 = day_bars(D, 959); i600 = day_bars(D, 600)
    t = day_table(mk)
    t = t[(t.gA >= gmin) & (t.k59 > t.k0)].copy()
    t['f'] = filled(t, lev)
    t['mv'] = [D['o'][i959[di]] - D['o'][i600[di]] if (di in i959 and di in i600) else np.nan for di in t.dayi]
    t['mv'] = t.d * t.mv / t.atr
    t['IS'] = t.ent_t < IS_END
    t = t[(t.ent_t >= START) & (t.ent_t < END)]
    out = {}
    for (f, s), g in t.groupby(['f', 'IS']):
        out[f"{'fill' if f else 'nofill'}_{'IS' if s else 'OOS'}"] = dict(n=int(g.mv.notna().sum()), mean_atr=round(float(g.mv.mean()), 3))
    return out


# ------------------------------------------------------------------ partial edges
def partials(mk, sig, days, entry):
    tr = simulate(mk, sig)
    m = days.set_index('k59' if entry == 'A' else 'k0')
    tr = tr.join(m[['atr', 'gA', 'g', 'R', 'pdh', 'pdl', 'trend', 'dow', 'o930', 'src']].rename(columns={'R': 'Rref'}), on='k')
    outside = np.where(tr.g > 0, tr.o930 > tr.pdh, tr.o930 < tr.pdl)
    hr = tr.entry_t.dt.hour * 60 + tr.entry_t.dt.minute
    return {
        'long(gap_down)': split_pf(tr, tr.d == 1), 'short(gap_up)': split_pf(tr, tr.d == -1),
        'with_trend_d': split_pf(tr, tr.d == tr.trend), 'against_trend_d': split_pf(tr, tr.d != tr.trend),
        'gap<0.3ATR': split_pf(tr, tr.gA < 0.3), 'gap0.3-0.6ATR': split_pf(tr, (tr.gA >= 0.3) & (tr.gA < 0.6)),
        'gap>=0.6ATR': split_pf(tr, tr.gA >= 0.6),
        'open_outside_prior_range': split_pf(tr, outside), 'open_inside_prior_range': split_pf(tr, ~outside),
        'Mon-Tue': split_pf(tr, tr.dow <= 1), 'Wed-Fri': split_pf(tr, tr.dow >= 2),
        'fill_before_10:00': split_pf(tr, hr < 600), 'fill_10:00+': split_pf(tr, hr >= 600),
    }


def by_year_str(rep):
    return ' '.join(f"{y}:{v['pf']}({v['n']})" for y, v in rep['by_year'].items())


# ------------------------------------------------------------------ main
GRID = [dict(gmin=gm, lev=lv, entry=en, ex=ex) for en in ('A', 'B') for lv in ('CE', 'FULL') for ex in ('close', 'tgt')
        for gm in (0.1, 0.2, 0.3)]


def run_grid():
    rows = []
    for p in GRID:
        r = dict(p)
        for mk in ('NQ', 'ES'):
            tr = simulate(mk, signals(mk, **p))
            i = tr[tr.entry_t < IS_END]; o = tr[tr.entry_t >= IS_END]
            r[f'{mk}_n'] = len(tr); r[f'{mk}_nIS'] = len(i)
            r[f'{mk}_IS'] = round(pf(i.R), 3); r[f'{mk}_OOS'] = round(pf(o.R), 3)
        r['geo_IS'] = float(np.sqrt(max(r['NQ_IS'], 0) * max(r['ES_IS'], 0)))
        rows.append(r)
    g = pd.DataFrame(rows)
    sm = []
    for _, r in g.iterrows():
        same = g[(g.lev == r.lev) & (g.entry == r.entry) & (g.ex == r.ex)].sort_values('gmin').reset_index(drop=True)
        pos = int(np.where(np.isclose(same.gmin, r.gmin))[0][0])
        nb = same.iloc[max(pos - 1, 0): pos + 2]
        sm.append(float(nb.geo_IS.mean()))
    g['smooth_IS'] = sm
    return g


if __name__ == '__main__':
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    grid = run_grid()
    print(grid.to_string())
    best = grid.sort_values(['smooth_IS', 'geo_IS'], ascending=False).iloc[0]
    P = dict(gmin=float(best.gmin), lev=best.lev, entry=best.entry, ex=best.ex)
    print('\nCHOSEN (IS plateau):', P)
    res = dict(slug=SLUG, chosen=P, variants_tested=len(grid),
               median_IS_pf={'NQ': round(float(grid.NQ_IS.median()), 3), 'ES': round(float(grid.ES_IS.median()), 3)},
               median_OOS_pf_info={'NQ': round(float(grid.NQ_OOS.median()), 3), 'ES': round(float(grid.ES_OOS.median()), 3)},
               grid=grid.to_dict('records'))
    for mk in ('NQ', 'ES'):
        sig, days = signals(mk, **P, return_days=True)
        rep = full_report(mk, sig)
        if rep.get('pf_opposite') is None or P['entry'] == 'B':
            rep['pf_opposite_manual'] = ctrl_opposite_manual(mk, sig, days, P['entry'], P['ex'])
            rep['pf_opposite'] = rep['pf_opposite_manual']['pf']
        if P['entry'] == 'B':
            rep['random_dir_B_same_fill'] = ctrl_random_B(mk, sig)
            rep['pf_random_dir_median'] = rep['random_dir_B_same_fill']['median']
        elif P['ex'] == 'tgt':
            rep['random_dir_A_tgt'] = ctrl_random_A_tgt(mk, sig)
            rep['pf_random_dir_median'] = rep['random_dir_A_tgt']['median']
        rep['nofill_days_same_10am_entry'] = ctrl_nofill(mk, P['gmin'], P['lev'], P['ex'])
        rep['move_10to1559_fill_dir_ATR'] = move_stats(mk, P['gmin'], P['lev'])
        rep['move_10to1559_fill_dir_ATR_CE'] = {f'gmin{gm}': move_stats(mk, gm, 'CE') for gm in (0.1, 0.2, 0.3)}
        rep['all_gap_days_10am_fill_dir'] = {f'gmin{gm}': ctrl_all_gap_days(mk, gm) for gm in (0.1, 0.2, 0.3)}
        rep['partials'] = partials(mk, sig, days, P['entry'])
        rep['by_year_str'] = by_year_str(rep)
        res[mk] = rep
        print('\n' + mk, json.dumps(rep, default=str, indent=1))
    # ---- information only (NOT used for selection or verdict): the original probe cell and the Entry-B CE family
    info = {}
    for name, p in (('probe_cell_0.2_CE_A_close', dict(gmin=0.2, lev='CE', entry='A', ex='close')),
                    ('B_CE_close_0.1', dict(gmin=0.1, lev='CE', entry='B', ex='close')),
                    ('B_CE_close_0.2', dict(gmin=0.2, lev='CE', entry='B', ex='close'))):
        info[name] = {}
        for mk in ('NQ', 'ES'):
            sig, days = signals(mk, **p, return_days=True)
            h = headline(simulate(mk, sig))
            x = dict(n=h['n'], pf=h['pf'], pf_IS=h['pf_IS'], pf_OOS=h['pf_OOS'],
                     pf_2x=round(pf(simulate(mk, sig, costmult=2.0).R), 3), partials=partials(mk, sig, days, p['entry']))
            if p['entry'] == 'B':
                x['random_dir_B_same_fill'] = ctrl_random_B(mk, sig)
                x['opposite_manual'] = ctrl_opposite_manual(mk, sig, days, 'B', p['ex'])
            else:
                x['nofill'] = ctrl_nofill(mk, p['gmin'], p['lev'], p['ex'])
            info[name][mk] = x
    # selection sensitivity: what alternative IS-only rules with a minimum IS sample would have picked
    sens = {}
    for nmin in (30, 50):
        e = grid[(grid.NQ_nIS >= nmin) & (grid.ES_nIS >= nmin)]
        if len(e):
            b = e.sort_values('geo_IS', ascending=False).iloc[0]
            sens[f'min_nIS_{nmin}_best_geo'] = {k: (v if not isinstance(v, (np.floating, np.integer)) else float(v)) for k, v in b.items()}
    # unconditional gap fade (control that turned out to carry the effect): all |g| >= gmin days, 10:00 entry in the
    # fill direction, Entry-A stop, hold to close; full engine report + split by 20-day trend (info only, post hoc)
    ug = {}
    for gm in (0.1, 0.2):
        for mk in ('NQ', 'ES'):
            t = day_table(mk); t = t[(t.gA >= gm) & (t.k59 > t.k0)].copy()
            s_ = signals_frame(t.k59.to_numpy(), t.d.to_numpy(), etype=0, stop_pts=stopA(t, mk))
            r_ = full_report(mk, s_)
            tr_ = simulate(mk, s_).join(t.set_index('k59')[['trend']], on='k')
            wt = tr_.d == tr_.trend
            ug[f'{mk}_gmin{gm}'] = {k: r_[k] for k in ('n', 'per_week', 'pf', 'pf_IS', 'pf_OOS', 'pf_2x_cost', 'pf_opposite',
                                                        'pf_random_dir_median', 'pf_random_dir_95', 'p_pf_below_1')} | dict(
                by_year=by_year_str(r_), with_trend=split_pf(tr_, wt), against_trend=split_pf(tr_, ~wt),
                with_trend_2x=round(pf(simulate(mk, s_[np.isin(s_.k, t.k59[t.d == t.trend])], costmult=2.0).R), 3))
    res['uncond_gap_fade_post_hoc'] = ug
    print('\nUNCOND', json.dumps(ug, default=str, indent=1))
    res['info_not_selected'] = info; res['selection_sensitivity_post_hoc'] = sens
    print('\nINFO', json.dumps(info, default=str, indent=1)); print('\nSENS', json.dumps(sens, default=str, indent=1))
    json.dump(res, open(OUT, 'w'), default=str, indent=1)
    print('saved', OUT)
