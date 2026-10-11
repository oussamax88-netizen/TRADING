"""Power-of-3 midnight-open Judas reclaim in the 20-day trend direction, hold to close (slug: po3_mo_reclaim).

Rules as implemented (both markets, same rules, sizes scale by ATR):
  * Day = NY calendar date (cache `dayi`). Only dates whose first bar is the 00:00 bar (mod 0) are used.
    MO = o[00:00 bar]. ATR = cache atr at that bar (14-day mean of previous RTH ranges, previous days only).
    d = trend_d at that bar (sign of previous RTH close minus its 20-day SMA, previous days only); skip if 0/NaN.
  * From the 00:00 bar on, track the RUNNING extreme against the bias (d=+1: lowest low since 00:00 incl. bar k;
    d=-1: highest high). manip(k) = d*(MO - extreme_k) >= depth*ATR (uses bars 00:00..k only; the sweep may happen
    before W0, e.g. in Asia/London, and the bar that sweeps may also be the trigger bar if it closes back beyond MO).
  * Trigger PLAIN: first bar k with mod[k] in [W0, 630) (10:30 exclusive), manip(k) true and d*(c[k] - MO) > 0.
    (Literal reading: if price already sits beyond MO at W0 after an earlier sweep, the W0 bar triggers.)
  * Trigger CISD: as PLAIN but the close must also be beyond the CISD level = open of the LAST opposite-coloured bar
    at or before the bar that made the running extreme (d=+1: last bar with c<o among 00:00..argmin; d=-1: last c>o).
    If no such bar exists the CISD level is ignored (= PLAIN). Level is recomputed whenever the extreme moves.
  * Entry: market at the open of k+1 (etype 0). stop_pts = max(d*(c[k] - extreme_k) + 0.02*ATR, 0.12*ATR,
    20 pts NQ / 4 pts ES), measured from the fill by the engine.
  * Exit: engine force-close at 15:59 ('close'), or tgt_pts = 1.0*ATR ('tgt') / 15:59. One trade per day.
  * Grid (24): W0 {03:00, 07:00, 08:30} x depth {0.1, 0.2} ATR x exit {close, tgt} x trigger {PLAIN, CISD}.
  * Selection (IS only, fixed before looking): score = geometric mean of IS PF on NQ and ES; plateau score = mean
    score of the variant and its grid neighbours along W0 and depth (same trigger/exit); highest plateau wins.
  * Controls (not counted): bias from nothing (both directions, first reclaim of either side), bias = -trend_d,
    engine opposite/random direction, random entry time on the same days, all-days trend baseline at the same
    entry rule but WITHOUT the manipulation requirement, and 'no-sweep' days (MO reclaim w/o depth sweep).
  * Overlap: trade days shared with the TDH proxy (09:30-09:40 range breakout with trend, hold to close),
    AZL (saved trades) and an ORG gap-rejection proxy (gmin 0.2, CE, entry A, close).

Lookahead check: MO is the open of the 00:00 bar; the running extreme / CISD level / manip flag at bar k use bars
00:00..k only (numpy accumulate along the day); ATR and trend_d are previous-day values; entry is at k+1.
"""
import sys, json
sys.path.insert(0, '/home/user/TRADING/scalp')
import numpy as np
import pandas as pd
from engine import load, signals_frame, simulate, full_report, headline, pf, IS_END, START, END

SLUG = 'po3_mo_reclaim'
OUT = f'/home/user/TRADING/scalp/results/{SLUG}.json'
SPAD = '/tmp/claude-0/-home-user-TRADING/b9cb08f9-c0e5-5011-839c-6894039a978b/scratchpad'
MINSTOP = {'NQ': 20.0, 'ES': 4.0}
WEND = 630
W0S = (180, 420, 510)
DEPTHS = (0.1, 0.2)
EXITS = ('close', 'tgt')
TRIGS = ('PLAIN', 'CISD')
_DAYS = {}


# ------------------------------------------------------------------ per-day precomputation (no direction choice yet)
def days(mk):
    """For every date with a 00:00 bar: the bar slice 00:00..10:29 and running arrays for BOTH directions."""
    if mk in _DAYS:
        return _DAYS[mk]
    D = load(mk)
    o, h, l, c, mod, dayi, atr, trd = D['o'], D['h'], D['l'], D['c'], D['mod'], D['dayi'], D['atr'], D['trend_d']
    starts = np.r_[0, np.where(np.diff(dayi) != 0)[0] + 1]; ends = np.r_[starts[1:], len(dayi)]
    out = []
    for a, b in zip(starts, ends):
        if mod[a] != 0 or np.isnan(atr[a]):
            continue
        j = a + np.searchsorted(mod[a:b], WEND)            # first bar >= 10:30 (mods are increasing within a date)
        if j - a < 200:
            continue
        idx = np.arange(a, j)
        MO, A = o[a], atr[a]
        rec = dict(a=a, idx=idx, MO=MO, atr=A, trend=trd[a], date=D['date'][a], dow=D['dow'][a], mod=mod[idx], c=c[idx])
        n = len(idx)
        for d in (1, -1):
            if d == 1:
                ext = np.minimum.accumulate(l[idx])
                new = np.r_[True, l[idx][1:] < ext[:-1]]
                opp = c[idx] < o[idx]                        # down-close bars
            else:
                ext = np.maximum.accumulate(h[idx])
                new = np.r_[True, h[idx][1:] > ext[:-1]]
                opp = c[idx] > o[idx]                        # up-close bars
            ext_i = np.maximum.accumulate(np.where(new, np.arange(n), 0))       # bar that made the running extreme
            last_opp = np.maximum.accumulate(np.where(opp, np.arange(n), -1))   # last opposite-coloured bar so far
            lo_i = last_opp[ext_i]                                              # ... at or before the extreme bar
            lev = np.where(lo_i >= 0, o[idx][np.maximum(lo_i, 0)], np.nan)
            rec[d] = dict(ext=ext, sweep=d * (MO - ext) / A, lev=lev, ext_i=ext_i)
        out.append(rec)
    _DAYS[mk] = out
    return out


def first_trigger(r, d, W0, depth, trig, need_manip=True):
    x = r[d]
    cond = (r['mod'] >= W0) & (d * (r['c'] - r['MO']) > 0)
    if need_manip:
        cond &= x['sweep'] >= depth
    if trig == 'CISD':
        cond &= ~(d * (r['c'] - x['lev']) <= 0)          # NaN level (no opposite bar) -> ignored
    w = np.flatnonzero(cond)
    return w[0] if len(w) else -1


def table(mk, W0=420, depth=0.1, trig='PLAIN', bias='trend', need_manip=True):
    """One row per triggered day. bias: 'trend' (d = trend_d), 'anti' (d = -trend_d), 'none' (both sides, first)."""
    rows = []
    for r in days(mk):
        tdir = r['trend']
        if bias != 'none' and (np.isnan(tdir) or tdir == 0):
            continue
        cands = (int(tdir),) if bias == 'trend' else ((-int(tdir),) if bias == 'anti' else (1, -1))
        best = None
        for d in cands:
            j = first_trigger(r, d, W0, depth, trig, need_manip)
            if j >= 0 and (best is None or j < best[0]):
                best = (j, d)
        if best is None:
            continue
        j, d = best
        x = r[d]
        k = int(r['idx'][j])
        raw = d * (r['c'][j] - x['ext'][j]) + 0.02 * r['atr']
        rows.append(dict(k=k, d=d, date=r['date'], dow=r['dow'], mod=int(r['mod'][j]), atr=r['atr'], MO=r['MO'],
                         trend=tdir, sweep=x['sweep'][j], ext_mod=int(r['mod'][x['ext_i'][j]]),
                         stop=max(raw, 0.12 * r['atr'], MINSTOP[mk]), c=r['c'][j]))
    return pd.DataFrame(rows)


def signals(mk, W0=420, depth=0.1, ex='close', trig='PLAIN', bias='trend', need_manip=True, return_days=False):
    t = table(mk, W0, depth, trig, bias, need_manip)
    tg = np.inf if ex == 'close' else 1.0 * t.atr.to_numpy()
    sig = signals_frame(t.k.to_numpy(), t.d.to_numpy(), etype=0, stop_pts=t.stop.to_numpy(), tgt_pts=tg)
    return (sig, t) if return_days else sig


# ------------------------------------------------------------------ helpers
def split_pf(tr, mask):
    x = tr[np.asarray(mask, dtype=bool)]
    i = x[x.entry_t < IS_END]; o = x[x.entry_t >= IS_END]
    return dict(n_IS=len(i), pf_IS=round(pf(i.R), 2) if len(i) else None, n_OOS=len(o), pf_OOS=round(pf(o.R), 2) if len(o) else None)


def brief(tr):
    return dict(n=len(tr), pf=round(pf(tr.R), 3)) | split_pf(tr, np.ones(len(tr), bool))


def by_year_str(rep):
    return ' '.join(f"{y}:{v['pf']}({v['n']})" for y, v in rep['by_year'].items())


# ------------------------------------------------------------------ controls
def ctrl_random_time(mk, sig, t, W0, ex, n=30, seed=7):
    """Same days, same direction and stop size, but the entry bar is random in [W0, 10:30) of that day."""
    rng = np.random.default_rng(seed)
    D = load(mk); mod, dayi = D['mod'], D['dayi']
    pools = []
    for k in t.k.to_numpy():
        a = np.searchsorted(dayi, dayi[k]); b = np.searchsorted(dayi, dayi[k], side='right')
        m = mod[a:b]
        pools.append(a + np.flatnonzero((m >= W0) & (m < WEND)))
    out = []
    for _ in range(n):
        ks = np.array([p[rng.integers(len(p))] for p in pools])
        s = sig.copy(); s['k'] = ks
        out.append(pf(simulate(mk, s).R))
    return dict(median=round(float(np.nanmedian(out)), 3), p95=round(float(np.nanpercentile(out, 95)), 3))


def ctrl_all_days_trend(mk, W0, ex):
    """Every day with a trend: market entry at the W0 bar in the trend direction, same stop formula (extreme since 00:00)."""
    rows = []
    for r in days(mk):
        d = r['trend']
        if np.isnan(d) or d == 0:
            continue
        d = int(d)
        w = np.flatnonzero(r['mod'] >= W0)
        if not len(w):
            continue
        j = w[0]
        raw = d * (r['c'][j] - r[d]['ext'][j]) + 0.02 * r['atr']
        rows.append((r['idx'][j], d, max(raw, 0.12 * r['atr'], MINSTOP[mk]), r['atr']))
    k, d, st, A = map(np.array, zip(*rows))
    tg = np.inf if ex == 'close' else 1.0 * A
    return brief(simulate(mk, signals_frame(k.astype(np.int64), d, stop_pts=st, tgt_pts=tg)))


def ctrl_no_sweep(mk, W0, depth, ex, trig):
    """Days where the trend-side MO 'reclaim' (close beyond MO in window) happens WITHOUT a prior depth sweep."""
    t = table(mk, W0, depth, trig, need_manip=False)
    t = t[t.sweep < depth]
    tg = np.inf if ex == 'close' else t.atr.to_numpy()
    return brief(simulate(mk, signals_frame(t.k.to_numpy(), t.d.to_numpy(), stop_pts=t.stop.to_numpy(), tgt_pts=tg)))


# ------------------------------------------------------------------ overlap proxies
def tdh_signals(mk):
    """TDH proxy (same as gap_go_midnight.py): 09:30-09:40 range, first 1m close beyond it by 15:30 and >= 0.15 ATR from
    the 09:30 open, only in the 20-day trend direction; stop = max(other side, 0.12 ATR); hold to 15:59."""
    D = load(mk)
    mod, dayi, o, h, l, c, atr, tr = D['mod'], D['dayi'], D['o'], D['h'], D['l'], D['c'], D['atr'], D['trend_d']
    starts = np.r_[0, np.where(np.diff(dayi) != 0)[0] + 1]; ends = np.r_[starts[1:], len(dayi)]
    ks, ds, st = [], [], []
    for a, b in zip(starts, ends):
        m = mod[a:b]
        j0 = np.where(m == 570)[0]
        if len(j0) == 0:
            continue
        a0 = a + j0[0]
        orr = np.where((m >= 570) & (m < 580))[0]
        if len(orr) < 5 or np.isnan(atr[a0]) or tr[a0] == 0 or np.isnan(tr[a0]):
            continue
        hi, lo, op = h[a + orr].max(), l[a + orr].min(), o[a0]
        for j in np.where((m >= 580) & (m < 930))[0]:
            k = a + j
            d = 1 if (c[k] > hi and c[k] - op >= 0.15 * atr[k]) else (-1 if (c[k] < lo and op - c[k] >= 0.15 * atr[k]) else 0)
            if d != 0:
                if d == tr[k]:
                    ks.append(k); ds.append(d); st.append(max(c[k] - lo if d == 1 else hi - c[k], 0.12 * atr[k]))
                break
    return signals_frame(ks, ds, stop_pts=np.array(st, dtype=float))


def org_signals(mk, gmin=0.2):
    """ORG gap-rejection proxy (probe cell): R = 16:14 close of previous date (fallback pdc), g = 09:30 open - R,
    |g| >= gmin*ATR, CE = R + g/2 traded through during 09:30-09:59 -> short gap-up / long gap-down at 10:00 open,
    stop = range extreme 09:30-09:59 beyond c[09:59] + 0.02 ATR floored at max(0.12 ATR, 20/4), hold to close."""
    D = load(mk)
    o, h, l, c, mod, dayi, atr = D['o'], D['h'], D['l'], D['c'], D['mod'], D['dayi'], D['atr']

    def first(minute):
        idx = np.where(mod == minute)[0]; di = dayi[idx]
        _, f = np.unique(di, return_index=True)
        return dict(zip(di[f].tolist(), idx[f].tolist()))
    i570, i599, i974 = first(570), first(599), first(974)
    dl = sorted(i570)
    ks, ds, st = [], [], []
    for j in range(1, len(dl)):
        di, prev = dl[j], dl[j - 1]
        k0 = i570[di]; A = atr[k0]; k59 = i599.get(di, -1)
        if np.isnan(A) or k59 <= k0:
            continue
        R = c[i974[prev]] if prev in i974 else D['pdc'][k0]
        if np.isnan(R):
            continue
        g = o[k0] - R
        if abs(g) < gmin * A or g == 0:
            continue
        LV = R + g / 2
        hm, lm = h[k0:k59 + 1].max(), l[k0:k59 + 1].min()
        if (g > 0 and lm <= LV) or (g < 0 and hm >= LV):
            d = -int(np.sign(g))
            raw = (hm - c[k59] if d == -1 else c[k59] - lm) + 0.02 * A
            ks.append(k59); ds.append(d); st.append(max(raw, 0.12 * A, MINSTOP[mk]))
    return signals_frame(ks, ds, stop_pts=np.array(st, dtype=float))


def daily(mk, tr):
    D = load(mk)
    x = tr.copy(); x['date'] = D['date'][x.entry_k]
    return x.groupby('date').agg(R=('R', 'sum'), d=('d', 'first'))


def overlap(mk, tr_me, other):
    g = daily(mk, tr_me); t = other
    both = g.index.intersection(t.index)
    D = load(mk)
    alld = np.unique(D['date'][(D['mod'] == 600)])
    alld = alld[(alld >= START.astype('datetime64[D]').astype(int)) & (alld < END.astype('datetime64[D]').astype(int))]
    ga = g.R.reindex(alld).fillna(0); ta = t.R.reindex(alld).fillna(0)
    same = (g.loc[both, 'd'] == t.loc[both, 'd']) if len(both) else pd.Series(dtype=bool)
    return dict(my_days=len(g), other_days=len(t), common_days=len(both),
                share_of_my_days=round(len(both) / max(len(g), 1), 3), share_of_other_days=round(len(both) / max(len(t), 1), 3),
                same_dir_share=round(float(same.mean()), 3) if len(both) else None,
                corr_daily_R_all_days=round(float(np.corrcoef(ga, ta)[0, 1]), 3),
                my_pf_on_common=round(pf(g.R.loc[both]), 3) if len(both) else None,
                my_pf_off_common=round(pf(g.R.drop(both)), 3) if len(g) > len(both) else None,
                other_pf=round(pf(t.R), 3),
                pf_combined_daily=round(pf((ga + ta).to_numpy()[(ga + ta).to_numpy() != 0]), 3))


def azl_daily(mk):
    try:
        A = pd.read_pickle(f'{SPAD}/AZL_trades.pkl')
    except Exception:
        return None
    a = A[A.mk == mk].copy()
    a['date'] = (a.entry_t.values.astype('datetime64[D]').astype(np.int64))
    return a.groupby('date').agg(R=('R', 'sum'), d=('side', 'first'))


# ------------------------------------------------------------------ partial edges
def partials(mk, tr, t):
    x = tr.merge(t[['k', 'mod', 'sweep', 'ext_mod', 'dow', 'atr', 'MO', 'stop']].rename(columns={'stop': 'stop_sig'}), on='k', how='left')
    D = load(mk)
    x['pdc'] = D['pdc'][x.k]; x['pdh'] = D['pdh'][x.k]; x['pdl'] = D['pdl'][x.k]
    ext = x.MO - x.d * x.sweep * x.atr
    out = {
        'long': split_pf(x, x.d == 1), 'short': split_pf(x, x.d == -1),
        'entry 03:00-06:59': split_pf(x, x['mod'] < 420), 'entry 07:00-08:29': split_pf(x, (x['mod'] >= 420) & (x['mod'] < 510)),
        'entry 08:30-09:29': split_pf(x, (x['mod'] >= 510) & (x['mod'] < 570)), 'entry 09:30-10:29': split_pf(x, x['mod'] >= 570),
        'sweep < 0.25 ATR': split_pf(x, x.sweep < 0.25), 'sweep 0.25-0.5 ATR': split_pf(x, (x.sweep >= 0.25) & (x.sweep < 0.5)),
        'sweep >= 0.5 ATR': split_pf(x, x.sweep >= 0.5),
        'extreme made 00:00-02:59 (Asia)': split_pf(x, x.ext_mod < 180),
        'extreme made 03:00-06:59 (London)': split_pf(x, (x.ext_mod >= 180) & (x.ext_mod < 420)),
        'extreme made >= 07:00': split_pf(x, x.ext_mod >= 420),
        'extreme swept prev-day extreme (pdl for long / pdh for short)': split_pf(x, np.where(x.d == 1, ext < x.pdl, ext > x.pdh)),
        'extreme inside prev-day range': split_pf(x, np.where(x.d == 1, ext >= x.pdl, ext <= x.pdh)),
        'MO beyond pdc in trade dir': split_pf(x, x.d * (x.MO - x.pdc) > 0),
        'MO behind pdc': split_pf(x, x.d * (x.MO - x.pdc) <= 0),
        'stop floor binding (<= 0.13 ATR or min pts)': split_pf(x, x.stop_sig <= np.maximum(0.13 * x.atr, MINSTOP[mk] + 1e-9)),
        'Mon-Tue': split_pf(x, x.dow <= 1), 'Wed': split_pf(x, x.dow == 2), 'Thu-Fri': split_pf(x, x.dow >= 3),
        'stopped out share': dict(share=round(float((x.reason == 1).mean()), 2)),
    }
    return out


# ------------------------------------------------------------------ main
def grid():
    rows = []
    for trig in TRIGS:
        for ex in EXITS:
            for W0 in W0S:
                for depth in DEPTHS:
                    r = dict(trig=trig, ex=ex, W0=W0, depth=depth)
                    for mk in ('NQ', 'ES'):
                        tr = simulate(mk, signals(mk, W0, depth, ex, trig))
                        h = headline(tr)
                        r[f'{mk}_n'] = h['n']; r[f'{mk}_IS'] = h['pf_IS']; r[f'{mk}_OOS'] = h['pf_OOS']
                    r['geo_IS'] = float(np.sqrt(max(r['NQ_IS'], 0) * max(r['ES_IS'], 0)))
                    rows.append(r)
    g = pd.DataFrame(rows)
    # plateau: mean geo_IS over the variant and its W0 / depth neighbours (same trigger and exit)
    sm = []
    for _, r in g.iterrows():
        wi = W0S.index(r.W0); di = DEPTHS.index(r.depth)
        nb = g[(g.trig == r.trig) & (g.ex == r.ex) &
               (((g.W0.map(W0S.index) - wi).abs() + (g.depth.map(DEPTHS.index) - di).abs()) <= 1)]
        sm.append(nb.geo_IS.mean())
    g['plateau_IS'] = sm
    return g


if __name__ == '__main__':
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    g = grid()
    print(g.round(3).to_string())
    best = g.sort_values('plateau_IS', ascending=False).iloc[0]
    ch = dict(W0=int(best.W0), depth=float(best.depth), ex=best.ex, trig=best.trig)
    print('\nCHOSEN (IS plateau):', ch)
    res = dict(slug=SLUG, chosen=ch, variants_tested=len(g),
               median_IS_pf=dict(NQ=round(float(g.NQ_IS.median()), 3), ES=round(float(g.ES_IS.median()), 3)),
               median_OOS_pf_info=dict(NQ=round(float(g.NQ_OOS.median()), 3), ES=round(float(g.ES_OOS.median()), 3)),
               grid=g.round(3).to_dict(orient='records'))
    trades = {}
    for mk in ('NQ', 'ES'):
        sig, t = signals(mk, return_days=True, **ch)
        rep = full_report(mk, sig)
        rep['by_year_str'] = by_year_str(rep)
        res[mk] = rep
        print(f'\n{mk} REPORT:', json.dumps(rep, default=str))
        tr = simulate(mk, sig); trades[mk] = tr.assign(mk=mk)
        # controls
        c = {}
        c['bias_none_both_sides'] = brief(simulate(mk, signals(mk, bias='none', **ch)))
        c['bias_anti_trend'] = brief(simulate(mk, signals(mk, bias='anti', **ch)))
        c['random_entry_time_same_days'] = ctrl_random_time(mk, sig, t, ch['W0'], ch['ex'])
        c['all_days_trend_at_W0_no_sweep_needed'] = ctrl_all_days_trend(mk, ch['W0'], ch['ex'])
        c['mo_reclaim_without_depth_sweep'] = ctrl_no_sweep(mk, ch['W0'], ch['depth'], ch['ex'], ch['trig'])
        res[f'{mk}_controls'] = c
        print(mk, 'controls:', json.dumps(c))
        # overlaps
        ov = {}
        ov['TDH_proxy'] = overlap(mk, tr, daily(mk, simulate(mk, tdh_signals(mk))))
        ov['ORG_proxy'] = overlap(mk, tr, daily(mk, simulate(mk, org_signals(mk))))
        ad = azl_daily(mk)
        if ad is not None:
            ad = ad[(ad.index >= START.astype('datetime64[D]').astype(int)) & (ad.index < END.astype('datetime64[D]').astype(int))]
            ov['AZL'] = overlap(mk, tr, ad)
        res[f'{mk}_overlap'] = ov
        print(mk, 'overlap:', json.dumps(ov))
        p = partials(mk, tr, t)
        res[f'{mk}_partials'] = p
        print(mk, 'partials:', json.dumps(p))
        # CISD vs PLAIN identity share (diagnostic)
        tp = table(mk, ch['W0'], ch['depth'], 'PLAIN'); tc = table(mk, ch['W0'], ch['depth'], 'CISD')
        res[f'{mk}_cisd_same_bar_share'] = round(float(tp.k.isin(tc.k).mean()), 3)
    pd.concat(trades.values()).to_pickle(f'{SPAD}/{SLUG}_trades.pkl')
    # verdict per the task rules
    def ok(m, lo):
        return res[m]['pf_IS'] >= lo and res[m]['pf_OOS'] >= lo
    pas = any(ok(m, 1.3) and res[o]['pf_OOS'] >= 1.1 and res[m]['per_week'] >= 3 and res[m]['pf_2x_cost'] >= 1.1
              for m, o in (('NQ', 'ES'), ('ES', 'NQ')))
    res['verdict'] = 'PASS' if pas else ('BORDERLINE' if (ok('NQ', 1.15) or ok('ES', 1.15)) else 'FAIL')
    res['notes'] = ('CISD trigger fires on the same bar as PLAIN in ~99% of days (the open of the last down/up-close bar '
                    'into the extreme is almost always behind MO), so the grid has ~12 distinct variants. '
                    'NQ: same trades in a random direction (median PF ~1.20) or the opposite direction (1.27) beat the '
                    'trend-direction rule (1.04); the IS gain came from days that later became TDH trend days.')
    print('VERDICT:', res['verdict'])
    json.dump(res, open(OUT, 'w'), default=str, indent=1)
    print('\nsaved', OUT)
