"""Gap-and-go outside the prior RTH range with midnight-open acceptance (slug: gap_go_midnight).

Rules as implemented (both markets, same rules, sizes scale by ATR):
  * Per calendar date (NY), M = open of the 00:00 bar (mod 0) of that date; the day is skipped if that bar is missing.
  * Signal bar k = the bar with NY minute 569 (09:29, i.e. last pre-market bar; its close is known at 09:30) or,
    variant, 584 (09:44). The day is skipped if that exact bar is missing.
  * pdh / pdl / pdc = previous RTH day (09:30-15:59) high / low / close (engine cache, completed sessions only).
  * STRICT: LONG if c[k] > pdh and M > pdh; SHORT if c[k] < pdl and M < pdl.
    LOOSE : LONG if c[k] > pdh and M > pdc; SHORT if c[k] < pdl and M < pdc.
    Control (not counted in the grid): no midnight filter, LONG if c[k] > pdh, SHORT if c[k] < pdl.
  * Entry: market at the open of k+1 (09:30 or 09:45), etype 0.
  * Stop: stop_pts = max(s*ATR, 0.12*ATR, 20 pts NQ / 4 pts ES) from the fill (ATR = 14-day mean prior RTH range).
  * Exit: target tgt_pts = 0.6*ATR (variant) or none; otherwise engine force-close at 15:59. One trade per day.
  * Grid (24): s {0.15, 0.25, 0.40} x signal bar {09:29, 09:44} x target {close, 0.6 ATR} x {STRICT, LOOSE}.
  * Selection (IS only, fixed before looking): score = geometric mean of IS PF over NQ and ES, smoothed over the
    variant and its neighbours on the s axis (same other settings); pick the max smoothed score.

Lookahead check: M (00:00 open) and c[k] are both known at the close of bar k (k = 09:29/09:44 of the same date);
pdh/pdl/pdc/atr/trend_d are from completed previous RTH days (build_cache.py, shift(1)).
"""
import sys, json
sys.path.insert(0, '/home/user/TRADING/scalp')
import numpy as np
import pandas as pd
from engine import load, signals_frame, simulate, full_report, headline, pf, IS_END, START, END

MINSTOP = {'NQ': 20.0, 'ES': 4.0}
SLUG = 'gap_go_midnight'


def day_bars(D, minute):
    """dict dayi -> bar index of the exact NY minute (first occurrence)."""
    idx = np.where(D['mod'] == minute)[0]
    di = D['dayi'][idx]
    _, first = np.unique(di, return_index=True)
    return dict(zip(di[first].tolist(), idx[first].tolist()))


def raw_days(mk, sigmod=569):
    """Per-day table with the features known at the signal bar (no direction decision yet)."""
    D = load(mk)
    mid = day_bars(D, 0); sb = day_bars(D, sigmod)
    rows = []
    for di, k in sb.items():
        if di not in mid:
            continue
        m0 = mid[di]
        if m0 >= k:
            continue
        atr = D['atr'][k]
        if np.isnan(atr) or np.isnan(D['pdh'][k]):
            continue
        rows.append(dict(dayi=di, k=k, M=D['o'][m0], c=D['c'][k], pdh=D['pdh'][k], pdl=D['pdl'][k], pdc=D['pdc'][k],
                         atr=atr, trend=D['trend_d'][k], date=D['date'][k], dow=D['dow'][k]))
    return pd.DataFrame(rows)


def signals(mk, s=0.25, sigmod=569, tgt=None, conf='STRICT', return_days=False):
    """conf in {'STRICT','LOOSE','NONE'}; tgt None = hold to close, else multiple of ATR."""
    t = raw_days(mk, sigmod)
    up = t.c > t.pdh; dn = t.c < t.pdl
    if conf == 'STRICT':
        lg = up & (t.M > t.pdh); sh = dn & (t.M < t.pdl)
    elif conf == 'LOOSE':
        lg = up & (t.M > t.pdc); sh = dn & (t.M < t.pdc)
    else:
        lg = up; sh = dn
    t['d'] = np.where(lg, 1, np.where(sh, -1, 0))
    t = t[t.d != 0].copy()
    t['stop'] = np.maximum.reduce([s * t.atr, 0.12 * t.atr, np.full(len(t), MINSTOP[mk])])
    t['tgt'] = np.inf if tgt is None else tgt * t.atr
    sig = signals_frame(t.k.to_numpy(), t.d.to_numpy(), etype=0, stop_pts=t.stop.to_numpy(), tgt_pts=t.tgt.to_numpy())
    return (sig, t) if return_days else sig


# ---------------------------------------------------------------- TDH proxy (for overlap / correlation)
def tdh_signals(mk):
    """TDH proxy (obchoch/research.py tdh_days): 09:30-09:40 range, first 1m close beyond it (by 15:30) and
    >= 0.15 ATR from the 09:30 open, only in the 20-day trend direction (trend_d); market at k+1,
    stop = max(other side of the 10-min range, 0.12 ATR), hold to 15:59."""
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


def daily(mk, tr):
    D = load(mk)
    x = tr.copy(); x['date'] = D['date'][x.entry_k]
    return x.groupby('date').agg(R=('R', 'sum'), d=('d', 'first'))


def overlap_stats(mk, tr_g, tr_t):
    g = daily(mk, tr_g); t = daily(mk, tr_t)
    both = g.index.intersection(t.index)
    same = int((g.loc[both, 'd'] == t.loc[both, 'd']).sum())
    u = g.index.union(t.index)
    gr = g.R.reindex(u).fillna(0); trr = t.R.reindex(u).fillna(0)
    # correlation on ALL trading days (zero on days without a trade)
    D = load(mk)
    alld = np.unique(D['date'][(D['mod'] == 600)])
    alld = alld[(alld >= START.astype('datetime64[D]').astype(int)) & (alld < END.astype('datetime64[D]').astype(int))]
    ga = g.R.reindex(alld).fillna(0); ta = t.R.reindex(alld).fillna(0)
    return dict(gap_days=len(g), tdh_days=len(t), overlap_days=len(both), overlap_share_of_gap=round(len(both) / max(len(g), 1), 3),
                same_dir_on_overlap=same,
                corr_daily_R_all_days=round(float(np.corrcoef(ga, ta)[0, 1]), 3),
                corr_daily_R_overlap=round(float(np.corrcoef(g.R.loc[both], t.R.loc[both])[0, 1]), 3) if len(both) > 2 else None,
                pf_gap_on_overlap=round(pf(g.R.loc[both]), 3) if len(both) else None,
                pf_gap_non_overlap=round(pf(g.R.drop(both)), 3) if len(g) > len(both) else None,
                pf_tdh_alone=round(pf(t.R), 3),
                pf_combined_daily=round(pf((gr + trr).to_numpy()), 3))


def split_pf(tr, mask):
    x = tr[mask]
    i = x[x.entry_t < IS_END]; o = x[x.entry_t >= IS_END]
    return dict(n_IS=len(i), pf_IS=round(pf(i.R), 2) if len(i) else None, n_OOS=len(o), pf_OOS=round(pf(o.R), 2) if len(o) else None)


def partials(mk, sig, days):
    tr = simulate(mk, sig)
    m = days.set_index('k')
    tr = tr.join(m[['atr', 'pdh', 'pdl', 'pdc', 'M', 'c', 'trend', 'dow']], on='k')
    gap = np.where(tr.d == 1, tr.c - tr.pdh, tr.pdl - tr.c) / tr.atr        # distance beyond the prior range, ATR units
    out = {
        'long': split_pf(tr, tr.d == 1), 'short': split_pf(tr, tr.d == -1),
        'with_trend_d': split_pf(tr, tr.d == tr.trend), 'against_trend_d': split_pf(tr, tr.d != tr.trend),
        'beyond<0.15ATR': split_pf(tr, gap < 0.15), 'beyond0.15-0.4ATR': split_pf(tr, (gap >= 0.15) & (gap < 0.4)),
        'beyond>=0.4ATR': split_pf(tr, gap >= 0.4),
        'Mon-Tue': split_pf(tr, tr.dow <= 1), 'Wed-Fri': split_pf(tr, tr.dow >= 2),
    }
    return out


def grid():
    rows = []
    for conf in ('STRICT', 'LOOSE'):
        for sigmod in (569, 584):
            for tgt in (None, 0.6):
                for s in (0.15, 0.25, 0.40):
                    r = dict(conf=conf, sigmod=sigmod, tgt='close' if tgt is None else tgt, s=s)
                    for mk in ('NQ', 'ES'):
                        tr = simulate(mk, signals(mk, s, sigmod, tgt, conf))
                        h = headline(tr)
                        r[f'{mk}_n'] = h.get('n', 0); r[f'{mk}_IS'] = h.get('pf_IS'); r[f'{mk}_OOS'] = h.get('pf_OOS')
                    rows.append(r)
    g = pd.DataFrame(rows)
    g['geo_IS'] = np.sqrt(g.NQ_IS.clip(lower=1e-6) * g.ES_IS.clip(lower=1e-6))
    # neighbour smoothing on the s axis (same conf / sigmod / tgt)
    g['smooth_IS'] = g.groupby(['conf', 'sigmod', 'tgt'], sort=False).geo_IS.transform(
        lambda v: v.rolling(3, center=True, min_periods=2).mean())   # IS-only metric; centre window over the parameter axis
    return g


if __name__ == '__main__':
    pd.set_option('display.width', 200)
    g = grid()
    print(g.to_string())
    best = g.loc[g.smooth_IS.idxmax()]
    tgt = None if best.tgt == 'close' else float(best.tgt)
    ch = dict(s=float(best.s), sigmod=int(best.sigmod), tgt=tgt, conf=best.conf)
    print('CHOSEN (IS only):', ch)
    res = dict(slug=SLUG, chosen=ch, variants_tested=len(g),
               median_IS_pf=dict(NQ=round(float(g.NQ_IS.median()), 3), ES=round(float(g.ES_IS.median()), 3)),
               median_OOS_pf_info=dict(NQ=round(float(g.NQ_OOS.median()), 3), ES=round(float(g.ES_OOS.median()), 3)),
               grid=g.to_dict('records'))
    for mk in ('NQ', 'ES'):
        sig, days = signals(mk, return_days=True, **ch)
        rep = full_report(mk, sig)
        print('\n', mk, 'CHOSEN', json.dumps(rep, default=str))
        res[mk] = rep
        # control: no midnight filter (not counted)
        sc = signals(mk, ch['s'], ch['sigmod'], ch['tgt'], 'NONE')
        hc = headline(simulate(mk, sc))
        res[f'{mk}_control_no_midnight'] = hc
        print(mk, 'control no-midnight:', json.dumps(hc, default=str))
        # TDH proxy overlap / correlation
        trt = simulate(mk, tdh_signals(mk)); trg = simulate(mk, sig)
        res[f'{mk}_tdh_proxy'] = headline(trt)
        res[f'{mk}_tdh_overlap'] = overlap_stats(mk, trg, trt)
        print(mk, 'TDH proxy:', json.dumps(res[f'{mk}_tdh_proxy'], default=str))
        print(mk, 'overlap:', json.dumps(res[f'{mk}_tdh_overlap'], default=str))
        # random-direction control split IS / OOS (does the gap direction beat a coin flip on the same days?)
        rng = np.random.default_rng(1); rr = []
        for _ in range(50):
            x = sig.copy(); x['d'] = np.where(rng.random(len(x)) < 0.5, 1, -1); t = simulate(mk, x)
            rr.append((pf(t[t.entry_t < IS_END].R), pf(t[t.entry_t >= IS_END].R)))
        rr = np.nanmedian(np.array(rr), 0)
        res[f'{mk}_random_dir_IS_OOS_median'] = [round(float(rr[0]), 3), round(float(rr[1]), 3)]
        print(mk, 'random direction median PF IS/OOS:', res[f'{mk}_random_dir_IS_OOS_median'])
        # TDH proxy trades split by whether the day was a STRICT gap day (gap known at 09:29, before any TDH entry >= 09:40)
        trt['gap'] = pd.Series(load(mk)['dayi'][trt.k]).map(days.set_index('dayi').d).fillna(0).to_numpy()
        res[f'{mk}_tdh_by_gapday'] = {nm: split_pf(trt, msk) for nm, msk in (
            ('tdh_on_gapday_same_dir', trt.gap == trt.d), ('tdh_on_gapday_opp_dir', trt.gap == -trt.d),
            ('tdh_on_gapday_any_dir', trt.gap != 0), ('tdh_no_gapday', trt.gap == 0))}
        print(mk, 'TDH by gap day:', json.dumps(res[f'{mk}_tdh_by_gapday']))
        res[f'{mk}_partials'] = partials(mk, sig, days)
        print(mk, 'partials:', json.dumps(res[f'{mk}_partials'], default=str))
    json.dump(res, open(f'/home/user/TRADING/scalp/results/{SLUG}.json', 'w'), default=str, indent=1)
