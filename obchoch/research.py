"""Research table: every OB + 1m CHOCH signal with entry-time features and several exit styles.

Signals use the same logic as engine.run_zone (first touch in window, 5m-close cancel, CHOCH = 1m close
beyond the last confirmed swing, entry next 1m open). No position overlap filter here: that is applied
when a rule set is evaluated. All features are known at the CHOCH bar close (no lookahead).
"""
import sys, pickle
import numpy as np
import pandas as pd
from numba import njit
sys.path.insert(0, 'obchoch')
from engine import WIN_START, WIN_END, FORCE_CLOSE

NS = 60_000_000_000


@njit(cache=True)
def find_signal(start, zlo, zhi, exp_day, mod, dayi, o, h, l, c, is5, f5c, sh, win_start, win_end):
    """Demand side. Returns (touch_k, choch_j, away_before_touch, low_after_touch)."""
    n = len(o)
    away = -1e18
    k = start
    touch_k = -1
    lowest = 1e18
    while k < n:
        if dayi[k] > exp_day:
            return -1, -1, 0.0, 0.0
        if touch_k < 0 and l[k] <= zhi:
            if mod[k] < win_start or mod[k] >= win_end:
                return -1, -1, 0.0, 0.0
            touch_k = k
        if is5[k] and f5c[k] < zlo:
            return -1, -1, 0.0, 0.0
        if touch_k < 0:
            if h[k] > away:
                away = h[k]
        else:
            if l[k] < lowest:
                lowest = l[k]
            if mod[k] >= win_end or dayi[k] != dayi[touch_k]:
                return -1, -1, 0.0, 0.0
            if not np.isnan(sh[k]) and c[k] > sh[k]:
                return touch_k, k, away, lowest
        k += 1
    return -1, -1, 0.0, 0.0


@njit(cache=True)
def sim_exit(e, px, stop, tgt, be_at, dayi, mod, o, h, l, c):
    """Demand side. Stop first, no target on entry bar, gaps fill at open, forced close 15:59.
    be_at > 0: once price reaches px + be_at, stop moves to entry. Returns (exit_k, exit_px, reason)."""
    n = len(o)
    if l[e] <= stop:
        return e, stop, 1
    st = stop
    if be_at > 0 and h[e] >= px + be_at:
        st = px
    m = e + 1
    while m < n:
        if dayi[m] != dayi[e]:
            return m - 1, c[m - 1], 3
        if mod[m] >= FORCE_CLOSE:
            return m, o[m], 3
        if o[m] <= st:
            return m, o[m], 1
        if l[m] <= st:
            return m, st, 1
        if h[m] >= tgt:
            return m, tgt, 2
        if be_at > 0 and h[m] >= px + be_at:
            st = max(st, px)
        m += 1
    return n - 1, c[n - 1], 3


def tdh_days(a):
    """US-open opening-range direction per day (TDH style): 09:30-09:40 range, first 1m close beyond it
    and >= 0.15 ATR from the 09:30 open. Returns per-day (dir, minute index known)."""
    mod, dayi, o, c = a['mod'], a['dayi'], a['o'], a['c']
    h, l = a['h'], a['l']
    nd = dayi.max() + 1
    dirs = np.zeros(nd); when = np.full(nd, -1, dtype=np.int64); open930 = np.full(nd, np.nan)
    idx = np.where((mod >= 570) & (mod < 960))[0]
    df = pd.DataFrame({'k': idx, 'd': dayi[idx], 'm': mod[idx]})
    for d, g in df.groupby('d'):
        ks = g.k.to_numpy(); ms = g.m.to_numpy()
        if ms[0] != 570:
            continue
        orr = ks[ms < 580]
        if len(orr) < 5:
            continue
        hi, lo, op = h[orr].max(), l[orr].min(), o[ks[0]]
        atr = a['atr'][ks[0]]
        open930[d] = op
        if np.isnan(atr):
            continue
        for k in ks[ms >= 580]:
            if c[k] > hi and c[k] - op >= 0.15 * atr:
                dirs[d], when[d] = 1, k; break
            if c[k] < lo and op - c[k] >= 0.15 * atr:
                dirs[d], when[d] = -1, k; break
    return dirs, when, open930


def build_table(S, mk, N=5, life=5):
    LIFE = {'30m': life, '1h': life, '4h': 10, '1d': 20}
    a, z = pickle.load(open(f'{S}/{mk}.pkl', 'rb'))
    tmin, mod, dayi = a['tmin'], a['mod'], a['dayi']
    n = len(tmin)
    # day-level context (prev day high/low/close from 09:30-16:00, known before the day)
    rth = (mod >= 570) & (mod < 960)
    dd = pd.DataFrame({'d': dayi[rth], 'h': a['h'][rth], 'l': a['l'][rth], 'c': a['c'][rth]}).groupby('d').agg(h=('h', 'max'), l=('l', 'min'), c=('c', 'last'))
    dd = dd.reindex(range(dayi.max() + 1))
    pdh, pdl, pdc = dd.h.shift(1).to_numpy(), dd.l.shift(1).to_numpy(), dd.c.shift(1).to_numpy()
    tdir, twhen, open930 = tdh_days(a)
    rows = []
    for side in (1, -1):
        if side == 1:
            o, h, l, c, f5c, sh, sw = a['o'], a['h'], a['l'], a['c'], a['f5c'], a[f'sh{N}'], a[f'sl{N}']
        else:
            o, h, l, c, f5c, sh, sw = -a['o'], -a['l'], -a['h'], -a['c'], -a['f5c'], -a[f'sl{N}'], -a[f'sh{N}']
        zz = z[z.dir == side]
        for zr in zz.itertuples(index=False):
            zlo, zhi = (zr.zlo, zr.zhi) if side == 1 else (-zr.zhi, -zr.zlo)
            tk, j, away, lowest = find_signal(zr.start, zlo, zhi, dayi[zr.start] + LIFE[zr.tf], mod, dayi, o, h, l, c,
                                              a['is5'], f5c, sh, WIN_START, WIN_END)
            if j < 0 or j + 1 >= n or dayi[j + 1] != dayi[j] or mod[j + 1] >= FORCE_CLOSE:
                continue
            e = j + 1; px = o[e]
            d = dayi[j]
            atr = a['atr'][j]          # ATR of the trade date (zone may be created on a date with no RTH, e.g. Sunday evening)
            if np.isnan(atr):
                continue
            r = dict(side=side, tf=zr.tf, created=zr.created, cmod=pd.Timestamp(zr.created).hour * 60 + pd.Timestamp(zr.created).minute, disp=zr.disp, fvg=zr.fvg,
                     sweep_pd=(lowest < (pdl[d] if side == 1 else -pdh[d])) if not np.isnan(pdl[d]) else False,
                     zone_beyond_pd=(zlo < (pdl[d] if side == 1 else -pdh[d])) if not np.isnan(pdl[d]) else False, touch_k=tk, choch_k=j, entry_k=e, px=px, zlo=zlo, zhi=zhi,
                     atr=atr, zone_h=(zhi - zlo) / atr, age_min=(tmin[tk] - (pd.Timestamp(zr.created).value // NS)),
                     away=(away - zhi) / (zhi - zlo) if zhi > zlo else np.nan,
                     pen=(zhi - lowest) / (zhi - zlo) if zhi > zlo else np.nan,   # how deep into the zone
                     choch_min=tmin[j] - tmin[tk], mod=mod[j], dow=pd.Timestamp(tmin[j] * NS).dayofweek,
                     ext=(px - sh[j]) / atr,                                        # entry beyond broken level
                     swing_stop=sw[j], trend_d=side * a['trend_d'][j], trend_h=side * a['trend_h'][j],
                     tdh=side * tdir[d] if (twhen[d] >= 0 and twhen[d] <= j) else 0,
                     tdh_any=side * tdir[d] if twhen[d] >= 0 else 0,                # NOT known before 09:40 -> only for analysis
                     day_move=(side * (a['c'][j] - open930[d]) / atr) if (mod[j] >= 570 and not np.isnan(open930[d])) else np.nan,
                     loc_pd=(a['c'][j] - pdl[d]) / (pdh[d] - pdl[d]) if pdh[d] > pdl[d] else np.nan,
                     gap=side * (open930[d] - pdc[d]) / atr if (mod[j] >= 570 and not np.isnan(open930[d])) else np.nan,
                     pcm=side * (a['c'][j] - pdc[d]) / atr)                       # price now vs prev 16:00 close
            if side == -1:
                r['loc_pd'] = 1 - r['loc_pd'] if not np.isnan(r['loc_pd']) else np.nan
            # exits: stop variants x targets
            stops = {'zone': zlo, 'swing': sw[j] if (not np.isnan(sw[j]) and sw[j] < px) else zlo, 'atr02': px - 0.2 * atr}
            for sn, st in stops.items():
                risk = px - st
                if risk <= 0:
                    continue
                r[f'risk_{sn}'] = risk
                for tn, tg in (('1R', 1), ('2R', 2), ('3R', 3), ('EOD', 1e9)):
                    xk, xp, why = sim_exit(e, px, st, px + tg * risk, 0.0, dayi, mod, o, h, l, c)
                    r[f'g_{sn}_{tn}'] = (xp - px) / risk          # gross R
                    r[f'x_{sn}_{tn}'] = xk
                xk, xp, why = sim_exit(e, px, st, px + 1e9, risk, dayi, mod, o, h, l, c)
                r[f'g_{sn}_EODbe'] = (xp - px) / risk
                r[f'x_{sn}_EODbe'] = xk
            rows.append(r)
    t = pd.DataFrame(rows).sort_values('entry_k').reset_index(drop=True)
    t['entry_t'] = pd.to_datetime(tmin[t.entry_k] * NS)
    t['rr_tr'] = np.where(t.trend_d > 0, 2, 1)
    return t


if __name__ == '__main__':
    S = sys.argv[1]
    for mk in ('NQ', 'ES'):
        for N in (3, 5):
            t = build_table(S, mk, N=N)
            t.to_pickle(f'{S}/{mk}_table_N{N}.pkl')
            print(mk, N, len(t), 'signals')
