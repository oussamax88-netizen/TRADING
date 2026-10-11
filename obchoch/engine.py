"""OB + 1m CHOCH backtest engine (1-minute simulation, New York time).

Written for the demand (long) side only; supply (short) trades run through the same
kernel on price-negated arrays, so both sides follow identical logic.

Per zone, first touch only:
  departure  optional: a touch only counts after price has moved >= depk x zone height away from the zone
  touch      first 1m bar whose low reaches the zone top, inside the entry window
  big-candle optional: when the 5m bar containing the touch closes, if the arrival was 'fast'
             the touch is ignored; the zone survives and a new touch can come after a 5m bar
             closes fully back outside the zone ('slow momentum coming back')
  cancel     any 5m close beyond the far side of the zone kills it (also before the touch)
  CHOCH      first 1m close above the latest confirmed 1m swing high (fractal N each side)
  entry      A: next 1m open   B: limit at the broken swing level within 30 bars
  stop       far side of the zone minus buffer
  target     2R with trend, 1R against trend
  exit       stop / target / forced close at 15:59 NY. Stop wins if both hit in the same bar,
             no target fill on the entry bar, a gap through the stop fills at the bar open.
"""
import numpy as np
import pandas as pd
from numba import njit

WIN_START = 3 * 60          # 03:00 NY  (London open)
WIN_END = 15 * 60 + 30      # 15:30 NY  last new entry
FORCE_CLOSE = 15 * 60 + 59  # 15:59 NY


def swing_levels(h, l, N):
    """Latest confirmed swing high / low level as known at each bar's close."""
    n = len(h)
    hs = pd.Series(h)
    ls = pd.Series(l)
    left_h = hs.shift(1).rolling(N).max()
    right_h = hs[::-1].shift(1).rolling(N).max()[::-1]
    left_l = ls.shift(1).rolling(N).min()
    right_l = ls[::-1].shift(1).rolling(N).min()[::-1]
    ph = (hs > left_h) & (hs >= right_h)
    pl = (ls < left_l) & (ls <= right_l)
    sh = np.full(n, np.nan)
    sl = np.full(n, np.nan)
    idx_h = np.where(ph.to_numpy())[0]
    idx_l = np.where(pl.to_numpy())[0]
    sh[idx_h + N] = h[idx_h]          # known once N bars to the right have closed
    sl[idx_l + N] = l[idx_l]
    sh = pd.Series(sh[:n]).ffill().to_numpy()
    sl = pd.Series(sl[:n]).ffill().to_numpy()
    return sh, sl


def build_arrays(d, ctx):
    t = d.t
    a = {}
    a['tmin'] = (t.astype('int64') // 60_000_000_000).to_numpy() if t.dtype == 'datetime64[ns]' else \
        (t.astype('datetime64[ns]').astype('int64') // 60_000_000_000).to_numpy()
    a['mod'] = (t.dt.hour * 60 + t.dt.minute).to_numpy().astype(np.int64)
    dates = t.dt.date
    udates = pd.Index(sorted(dates.unique()))
    a['dayi'] = udates.get_indexer(dates).astype(np.int64)
    a['dates'] = udates
    for x in ('open', 'high', 'low', 'close'):
        a[x[0]] = d[x].to_numpy().astype(np.float64)
    o, h, l, c = a['o'], a['h'], a['l'], a['c']
    n = len(d)
    # 5-minute bars, evaluated at the 1m bar that completes them
    b5 = a['tmin'] // 5
    is5 = np.ones(n, dtype=np.bool_)
    is5[:-1] = b5[1:] != b5[:-1]
    g = pd.DataFrame({'b': b5, 'o': o, 'h': h, 'l': l, 'c': c}).groupby('b', sort=True)
    f5 = g.agg(o=('o', 'first'), h=('h', 'max'), l=('l', 'min'), c=('c', 'last'))
    f5['rng'] = f5.h - f5.l
    f5['avg'] = f5.rng.shift(1).rolling(20).mean()
    f5['po'] = f5.o.shift(1); f5['pc'] = f5.c.shift(1)
    f5['prng'] = f5.rng.shift(1); f5['pavg'] = f5.avg.shift(1)
    m = f5.reindex(b5)
    for col in ('o', 'h', 'l', 'c', 'rng', 'avg', 'po', 'pc', 'prng', 'pavg'):
        a['f5' + col] = m[col].to_numpy()
    a['is5'] = is5
    for N in (3, 5):
        a[f'sh{N}'], a[f'sl{N}'] = swing_levels(h, l, N)
    # 1h trend: close of last completed 1h bar vs EMA50 of 1h closes
    b60 = a['tmin'] // 60
    h1 = pd.DataFrame({'b': b60, 'c': c}).groupby('b').c.last()
    ema = h1.ewm(span=50, adjust=False).mean()
    tr = np.sign(h1 - ema)
    is60 = np.ones(n, dtype=np.bool_)
    is60[:-1] = b60[1:] != b60[:-1]
    # value for bar j = trend of the hour whose last 1m bar is j or earlier
    tr_at_close = np.where(is60, tr.reindex(b60).to_numpy(), np.nan)
    a['trend_h'] = pd.Series(tr_at_close).ffill().fillna(0).to_numpy()
    cd = ctx.reindex(udates)
    a['trend_d'] = cd.trend_d.to_numpy()[a['dayi']]
    a['trend_d'] = np.nan_to_num(a['trend_d'])
    a['atr'] = cd.atr.to_numpy()[a['dayi']]
    a['atr_next'] = pd.Series(a['atr']).bfill().to_numpy()
    return a


@njit(cache=True)
def run_zone(start, zlo, zhi, exp_day, tmin, mod, dayi, o, h, l, c,
             is5, f5o, f5h, f5l, f5c, f5rng, f5avg, f5po, f5pc, f5prng, f5pavg,
             sh, trend, buf, big, entry_mode, deadline, dep):
    """Demand side. Returns (touch_k, choch_j, entry_k, entry_px, stop, rr, exit_k, exit_px, reason)."""
    n = len(o)
    NONE = (-1, -1, -1, 0.0, 0.0, 0.0, -1, 0.0, 0)
    state = 0
    touch_k = -1
    away = -1e18
    k = start
    while k < n:
        if dayi[k] > exp_day:
            return NONE
        if state == 0 and dep > 0 and l[k] <= zhi and away - zhi < dep:
            pass                                 # price has not left the zone yet: not a return
        elif state == 0 and l[k] <= zhi:
            if mod[k] < WIN_START or mod[k] >= WIN_END:
                return NONE                      # touched while we are not trading: zone used up
            touch_k = k
            state = 2 if big == 0 else 1
        if is5[k] and f5c[k] < zlo:
            return NONE                          # 5m close beyond the zone: cancelled
        if h[k] > away:
            away = h[k]
        if state == 1 and is5[k]:
            fast = False
            if big == 1:
                fast = f5rng[k] > 2.0 * f5avg[k]
            elif big == 2:
                fast = (f5c[k] < f5o[k] and f5pc[k] < f5po[k] and
                        f5rng[k] > 1.5 * f5avg[k] and f5prng[k] > 1.5 * f5pavg[k])
            state = 3 if fast else 2
            if state == 3:
                k += 1
                continue
        if state == 3:
            if is5[k] and f5l[k] > zhi:
                state = 0                        # price left the zone; wait for a slower return
            k += 1
            continue
        if state == 2:
            if deadline > 0 and tmin[k] - tmin[touch_k] > deadline:
                return NONE
            if mod[k] >= WIN_END or dayi[k] != dayi[touch_k]:
                return NONE
            if not np.isnan(sh[k]) and c[k] > sh[k]:
                j = k
                stop = zlo - buf
                e = -1
                px = 0.0
                if entry_mode == 0:
                    e = j + 1
                    if e >= n or dayi[e] != dayi[j] or mod[e] >= FORCE_CLOSE:
                        return NONE
                    px = o[e]
                else:
                    lvl = sh[j]
                    m = j + 1
                    while m < n and m <= j + 30 and dayi[m] == dayi[j] and mod[m] < WIN_END:
                        if o[m] <= lvl:
                            px = o[m]; e = m; break
                        if l[m] <= lvl:
                            px = lvl; e = m; break
                        m += 1
                    if e < 0:
                        return NONE
                if px <= stop:
                    return NONE
                risk = px - stop
                rr = 2.0 if trend[j] > 0 else 1.0
                tgt = px + rr * risk
                # --- exit simulation
                if l[e] <= stop:
                    return (touch_k, j, e, px, stop, rr, e, stop, 1)
                m = e + 1
                while m < n:
                    if dayi[m] != dayi[e]:
                        return (touch_k, j, e, px, stop, rr, m - 1, c[m - 1], 3)
                    if mod[m] >= FORCE_CLOSE:
                        return (touch_k, j, e, px, stop, rr, m, o[m], 3)
                    if o[m] <= stop:
                        return (touch_k, j, e, px, stop, rr, m, o[m], 1)
                    if l[m] <= stop:
                        return (touch_k, j, e, px, stop, rr, m, stop, 1)
                    if h[m] >= tgt:
                        return (touch_k, j, e, px, stop, rr, m, tgt, 2)
                    m += 1
                return (touch_k, j, e, px, stop, rr, n - 1, c[n - 1], 3)
        k += 1
    return NONE


def run_variant(a, zones, p, cost):
    """p: dict(N, deadline, entry, trend, big, life, bufk). Returns trade DataFrame (one position at a time)."""
    tmin = a['tmin']
    rows = []
    sides = {}
    for side in (1, -1):
        s = side
        if s == 1:
            arr = dict(o=a['o'], h=a['h'], l=a['l'], c=a['c'], f5o=a['f5o'], f5h=a['f5h'], f5l=a['f5l'],
                       f5c=a['f5c'], f5po=a['f5po'], f5pc=a['f5pc'], sh=a[f"sh{p['N']}"])
        else:
            arr = dict(o=-a['o'], h=-a['l'], l=-a['h'], c=-a['c'], f5o=-a['f5o'], f5h=-a['f5l'], f5l=-a['f5h'],
                       f5c=-a['f5c'], f5po=-a['f5po'], f5pc=-a['f5pc'], sh=-a[f"sl{p['N']}"])
        arr['trend'] = s * a['trend_h' if p['trend'] == 'H' else 'trend_d']
        sides[s] = arr
    life = 5 if p['life'] == 'A' else 60
    for z in zones.itertuples(index=False):
        s = z.dir
        arr = sides[s]
        zlo, zhi = (z.zlo, z.zhi) if s == 1 else (-z.zhi, -z.zlo)
        atr = a['atr_next'][z.start]   # first known ATR from the zone's start (Sunday-evening zones have no RTH date)
        if np.isnan(atr):
            continue
        r = run_zone(z.start, zlo, zhi, a['dayi'][z.start] + life, tmin, a['mod'], a['dayi'],
                     arr['o'], arr['h'], arr['l'], arr['c'], a['is5'], arr['f5o'], arr['f5h'], arr['f5l'], arr['f5c'],
                     a['f5rng'], a['f5avg'], arr['f5po'], arr['f5pc'], a['f5prng'], a['f5pavg'],
                     arr['sh'], arr['trend'], p['bufk'] * atr, p['big'], p['entry'], p['deadline'],
                     p.get('depk', 0.0) * (z.zhi - z.zlo))
        if r[2] < 0:
            continue
        touch_k, j, e, px, stop, rr, xk, xpx, why = r
        risk = px - stop
        pts = (xpx - px) - cost
        rows.append(dict(side=s, tf=z.tf, zlo=z.zlo, zhi=z.zhi, created=z.created, ob_t=z.ob_t,
                         touch_k=touch_k, choch_k=j, entry_k=e, exit_k=xk,
                         entry=s * px, stop=s * stop, target=s * (px + rr * risk), exit=s * xpx,
                         risk=risk, rr=rr, pts=pts, R=pts / risk, reason=why, atr=atr))
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    t = t.sort_values(['entry_k', 'tf']).reset_index(drop=True)
    keep = []
    last_exit = -1
    for i, e, x in zip(t.index, t.entry_k, t.exit_k):
        if e > last_exit:
            keep.append(i)
            last_exit = x
    t = t.loc[keep].reset_index(drop=True)
    t['entry_t'] = pd.to_datetime(tmin[t.entry_k] * 60_000_000_000)
    t['exit_t'] = pd.to_datetime(tmin[t.exit_k] * 60_000_000_000)
    return t


def prepare_zones(zones, a):
    z = zones.copy()
    created_min = (z.created.astype('datetime64[ns]').astype('int64') // 60_000_000_000).to_numpy()
    z['start'] = np.searchsorted(a['tmin'], created_min, side='left')
    return z[z.start < len(a['tmin'])].reset_index(drop=True)
