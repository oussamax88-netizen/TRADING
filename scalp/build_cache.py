"""Build fast-loading feature caches (New York time, 1-minute) for the scalping research.

Every feature in the cache is KNOWN AT THE CLOSE OF ITS BAR (no lookahead):
  daily values (atr, trend_d, pdh, pdl, pdc) come from previous RTH days only,
  running values (on_high, vwap, rth_open, ...) only use bars up to and including the current bar.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/user/TRADING/obchoch')
from data import load_mt5  # noqa: E402

S = '/tmp/claude-0/-home-user-TRADING/b9cb08f9-c0e5-5011-839c-6894039a978b/scratchpad'
SRC = {'NQ': f'{S}/x2/hada.csv', 'ES': f'{S}/x1/US500 2023.csv'}


def build(mk):
    d = load_mt5(SRC[mk])
    t = (d.t.astype('datetime64[ns]').astype('int64') // 60_000_000_000).to_numpy()
    o, h, l, c = (d[x].to_numpy(np.float64) for x in ('open', 'high', 'low', 'close'))
    vol = d.tickvol.to_numpy(np.float64)
    mod = (t % 1440).astype(np.int64)
    date = (t // 1440).astype(np.int64)
    udates, dayi = np.unique(date, return_inverse=True)
    dow = ((date + 3) % 7).astype(np.int64)          # 0 = Monday (1970-01-01 was a Thursday)
    rth = (mod >= 570) & (mod < 960)
    df = pd.DataFrame({'date': date, 'h': h, 'l': l, 'c': c, 'o': o, 'mod': mod, 'rth': rth})

    # ---- previous-RTH-day context (known before the day starts)
    r = df[rth].groupby('date').agg(hi=('h', 'max'), lo=('l', 'min'), cl=('c', 'last'), op=('o', 'first'))
    r['rng'] = r.hi - r.lo
    ctx = pd.DataFrame(index=r.index)
    ctx['atr'] = r.rng.rolling(14).mean().shift(1)
    ctx['trend_d'] = np.sign(r.cl - r.cl.rolling(20).mean()).shift(1)
    ctx['pdh'] = r.hi.shift(1); ctx['pdl'] = r.lo.shift(1); ctx['pdc'] = r.cl.shift(1)
    # dates without RTH (Sunday evening, holidays) take the NEXT RTH date's context (all of it is from earlier days)
    ctx = ctx.reindex(udates).bfill()
    out = {k: ctx[k].to_numpy()[dayi] for k in ctx.columns}

    # ---- 09:30 open of the current date, known from 09:30 on
    op930 = r.op.reindex(udates).to_numpy()[dayi]
    out['rth_open'] = np.where(mod >= 570, op930, np.nan)

    # ---- overnight session (18:00 previous evening -> 09:29), running high/low; final values during RTH
    sid = np.where(mod >= 1080, date + 1, date)
    sid = np.where(dow == 6, sid, sid)  # Sunday 18:00 -> Monday session (date+1 already)
    on = (mod >= 1080) | (mod < 570)
    g = pd.DataFrame({'sid': sid, 'h': np.where(on, h, np.nan), 'l': np.where(on, l, np.nan)})
    onh_run = g.groupby('sid').h.cummax().to_numpy()
    onl_run = g.groupby('sid').l.cummin().to_numpy()
    fin = g.groupby('sid').agg(H=('h', 'max'), L=('l', 'min'))
    finH = fin.H.reindex(sid).to_numpy(); finL = fin.L.reindex(sid).to_numpy()
    out['on_high'] = np.where(on, onh_run, np.where(mod < 1080, finH, np.nan))
    out['on_low'] = np.where(on, onl_run, np.where(mod < 1080, finL, np.nan))
    # make RTH values strictly from the finished overnight session (finH covers only overnight bars)
    # ---- RTH session VWAP (typical price weighted by tick volume), from 09:30
    tp = (h + l + c) / 3
    w = np.where(rth, vol, 0.0); pv = np.where(rth, tp * vol, 0.0)
    cw = pd.Series(w).groupby(date).cumsum().to_numpy(); cpv = pd.Series(pv).groupby(date).cumsum().to_numpy()
    out['vwap'] = np.where(rth & (cw > 0), cpv / np.maximum(cw, 1e-9), np.nan)
    # ---- RTH running high/low of the current day
    rh = pd.Series(np.where(rth, h, np.nan)).groupby(date).cummax().to_numpy()
    rl = pd.Series(np.where(rth, l, np.nan)).groupby(date).cummin().to_numpy()
    out['rth_high'] = np.where(rth, rh, np.nan); out['rth_low'] = np.where(rth, rl, np.nan)

    np.savez_compressed(f'{S}/scalp_{mk}.npz', t=t, o=o, h=h, l=l, c=c, vol=vol, mod=mod, date=date, dayi=dayi.astype(np.int64),
                        dow=dow, spread=d.spread.to_numpy(np.float64), **out)
    print(mk, len(t), 'bars cached')


if __name__ == '__main__':
    for mk in ('NQ', 'ES'):
        build(mk)
