"""Load MT5 1-minute exports and convert to New York time (server = NY + 7h, verified)."""
import numpy as np
import pandas as pd

SERVER_MINUS_NY = 7


def load_mt5(path):
    d = pd.read_csv(path, sep='\t')
    d.columns = [c.strip('<>').lower() for c in d.columns]
    t = pd.to_datetime(d['date'] + ' ' + d['time'], format='%Y.%m.%d %H:%M:%S')
    d['t'] = t - pd.Timedelta(hours=SERVER_MINUS_NY)  # New York time
    d = d[['t', 'open', 'high', 'low', 'close', 'tickvol', 'spread']].sort_values('t')
    return d.drop_duplicates('t').reset_index(drop=True)


def resample(d, minutes):
    """Bars in NY time, labelled by open time; 'end' = time the bar is complete."""
    g = d.set_index('t').resample(f'{minutes}min', label='left', closed='left')
    b = g.agg(open=('open', 'first'), high=('high', 'max'), low=('low', 'min'),
              close=('close', 'last')).dropna().reset_index()
    b['end'] = b['t'] + pd.Timedelta(minutes=minutes)
    return b


def daily_context(d):
    """Per NY date: ATR (14-day mean of 09:30-16:00 range) and 20-day trend, both known before the day starts."""
    mod = d.t.dt.hour * 60 + d.t.dt.minute
    rth = d[(mod >= 570) & (mod < 960)]
    day = rth.groupby(rth.t.dt.date).agg(hi=('high', 'max'), lo=('low', 'min'), cl=('close', 'last'))
    day['rng'] = day.hi - day.lo
    ctx = pd.DataFrame(index=day.index)
    ctx['atr'] = day.rng.rolling(14).mean().shift(1)
    sma = day.cl.rolling(20).mean()
    ctx['trend_d'] = np.sign(day.cl - sma).shift(1)  # +1 up, -1 down (prev 16:00 close vs its 20-day SMA)
    return ctx


def resample_daily(d):
    """Futures-style daily bars 18:00-17:00 NY, labelled by start; 'end' = 18:00 next day (bar complete by 17:00)."""
    g = d.set_index('t').resample('24h', offset='18h', label='left', closed='left')
    b = g.agg(open=('open', 'first'), high=('high', 'max'), low=('low', 'min'),
              close=('close', 'last')).dropna().reset_index()
    b['end'] = b['t'] + pd.Timedelta(days=1)
    return b
