"""Order-block zones on a higher timeframe (30m / 1h), built without lookahead.

Demand OB: a bar closes above the last *confirmed* swing high (fractal, 2 bars each side)
-> the last bearish candle between that swing high and the breaking bar is the order block.
Zone = that candle's full range (high..low). Zone becomes usable when the breaking bar has closed.
Supply OB is the mirror image.
"""
import numpy as np
import pandas as pd

K = 2          # fractal: bars on each side of a swing point
MAX_BACK = 15  # how far back from the breaking bar we search for the OB candle


def find_zones(b, tf_label):
    o, h, l, c = (b[x].to_numpy() for x in ('open', 'high', 'low', 'close'))
    n = len(b)
    avg = pd.Series(h - l).shift(1).rolling(20).mean().to_numpy()   # mean range of the 20 bars before
    out = []
    sh = sl = None          # (index, level) of the latest confirmed swing high / low, not yet broken
    for i in range(n):
        # 1) break of structure on bar i, using swings confirmed before bar i
        if sh is not None and c[i] > sh[1]:
            for j in range(i - 1, max(sh[0], i - MAX_BACK) - 1, -1):
                if c[j] < o[j]:
                    fvg = any(l[m + 1] > h[m - 1] for m in range(j + 1, i))
                    out.append((b.t.iat[j], b.end.iat[i], 1, l[j], h[j], tf_label, sh[1], (c[i] - l[j]) / avg[i], fvg))
                    break
            sh = None
        if sl is not None and c[i] < sl[1]:
            for j in range(i - 1, max(sl[0], i - MAX_BACK) - 1, -1):
                if c[j] > o[j]:
                    fvg = any(h[m + 1] < l[m - 1] for m in range(j + 1, i))
                    out.append((b.t.iat[j], b.end.iat[i], -1, l[j], h[j], tf_label, sl[1], (h[j] - c[i]) / avg[i], fvg))
                    break
            sl = None
        # 2) swing point at p = i-K is confirmed once bar i has closed
        p = i - K
        if p >= K:
            if h[p] > h[p - K:p].max() and h[p] >= h[p + 1:i + 1].max():
                sh = (p, h[p])
            if l[p] < l[p - K:p].min() and l[p] <= l[p + 1:i + 1].min():
                sl = (p, l[p])
    z = pd.DataFrame(out, columns=['ob_t', 'created', 'dir', 'zlo', 'zhi', 'tf', 'broken_level', 'disp', 'fvg'])
    return z
