import sys, numpy as np, pandas as pd
sys.path.insert(0, 'obchoch')
from evalr import trades, pf, IS_END
S = sys.argv[1]; pd.set_option('display.width', 250)
for f, bins in (('pcm', [-9, -1, -0.5, -0.2, 0, 0.2, 0.5, 1, 9]), ('age_min', [0, 30, 90, 240, 600, 1500, 99999]), ('risk_atr', [0, .1, .2, .3, .5, 9]), ('loc_pd', [-9, 0, .25, .5, .75, 1, 9])):
    out = []
    for mk in ('NQ', 'ES'):
        t = pd.read_pickle(f'{S}/{mk}_table_N5.pkl'); t['risk_atr'] = t.risk_zone / t.atr
        s = trades(t, mk, t.side != 0, 'zone', 'TR', overlap=False)
        s['b'] = pd.cut(s[f], bins); s['half'] = np.where(s.entry_t < IS_END, 'IS', 'OOS')
        for (b, h), g in s.groupby(['b', 'half'], observed=True):
            out.append(dict(mk=mk, b=b, half=h, pf=pf(g.R), n=len(g)))
    o = pd.DataFrame(out)
    print(f, '\n', o.pivot_table(index='b', columns=['mk', 'half'], values=['pf', 'n'], observed=True).round(2).to_string(), '\n')
