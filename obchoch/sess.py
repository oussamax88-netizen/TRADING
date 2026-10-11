import sys, numpy as np, pandas as pd
sys.path.insert(0, 'obchoch')
from evalr import trades, pf, IS_END
S = sys.argv[1]; pd.set_option('display.width', 250)
def sess(m):
    return np.select([(m >= 1080) | (m < 180), m < 570, m < 960], ['1 Asia 18-03', '2 London 03-09:30', '3 NY 09:30-16'], '0 16-18')
for ex in ('zone_EOD', 'zone_TR'):
    st, tg = ex.split('_'); out = []
    for mk in ('NQ', 'ES'):
        t = pd.read_pickle(f'{S}/{mk}_table_N5.pkl')
        s = trades(t, mk, t.side != 0, st, tg, overlap=False)
        tmod = (s.entry_t.dt.hour * 60 + s.entry_t.dt.minute).to_numpy()
        cmod = ((tmod - s.age_min.to_numpy() - s.choch_min.to_numpy()) % 1440)
        s['cs'] = sess(cmod); s['es'] = np.where(tmod < 570, 'enter London', 'enter NY')
        s['sameday'] = (s.age_min + s.choch_min) < tmod + 360  # created after 18:00 previous evening
        s['half'] = np.where(s.entry_t < IS_END, 'IS', 'OOS')
        for (a, b, h), g in s.groupby(['cs', 'es', 'half']):
            out.append(dict(mk=mk, create=a, enter=b, half=h, pf=pf(g.R), n=len(g)))
    o = pd.DataFrame(out)
    print(ex); print(o.pivot_table(index=['create', 'enter'], columns=['mk', 'half'], values=['pf', 'n']).round(2).to_string(), '\n')
