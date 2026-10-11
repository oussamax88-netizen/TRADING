import sys, pickle, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
sys.path.insert(0, 'obchoch')
from charts import plot_trade
S, out = sys.argv[1], sys.argv[2]
A = pd.read_pickle(f'{S}/AZL_trades.pkl')
rng = np.random.default_rng(5)
samp = pd.concat([A[A.R > 0].sample(4, random_state=1), A[A.R <= 0].sample(6, random_state=1)]).sort_values('entry_t')
with PdfPages(out) as pdf:
    for mk in ('NQ', 'ES'):
        a, z = pickle.load(open(f'{S}/{mk}.pkl', 'rb'))
        for i, r in samp[samp.mk == mk].iterrows():
            sd = r.side
            zlo, zhi = (r.zlo, r.zhi) if sd == 1 else (-r.zhi, -r.zlo)
            zm = z[(z.created == r.created) & (z.dir == sd) & np.isclose(z.zlo, zlo) & np.isclose(z.zhi, zhi)]
            px = r.px; rk = r.risk_atr02; g = r.g_atr02_EOD
            tr = pd.Series(dict(tf=r.tf, ob_t=zm.ob_t.iat[0], created=r.created, side=sd, zlo=zlo, zhi=zhi, touch_k=r.touch_k, choch_k=r.choch_k,
                                entry_k=r.entry_k, exit_k=int(r.x_atr02_EOD), entry=sd * px, stop=sd * (px - rk), target=np.nan, rr=0,
                                exit=sd * (px + g * rk), reason=1 if g <= -0.999 else 3, R=r.R), name=i)
            plot_trade(pdf, a, tr, mk, '  |  AZL: Asia zone -> London CHOCH, with trend, stop 0.2 ATR, hold to 15:59')
print(samp[['mk', 'entry_t', 'side', 'tf', 'R']].to_string())
