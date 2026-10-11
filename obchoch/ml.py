"""Walk-forward check: can ANY combination of entry-time features pick winners out of sample?
Model trained on the first half only (both markets pooled), then ranks second-half signals."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'obchoch')
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from evalr import trades, pf, IS_END, COST
S = sys.argv[1]
F = ['zone_h', 'age_min', 'away', 'pen', 'choch_min', 'mod', 'ext', 'loc_pd', 'pcm', 'risk_atr', 'side', 'tfc',
     'trend_d', 'trend_h', 'tdh', 'disp', 'fvg', 'sweep_pd', 'zone_beyond_pd', 'dow']
for ex in ('zone_TR', 'zone_EOD', 'atr02_EOD'):
    st, tg = ex.split('_')
    D = []
    for mk in ('NQ', 'ES'):
        t = pd.read_pickle(f'{S}/{mk}_table_N5.pkl')
        t['risk_atr'] = t[f'risk_{st}'] / t.atr; t['tfc'] = t.tf.map({'30m': 0, '1h': 1, '4h': 2, '1d': 3})
        s = trades(t, mk, t.side != 0, st, tg, overlap=False); s['mk'] = mk
        D.append(s)
    D = pd.concat(D)
    X = D[F].astype(float).to_numpy(); y = D.R.clip(-3, 5).to_numpy()
    tr = (D.entry_t < IS_END).to_numpy(); te = ~tr
    res = []
    for name, m in (('GBM', HistGradientBoostingRegressor(max_depth=3, max_iter=150, learning_rate=0.05, min_samples_leaf=60, random_state=0)),
                    ('Logit', None)):
        if name == 'Logit':
            Xf = np.nan_to_num(X); mu, sd = Xf[tr].mean(0), Xf[tr].std(0) + 1e-9
            m = LogisticRegression(C=0.1, max_iter=2000).fit((Xf[tr] - mu) / sd, (y[tr] > 0))
            p = m.predict_proba((Xf - mu) / sd)[:, 1]
        else:
            m.fit(X[tr], y[tr]); p = m.predict(X)
        cut = {q: np.quantile(p[tr], q) for q in (0.5, 0.7, 0.8)}   # thresholds fixed on IS
        for q, c in cut.items():
            for mk in ('NQ', 'ES'):
                sel_is = tr & (p >= c) & (D.mk == mk).to_numpy(); sel_oos = te & (p >= c) & (D.mk == mk).to_numpy()
                res.append(dict(model=name, top=f'{int((1-q)*100)}%', mk=mk, pf_IS=pf(D.R[sel_is]), pf_OOS=pf(D.R[sel_oos]), n_OOS=sel_oos.sum(),
                                pf_OOS_all=pf(D.R[te & (D.mk == mk).to_numpy()])))
    print('\nExit', ex)
    print(pd.DataFrame(res).pivot_table(index=['model', 'top'], columns='mk', values=['pf_IS', 'pf_OOS', 'n_OOS', 'pf_OOS_all']).round(2).to_string())
