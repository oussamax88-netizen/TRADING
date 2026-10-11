import sys, numpy as np, pandas as pd
sys.path.insert(0, 'obchoch')
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from evalr import trades, IS_END
S = sys.argv[1]
F = ['zone_h', 'age_min', 'away', 'pen', 'choch_min', 'mod', 'ext', 'loc_pd', 'pcm', 'risk_atr', 'side', 'tfc',
     'trend_d', 'trend_h', 'tdh', 'disp', 'fvg', 'sweep_pd', 'zone_beyond_pd', 'dow']
st, tg = 'zone', 'TR'
D = []
for mk in ('NQ', 'ES'):
    t = pd.read_pickle(f'{S}/{mk}_table_N5.pkl')
    t['risk_atr'] = t[f'risk_{st}'] / t.atr; t['tfc'] = t.tf.map({'30m': 0, '1h': 1, '4h': 2, '1d': 3})
    s = trades(t, mk, t.side != 0, st, tg, overlap=False); s['mk'] = mk; D.append(s)
D = pd.concat(D); X = D[F].astype(float); y = D.R.clip(-3, 5)
tr = D.entry_t < IS_END
m = HistGradientBoostingRegressor(max_depth=3, max_iter=150, learning_rate=0.05, min_samples_leaf=60, random_state=0).fit(X[tr], y[tr])
for nm, mask in (('IS', tr), ('OOS', ~tr)):
    pi = permutation_importance(m, X[mask], y[mask], n_repeats=20, random_state=0, scoring='r2')
    print(nm, pd.Series(pi.importances_mean, F).sort_values(ascending=False).round(4).head(8).to_dict())
