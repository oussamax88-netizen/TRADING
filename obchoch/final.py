"""Final candidate 'AZL' (Asia Zone -> London CHOCH, with 20-day trend, 0.2 ATR stop, hold to 15:59) full report."""
import sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, 'obchoch')
from evalr import trades, summ, pf, IS_END, COST
from validate_helpers import full
from validate import sim
S = sys.argv[1]; pd.set_option('display.width', 250)
rng = np.random.default_rng(11)
def rule(t): return ((t.cmod >= 1080) | (t.cmod <= 180)) & (t['mod'] < 570) & (t.trend_d > 0)
ALL = []
for mk in ('NQ', 'ES'):
    a, _ = pickle.load(open(f'{S}/{mk}.pkl', 'rb'))
    t = pd.read_pickle(f'{S}/{mk}_table_N5.pkl')
    s = trades(t, mk, rule(t), 'atr02', 'EOD'); s['mk'] = mk
    print(f'\n######## {mk}', full(s))
    s['half'] = np.where(s.entry_t < IS_END, '1st half (IS)', '2nd half (OOS)'); s['year'] = s.entry_t.dt.year
    s['hour'] = s.entry_t.dt.hour; s['dow'] = s.entry_t.dt.day_name().str[:3]; s['tfz'] = s.tf
    for by in ('half', 'year', 'side', 'tfz', 'hour', 'dow'):
        g = s.groupby(by).R.agg(n='size', pf=pf, netR='sum', win=lambda x: (x > 0).mean() * 100).round(2)
        print(' ', by, g.to_dict('index'))
    s2 = trades(t, mk, rule(t), 'atr02', 'EOD', costmult=2.0); print('  double cost PF', round(pf(s2.R), 2), '| median risk pts', round(s.risk_atr02.median(), 1))
    R = s.R.to_numpy(); bs = np.array([pf(rng.choice(R, len(R))) for _ in range(5000)])
    print('  bootstrap PF 5/50/95%', np.round(np.percentile(bs, [5, 50, 95]), 2), 'P(PF<1)', round((bs < 1).mean(), 3), 'P(PF<1.7)', round((bs < 1.7).mean(), 3))
    rk = s.risk_atr02.to_numpy()
    opp = np.array([sim(a, e, -sd, r, COST[mk]) for e, sd, r in zip(s.entry_k, s.side, rk)])
    rc = [pf(np.array([sim(a, e, sd * (1 if rng.random() < .5 else -1), r, COST[mk]) for e, sd, r in zip(s.entry_k, s.side, rk)])) for _ in range(200)]
    dayi, mod = a['dayi'], a['mod']; idx = np.arange(len(mod))[(mod >= 180) & (mod < 570)]
    rngd = pd.Series(idx).groupby(dayi[idx]).agg(['min', 'max'])
    nz = [pf(np.array([sim(a, rng.integers(*rngd.loc[dayi[e]].to_numpy() + [0, 1]), sd, r, COST[mk]) for e, sd, r in zip(s.entry_k, s.side, rk)])) for _ in range(200)]
    nof = trades(t, mk, ((t.cmod >= 1080) | (t.cmod <= 180)) & (t['mod'] < 570), 'atr02', 'EOD')
    allz = trades(t, mk, t.side != 0, 'atr02', 'EOD')
    print(f'  CONTROLS: strategy {pf(s.R):.2f} | opposite direction {pf(opp):.2f} | random direction median {np.median(rc):.2f} (beats {np.mean(np.array(rc) < pf(s.R))*100:.0f}%)'
          f' | trend direction at random London minute (no zone, no CHOCH) median {np.median(nz):.2f} (95th {np.percentile(nz, 95):.2f}, beats {np.mean(np.array(nz) < pf(s.R))*100:.0f}%)'
          f' | without trend filter {pf(nof.R):.2f} ({len(nof)} tr) | all zones/any time, same exit {pf(allz.R):.2f} ({len(allz)} tr)')
    ALL.append(s)
A = pd.concat(ALL).sort_values('entry_t')
print('\n######## PORTFOLIO NQ+ES', full(A))
dd = A.groupby(A.entry_t.dt.date).mk.nunique(); print('  days with a trade', len(dd), '| days with both markets', (dd == 2).sum())
both = A[A.entry_t.dt.date.isin(dd[dd == 2].index)]
p = both.pivot_table(index=both.entry_t.dt.date, columns='mk', values='R', aggfunc='sum'); print('  same-day R correlation NQ vs ES', round(p.corr().iloc[0, 1], 2))
A.to_pickle(f'{S}/AZL_trades.pkl')
