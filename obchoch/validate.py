"""Full validation of the 'Asia zone -> London CHOCH, hold to 15:59' candidate."""
import sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, 'obchoch')
from research import sim_exit
from evalr import trades, summ, pf, IS_END, COST
pd.set_option('display.width', 250)
rng = np.random.default_rng(1)

def asia(t, c0=1080, c1=180, e1=570):
    cm = t.cmod
    return ((cm >= c0) | (cm <= c1)) & (t['mod'] < e1)

def sim(a, e, d, risk, cost):
    if d == 1: o, h, l, c = a['o'], a['h'], a['l'], a['c']
    else:      o, h, l, c = -a['o'], -a['l'], -a['h'], -a['c']
    px = o[e]
    xk, xp, _ = sim_exit(e, px, px - risk, px + 1e9, 0.0, a['dayi'], a['mod'], o, h, l, c)
    return (xp - px - cost) / risk

def stats_full(s):
    s = s.sort_values('entry_t'); R = s.R
    eq = R.cumsum(); dd = (eq - eq.cummax()).min()
    los = (R <= 0).astype(int); streak = los.groupby((los != los.shift()).cumsum()).cumsum().max()
    m = s.groupby(s.entry_t.dt.to_period('M')).R.sum()
    return dict(n=len(s), wk=round(len(s) / 156.4, 2), win=round((R > 0).mean() * 100), avgW=round(R[R > 0].mean(), 2), avgL=round(R[R <= 0].mean(), 2),
                pf=round(pf(R), 2), netR=round(R.sum(), 1), maxDD=round(dd, 1), streak=int(streak), losing_months=f'{(m < 0).mean()*100:.0f}%')

if __name__ == '__main__':
    S = sys.argv[1]
    for N in (5, 3):
      for mk in ('NQ', 'ES'):
        a, _ = pickle.load(open(f'{S}/{mk}.pkl', 'rb'))
        t = pd.read_pickle(f'{S}/{mk}_table_N{N}.pkl')
        s = trades(t, mk, asia(t), 'zone', 'EOD')
        print(f'\n################ {mk}  (CHOCH swing N={N})')
        print('HEADLINE', stats_full(s))
        if N == 3:
            continue
        s['half'] = np.where(s.entry_t < IS_END, 'IS', 'OOS'); s['year'] = s.entry_t.dt.year
        s['hour'] = s.entry_t.dt.hour; s['dow'] = s.entry_t.dt.day_name().str[:3]; s['trendwith'] = s.trend_d > 0
        for by in ('half', 'year', 'side', 'tf', 'hour', 'dow', 'trendwith'):
            g = s.groupby(by).R.agg(n='size', pf=pf, netR='sum', win=lambda x: (x > 0).mean() * 100).round(2)
            print(by, g.T.to_dict())
        # costs
        s2 = trades(t, mk, asia(t), 'zone', 'EOD', costmult=2.0)
        print('double cost PF', round(pf(s2.R), 2), '| cost per trade in R (median)', round((COST[mk] / s.risk_zone).median(), 3))
        # bootstrap
        R = s.R.to_numpy(); bs = np.array([pf(rng.choice(R, len(R))) for _ in range(5000)])
        print('bootstrap PF 5/50/95%:', np.round(np.percentile(bs, [5, 50, 95]), 2), ' P(PF<1)=', round((bs < 1).mean(), 3), ' P(PF<1.7)=', round((bs < 1.7).mean(), 3))
        # neighbours
        nb = []
        for nm, m in (('Asia 18-03 (base)', asia(t)), ('create 17-03', asia(t, 1020)), ('create 19-03', asia(t, 1140)), ('create 18-02', asia(t, 1080, 120)),
                      ('create 18-04', asia(t, 1080, 240)), ('enter until 08:00', asia(t, e1=480)), ('enter until 11:00', asia(t, e1=660)), ('enter until 15:30', asia(t, e1=930))):
            for st in ('zone', 'atr02', 'swing'):
                x = trades(t, mk, m, st, 'EOD'); r = summ(x); nb.append(dict(v=nm, stop=st, pf=r['pf'], IS=r['pf_IS'], OOS=r['pf_OOS'], n=r['n']))
            for tg in ('2R', '3R', 'TR', 'EODbe'):
                x = trades(t, mk, m, 'zone', tg); r = summ(x); nb.append(dict(v=nm, stop='zone/' + tg, pf=r['pf'], IS=r['pf_IS'], OOS=r['pf_OOS'], n=r['n']))
        nb = pd.DataFrame(nb)
        print(nb.pivot_table(index='v', columns='stop', values='pf', sort=False).round(2).to_string())
        print('family median PF', round(nb.pf.median(), 2))
        # controls
        rc = []
        for _ in range(300):
            flip = rng.random(len(s)) < 0.5
            rr = [sim(a, e, -sd if f else sd, rk, COST[mk]) for e, sd, rk, f in zip(s.entry_k, s.side, s.risk_zone, flip)]
            rc.append(pf(np.array(rr)))
        opp = np.array([sim(a, e, -sd, rk, COST[mk]) for e, sd, rk in zip(s.entry_k, s.side, s.risk_zone)])
        notouch = np.array([sim(a, tk + 1, sd, abs(a['o'][tk + 1] - (zl if sd == 1 else -zl)), COST[mk]) if abs(a['o'][tk + 1] - (zl if sd == 1 else -zl)) > 0 else 0
                            for tk, sd, zl in zip(s.touch_k, s.side, s.zlo)])
        # no zone: same day, random minute 03:00-09:30, same direction, same risk
        dayi, mod = a['dayi'], a['mod']
        first = pd.Series(np.arange(len(mod))[(mod >= 180) & (mod < 570)]).groupby(dayi[(mod >= 180) & (mod < 570)]).agg(['min', 'max'])
        nz = []
        for _ in range(100):
            rr = []
            for e, sd, rk in zip(s.entry_k, s.side, s.risk_zone):
                lo, hi = first.loc[dayi[e]]; rr.append(sim(a, rng.integers(lo, hi + 1), sd, rk, COST[mk]))
            nz.append(pf(np.array(rr)))
        longonly = np.array([sim(a, e, 1, rk, COST[mk]) for e, rk in zip(s.entry_k, s.risk_zone)])
        print(f'CONTROLS  strategy PF {pf(s.R):.2f} | opposite direction {pf(opp):.2f} | random direction median {np.median(rc):.2f} (95th pct {np.percentile(rc,95):.2f}, strategy beats {np.mean(np.array(rc) < pf(s.R))*100:.0f}%)'
              f' | enter at touch, no CHOCH {pf(notouch):.2f} | no zone, random London minute same side median {np.median(nz):.2f} (95th {np.percentile(nz,95):.2f}) | always long same times {pf(longonly):.2f}')
        s.to_pickle(f'{S}/{mk}_cand.pkl')
