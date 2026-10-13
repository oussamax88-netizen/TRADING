import sys, json
sys.path.insert(0, '/home/user/TRADING/scalp'); sys.path.insert(0, '/home/user/TRADING/combo')
import numpy as np, pandas as pd
from engine import load, simulate, pf, IS_END
from tdh import signals
S = '/tmp/claude-0/-home-user-TRADING/b9cb08f9-c0e5-5011-839c-6894039a978b/scratchpad'
A = pd.read_pickle(f'{S}/AZL_trades.pkl')
rows = []
for mk in ('NQ', 'ES'):
    D = load(mk); a = A[A.mk == mk]
    assert (pd.to_datetime(D['t'][a.entry_k.to_numpy()] * 60_000_000_000) == a.entry_t.to_numpy()).all()
    rows.append(pd.DataFrame({'sys': 'AZL', 'mk': mk, 'entry_k': a.entry_k, 'exit_k': a.x_atr02_EOD.astype(int), 'entry_t': a.entry_t, 'd': a.side, 'R': a.R}))
t = simulate('NQ', signals('NQ'))
rows.append(pd.DataFrame({'sys': 'TDH', 'mk': 'NQ', 'entry_k': t.entry_k, 'exit_k': t.exit_k, 'entry_t': t.entry_t, 'd': t.d, 'R': t.R}))
T = pd.concat(rows).sort_values('entry_t').reset_index(drop=True)
T['date'] = T.entry_t.dt.date
# overlap on NQ: TDH entry while AZL NQ position still open
nq = T[T.mk == 'NQ']
azl = nq[nq.sys == 'AZL']; tdh = nq[nq.sys == 'TDH']
ov = [any((r.entry_k > x.entry_k) & (r.entry_k <= x.exit_k) for x in azl[azl.date == r.date].itertuples()) for r in tdh.itertuples()]
tdh = tdh.assign(overlap=ov)
same = [ (azl[(azl.date == r.date)].d == r.d).all() for r in tdh[tdh.overlap].itertuples()]
print('TDH NQ trades', len(tdh), '| overlap with open AZL NQ position', int(sum(ov)), '| same direction', int(sum(same)))
print('  TDH PF when overlapping', round(pf(tdh[tdh.overlap].R), 2), 'vs not', round(pf(tdh[~tdh.overlap].R), 2))
def full(x, label):
    x = x.sort_values('entry_t'); R = x.R.to_numpy(); eq = R.cumsum(); dd = (eq - np.maximum.accumulate(eq)).min()
    los = (R <= 0).astype(int); s = c = 0
    for v in los: c = c + 1 if v else 0; s = max(s, c)
    m = x.groupby(x.entry_t.dt.to_period('M')).R.sum(); i = x[x.entry_t < IS_END]; o = x[x.entry_t >= IS_END]
    yr = {int(y): f"{pf(g.R):.2f} ({len(g)})" for y, g in x.groupby(x.entry_t.dt.year)}
    print(f"{label:34} n {len(x):4} | {len(x)/156.4:.2f}/wk | win {np.mean(R>0)*100:.0f}% | PF {pf(R):.2f} | IS {pf(i.R):.2f} OOS {pf(o.R):.2f} | net {R.sum():+.0f}R | maxDD {dd:.1f}R | streak {s} | losing months {np.mean(m<0)*100:.0f}% | {yr}")
    return x
full(T[(T.sys == 'TDH')], 'TDH NQ alone')
full(T[(T.sys == 'AZL') & (T.mk == 'NQ')], 'AZL NQ alone')
full(T[(T.sys == 'AZL') & (T.mk == 'ES')], 'AZL ES alone')
Tb = pd.concat([T[T.sys == 'AZL'], tdh.drop(columns='overlap')])
full(Tb, 'A) ALL (separate positions)')
Ta = pd.concat([T[T.sys == 'AZL'], tdh[~tdh.overlap].drop(columns='overlap')])
full(Ta, 'B) ALL, skip TDH if AZL NQ open')
full(pd.concat([T[(T.sys == 'AZL') & (T.mk == 'NQ')], tdh.drop(columns='overlap')]), 'C) NQ only: TDH + AZL')
# daily correlation
dd_ = Tb.pivot_table(index='date', columns=[Tb.sys + '_' + Tb.mk], values='R', aggfunc='sum').fillna(0)
print('daily R correlation (days with any trade):'); print(dd_.corr().round(2).to_string())
Tb.to_pickle(f'{S}/combo_A.pkl'); Ta.to_pickle(f'{S}/combo_B.pkl')
