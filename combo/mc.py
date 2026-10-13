import sys; sys.path.insert(0, '/home/user/TRADING/scalp')
import numpy as np, pandas as pd
from engine import pf
S = '/tmp/claude-0/-home-user-TRADING/b9cb08f9-c0e5-5011-839c-6894039a978b/scratchpad'
T = pd.read_pickle(f'{S}/combo_A.pkl')
R = T.R.to_numpy().copy(); w = R > 0
target = 1.45; R[w] *= target / pf(T.R)          # live haircut: PF 1.74 -> 1.45
T = T.assign(Rh=R)
days = [g.Rh.to_numpy() for _, g in T.groupby('date')]
ndays = len(days); per_year = ndays / 3.0
print(f'backtest PF {pf(T.R):.2f} -> haircut PF {pf(T.Rh):.2f}; trade days {ndays} ({per_year:.0f}/yr), trades/yr {len(T)/3:.0f}')
rng = np.random.default_rng(7); N = 10000
def run(risk, brake, horizon_days):
    idx = rng.integers(0, ndays, (N, horizon_days))
    eq = np.ones(N); pk = np.ones(N); mdd = np.zeros(N)
    for t in range(horizon_days):
        r = np.where(brake & (eq < pk * 0.75), risk / 2, risk)
        dayR = np.array([days[i].sum() for i in idx[:, t]])  # trades on the same day share the starting equity
        eq = np.maximum(eq * (1 + r * dayR), 0); pk = np.maximum(pk, eq); mdd = np.maximum(mdd, 1 - eq / pk)
    return eq, mdd
H = {'1 month': round(per_year / 12), '3 months': round(per_year / 4), '6 months': round(per_year / 2), '1 year': round(per_year)}
for risk in (0.01, 0.02, 0.03, 0.05):
    for brake in (False, True):
        if risk < 0.03 and brake: continue
        line = f'{risk*100:.0f}% {"brake" if brake else "     "} |'
        for nm, hd in H.items():
            eq, mdd = run(risk, brake, hd)
            line += f' {nm}: P(profit) {np.mean(eq>1)*100:3.0f}% med {np.median(eq)*100-100:+5.0f}% |'
        eq, mdd = run(risk, brake, H['1 year'])
        line += f' 1y worst10% {np.percentile(eq,10)*100-100:+4.0f}% best10% {np.percentile(eq,90)*100-100:+5.0f}% | DD>30% {np.mean(mdd>.3)*100:3.0f}% DD>50% {np.mean(mdd>.5)*100:3.0f}% DD>70% {np.mean(mdd>.7)*100:3.0f}% | typical maxDD {np.median(mdd)*100:.0f}%'
        print(line)
fs = np.arange(0.005, 0.2, 0.0025)
g = [np.mean(np.log1p(f * np.array([d.sum() for d in days]))) for f in fs]
print('Kelly per trade (haircut, day-level):', round(fs[np.argmax(g)] * 100, 1), '%')
