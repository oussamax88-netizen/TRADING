"""2027 scenarios for TDH(NQ)+AZL(NQ,ES): 10,000 simulated years from the real daily trade results (with no-trade days),
live haircut PF 1.45, risk 2%/trade with drawdown brake (-25% from peak -> half risk). Also an 'edge disappears' case."""
import sys; sys.path.insert(0, '/home/user/TRADING/scalp')
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from engine import pf, load
S = '/tmp/claude-0/-home-user-TRADING/b9cb08f9-c0e5-5011-839c-6894039a978b/scratchpad'
T = pd.read_pickle(f'{S}/combo_A.pkl')
D = load('NQ'); rth_days = pd.Series(pd.to_datetime(np.unique(D['date'][(D['mod'] == 600)]) * 86400 * 10**9)).dt.date
rth_days = rth_days[(rth_days >= pd.Timestamp('2023-10-01').date()) & (rth_days < pd.Timestamp('2026-10-01').date())]
def pool(target):
    R = T.R.to_numpy().copy(); w = R > 0; R[w] *= target / pf(T.R)
    s = pd.Series(R, index=T.index).groupby(T.date).sum()
    return s.reindex(rth_days, fill_value=0.0).to_numpy()
days27 = pd.bdate_range('2027-01-01', '2027-12-31'); H = len(days27)
rng = np.random.default_rng(2027); N = 10000; START = 1000.0
def sim(p, risk, brake=True):
    idx = rng.integers(0, len(p), (N, H)); eq = np.full(N, START); pk = eq.copy(); path = np.empty((N, H)); mdd = np.zeros(N)
    for t in range(H):
        r = np.where(brake & (eq < pk * .75), risk / 2, risk)
        eq = np.maximum(eq * (1 + r * p[idx[:, t]]), 0); pk = np.maximum(pk, eq); mdd = np.maximum(mdd, 1 - eq / pk); path[:, t] = eq
    return path, mdd
out = {}
for nm, target, risk in (('live', 1.45, 0.05), ('nobrake', 1.45, 0.05), ('dead', 0.95, 0.05)):
    path, mdd = sim(pool(target), risk, brake=(nm != 'nobrake')); out[nm] = (path, mdd)
    fin = path[:, -1]; q = np.percentile(fin, [1, 5, 10, 50, 90, 95, 99])
    print(f"{nm:6} risk {risk*100:.0f}% | 1% {q[0]:,.0f} | 5% {q[1]:,.0f} | 10% {q[2]:,.0f} | median {q[3]:,.0f} | 90% {q[4]:,.0f} | 95% {q[5]:,.0f} | 99% {q[6]:,.0f} | P(profit) {np.mean(fin>START)*100:.0f}% | median maxDD {np.median(mdd)*100:.0f}% | 90th pct maxDD {np.percentile(mdd,90)*100:.0f}% | P(DD>50%) {np.mean(mdd>.5)*100:.0f}%")
# pick real simulated paths nearest to each percentile (shape is realistic, not a smoothed line)
path, _ = out['live']; fin = path[:, -1]
pick = {}
for lab, q in (('Best case (top 10%)', 90), ('Typical (median)', 50), ('Bad year (worst 10%)', 10), ('Very bad year (worst 1%)', 1)):
    target_v = np.percentile(fin, q); pick[lab] = path[np.argmin(np.abs(fin - target_v))]
dpath, _ = out['dead']; dfin = dpath[:, -1]; pick['Edge disappears (PF 0.95), median'] = dpath[np.argmin(np.abs(dfin - np.median(dfin)))]
C = {'Best case (top 10%)': '#2a78d6', 'Typical (median)': '#1baf7a', 'Bad year (worst 10%)': '#eda100', 'Very bad year (worst 1%)': '#eb6834', 'Edge disappears (PF 0.95), median': '#6250d6'}
fig, ax = plt.subplots(figsize=(12, 6.5)); fig.patch.set_facecolor('#fcfcfb'); ax.set_facecolor('#fcfcfb')
lo, hi = np.percentile(path, 10, axis=0), np.percentile(path, 90, axis=0)
ax.fill_between(days27, lo, hi, color='#2a78d6', alpha=.08, lw=0, label='80% of outcomes (10th–90th pct)')
for lab, p in pick.items():
    ax.plot(days27, p, color=C[lab], lw=2.2 if 'Typical' in lab else 1.6, ls='--' if 'Edge' in lab else '-', label=lab)
    ax.annotate(f'${p[-1]:,.0f}', (days27[-1], p[-1]), xytext=(6, 0), textcoords='offset points', va='center', fontsize=9, color='#0b0b0b')
ax.axhline(START, color='#52514e', lw=1, ls=':')
ax.grid(axis='y', color='#e4e3df', lw=.6); ax.spines[['top', 'right']].set_visible(False); ax.spines[['left', 'bottom']].set_color('#b5b4ae'); ax.tick_params(colors='#52514e', labelsize=9)
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'${v:,.0f}'))
ax.set_xlim(days27[0], days27[-1] + pd.Timedelta(days=40))
ax.set_title('2027 scenarios: TDH + AZL, $1,000 start, 5% risk per trade, drawdown brake (simulated, not a promise)', loc='left', fontsize=12, color='#0b0b0b')
ax.legend(frameon=False, fontsize=9, loc='upper left'); fig.tight_layout(); fig.savefig('reports/SCENARIOS_2027_1000_5pct.png', dpi=130, facecolor=fig.get_facecolor())
