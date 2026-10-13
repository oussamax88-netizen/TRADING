import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
S = '/tmp/claude-0/-home-user-TRADING/b9cb08f9-c0e5-5011-839c-6894039a978b/scratchpad'
T = pd.read_pickle(f'{S}/combo_A.pkl').sort_values('entry_t'); T['key'] = T.sys + ' ' + T.mk
C = {'TDH NQ': '#2a78d6', 'AZL NQ': '#eb6834', 'AZL ES': '#1baf7a', 'Combined': '#eda100'}
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 7.5), gridspec_kw={'height_ratios': [3, 1.2]}, sharex=True); fig.patch.set_facecolor('#fcfcfb')
for a in (ax, ax2):
    a.set_facecolor('#fcfcfb'); a.grid(axis='y', color='#e4e3df', lw=.6); a.spines[['top', 'right']].set_visible(False)
    a.spines[['left', 'bottom']].set_color('#b5b4ae'); a.tick_params(colors='#52514e', labelsize=9); a.axvline(pd.Timestamp('2025-04-01'), color='#52514e', lw=1, ls=':')
for nm in ('TDH NQ', 'AZL NQ', 'AZL ES', 'Combined'):
    d = T if nm == 'Combined' else T[T.key == nm]; eq = d.R.cumsum()
    ax.plot(d.entry_t, eq, color=C[nm], lw=2.6 if nm == 'Combined' else 1.6, label=nm)
    ax.annotate(f'{nm}  {eq.iat[-1]:+.0f}R', (d.entry_t.iat[-1], eq.iat[-1]), xytext=(6, 0), textcoords='offset points', va='center', fontsize=9, color='#0b0b0b')
eq = T.R.cumsum(); dd = eq - eq.cummax()
ax2.fill_between(T.entry_t, dd, 0, color='#eda100', alpha=.35, lw=0); ax2.plot(T.entry_t, dd, color='#c98500', lw=1.4)
ax2.set_ylabel('Drawdown (R)\ncombined', color='#52514e', fontsize=9); ax.set_ylabel('Cumulative result (R, after costs)', color='#52514e', fontsize=9)
ax.text(pd.Timestamp('2025-04-08'), ax.get_ylim()[1] * .95, 'second half (unseen when rules were set)', fontsize=9, color='#52514e', va='top')
ax.legend(frameon=False, fontsize=9, loc='upper left'); ax.set_xlim(T.entry_t.min(), T.entry_t.max() + pd.Timedelta(days=90))
ax.set_title('TDH (NQ) + AZL (NQ, ES): equity curve Oct 2023 – Sep 2026, after costs', loc='left', fontsize=12, color='#0b0b0b')
fig.tight_layout(); fig.savefig('reports/COMBO_equity.png', dpi=130, facecolor=fig.get_facecolor())
