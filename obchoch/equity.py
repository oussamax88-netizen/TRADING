import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
S, out = sys.argv[1], sys.argv[2]
A = pd.read_pickle(f'{S}/AZL_trades.pkl').sort_values('entry_t')
C = {'NQ': '#2a78d6', 'ES': '#eb6834', 'NQ + ES': '#1baf7a'}
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 7.5), gridspec_kw={'height_ratios': [3, 1.2]}, sharex=True)
fig.patch.set_facecolor('#fcfcfb')
for a_ in (ax, ax2):
    a_.set_facecolor('#fcfcfb'); a_.grid(axis='y', color='#e4e3df', lw=0.6); a_.spines[['top', 'right']].set_visible(False)
    a_.spines[['left', 'bottom']].set_color('#b5b4ae'); a_.tick_params(colors='#52514e', labelsize=9)
    a_.axvline(pd.Timestamp('2025-04-01'), color='#52514e', lw=1, ls=':')
for nm, d in (('NQ', A[A.mk == 'NQ']), ('ES', A[A.mk == 'ES']), ('NQ + ES', A)):
    eq = d.R.cumsum()
    ax.plot(d.entry_t, eq, color=C[nm], lw=2, label=nm)
    ax.annotate(f'{nm}  {eq.iat[-1]:+.0f}R', (d.entry_t.iat[-1], eq.iat[-1]), xytext=(6, 0), textcoords='offset points', va='center', fontsize=9, color='#0b0b0b')
eq = A.R.cumsum(); dd = eq - eq.cummax()
ax2.fill_between(A.entry_t, dd, 0, color='#1baf7a', alpha=0.35, lw=0); ax2.plot(A.entry_t, dd, color='#1baf7a', lw=1.5)
ax2.set_ylabel('Drawdown (R)\nNQ + ES', color='#52514e', fontsize=9)
ax.set_ylabel('Cumulative result (R, after costs)', color='#52514e', fontsize=9)
ax.text(pd.Timestamp('2025-04-08'), ax.get_ylim()[1] * 0.95, 'second half: never used to choose settings', fontsize=9, color='#52514e', va='top')
ax.legend(frameon=False, fontsize=9, loc='upper left', labelcolor='#0b0b0b')
ax.set_title('AZL strategy (Asia zone, London CHOCH): equity curve Oct 2023 – Sep 2026, New York time', loc='left', fontsize=12, color='#0b0b0b')
ax.set_xlim(A.entry_t.min(), A.entry_t.max() + pd.Timedelta(days=75))
fig.tight_layout(); fig.savefig(out, dpi=130, facecolor=fig.get_facecolor())
