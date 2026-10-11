"""Sample-trade charts: left = zone timeframe (30m/1h) with the order block, right = 1m execution. NY time."""
import sys, pickle, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
sys.path.insert(0, 'obchoch')

UP, DN = '#2e9d6a', '#d0453a'

def candles(ax, x, o, h, l, c, w=0.6):
    for xi, oi, hi, li, ci in zip(x, o, h, l, c):
        col = UP if ci >= oi else DN
        ax.plot([xi, xi], [li, hi], color=col, lw=0.7)
        ax.add_patch(Rectangle((xi - w/2, min(oi, ci)), w, max(abs(ci - oi), 1e-9), color=col))

def frame(a, i0, i1):
    t = pd.to_datetime(a['tmin'][i0:i1] * 60_000_000_000)
    return pd.DataFrame({'t': t, 'open': a['o'][i0:i1], 'high': a['h'][i0:i1], 'low': a['l'][i0:i1], 'close': a['c'][i0:i1]})

def plot_trade(pdf, a, tr, mk, title_extra=''):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [1, 1.4]})
    tfm = 60 if tr.tf == '1h' else 30
    ob = pd.Timestamp(tr.ob_t); created = pd.Timestamp(tr.created)
    ent_t = pd.Timestamp(a['tmin'][tr.entry_k] * 60_000_000_000)
    # ---- HTF panel
    i0 = np.searchsorted(a['tmin'], (ob - pd.Timedelta(minutes=tfm*25)).value // 60_000_000_000)
    i1 = tr.exit_k + 1
    d = frame(a, i0, i1)
    b = d.set_index('t').resample(f'{tfm}min').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna().reset_index()
    x = np.arange(len(b)); candles(ax1, x, b.open, b.high, b.low, b.close)
    xo = b.index[b.t == ob]; xo = xo[0] if len(xo) else 0
    ax1.add_patch(Rectangle((xo - 0.5, tr.zlo), len(b) - xo, tr.zhi - tr.zlo, color='#4a7bd0' if tr.side == 1 else '#d08a2e', alpha=0.25))
    xc = b.index[b.t == created - pd.Timedelta(minutes=tfm)]
    if len(xc): ax1.annotate('BOS (zone created)', (xc[0], b.close[xc[0]]), xytext=(0, 25), textcoords='offset points', ha='center', fontsize=8, arrowprops=dict(arrowstyle='->'))
    ax1.annotate('OB candle', (xo, b.low[xo] if tr.side == 1 else b.high[xo]), xytext=(0, -25 if tr.side == 1 else 25), textcoords='offset points', ha='center', fontsize=8, arrowprops=dict(arrowstyle='->'))
    xe = b.index[b.t <= ent_t].max(); ax1.axvline(xe, color='k', ls=':', lw=0.8)
    step = max(1, len(b)//8); ax1.set_xticks(x[::step]); ax1.set_xticklabels(b.t.dt.strftime('%m-%d %H:%M')[::step], rotation=45, fontsize=7)
    ax1.set_title(f'{tr.tf} chart: {"DEMAND" if tr.side==1 else "SUPPLY"} zone {tr.zlo:.1f}-{tr.zhi:.1f}', fontsize=10)
    # ---- 1m panel
    j0 = max(tr.touch_k - 60, 0); j1 = min(tr.exit_k + 15, len(a['o']))
    m = frame(a, j0, j1); x = np.arange(len(m))
    candles(ax2, x, m.open, m.high, m.low, m.close, w=0.7)
    ax2.axhspan(tr.zlo, tr.zhi, color='#4a7bd0' if tr.side == 1 else '#d08a2e', alpha=0.18, label='zone')
    xe, xx, xt, xch = tr.entry_k - j0, tr.exit_k - j0, tr.touch_k - j0, tr.choch_k - j0
    ax2.hlines(tr.entry, xe, xx, color='k', lw=1, label=f'entry {tr.entry:.1f}')
    ax2.hlines(tr.stop, xe, xx, color=DN, lw=1.2, ls='--', label=f'stop {tr.stop:.1f}')
    if not np.isnan(tr.target):
        ax2.hlines(tr.target, xe, xx, color=UP, lw=1.2, ls='--', label=f'target {tr.target:.1f} ({tr.rr:.0f}R)')
    ax2.annotate('touch', (xt, m.low[xt] if tr.side == 1 else m.high[xt]), xytext=(0, -22 if tr.side == 1 else 22), textcoords='offset points', ha='center', fontsize=8, arrowprops=dict(arrowstyle='->'))
    ax2.annotate('CHOCH close', (xch, m.close[xch]), xytext=(-30, 28 if tr.side == 1 else -28), textcoords='offset points', fontsize=8, arrowprops=dict(arrowstyle='->', color='purple'), color='purple')
    ax2.scatter([xe], [tr.entry], marker='^' if tr.side == 1 else 'v', s=90, color='k', zorder=5)
    ax2.scatter([xx], [tr.exit], marker='X', s=90, color='purple', zorder=5)
    step = max(1, len(m)//10); ax2.set_xticks(x[::step]); ax2.set_xticklabels(m.t.dt.strftime('%H:%M')[::step], rotation=45, fontsize=7)
    why = {1: 'STOP', 2: 'TARGET', 3: 'TIME 15:59'}[tr.reason]
    ax2.set_title(f'1m execution  {ent_t:%a %Y-%m-%d}  {"LONG" if tr.side==1 else "SHORT"}  exit {why}  result {tr.R:+.2f}R', fontsize=10)
    ax2.legend(fontsize=8, loc='best')
    fig.suptitle(f'{mk}  trade #{tr.name}  (New York time){title_extra}', fontsize=12)
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

if __name__ == '__main__':
    S, mk, tfile, out, nsamp = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
    a, z = pickle.load(open(f'{S}/{mk}.pkl', 'rb'))
    t = pd.read_pickle(tfile)
    samp = t.sample(nsamp, random_state=7).sort_values('entry_k')
    with PdfPages(out) as pdf:
        for _, tr in samp.iterrows():
            plot_trade(pdf, a, tr, mk)
    print(samp[['entry_t','side','tf','R','reason']].to_string())
