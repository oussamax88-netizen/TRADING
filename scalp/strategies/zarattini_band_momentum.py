"""Time-of-day volatility-band momentum with trailing exit (Zarattini-Aziz-Barbon 2024 'Beat the Market' noise area,
Zarattini-Aziz 2023 VWAP).  Slug: zarattini_band_momentum.  Both markets, same rules.

Mechanical rules as implemented
-------------------------------
Days: only FULL RTH days (>= 380 one-minute bars between 09:30 and 15:59 NY). Holidays with partial CFD sessions and
      half days are neither traded nor used as history.  Missing minutes inside a day are forward-filled with the
      last bar at or before that minute (that bar is known at the minute's close).
L = 14: every sigma is the mean over the 14 previous full RTH days (strictly earlier dates) at the same minute of day.
REF NOISE:  O = 09:30 open, pdc = 15:59 close of the previous full RTH day.
            move_m(day) = |close(minute m)/O - 1|,  sigma_m = mean(move_m, previous 14 days)
            UB_m = max(O, pdc) * (1 + V*sigma_m),  LB_m = min(O, pdc) * (1 - V*sigma_m)   (gap-adjusted)
REF VWAP:   dev_m(day) = |close(m) - vwap(m)| / vwap(m),  sigma_m = mean(dev_m, previous 14 days)
            UB_m = vwap_m * (1 + V*sigma_m),  LB_m = vwap_m * (1 - V*sigma_m)
            (vwap = cache RTH session VWAP, tick-volume weighted, from 09:30, running value at the bar)
Decision bars: closes of the bars at 09:59, 10:29, ..., 15:29 (12 per day).
Entry: c[k] > UB -> long, c[k] < LB -> short, market at the open of k+1 (etype 0).  'Flat' means no open position at
       the close of k; the engine enforces this (a signal is skipped when its entry bar is not after the previous
       exit bar), so the opposite trade cannot be opened on the very decision bar that closes a position: it can be
       taken at the next decision bar 30 min later if the condition still holds.  Stop-outs free the slot at once.
Exit (evaluated only at later decision bars j > k of the same day, exit at the open of j+1 via maxbars = j - k):
   TRAIL: NOISE long  c[j] < max(UB_j, vwap_j), short c[j] > min(LB_j, vwap_j)
          VWAP  long  c[j] < vwap_j,             short c[j] > vwap_j
   BAND:  NOISE long  c[j] < UB_j,               short c[j] > LB_j
          VWAP  long  c[j] < vwap_j*(1 + 0.5*V*sigma_j), short c[j] > vwap_j*(1 - 0.5*V*sigma_j)
   no exit bar -> engine closes at the 15:59 open.
Catastrophic stop: stop_pts = max(0.35*ATR14, 20 pts NQ / 4 pts ES) from the fill.  No target.
Regime filter (optional): ATR14(day) / median(ATR14 of the previous 100 full RTH days) <= 1.0, no entries otherwise.
   (ATR14 = cache 'atr' = 14-day mean RTH range of previous days; the median window excludes the current day.)
Grid (24): REF {NOISE, VWAP} x V {1.0, 1.25, 1.5} x exit {TRAIL, BAND} x filter {off, on}.
Selection (IS only, fixed before looking at OOS): score = min(IS PF NQ, IS PF ES); plateau score = mean of that score
   over the variant and its V-neighbours (same REF/exit/filter); highest plateau score wins, tie -> own score.
"""
import sys, json
sys.path.insert(0, '/home/user/TRADING/scalp')
import numpy as np
import pandas as pd
from engine import load, signals_frame, simulate, full_report, headline, pf, IS_END, START, END

L = 14
DEC = np.arange(599, 930, 30)          # 09:59 ... 15:29 closes -> entries 10:00 ... 15:30
DC = DEC - 570
MINSTOP = {'NQ': 20.0, 'ES': 4.0}
OUT = '/home/user/TRADING/scalp/results/zarattini_band_momentum.json'
_P = {}


def roll_prev_mean(x, n):
    """mean of rows d-n .. d-1 (strictly earlier rows) for every row d; NaN for d < n."""
    cs = np.vstack([np.zeros((1,) + x.shape[1:]), np.cumsum(x, axis=0)])
    out = np.full(x.shape, np.nan)
    out[n:] = (cs[n:-1] - cs[:-n - 1]) / n
    return out


def prep(mk):
    if mk in _P:
        return _P[mk]
    D = load(mk)
    mod, date, c, vw = D['mod'], D['date'], D['c'], D['vwap']
    rth = (mod >= 570) & (mod < 960)
    ir = np.where(rth)[0]
    ud, inv = np.unique(date[ir], return_inverse=True)
    fdates = ud[np.bincount(inv) >= 380]
    nd = len(fdates)
    pos = np.searchsorted(fdates, date[ir])
    ok = (pos < nd) & (fdates[np.minimum(pos, nd - 1)] == date[ir])
    IDX = np.full((nd, 390), -1, dtype=np.int64)
    IDX[pos[ok], mod[ir[ok]] - 570] = ir[ok]
    assert (IDX[:, 0] >= 0).all()
    IDX = np.maximum.accumulate(IDX, axis=1)                 # forward fill: last bar at or before each minute
    C = c[IDX]; VW = vw[IDX]
    O = D['rth_open'][IDX[:, 0]]
    assert np.allclose(O, D['o'][IDX[:, 0]])
    pdc = np.r_[np.nan, C[:-1, 389]]                          # previous full RTH day's 15:59 close
    sigN = roll_prev_mean(np.abs(C / O[:, None] - 1.0), L)
    sigV = roll_prev_mean(np.abs(C - VW) / VW, L)
    atr = D['atr'][IDX[:, 0]]
    med100 = pd.Series(atr).rolling(100, min_periods=100).median().shift(1).to_numpy()
    ratio = atr / med100
    K = IDX[:, DC]
    kok = D['mod'][K] >= DEC - 5                             # decision bar must exist (<= 5 missing minutes)
    _P[mk] = dict(K=K, kok=kok, c=C[:, DC], vw=VW[:, DC], O=O, pdc=pdc, sigN=sigN[:, DC], sigV=sigV[:, DC], atr=atr,
                  ratio=ratio, dates=fdates, nd=nd)
    return _P[mk]


def bands(P, ref, V):
    c, vw = P['c'], P['vw']
    if ref == 'NOISE':
        s = P['sigN']
        UB = np.maximum(P['O'], P['pdc'])[:, None] * (1 + V * s)
        LB = np.minimum(P['O'], P['pdc'])[:, None] * (1 - V * s)
    else:
        s = P['sigV']
        UB = vw * (1 + V * s); LB = vw * (1 - V * s)
    return UB, LB, s


def candidates(mk, ref='NOISE', V=1.0, exit='TRAIL', filt=False):
    """Every decision bar whose entry condition holds -> one candidate signal with its own exit (maxbars).
    Returns a DataFrame with k, d, stop_pts, maxbars and descriptors (engine skips overlapping ones)."""
    P = prep(mk)
    c, vw, K = P['c'], P['vw'], P['K']
    UB, LB, s = bands(P, ref, V)
    if exit == 'TRAIL':
        if ref == 'NOISE':
            xl = c < np.maximum(UB, vw); xs = c > np.minimum(LB, vw)
        else:
            xl = c < vw; xs = c > vw
    else:
        if ref == 'NOISE':
            xl = c < UB; xs = c > LB
        else:
            xl = c < vw * (1 + 0.5 * V * s); xs = c > vw * (1 - 0.5 * V * s)
    rowok = np.isfinite(P['pdc']) & np.isfinite(P['atr']) & np.isfinite(s).all(axis=1)
    if filt:
        rowok &= (P['ratio'] <= 1.0)          # NaN ratio -> False
    valid = rowok[:, None] & P['kok'] & np.isfinite(UB) & np.isfinite(LB)
    ent = np.where(valid & (c > UB), 1, np.where(valid & (c < LB), -1, 0))
    rows = []
    for d, i in zip(*np.nonzero(ent)):
        side = ent[d, i]
        xx = xl[d] if side == 1 else xs[d]
        js = np.nonzero(xx[i + 1:] & P['kok'][d, i + 1:])[0]
        mb = int(K[d, i + 1 + js[0]] - K[d, i]) if len(js) else 0
        rows.append((K[d, i], side, max(0.35 * P['atr'][d], MINSTOP[mk]), mb, d, i))
    return pd.DataFrame(rows, columns=['k', 'd', 'stop', 'mb', 'row', 'slot'])


def to_sig(cand):
    return signals_frame(cand.k.to_numpy(), cand.d.to_numpy(), etype=0, stop_pts=cand.stop.to_numpy(),
                         maxbars=cand.mb.to_numpy())


VARIANTS = [dict(ref=r, V=v, exit=x, filt=f) for r in ('NOISE', 'VWAP') for v in (1.0, 1.25, 1.5)
            for x in ('TRAIL', 'BAND') for f in (False, True)]


def vname(v):
    return f"{v['ref']}_V{v['V']}_{v['exit']}_{'atrF' if v['filt'] else 'noF'}"


def is_pf(tr):
    return pf(tr[tr.entry_t < IS_END].R)


def run_grid():
    rows = []
    for v in VARIANTS:
        r = dict(name=vname(v), **v)
        for mk in ('NQ', 'ES'):
            tr = simulate(mk, to_sig(candidates(mk, **v)))
            i = tr[tr.entry_t < IS_END]
            r[f'{mk}_nIS'] = len(i); r[f'{mk}_pfIS'] = round(pf(i.R), 3)
            r[f'{mk}_wkIS'] = round(len(i) / (78.0), 2)     # IS = 2023-10-01..2025-03-31 ~ 78 weeks
        rows.append(r)
    g = pd.DataFrame(rows)
    g['score'] = g[['NQ_pfIS', 'ES_pfIS']].min(axis=1)
    plateau = []
    for _, r in g.iterrows():
        nb = g[(g.ref == r.ref) & (g.exit == r.exit) & (g.filt == r.filt) & ((g.V - r.V).abs() <= 0.26)]
        plateau.append(nb.score.mean())
    g['plateau'] = np.round(plateau, 3)
    return g


def oos_pf(mk, v):
    tr = simulate(mk, to_sig(candidates(mk, **v)))
    return round(pf(tr[tr.entry_t >= IS_END].R), 3)


def partial_edges(mk, v):
    """IS/OOS PF of sub-conditions of the chosen variant (diagnostic only, never used for selection)."""
    D = load(mk)
    cand = candidates(mk, **v)
    tr = simulate(mk, to_sig(cand))
    m = cand.set_index('k')
    tr['slot'] = m.loc[tr.k, 'slot'].to_numpy(); tr['row'] = m.loc[tr.k, 'row'].to_numpy()
    P = prep(mk)
    tr['ratio'] = P['ratio'][tr.row]
    tr['trend'] = D['trend_d'][tr.k]
    tr['gap'] = np.sign(P['O'][tr.row] - P['pdc'][tr.row])
    tr['first'] = ~tr.duplicated('row')
    tr['IS'] = tr.entry_t < IS_END
    subs = {
        'long': tr.d == 1, 'short': tr.d == -1,
        '10:00-11:30': tr.slot <= 2, '11:30-13:30': (tr.slot >= 3) & (tr.slot <= 6), '13:30-15:30': tr.slot >= 7,
        'first_of_day': tr['first'], 'reentry': ~tr['first'],
        'with_20d_trend': tr.d == tr.trend, 'against_20d_trend': tr.d == -tr.trend,
        'with_gap': tr.d == tr.gap, 'against_gap': tr.d == -tr.gap,
        'atr_ratio<0.9': tr.ratio < 0.9, 'atr_ratio>=0.9': tr.ratio >= 0.9,
        'exit_15:59': tr.reason == 3, 'exit_trail': tr.reason == 4, 'exit_stop': tr.reason == 1,
    }
    out = {}
    for name, msk in subs.items():
        a = tr[msk & tr.IS]; b = tr[msk & ~tr.IS]
        out[name] = dict(n_IS=len(a), pf_IS=round(pf(a.R), 2), n_OOS=len(b), pf_OOS=round(pf(b.R), 2))
    return out


def by_year_str(rep):
    return ' '.join(f"{y}:{v['pf']}({v['n']})" for y, v in rep['by_year'].items())


if __name__ == '__main__':
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    g = run_grid()
    print('=== GRID (IN-SAMPLE ONLY, 24 variants) ===')
    print(g.drop(columns=['ref', 'V', 'exit', 'filt']).to_string(index=False))
    med_is = {mk: round(float(g[f'{mk}_pfIS'].median()), 3) for mk in ('NQ', 'ES')}
    print('median IS PF of all 24 variants:', med_is)
    best = g.sort_values(['plateau', 'score'], ascending=False).iloc[0]
    chosen = dict(ref=best.ref, V=float(best.V), exit=best.exit, filt=bool(best.filt))
    print('CHOSEN on IS:', vname(chosen), 'plateau', best.plateau, 'score', best.score)

    res = dict(slug='zarattini_band_momentum', chosen=vname(chosen), variants_tested=len(VARIANTS), median_IS_pf=med_is,
               grid_IS=g.to_dict('records'))
    for mk in ('NQ', 'ES'):
        rep = full_report(mk, to_sig(candidates(mk, **chosen)))
        rep['by_year_str'] = by_year_str(rep)
        res[mk] = rep
        print(f'\n=== {mk} full_report ({vname(chosen)}) ===')
        print(json.dumps(rep, default=str))
    # sanity: every signal the engine took must be a candidate, and exits must happen at the planned bar
    for mk in ('NQ', 'ES'):
        cand = candidates(mk, **chosen); tr = simulate(mk, to_sig(cand), period=False)
        m = cand.set_index('k')
        planned = np.where(m.loc[tr.k, 'mb'].to_numpy() > 0, tr.k.to_numpy() + m.loc[tr.k, 'mb'].to_numpy() + 1, -1)
        tr4 = tr.reason == 4
        assert (tr.exit_k[tr4].to_numpy() == planned[tr4.to_numpy()]).all()
        print(mk, 'sanity ok: candidates', len(cand), 'taken', len(tr), 'reasons', tr.reason.value_counts().to_dict())

    # informational only (after selection): OOS of every variant
    g['NQ_pfOOS'] = [oos_pf('NQ', v) for v in VARIANTS]
    g['ES_pfOOS'] = [oos_pf('ES', v) for v in VARIANTS]
    print('\n=== GRID with OOS (informational, NOT used for selection) ===')
    print(g[['name', 'NQ_pfIS', 'NQ_pfOOS', 'ES_pfIS', 'ES_pfOOS', 'NQ_wkIS', 'ES_wkIS']].to_string(index=False))
    res['grid_OOS_info'] = g[['name', 'NQ_pfIS', 'NQ_pfOOS', 'ES_pfIS', 'ES_pfOOS']].to_dict('records')

    res['partial_edges'] = {}
    for mk in ('NQ', 'ES'):
        pe = partial_edges(mk, chosen); res['partial_edges'][mk] = pe
        print(f'\n=== {mk} partial edges (chosen variant) ===')
        for k_, v_ in pe.items():
            print(f'  {k_:20s} IS {v_["pf_IS"]}({v_["n_IS"]})  OOS {v_["pf_OOS"]}({v_["n_OOS"]})')
    # verdict (task rules)
    def ok13(r): return r['pf_IS'] >= 1.3 and r['pf_OOS'] >= 1.3
    passed = any(ok13(res[a]) and res[b]['pf_OOS'] >= 1.1 and res[a]['per_week'] >= 3 and res[a]['pf_2x_cost'] >= 1.1
                 for a, b in (('NQ', 'ES'), ('ES', 'NQ')))
    border = any(res[m]['pf_IS'] >= 1.15 and res[m]['pf_OOS'] >= 1.15 for m in ('NQ', 'ES'))
    res['verdict'] = 'PASS' if passed else ('BORDERLINE' if border else 'FAIL')
    print('VERDICT:', res['verdict'])
    json.dump(res, open(OUT, 'w'), default=str, indent=1)
    print('saved', OUT)
