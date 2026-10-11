"""London-open drive fade (03:00 NY 'Judas swing') held into the US pre-market.  Both markets, same rules.

Mechanical rules (as implemented):
  * Day = NY calendar date (engine dayi). Need a bar at exactly 03:00 (mod 180), a window-end bar and an exit bar.
  * Window start bar s = the 03:00 bar; window end bar e = last bar of the day with 180 < mod <= W
    (W = 194 / 209 / 239, i.e. the 03:14 / 03:29 / 03:59 bar; exact bar present on all but 1-3 days).
  * Drive r = (close[e] - open[s]) / ATR, ATR = 14-day mean of previous RTH ranges (cache 'atr', known before the day).
  * If |r| >= thr: signal at k = e (decided on the close of e), direction d = -sign(r) (FADE), market entry at open of k+1.
  * Stop = max(sl * ATR, floor) points from the fill, floor = 20 pts NQ / 4 pts ES. No target.
  * Time exit at the OPEN of the first bar with mod >= X (X = 510 -> 08:30 open, i.e. flat before the 08:30 data;
    X = 570 -> 09:30 open): maxbars = idx(X) - (k+1). Engine: stop first, gap through stop exits at the open.
  * One trade per day per market (only one signal per day is generated). Engine costs NQ 1.5 / ES 0.6 pts.
Lookahead audit: s, e, k are all <= the signal bar; ATR/trend_d are previous-day values; the overnight (Asia) range
used in diagnostics is the running on_high/on_low at bar s-1 (02:59). The exit index is a clock time, not data.

Grid (24 variants): W {03:14, 03:29, 03:59} x thr {0.05, 0.15} ATR x X {08:30, 09:30} x sl {0.20, 0.35} ATR.
Selection (IS only): plateau score = mean over NQ+ES of IS PF, averaged over the variant and its one-step grid
neighbours; the highest plateau score wins.
"""
import sys, json, itertools
sys.path.insert(0, '/home/user/TRADING/scalp')
import numpy as np
import pandas as pd
from engine import load, signals_frame, simulate, full_report, headline, pf, IS_END

S_PAD = '/tmp/claude-0/-home-user-TRADING/b9cb08f9-c0e5-5011-839c-6894039a978b/scratchpad'
FLOOR = {'NQ': 20.0, 'ES': 4.0}
WINDOWS = [194, 209, 239]
THRS = [0.05, 0.15]
EXITS = [510, 570]
SLS = [0.20, 0.35]
MKS = ('NQ', 'ES')
_DAYS = {}


def days(mk, start=180):
    """Per-day bar indices: window start (exact `start` minute), first bars at/after each clock time."""
    key = (mk, start)
    if key in _DAYS:
        return _DAYS[key]
    D = load(mk)
    mod, dayi = D['mod'], D['dayi']
    starts = np.r_[0, np.where(np.diff(dayi) != 0)[0] + 1]; ends = np.r_[starts[1:], len(dayi)]
    rows = []
    for a, b in zip(starts, ends):
        m = mod[a:b]
        js = np.where(m == start)[0]
        if len(js) == 0:
            continue
        s = a + js[0]
        row = dict(day=dayi[s], s=s, a=a, b=b)
        for off in (14, 29, 59):
            je = np.where((m > start) & (m <= start + off))[0]
            row[f'e{off}'] = a + je[-1] if len(je) else -1
        for X in (510, 570):
            jx = np.where(m >= X)[0]
            row[f'x{X}'] = a + jx[0] if len(jx) else -1
        rows.append(row)
    _DAYS[key] = pd.DataFrame(rows)
    return _DAYS[key]


def signals(mk, W=209, thr=0.05, X=570, sl=0.20, fade=True, start=180, meta=False):
    """W = window-end minute (absolute NY minute), X = exit minute. start != 180 only for timing controls."""
    D = load(mk)
    o, c, h, l, atr = D['o'], D['c'], D['h'], D['l'], D['atr']
    T = days(mk, start)
    off = W - start
    ks, ds, stops, mbs, info = [], [], [], [], []
    for r in T.itertuples():
        s = r.s; e = getattr(r, f'e{off}'); x = getattr(r, f'x{X}') if start == 180 else -1
        if start != 180:                      # timing control: X is already shifted by the caller
            j = r.a + int(np.searchsorted(D['mod'][r.a:r.b], X))
            x = j if j < r.b else -1
        A = atr[s]
        if e < 0 or x < 0 or not np.isfinite(A) or A <= 0:
            continue
        rr = (c[e] - o[s]) / A
        if abs(rr) < thr or x - (e + 1) < 1:
            continue
        d = -int(np.sign(rr)) if fade else int(np.sign(rr))
        ks.append(e); ds.append(d); stops.append(max(sl * A, FLOOR[mk])); mbs.append(x - (e + 1))
        if meta:
            onh, onl = D['on_high'][s - 1], D['on_low'][s - 1]           # overnight range up to 02:59 (running)
            wh, wl = h[s:e + 1].max(), l[s:e + 1].min()
            info.append(dict(k=e, r=rr, absr=abs(rr), drive=int(np.sign(rr)), trend=D['trend_d'][s], dow=D['dow'][s],
                             sweep=int((rr > 0 and wh > onh) or (rr < 0 and wl < onl)),
                             on_rng_atr=(onh - onl) / A, atr=A))
    sig = signals_frame(ks, ds, stop_pts=np.array(stops, dtype=float), maxbars=np.array(mbs, dtype=np.int64))
    if meta:
        return sig, pd.DataFrame(info)
    return sig


def vname(W, thr, X, sl):
    return f'W{W // 60:02d}:{W % 60:02d}_thr{thr}_X{X // 60:02d}:{X % 60:02d}_sl{sl}'


def grid():
    rows = []
    for W, thr, X, sl in itertools.product(WINDOWS, THRS, EXITS, SLS):
        row = dict(W=W, thr=thr, X=X, sl=sl, name=vname(W, thr, X, sl))
        for mk in MKS:
            hd = headline(simulate(mk, signals(mk, W, thr, X, sl)))
            row[f'{mk}_IS'] = hd['pf_IS']; row[f'{mk}_OOS'] = hd['pf_OOS']; row[f'{mk}_n'] = hd['n']; row[f'{mk}_pw'] = hd['per_week']
        rows.append(row)
    G = pd.DataFrame(rows)
    G['IS_mean'] = (G.NQ_IS + G.ES_IS) / 2
    lv = {'W': WINDOWS, 'thr': THRS, 'X': EXITS, 'sl': SLS}
    plat = []
    for i, r in G.iterrows():
        nb = [G.IS_mean[i]]
        for p, vals in lv.items():
            j = vals.index(r[p])
            for jj in (j - 1, j + 1):
                if 0 <= jj < len(vals):
                    q = dict(W=r.W, thr=r.thr, X=r.X, sl=r.sl); q[p] = vals[jj]
                    m = (G.W == q['W']) & (G.thr == q['thr']) & (G.X == q['X']) & (G.sl == q['sl'])
                    nb.append(float(G.IS_mean[m].iloc[0]))
        plat.append(np.mean(nb))
    G['plateau'] = plat
    return G


def bystr(rep):
    return ' '.join(f"{y}:{v['pf']}({v['n']})" for y, v in rep['by_year'].items())


def sub_pf(t, mask):
    a = t[mask & (t.entry_t < IS_END)]; b = t[mask & (t.entry_t >= IS_END)]
    return dict(n_IS=len(a), pf_IS=round(pf(a.R), 2) if len(a) else None, n_OOS=len(b), pf_OOS=round(pf(b.R), 2) if len(b) else None)


def partials(mk, W, thr, X, sl):
    sig, meta = signals(mk, W, thr, X, sl, meta=True)
    t = simulate(mk, sig).merge(meta, on='k', how='left')
    t['with_trend'] = (t.d == t.trend)
    q = t.absr.quantile([1 / 3, 2 / 3]).to_numpy()            # tercile edges (descriptive only)
    out = {
        'long (fade down-drive)': sub_pf(t, t.d == 1), 'short (fade up-drive)': sub_pf(t, t.d == -1),
        'fade WITH 20d trend': sub_pf(t, t.with_trend), 'fade AGAINST 20d trend': sub_pf(t, ~t.with_trend),
        'drive swept overnight hi/lo (Judas)': sub_pf(t, t.sweep == 1), 'drive inside overnight range': sub_pf(t, t.sweep == 0),
        f'|r| low tercile (<{q[0]:.2f})': sub_pf(t, t.absr < q[0]), '|r| mid tercile': sub_pf(t, (t.absr >= q[0]) & (t.absr < q[1])),
        f'|r| high tercile (>={q[1]:.2f})': sub_pf(t, t.absr >= q[1]),
        'Mon-Tue': sub_pf(t, t.dow <= 1), 'Wed': sub_pf(t, t.dow == 2), 'Thu-Fri': sub_pf(t, t.dow >= 3),
        'exit reason stop': dict(share=round(float((t.reason == 1).mean()), 2)),
    }
    return out, t


def azl_overlap(trades):
    """Overlap/correlation of the chosen variant with the known AZL edge (trades saved by obchoch/final.py)."""
    try:
        A = pd.read_pickle(f'{S_PAD}/AZL_trades.pkl')
    except Exception as ex:                                     # pragma: no cover
        return {'error': str(ex)}
    out = {}
    for mk in MKS:
        t = trades[mk].copy(); t['date'] = t.entry_t.dt.date
        a = A[A.mk == mk].copy(); a['date'] = a.entry_t.dt.date
        ad = a.groupby('date').agg(azl_R=('R', 'sum'), azl_side=('side', 'first'))
        m = t.merge(ad, left_on='date', right_index=True, how='inner')
        lof_daily = t.groupby('date').R.sum()
        out[mk] = dict(lof_days=int(t.date.nunique()), azl_days=int(len(ad)), common_days=int(len(m)),
                       share_of_azl_days_with_lof=round(len(m) / max(len(ad), 1), 2),
                       same_direction_share=round(float((m.d == m.azl_side).mean()), 2) if len(m) else None,
                       corr_daily_R_common=round(float(np.corrcoef(m.R, m.azl_R)[0, 1]), 2) if len(m) > 5 else None,
                       lof_pf_on_azl_days=round(pf(m.R), 2) if len(m) else None,
                       lof_pf_off_azl_days=round(pf(t[~t.date.isin(ad.index)].R), 2),
                       lof_pf_same_dir_as_azl=round(pf(m[m.d == m.azl_side].R), 2) if (m.d == m.azl_side).any() else None,
                       lof_pf_opp_dir_to_azl=round(pf(m[m.d != m.azl_side].R), 2) if (m.d != m.azl_side).any() else None)
        # all-day daily R correlation (0 when a strategy does not trade that day)
        allday = pd.concat([lof_daily.rename('lof'), ad.azl_R.rename('azl')], axis=1).fillna(0)
        out[mk]['corr_daily_R_all_days'] = round(float(allday.corr().iloc[0, 1]), 2)
    return out


def timing_controls(mk, W, thr, X, sl):
    """Same rules shifted to other clock hours (same window length and hold length) -> is it London-open specific?"""
    res = {}
    for st in (60, 120, 240, 300):
        try:
            hd = headline(simulate(mk, signals(mk, st + (W - 180), thr, st + (X - 180), sl, start=st)))
        except Exception:
            continue
        res[f'start {st // 60:02d}:00'] = dict(n=hd.get('n'), pf=hd.get('pf'), pf_IS=hd.get('pf_IS'), pf_OOS=hd.get('pf_OOS'))
    return res


if __name__ == '__main__':
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    G = grid()
    print(G.drop(columns=['name']).round(3).to_string())
    med = {mk: round(float(G[f'{mk}_IS'].median()), 3) for mk in MKS}
    print('median IS PF of all 24 variants:', med)
    best = G.sort_values('plateau', ascending=False).iloc[0]
    W, thr, X, sl = int(best.W), float(best.thr), int(best.X), float(best.sl)
    print('CHOSEN (IS plateau):', best['name'], 'plateau score', round(best.plateau, 3))
    res = dict(slug='london_open_fade', chosen=dict(W=W, thr=thr, X=X, sl=sl, name=best['name']), variants_tested=len(G),
               median_IS_pf=med, grid=G.round(3).to_dict('records'))
    trades = {}
    for mk in MKS:
        sig = signals(mk, W, thr, X, sl)
        rep = full_report(mk, sig)
        rep['by_year_str'] = bystr(rep)
        print(f'\n##### {mk}', json.dumps(rep, default=str))
        trades[mk] = simulate(mk, sig)
        rng = np.random.default_rng(1); base = pf(trades[mk].R); rr = []
        for _ in range(300):                                    # own, larger random-direction control (same bars/stops/exits)
            xx = sig.copy(); xx['d'] = np.where(rng.random(len(xx)) < .5, 1, -1); rr.append(pf(simulate(mk, xx).R))
        rep['random_dir_300'] = dict(median=round(float(np.nanmedian(rr)), 3), share_beaten=round(float(np.mean(np.array(rr) < base)), 3))
        print('  random direction x300:', rep['random_dir_300'])
        p, _ = partials(mk, W, thr, X, sl)
        print('  partials:', json.dumps(p, default=str))
        tc = timing_controls(mk, W, thr, X, sl)
        print('  timing controls (same rules, other hours):', json.dumps(tc))
        res[mk] = dict(report=rep, partials=p, timing_controls=tc)
    ov = azl_overlap(trades)
    print('\nAZL overlap:', json.dumps(ov, default=str))
    res['azl_overlap'] = ov
    # portfolio NQ+ES
    P = pd.concat([trades[mk].assign(mk=mk) for mk in MKS]).sort_values('entry_t')
    res['portfolio'] = headline(P.reset_index(drop=True))
    print('\nPORTFOLIO NQ+ES', json.dumps(res['portfolio'], default=str))
    r_ = {mk: res[mk]['report'] for mk in MKS}
    def _pass(a, b):
        return (r_[a]['pf_IS'] >= 1.3 and r_[a]['pf_OOS'] >= 1.3 and r_[b]['pf_OOS'] >= 1.1 and r_[a]['per_week'] >= 3
                and r_[a]['pf_2x_cost'] >= 1.1)
    if _pass('NQ', 'ES') or _pass('ES', 'NQ'):
        res['verdict'] = 'PASS'
    elif any(r_[mk]['pf_IS'] >= 1.15 and r_[mk]['pf_OOS'] >= 1.15 for mk in MKS):
        res['verdict'] = 'BORDERLINE'
    else:
        res['verdict'] = 'FAIL'
    print('VERDICT', res['verdict'])
    json.dump(res, open('/home/user/TRADING/scalp/results/london_open_fade.json', 'w'), default=str, indent=1)
