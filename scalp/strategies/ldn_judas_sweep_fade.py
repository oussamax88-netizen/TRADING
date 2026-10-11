"""London-open Judas fade: the 03:00 drive that runs the overnight (Asia) high/low is faded into the US pre-market.

Mechanical rules (as implemented, both markets, same rules):
  * Day = NY calendar date. s = the 03:00 bar (mod 180, must exist); s-1 must be on the same date.
  * ONH / ONL = cache on_high / on_low at bar s-1 (running overnight range 18:00 -> 02:59). ATR = cache atr (prior days).
  * e = last bar of the day with 180 < mod <= W (W = 03:14 / 03:29 / 03:59). Drive r = (c[e] - o[s]) / ATR.
  * Judas filter: up-drive (r > 0) with max(h[s..e]) > ONH, or down-drive (r < 0) with min(l[s..e]) < ONL.
    |r| >= thr (thr = 0.0 means any non-zero drive). Signal k = e, d = -sign(r), market entry at the open of e+1.
  * Stop (points from the fill): 'ATR' = max(0.35*ATR, 20 NQ / 4 ES);
    'SWP' = (window extreme in the drive direction - c[e]) * drive sign + 0.05*ATR, floored at max(0.12*ATR, 20 / 4).
  * Exit: 'T930' = time exit at the 09:30 open (maxbars = idx(09:30) - (e+1)).
    'MID' = tgt_px at the overnight midpoint (ONH+ONL)/2 when d*(mid - c[e]) >= 0.05*ATR, otherwise no target; + same 09:30 exit.
    (Engine note: if the e+1 open is already beyond the midpoint the engine skips the trade; counted as 'mid_skipped'.)
  * One trade per day per market. Engine costs NQ 1.5 / ES 0.6 pts.
Lookahead audit: every input (o[s], c[e], h/l[s..e], on_high/on_low[s-1], atr[s]) has index <= k = e; exits are clock times.
on_high/on_low at s-1 are re-verified below against a direct max/min of the 18:00->02:59 bars.

Grid (24): W {03:14, 03:29, 03:59} x thr {0.0, 0.15} x stop {ATR, SWP} x exit {T930, MID}.
Selection (IS only): eligible if >= 25 IS trades on both markets; score = min(NQ IS PF, ES IS PF);
plateau = mean score over the variant and its W neighbours (same thr/stop/exit); highest plateau wins.
"""
import sys, json, itertools
sys.path.insert(0, '/home/user/TRADING/scalp'); sys.path.insert(0, '/home/user/TRADING/scalp/strategies')
import numpy as np
import pandas as pd
from engine import load, signals_frame, simulate, full_report, headline, pf, IS_END
import london_open_fade as LOF

S_PAD = '/tmp/claude-0/-home-user-TRADING/b9cb08f9-c0e5-5011-839c-6894039a978b/scratchpad'
FLOOR = {'NQ': 20.0, 'ES': 4.0}
WS = [194, 209, 239]; OFF = {194: 14, 209: 29, 239: 59}
THRS = [0.0, 0.15]; STOPS = ['ATR', 'SWP']; EXITS = ['T930', 'MID']
MKS = ('NQ', 'ES')
_T = {}


def table(mk):
    """One row per day with the 03:00 bar, the window stats for each W, overnight range and the 09:30 bar."""
    if mk in _T:
        return _T[mk]
    D = load(mk); o, h, l, c, atr, dayi = D['o'], D['h'], D['l'], D['c'], D['atr'], D['dayi']
    T = LOF.days(mk).copy()
    s = T.s.to_numpy()
    T['ONH'] = D['on_high'][s - 1]; T['ONL'] = D['on_low'][s - 1]; T['A'] = atr[s]
    T['trend'] = D['trend_d'][s]; T['dow'] = D['dow'][s]
    ok = (dayi[s - 1] == dayi[s]) & np.isfinite(T.ONH) & np.isfinite(T.ONL) & np.isfinite(T.A) & (T.A > 0) & (T.x570 > 0)
    T = T[ok].reset_index(drop=True)
    for W, off in OFF.items():
        e = T[f'e{off}'].to_numpy()
        good = e > T.s.to_numpy()
        T[f'wh{W}'] = [h[a:b + 1].max() if g else np.nan for a, b, g in zip(T.s, e, good)]
        T[f'wl{W}'] = [l[a:b + 1].min() if g else np.nan for a, b, g in zip(T.s, e, good)]
        T[f'ce{W}'] = np.where(good, c[np.maximum(e, 0)], np.nan)
        T[f'r{W}'] = (T[f'ce{W}'] - o[T.s]) / T.A
    _T[mk] = T
    return T


def signals(mk, W, thr, stop, ex, mode='sweep', pts_target=False):
    """mode: 'sweep' (strategy), 'inside' (complement: drive did not run the overnight extreme), 'all'.
    pts_target: express the MID target as points from the signal close (for opposite/random-direction controls)."""
    T = table(mk); e = T[f'e{OFF[W]}'].to_numpy(); r = T[f'r{W}'].to_numpy()
    wh, wl, ce, A = T[f'wh{W}'].to_numpy(), T[f'wl{W}'].to_numpy(), T[f'ce{W}'].to_numpy(), T.A.to_numpy()
    sweep = ((r > 0) & (wh > T.ONH.to_numpy())) | ((r < 0) & (wl < T.ONL.to_numpy()))
    mb = T.x570.to_numpy() - (e + 1)
    m = np.isfinite(r) & (r != 0) & (np.abs(r) >= thr) & (mb >= 1)
    m &= sweep if mode == 'sweep' else (~sweep if mode == 'inside' else True)
    d = -np.sign(np.nan_to_num(r)).astype(int); F = FLOOR[mk]
    if stop == 'ATR':
        sp = np.maximum(0.35 * A, F)
    else:
        dist = np.where(d == -1, wh - ce, ce - wl) + 0.05 * A
        sp = np.maximum(dist, np.maximum(0.12 * A, F))
    mid = (T.ONH.to_numpy() + T.ONL.to_numpy()) / 2
    has_t = (ex == 'MID') & (d * (mid - ce) >= 0.05 * A)
    tpx = np.where(has_t, mid, np.nan); tpts = np.full(len(T), np.inf)
    if pts_target:
        tpts = np.where(has_t, d * (mid - ce), np.inf); tpx = np.full(len(T), np.nan)
    sig = signals_frame(e[m], d[m], stop_pts=sp[m], tgt_px=tpx[m], tgt_pts=tpts[m], maxbars=mb[m])
    meta = pd.DataFrame(dict(k=e[m], a=T.a.to_numpy()[m], b=T.b.to_numpy()[m], x=T.x570.to_numpy()[m], mid=mid[m], A=A[m],
                             absr=np.abs(r[m]), trend=T.trend.to_numpy()[m], dow=T.dow.to_numpy()[m], has_t=has_t[m],
                             depth=(np.where(r > 0, wh - T.ONH, T.ONL - wl) / A)[m], onr=((T.ONH - T.ONL) / A).to_numpy()[m],
                             back_in=((ce <= T.ONH) & (ce >= T.ONL)).to_numpy()[m]))
    return sig, meta


def vname(W, thr, stop, ex):
    return f'W{W // 60:02d}:{W % 60:02d}_thr{thr}_{stop}_{ex}'


def grid():
    rows = []
    for W, thr, stop, ex in itertools.product(WS, THRS, STOPS, EXITS):
        row = dict(W=W, thr=thr, stop=stop, ex=ex, name=vname(W, thr, stop, ex))
        for mk in MKS:
            hd = headline(simulate(mk, signals(mk, W, thr, stop, ex)[0]))
            row.update({f'{mk}_IS': hd.get('pf_IS'), f'{mk}_OOS': hd.get('pf_OOS'), f'{mk}_nIS': hd.get('n_IS', 0), f'{mk}_n': hd['n']})
        rows.append(row)
    G = pd.DataFrame(rows)
    G['score'] = G[['NQ_IS', 'ES_IS']].min(axis=1)
    G['elig'] = (G.NQ_nIS >= 25) & (G.ES_nIS >= 25)
    pl = []
    for i, r in G.iterrows():
        j = WS.index(r.W); nb = [WS[jj] for jj in (j - 1, j, j + 1) if 0 <= jj < 3]
        q = G[G.W.isin(nb) & (G.thr == r.thr) & (G.stop == r.stop) & (G.ex == r.ex)]
        pl.append(q.score.mean())
    G['plateau'] = pl
    return G


def bystr(rep):
    return ' '.join(f"{y}:{v['pf']}({v['n']})" for y, v in rep['by_year'].items())


def isoos(t):
    a, b = t[t.entry_t < IS_END], t[t.entry_t >= IS_END]
    return dict(pf=round(pf(t.R), 2) if len(t) else None, n=len(t), pf_IS=round(pf(a.R), 2) if len(a) else None, n_IS=len(a),
                pf_OOS=round(pf(b.R), 2) if len(b) else None, n_OOS=len(b))


def random_time(mk, sig, meta, ex, reps=20, seed=7):
    """Same days, same direction and stop points; signal bar drawn uniformly from 03:29..08:58 (entry 03:30..08:59);
    T930 exit (maxbars to 09:30 open); MID target re-applied with the same 0.05*ATR rule at the random bar's close."""
    D = load(mk); mod, c = D['mod'], D['c']; rng = np.random.default_rng(seed); out = []
    cand = [np.where((mod[a:b] >= 209) & (mod[a:b] <= 538))[0] + a for a, b in zip(meta.a, meta.b)]
    for _ in range(reps):
        ks = np.array([rng.choice(cc) if len(cc) else -1 for cc in cand]); ok = ks >= 0
        d = sig.d.to_numpy()
        ht = (ex == 'MID') & (d * (meta.mid.to_numpy() - c[np.maximum(ks, 0)]) >= 0.05 * meta.A.to_numpy())
        s2 = signals_frame(ks[ok], d[ok], stop_pts=sig.stop_pts.to_numpy()[ok], tgt_px=np.where(ht, meta.mid, np.nan)[ok],
                           maxbars=(meta.x.to_numpy() - (ks + 1))[ok])
        out.append(isoos(simulate(mk, s2)))
    f = lambda key: round(float(np.nanmedian([x[key] for x in out if x[key] is not None])), 3)
    return dict(reps=reps, median_pf=f('pf'), median_pf_IS=f('pf_IS'), median_pf_OOS=f('pf_OOS'), all_pf=[x['pf'] for x in out])


def daily(t, col='R'):
    return t.assign(date=t.entry_t.dt.date).groupby('date')[col].sum()


def overlap(t, other, oside):
    """t: our trades; other: trades with entry_t, R, side column name oside."""
    t = t.assign(date=t.entry_t.dt.date); o = other.assign(date=other.entry_t.dt.date)
    od = o.groupby('date').agg(oR=('R', 'sum'), oside=(oside, 'first'))
    m = t.merge(od, left_on='date', right_index=True, how='inner')
    allday = pd.concat([daily(t).rename('a'), od.oR.rename('b')], axis=1).fillna(0)
    return dict(our_days=int(t.date.nunique()), other_days=int(len(od)), common=int(len(m)),
                share_of_ours_on_other_days=round(len(m) / max(t.date.nunique(), 1), 2),
                same_dir=round(float((m.d == m.oside).mean()), 2) if len(m) else None,
                corr_daily_R_common=round(float(np.corrcoef(m.R, m.oR)[0, 1]), 2) if len(m) > 5 else None,
                corr_daily_R_all=round(float(allday.corr().iloc[0, 1]), 2),
                our_pf_off_other_days=round(pf(t[~t.date.isin(od.index)].R), 2) if (~t.date.isin(od.index)).any() else None)


def partials(t, meta):
    t = t.merge(meta, on='k', how='left'); isd = t.entry_t < IS_END
    q = t.absr[isd].quantile([1 / 3, 2 / 3]).to_numpy(); dm = t.depth[isd].median(); om = t.onr[isd].median()
    P = {'long (fade down-sweep)': t.d == 1, 'short (fade up-sweep)': t.d == -1,
         'fade WITH 20d trend': t.d == t.trend, 'fade AGAINST 20d trend': t.d != t.trend,
         '|r| low tercile (IS edges)': t.absr < q[0], '|r| mid tercile': (t.absr >= q[0]) & (t.absr < q[1]),
         '|r| high tercile': t.absr >= q[1],
         'sweep depth < IS median': t.depth < dm, 'sweep depth >= IS median': t.depth >= dm,
         'overnight range < IS median': t.onr < om, 'overnight range >= IS median': t.onr >= om,
         'c[e] back inside overnight range': t.back_in, 'c[e] still beyond overnight extreme': ~t.back_in,
         'Mon-Tue': t.dow <= 1, 'Wed': t.dow == 2, 'Thu-Fri': t.dow >= 3}
    out = {k: isoos(t[v]) for k, v in P.items()}
    out['edges'] = dict(r_terciles=[round(float(x), 3) for x in q], depth_med=round(float(dm), 3), onr_med=round(float(om), 3))
    out['exit mix'] = {n: round(float((t.reason == i).mean()), 2) for i, n in ((1, 'stop'), (2, 'target'), (4, 'time'), (3, 'eod'))}
    return out


def audit(mk, n=40, seed=3):
    """Verify ONH/ONL at s-1 against a direct max/min of the bars from the previous 18:00 (or session start) to s-1."""
    D = load(mk); T = table(mk); rng = np.random.default_rng(seed); bad = 0
    for i in rng.choice(len(T), n, replace=False):
        s = int(T.s[i]); j = s - 1; t0 = D["t"][s] - 9 * 60    # 18:00 of the previous calendar day (NY minutes)
        lo = np.searchsorted(D['t'], t0)
        bad += int(not (np.isclose(D['h'][lo:s].max(), T.ONH[i]) and np.isclose(D['l'][lo:s].min(), T.ONL[i])))
    return dict(checked=n, mismatches=bad)


if __name__ == '__main__':
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    G = grid()
    print(G.drop(columns=['name']).round(3).to_string())
    med = {mk: round(float(G[f'{mk}_IS'].median()), 3) for mk in MKS}
    print('median IS PF of all 24 variants:', med)
    best = G[G.elig].sort_values('plateau', ascending=False).iloc[0]
    W, thr, stop, ex = int(best.W), float(best.thr), best.stop, best.ex
    print('CHOSEN (IS plateau):', best['name'], 'plateau', round(best.plateau, 3), 'score', round(best.score, 3))
    res = dict(slug='ldn_judas_sweep_fade', chosen=dict(W=W, thr=thr, stop=stop, exit=ex, name=best['name']),
               variants_tested=len(G), median_IS_pf=med, grid=G.round(3).to_dict('records'))
    lof_sig = {mk: LOF.signals(mk, 209, 0.15, 570, 0.35) for mk in MKS}
    try:
        AZL = pd.read_pickle(f'{S_PAD}/AZL_trades.pkl')
    except Exception:
        AZL = None
    trades = {}
    for mk in MKS:
        sig, meta = signals(mk, W, thr, stop, ex)
        rep = full_report(mk, sig); rep['by_year_str'] = bystr(rep)
        t = simulate(mk, sig); trades[mk] = t
        n_sig = int(((pd.to_datetime(load(mk)['t'][sig.k + 1] * 60_000_000_000) >= np.datetime64('2023-10-01'))
                     & (pd.to_datetime(load(mk)['t'][sig.k + 1] * 60_000_000_000) < np.datetime64('2026-10-01'))).sum())
        rep['mid_skipped'] = n_sig - len(t)
        if ex == 'MID':                                   # own controls with the target converted to points
            sp, _ = signals(mk, W, thr, stop, ex, pts_target=True)
            rp = full_report(mk, sp)
            rep['pts_version'] = dict(pf=rp['pf'], pf_IS=rp['pf_IS'], pf_OOS=rp['pf_OOS'], n=rp['n'])
            rep['pf_opposite'] = rp['pf_opposite']; rep['pf_random_dir_median'] = rp['pf_random_dir_median']
            rep['pf_random_dir_95'] = rp['pf_random_dir_95']
        print(f'\n##### {mk}', json.dumps({k: v for k, v in rep.items() if k != 'by_year'}, default=str))
        comp = isoos(simulate(mk, signals(mk, W, thr, stop, ex, mode='inside')[0]))
        allv = isoos(simulate(mk, signals(mk, W, thr, stop, ex, mode='all')[0]))
        rt = random_time(mk, sig, meta, ex)
        part = partials(t, meta)
        ov = dict(LOF=overlap(t, simulate(mk, lof_sig[mk]), 'd'))
        if AZL is not None:
            ov['AZL'] = overlap(t, AZL[AZL.mk == mk], 'side')
        print('  complement (drive inside ON range):', comp, '\n  all drives (no sweep filter):', allv)
        print('  random entry time 03:30-08:59 x20:', {k: v for k, v in rt.items() if k != 'all_pf'})
        print('  overlap:', json.dumps(ov))
        print('  partials:'); [print('    ', k, v) for k, v in part.items()]
        res[mk] = dict(report=rep, complement_inside=comp, all_drives=allv, random_time=rt, overlap=ov, partials=part, audit=audit(mk))
        print('  audit:', res[mk]['audit'])
        if ex != 'MID':                                   # same cell with the MID target (already in the grid) -> scalp-ness
            tm = simulate(mk, signals(mk, W, thr, stop, 'MID')[0]); hm = headline(tm)
            res[mk]['same_cell_MID'] = dict(pf_IS=hm['pf_IS'], pf_OOS=hm['pf_OOS'], med_hold=hm['med_hold_min'],
                                            target_share=round(float((tm.reason == 2).mean()), 2))
            print('  same cell with MID target:', res[mk]['same_cell_MID'])
    both = [k for k in res['NQ']['partials'] if k not in ('exit mix', 'edges') and all(
        (res[mk]['partials'][k]['pf_IS'] or 0) > 1 and (res[mk]['partials'][k]['pf_OOS'] or 0) > 1 for mk in MKS)]
    res['partials_positive_IS_OOS_both_markets'] = both
    print('\npartials PF>1 in IS and OOS on both markets:', both)
    P = pd.concat([trades[mk].assign(mk=mk) for mk in MKS]).sort_values('entry_t')
    res['portfolio'] = headline(P.reset_index(drop=True))
    print('PORTFOLIO NQ+ES', json.dumps(res['portfolio'], default=str))
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
    json.dump(res, open('/home/user/TRADING/scalp/results/ldn_judas_sweep_fade.json', 'w'), default=str, indent=1)
