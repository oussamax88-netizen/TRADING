"""Silver-Bullet failure-to-fail with the 20-day trend, only when volatility is not compressed (slug: ftf_trend_volx).

Rules as implemented (both markets, same rules, sizes scale by ATR):
  * Entry signals = silver_bullet_inverse.signals(mk, 'FTF', trend=True, stop, exit), imported unchanged:
    H9/L9 = high/low of 09:00-09:59 bars; in 10:00-10:59 the first raid beyond H9 (L9) whose bar closes back inside
    triggers at that close (bar k); FTF = stop-entry in the RAID direction at sweep extreme +/- 0.02*ATR, valid 30 bars;
    keep only if d == trend_d (prev RTH close vs 20d SMA). One signal per day per market.
    Stops: MID = max(|entry - 09:00 midpoint|, 0.12*ATR, 20 NQ / 4 ES); FIX = max(0.15*ATR, 20 / 4).
    Exits: '2R' = target 2x stop, maxbars 120; 'close' = engine force-close 15:59.
  * Vol filter (volx_band_momentum definition): ratio = ATR14(day) / median(ATR14 of the previous 100 full RTH days)
    (ATR14 = cache 'atr', previous days only; full RTH day = >= 380 RTH bars, as in zarattini_band_momentum.prep).
    Implemented for EVERY SB day (not only full days, so no 'will this day be full' knowledge): median over the 100
    full RTH days strictly before the date. Identical to Z.prep()['ratio'] on full days (asserted). Keep if ratio >= lb;
    NaN ratio -> no trade in filtered cells. lb = 'none' keeps every signal (must reproduce silver_bullet_inverse.json).
  * Grid (16): stop {MID, FIX} x exit {2R, close} x lb {none, 0.85, 0.95, 1.05}.
  * Selection (IS only, fixed before looking): eligible = >= 40 IS trades on BOTH markets; score = min(NQ IS PF, ES IS PF);
    lb order none < 0.85 < 0.95 < 1.05; plateau = mean score of the cell and its adjacent lb cells (same stop/exit);
    best plateau wins (tie -> score). 'none' is part of the grid and eligible.
  * Controls: complement (ratio < lb), hand random-direction / random-time (SB functions), vol-day trend baseline
    (market entry at the 10:00 close in trend direction on every filtered day, same stop/exit), TDH-proxy overlap,
    daily-R correlation and PF on non-TDH days, 2x cost / by side / by year (full_report).
Lookahead: H9/L9 from 09:00-09:59 only, signals at 10:00-10:59 closes using bars <= k; ATR, trend_d and the ratio
median use previous days only (asserted); entries at k+1 or later (engine, asserted on trades).
"""
import sys, json
sys.path.insert(0, '/home/user/TRADING/scalp'); sys.path.insert(0, '/home/user/TRADING/scalp/strategies')
import numpy as np
import pandas as pd
from engine import load, signals_frame, simulate, full_report, pf, IS_END
import silver_bullet_inverse as SB
import zarattini_band_momentum as Z
from volx_band_momentum import tdh_overlap, split, by_year_str

SLUG = 'ftf_trend_volx'
OUT = f'/home/user/TRADING/scalp/results/{SLUG}.json'
LBS = [None, 0.85, 0.95, 1.05]
GRID = [(s, e, b) for s in ('MID', 'FIX') for e in ('2R', 'close') for b in LBS]
_R = {}


def vname(v):
    return f"FTF|trend=on|{v[0]}|{v[1]}|lb={'none' if v[2] is None else v[2]}"


def day_ratio(mk):
    """dict dayi -> ratio for every SB day (prior-days-only median of ATR14 over 100 full RTH days)."""
    if mk in _R:
        return _R[mk]
    D = load(mk); P = Z.prep(mk); df = SB.days(mk)
    dt = D['date'][df.k10.to_numpy(int)]
    j = np.searchsorted(P['dates'], dt, side='left')          # count of full days strictly before the date
    med = np.array([np.median(P['atr'][i - 100:i]) if i >= 100 else np.nan for i in j])
    ratio = df.A.to_numpy() / med
    # checks: ATR at 10:00 equals the 09:30 ATR of the same date (day-constant), and equals Z ratio on full days
    infull = (j < P['nd']) & (P['dates'][np.minimum(j, P['nd'] - 1)] == dt)
    assert np.allclose(df.A.to_numpy()[infull], P['atr'][j[infull]]), 'atr not day-constant'
    zr = P['ratio'][j[infull]]; ok = np.isfinite(zr)
    assert np.allclose(ratio[infull][ok], zr[ok]), 'ratio mismatch vs zarattini prep'
    _R[mk] = dict(zip(D['dayi'][df.k10.to_numpy(int)], ratio))
    return _R[mk]


def ratio_of(mk, k):
    D = load(mk); m = day_ratio(mk)
    return np.array([m.get(x, np.nan) for x in D['dayi'][np.asarray(k, int)]])


def signals(mk, stop, exit, lb=None, mode='ge'):
    s = SB.signals(mk, 'FTF', True, stop, exit)
    if lb is None:
        return s
    r = ratio_of(mk, s.k)
    keep = (r >= lb) if mode == 'ge' else (r < lb)            # NaN -> False in both
    return s[keep].reset_index(drop=True)


def is_oos(mk, sig, **kw):
    tr = simulate(mk, sig, **kw)
    i = tr[tr.entry_t < IS_END]; o = tr[tr.entry_t >= IS_END]
    return dict(n=len(tr), pw=round(len(tr) / 156.4, 2), n_IS=len(i), IS=round(pf(i.R), 3), OOS=round(pf(o.R), 3),
                all=round(pf(tr.R), 3))


def vol_day_trend_baseline(mk, stop, exit, lb):
    """Every SB day with trend_d != 0 (and ratio >= lb): market entry after the 10:00 bar close in trend direction."""
    D = load(mk); c = D['c']; df = SB.days(mk); F = SB.FLOOR[mk]
    x = df[df.tr.isin([1, -1])]
    k = x.k10.to_numpy(int); d = x.tr.to_numpy(int); A = x.A.to_numpy(); mid = ((x.H9 + x.L9) / 2).to_numpy()
    if lb is not None:
        keep = ratio_of(mk, k) >= lb; k, d, A, mid = k[keep], d[keep], A[keep], mid[keep]
    sp = np.maximum(d * (c[k] - mid), np.maximum(0.12 * A, F)) if stop == 'MID' else np.maximum(0.15 * A, F)
    tp, mb = (2 * sp, 120) if exit == '2R' else (np.inf, 0)
    return is_oos(mk, signals_frame(k, d, stop_pts=sp, tgt_pts=tp, maxbars=mb))


def lookahead_checks(mk, sig, tr):
    D = load(mk); mod, h, l = D['mod'], D['h'], D['l']
    k = sig.k.to_numpy(int)
    assert ((mod[k] >= 600) & (mod[k] < 660)).all()
    assert (tr.entry_k > tr.k).all()
    # stop-entry price is beyond the sweep extreme, which is built from window bars <= k only
    df = SB.days(mk).set_index('ck')
    for kk, d, ep in zip(k[:50], sig.d.to_numpy()[:50], sig.eprice.to_numpy()[:50]):
        r = df.loc[kk]; w = np.arange(int(r.k10), kk + 1)
        ext = h[w].max() if d == 1 else l[w].min()
        assert d * (ep - ext) >= 0 and abs(ep - (r.sw + d * 0.02 * r.A)) < 1e-9
    return True


if __name__ == '__main__':
    res = dict(slug=SLUG, name='SB failure-to-fail with 20d trend, vol not compressed', variants_tested=len(GRID), grid={})
    for v in GRID:
        g = {mk: is_oos(mk, signals(mk, *v)) for mk in ('NQ', 'ES')}
        res['grid'][vname(v)] = g
        print(f"{vname(v):38s} NQ n={g['NQ']['n']:4d} IS={g['NQ']['IS']:.2f}({g['NQ']['n_IS']}) OOS={g['NQ']['OOS']:.2f} | "
              f"ES n={g['ES']['n']:4d} IS={g['ES']['IS']:.2f}({g['ES']['n_IS']}) OOS={g['ES']['OOS']:.2f}")
    G = res['grid']
    # sanity: 'none' cells reproduce silver_bullet_inverse.json
    ref = json.load(open('/home/user/TRADING/scalp/results/silver_bullet_inverse.json'))['grid']
    san = {}
    for s in ('MID', 'FIX'):
        for e in ('2R', 'close'):
            for mk in ('NQ', 'ES'):
                a = G[vname((s, e, None))][mk]; b = ref[f'FTF|trend=on|{s}|{e}'][mk]
                san[f'{s}|{e}|{mk}'] = (a['n'] == b['n'] and abs(a['IS'] - b['IS']) < 1e-3 and abs(a['OOS'] - b['OOS']) < 1e-3)
    res['sanity_none_reproduces_SB'] = all(san.values()); print('sanity none==SB json:', all(san.values()), san if not all(san.values()) else '')
    for mk in ('NQ', 'ES'):
        res[f'median_IS_pf_{mk}'] = round(float(np.median([G[vname(v)][mk]['IS'] for v in GRID])), 3)
        res[f'median_OOS_pf_{mk}_info'] = round(float(np.median([G[vname(v)][mk]['OOS'] for v in GRID])), 3)
    score = {vname(v): min(G[vname(v)]['NQ']['IS'], G[vname(v)]['ES']['IS']) for v in GRID}
    elig = {vname(v): min(G[vname(v)]['NQ']['n_IS'], G[vname(v)]['ES']['n_IS']) >= 40 for v in GRID}

    def nb(v):
        i = LBS.index(v[2]); return [(v[0], v[1], LBS[j]) for j in (i - 1, i + 1) if 0 <= j < len(LBS)]
    plat = {vname(v): float(np.mean([score[vname(u)] for u in [v] + nb(v)])) for v in GRID}
    cand = [v for v in GRID if elig[vname(v)]]
    best = max(cand, key=lambda v: (round(plat[vname(v)], 6), score[vname(v)]))
    res['selection'] = {vname(v): dict(score=round(score[vname(v)], 3), plateau=round(plat[vname(v)], 3), eligible=elig[vname(v)])
                        for v in GRID}
    res['chosen'] = vname(best); print('CHOSEN (IS plateau):', vname(best), 'score', round(score[vname(best)], 3),
                                       'plateau', round(plat[vname(best)], 3))
    # partial edges: every cell/market with PF >= 1.1 in BOTH IS and OOS (all listed, none chosen by OOS)
    res['partials_grid_IS_OOS_ge_1.1'] = {f'{k}|{mk}': [g[mk]['IS'], g[mk]['OOS'], g[mk]['n']] for k, g in G.items()
                                          for mk in ('NQ', 'ES') if g[mk]['IS'] >= 1.1 and g[mk]['OOS'] >= 1.1}
    s0, e0, lb0 = best
    for mk in ('NQ', 'ES'):
        sig = signals(mk, *best)
        rep = full_report(mk, sig)
        rep['pf_random_dir_hand'] = SB.rand_dir_control(mk, sig)
        rep['pf_random_time_hand'] = SB.rand_time_control(mk, sig)
        rep['by_year_str'] = by_year_str(rep)
        res[mk] = rep
        tr = simulate(mk, sig)
        res[f'{mk}_lookahead_ok'] = lookahead_checks(mk, sig, tr)
        ctl = {}
        for b in LBS[1:]:
            ctl[f'complement_ratio_lt_{b}'] = is_oos(mk, signals(mk, s0, e0, b, 'lt'))
        ctl['vol_day_trend_baseline_chosen_lb'] = vol_day_trend_baseline(mk, s0, e0, lb0)
        ctl['all_day_trend_baseline'] = vol_day_trend_baseline(mk, s0, e0, None)
        res[f'{mk}_controls'] = ctl
        res[f'{mk}_tdh'] = tdh_overlap(mk, tr)
        # fixed partial subsets of the unfiltered same stop/exit cell (ratio buckets) and of the chosen cell
        base = SB.signals(mk, 'FTF', True, s0, e0); tb = simulate(mk, base); rb = ratio_of(mk, tb.k)
        sub = {'long': split(tr[tr.d == 1]), 'short': split(tr[tr.d == -1])}
        for lo, hi in ((0, 0.85), (0.85, 0.95), (0.95, 1.05), (1.05, 1.3), (1.3, 99)):
            sub[f'unfiltered_ratio_{lo}-{hi}'] = split(tb[(rb >= lo) & (rb < hi)])
        sub['unfiltered_ratio_nan'] = split(tb[np.isnan(rb)])
        res[f'{mk}_partial_subsets'] = sub
        print(mk, json.dumps({k: rep[k] for k in rep if k not in ('boot_pf_5_50_95', 'by_year', 'by_side')}, default=str))
        print(mk, 'by_year', rep['by_year_str'], 'by_side', rep['by_side'])
        print(mk, 'controls', json.dumps(ctl))
        print(mk, 'tdh', json.dumps(res[f'{mk}_tdh']))
        print(mk, 'subsets', json.dumps(sub))
    N, E = res['NQ'], res['ES']

    def ok(a, b): return a['pf_IS'] >= 1.3 and a['pf_OOS'] >= 1.3 and b['pf_OOS'] >= 1.1 and a['per_week'] >= 3 and a['pf_2x_cost'] >= 1.1
    if ok(N, E) or ok(E, N): res['verdict'] = 'PASS'
    elif any(x['pf_IS'] >= 1.15 and x['pf_OOS'] >= 1.15 for x in (N, E)): res['verdict'] = 'BORDERLINE'
    else: res['verdict'] = 'FAIL'
    res['notes'] = 'Sanity: lb=none cells reproduce silver_bullet_inverse.json exactly. Chosen by the pre-set IS rule (min-PF score, lb-neighbour plateau; edge cell has one neighbour): FIX|close|lb=1.05 -> NQ IS 2.02 / OOS 0.63, ES 1.29 / 0.39, 0.6 tr/wk: FAIL. Middle-of-plateau alternative FIX|close|lb=0.95 also fails OOS (NQ 1.04, ES 0.54). Hold-to-close cells collapse OOS when filtered; the vol filter does not hold for this signal: unfiltered ratio buckets are non-monotone and flip between halves, complements are not weaker OOS (NQ ratio<1.05: 1.28/1.44). Info only (not chosen, not tradeable as a pick): the 2R cells stay >1.1 in both halves, e.g. FIX|2R|lb=0.95 NQ 1.65/1.38, ES 1.22/1.54, 0.8 tr/wk, hold ~35 min, 2x cost NQ 1.44 ES 1.17, ~77% overlap with TDH (same direction), non-TDH days NQ 1.66/1.65 (n 18/13), ES 1.25/1.56 (n 20/6) - small samples.'
    print('VERDICT', res['verdict'])
    json.dump(res, open(OUT, 'w'), default=str, indent=1)
