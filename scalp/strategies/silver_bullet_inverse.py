"""Inverse Silver Bullet: 09:00-hour range acceptance in 10:00-11:00 (slug: silver_bullet_inverse).

Rules as implemented (both markets, same rules, sizes scale by ATR):
  * Day = NY calendar date (cache `dayi`). H9/L9 = max high / min low of the bars with mod 540..599 (09:00-09:59),
    known at the 09:59 close; need >= 50 such bars and a valid ATR. MID = (H9+L9)/2. ATR = cache atr (14-day mean
    of previous RTH ranges), trend_d = sign(prev RTH close - 20d SMA) (previous days only). Floor = 20 pts NQ / 4 ES.
  * Window: signal bars k with mod 600..659 (10:00-10:59). At most one signal per day per family.
  * ACCEPT: first bar k in the window with c[k] > H9 (d=+1) or c[k] < L9 (d=-1). Market entry at open k+1 (etype 0).
  * Classic SB trigger (shared by FTF, MIRROR and the control): an UP raid starts at the first window bar j with
    h[j] > H9; sweep high = max high from j to k. Trigger at the first window bar k >= j with c[k] < H9 (the raid
    bar itself may be k if it wicks above and closes back below - literal reading 'k>j' is reported as a check).
    Mirror for L9 (DOWN raid, sweep low, c[k] > L9). The first trigger of either side is the day's only one.
    Classic side cs = -1 after an up raid (short fade), +1 after a down raid.
  * FTF: d = -cs; buy stop-entry (etype 2) at sweep high + 0.02*ATR (sell stop at sweep low - 0.02*ATR), ebars 30.
  * MIRROR: d = -cs, market at open k+1; stop_px = L9 for the long (H9 for the short), distance from c[k] floored
    at max(0.12*ATR, floor); tgt_px = sweep high + 0.02*ATR (sweep low - 0.02*ATR); maxbars 120.
  * Stops (ACCEPT, FTF): MID = max(d*(ref - MID), 0.12*ATR, floor) with ref = c[k] (ACCEPT) or the stop-entry price
    (FTF); FIX = max(0.15*ATR, floor) (floor added; it rarely binds). Exits: '2R' tgt_pts = 2*stop, maxbars 120;
    'close' no target, engine force-close at 15:59.
  * Trend filter 'on': keep the day's (single) signal only if d == trend_d (the first event is NOT re-searched).
  * Grid (18): ACCEPT, FTF x trend {off,on} x stop {MID,FIX} x exit {2R,close} = 16; MIRROR x trend {off,on} = 2.
  * Selection (IS only, fixed before looking): score = geometric mean of IS PF on NQ and ES; plateau score = mean
    score of the variant and its one-step grid neighbours (same family, one of trend/stop/exit changed); highest wins.
  * Controls (not counted): classic SB fade (d = cs, market k+1, stop_px = sweep extreme +- 0.02 ATR with distance
    floored at max(0.12 ATR, floor), tgt_px = L9 for the short / H9 for the long, maxbars 120), classic with the
    'k>j' trigger; hand-built random-direction control for the chosen variant (same k, same point distances,
    market entry at k+1, d random, 30 seeds; stop-entries converted to market entries); random-time control
    (random window bar, random d, same stop/exit spec).

Lookahead check: H9/L9 use bars 09:00-09:59 only, all signal bars are >= 10:00; the raid/sweep state at bar k uses
h/l/c of window bars <= k; ATR/trend_d are previous-day values; entries are at k+1 or later (engine).
"""
import sys, json
sys.path.insert(0, '/home/user/TRADING/scalp')
import numpy as np
import pandas as pd
from engine import load, signals_frame, simulate, full_report, headline, pf, IS_END

SLUG = 'silver_bullet_inverse'
OUT = f'/home/user/TRADING/scalp/results/{SLUG}.json'
FLOOR = {'NQ': 20.0, 'ES': 4.0}
_DAYS = {}


def days(mk, strict_later=False):
    key = (mk, strict_later)
    if key in _DAYS:
        return _DAYS[key]
    D = load(mk)
    h, l, c, mod, dayi, atr, trd = D['h'], D['l'], D['c'], D['mod'], D['dayi'], D['atr'], D['trend_d']
    starts = np.r_[0, np.where(np.diff(dayi) != 0)[0] + 1]; ends = np.r_[starts[1:], len(dayi)]
    rows = []
    for a, b in zip(starts, ends):
        m = mod[a:b]
        r9 = a + np.where((m >= 540) & (m < 600))[0]
        w = a + np.where((m >= 600) & (m < 660))[0]
        if len(r9) < 50 or len(w) < 30 or np.isnan(atr[w[0]]):
            continue
        H9, L9 = h[r9].max(), l[r9].min()
        rec = dict(H9=H9, L9=L9, A=atr[w[0]], tr=trd[w[0]], k10=w[0], ak=-1, ad=0, ck=-1, cs=0, sw=np.nan)
        for k in w:                                       # ACCEPT: first close outside
            if c[k] > H9 or c[k] < L9:
                rec['ak'] = k; rec['ad'] = 1 if c[k] > H9 else -1; break
        up = dn = False; uh = -np.inf; dl = np.inf
        for k in w:                                       # classic SB trigger
            if not strict_later:
                if h[k] > H9: up = True
                if l[k] < L9: dn = True
            if up: uh = max(uh, h[k])
            if dn: dl = min(dl, l[k])
            if up and c[k] < H9:
                rec['ck'] = k; rec['cs'] = -1; rec['sw'] = uh; break
            if dn and c[k] > L9:
                rec['ck'] = k; rec['cs'] = 1; rec['sw'] = dl; break
            if strict_later:
                if h[k] > H9 and not up: up = True; uh = h[k]
                if l[k] < L9 and not dn: dn = True; dl = l[k]
        rows.append(rec)
    df = pd.DataFrame(rows)
    _DAYS[key] = df
    return df


def signals(mk, fam, trend=False, stop='MID', exit='2R', strict_later=False):
    D = load(mk); c = D['c']; df = days(mk, strict_later); F = FLOOR[mk]
    if fam == 'ACCEPT':
        x = df[df.ak >= 0]
        k = x.ak.to_numpy(int); d = x.ad.to_numpy(int)
    else:
        x = df[df.ck >= 0]
        k = x.ck.to_numpy(int); d = (-x.cs if fam != 'CLASSIC' else x.cs).to_numpy(int)
    A = x.A.to_numpy(); H9 = x.H9.to_numpy(); L9 = x.L9.to_numpy(); mid = (H9 + L9) / 2; sw = x.sw.to_numpy()
    keep = (d == x.tr.to_numpy()) if trend else np.ones(len(k), bool)
    k, d, A, H9, L9, mid, sw = k[keep], d[keep], A[keep], H9[keep], L9[keep], mid[keep], sw[keep]
    fl = np.maximum(0.12 * A, F)
    if fam in ('MIRROR', 'CLASSIC'):
        # MIRROR: d = -cs; stop on the classic target side; target just beyond the sweep extreme.
        # CLASSIC: d = cs; stop just beyond the sweep extreme; target the opposite side of the 09:00 range.
        if fam == 'MIRROR':
            stop_px = np.where(d == 1, np.minimum(L9, c[k] - fl), np.maximum(H9, c[k] + fl))
            tgt_px = sw + d * 0.02 * A
        else:
            stop_px = np.where(d == -1, np.maximum(sw + 0.02 * A, c[k] + fl), np.minimum(sw - 0.02 * A, c[k] - fl))
            tgt_px = np.where(d == -1, L9, H9)
        return signals_frame(k, d, stop_px=stop_px, tgt_px=tgt_px, maxbars=120)
    if fam == 'ACCEPT':
        ref = c[k]; et, ep, eb = 0, np.nan, 0
    else:  # FTF
        ref = sw + d * 0.02 * A; et, ep, eb = 2, ref, 30
    sp = np.maximum(d * (ref - mid), fl) if stop == 'MID' else np.maximum(0.15 * A, F)
    tp, mb = (2 * sp, 120) if exit == '2R' else (np.inf, 0)
    return signals_frame(k, d, etype=et, eprice=ep, ebars=eb, stop_pts=sp, tgt_pts=tp, maxbars=mb)


GRID = [(f, t, s, e) for f in ('ACCEPT', 'FTF') for t in (False, True) for s in ('MID', 'FIX') for e in ('2R', 'close')] + \
       [('MIRROR', t, '-', '-') for t in (False, True)]


def vname(v):
    return f"{v[0]}|trend={'on' if v[1] else 'off'}|{v[2]}|{v[3]}"


def neighbours(v):
    out = []
    for u in GRID:
        if u[0] == v[0] and u != v and sum(a != b for a, b in zip(u[1:], v[1:])) == 1:
            out.append(u)
    return out


def rand_dir_control(mk, sig, seeds=30):
    """Same k, same point distances, market entry at k+1, random direction."""
    D = load(mk); c = D['c']; s = sig.copy()
    if s.stop_px.notna().any():
        s['stop_pts'] = np.abs(c[s.k] - s.stop_px); s['tgt_pts'] = np.abs(s.tgt_px - c[s.k])
        s['stop_px'] = np.nan; s['tgt_px'] = np.nan
    s['etype'] = 0; s['eprice'] = np.nan; s['ebars'] = 0
    rng = np.random.default_rng(1); r = []
    for _ in range(seeds):
        x = s.copy(); x['d'] = np.where(rng.random(len(x)) < 0.5, 1, -1); r.append(pf(simulate(mk, x).R))
    return round(float(np.median(r)), 3)


def rand_time_control(mk, sig, seeds=30):
    """Random window bar (10:00-10:59) on the same days, random direction, same stop/target points."""
    D = load(mk); mod, dayi = D['mod'], D['dayi']; c = D['c']
    s = sig.copy()
    if s.stop_px.notna().any():
        s['stop_pts'] = np.abs(c[s.k] - s.stop_px); s['tgt_pts'] = np.abs(s.tgt_px - c[s.k])
        s['stop_px'] = np.nan; s['tgt_px'] = np.nan
    s['etype'] = 0; s['eprice'] = np.nan; s['ebars'] = 0
    rng = np.random.default_rng(2); r = []
    k0 = s.k.to_numpy() - (mod[s.k.to_numpy()] - 600)          # approx 10:00 bar of that day
    for _ in range(seeds):
        kk = np.clip(k0 + rng.integers(0, 60, len(s)), 0, len(mod) - 2)
        ok = (dayi[kk] == dayi[s.k.to_numpy()]) & (mod[kk] >= 600) & (mod[kk] < 660)
        x = s[ok].copy(); x['k'] = kk[ok]; x['d'] = np.where(rng.random(len(x)) < 0.5, 1, -1)
        r.append(pf(simulate(mk, x).R))
    return round(float(np.median(r)), 3)


def trend_baselines(mk, v):
    """Is it just 'trend hold'? (a) chosen signals, market entry at k+1 instead of the stop-entry (same d/stop/exit);
    (b) every valid day with trend_d != 0, market entry after the 10:00 bar close in trend direction, same stop/exit."""
    D = load(mk); c = D['c']; df = days(mk); F = FLOOR[mk]; out = {}
    s = signals(mk, *v).copy(); s['etype'] = 0; s['eprice'] = np.nan; s['ebars'] = 0
    out['same_signals_market_entry'] = is_oos(mk, s)
    x = df[df.tr.isin([1, -1])]; k = x.k10.to_numpy(int); d = x.tr.to_numpy(int); A = x.A.to_numpy()
    mid = ((x.H9 + x.L9) / 2).to_numpy()
    sp = np.maximum(d * (c[k] - mid), np.maximum(0.12 * A, F)) if v[2] == 'MID' else np.maximum(0.15 * A, F)
    tp, mb = (2 * sp, 120) if v[3] == '2R' else (np.inf, 0)
    out['all_days_trend_1000'] = is_oos(mk, signals_frame(k, d, stop_pts=sp, tgt_pts=tp, maxbars=mb))
    return out


def is_oos(mk, sig, **kw):
    tr = simulate(mk, sig, **kw)
    i = tr[tr.entry_t < IS_END]; o = tr[tr.entry_t >= IS_END]
    return dict(n=len(tr), pw=round(len(tr) / 156.4, 2), IS=round(pf(i.R), 3), OOS=round(pf(o.R), 3), all=round(pf(tr.R), 3))


if __name__ == '__main__':
    res = dict(slug=SLUG, variants_tested=len(GRID), grid={})
    for v in GRID:
        g = {mk: is_oos(mk, signals(mk, *v)) for mk in ('NQ', 'ES')}
        res['grid'][vname(v)] = g
        print(f"{vname(v):34s} NQ n={g['NQ']['n']:4d} IS={g['NQ']['IS']:.2f} OOS={g['NQ']['OOS']:.2f} | "
              f"ES n={g['ES']['n']:4d} IS={g['ES']['IS']:.2f} OOS={g['ES']['OOS']:.2f}")
    G = res['grid']
    for mk in ('NQ', 'ES'):
        res[f'median_IS_pf_{mk}'] = round(float(np.median([G[vname(v)][mk]['IS'] for v in GRID])), 3)
        res[f'median_OOS_pf_{mk}_info'] = round(float(np.median([G[vname(v)][mk]['OOS'] for v in GRID])), 3)
    score = {vname(v): float(np.sqrt(max(G[vname(v)]['NQ']['IS'], 0) * max(G[vname(v)]['ES']['IS'], 0))) for v in GRID}
    plat = {vname(v): float(np.mean([score[vname(u)] for u in [v] + neighbours(v)])) for v in GRID}
    best = max(GRID, key=lambda v: plat[vname(v)])
    res['chosen'] = vname(best); res['chosen_IS_score'] = round(score[vname(best)], 3); res['chosen_plateau'] = round(plat[vname(best)], 3)
    print('CHOSEN (IS plateau):', vname(best), round(plat[vname(best)], 3))
    # partial edges: every variant/market with PF > 1 in BOTH IS and OOS (all listed, none chosen by OOS)
    res['partials'] = {f'{k}|{mk}': [g[mk]['IS'], g[mk]['OOS'], g[mk]['n']] for k, g in G.items() for mk in ('NQ', 'ES')
                       if g[mk]['IS'] > 1 and g[mk]['OOS'] > 1}
    for mk in ('NQ', 'ES'):
        sig = signals(mk, *best)
        rep = full_report(mk, sig)
        rep['pf_random_dir_hand'] = rand_dir_control(mk, sig)
        rep['pf_random_time_hand'] = rand_time_control(mk, sig)
        res[mk] = rep
        ctl = {}
        ctl['classic_fade'] = is_oos(mk, signals(mk, 'CLASSIC'))
        ctl['classic_fade_nocost'] = is_oos(mk, signals(mk, 'CLASSIC'), cost=0.0)
        ctl['classic_fade_k>j'] = is_oos(mk, signals(mk, 'CLASSIC', strict_later=True))
        ctl['mirror_k>j'] = is_oos(mk, signals(mk, 'MIRROR', strict_later=True))
        ctl['accept_rand_dir'] = rand_dir_control(mk, signals(mk, 'ACCEPT', False, 'MID', '2R'))
        ctl['chosen_k>j'] = is_oos(mk, signals(mk, *best, strict_later=True))
        ctl.update(trend_baselines(mk, best))
        res[f'{mk}_controls'] = ctl
        print(mk, json.dumps({k: rep[k] for k in rep if k not in ('boot_pf_5_50_95',)}, default=str))
        print(mk, 'controls', json.dumps(ctl))
    N, E = res['NQ'], res['ES']
    def ok(a, b): return a['pf_IS'] >= 1.3 and a['pf_OOS'] >= 1.3 and b['pf_OOS'] >= 1.1 and a['per_week'] >= 3 and a['pf_2x_cost'] >= 1.1
    if ok(N, E) or ok(E, N): res['verdict'] = 'PASS'
    elif any(x['pf_IS'] >= 1.15 and x['pf_OOS'] >= 1.15 for x in (N, E)): res['verdict'] = 'BORDERLINE'
    else: res['verdict'] = 'FAIL'
    res['notes'] = ('Classic SB fade replicates the proposer (IS PF NQ 0.64 / ES 0.56) but its failure decays OOS '
                    '(0.90 / 1.01), so the mirror only holds on NQ. Chosen FTF trend-on FIX close: NQ 1.61/1.07, '
                    'ES 1.23/0.95, 1.3 trades/week. FTF family is >1 IS and OOS on both markets in most cells '
                    '(e.g. trend-on 2R exits) but not chosen and below PASS levels.')
    print('VERDICT', res['verdict'])
    json.dump(res, open(OUT, 'w'), default=str, indent=1)
