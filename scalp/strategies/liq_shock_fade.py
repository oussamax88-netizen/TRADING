"""liq_shock_fade -- Liquidity-shock fade: extreme 3-5 minute move with a relative-activity spike, partial-retrace
(or VWAP) target, optional CALM (positive-gamma proxy) regime. Both markets, engine only.

Hypothesis: an extreme short-horizon move on abnormal activity is forced liquidity demand (stop cascades,
liquidations) rather than information, so liquidity providers are paid by a partial reversal; dealer long gamma
(calm up-markets) should strengthen it.

Mechanical rules (all features at bar k use data <= close of bar k):
  * Signal bars: NY minute mod in [600, 900], excluding 600-604 (10:00 data) and 840-849 (14:00 FOMC/minutes).
  * sigma1: std of 1-minute close-to-close changes over the previous 1,950 RTH bars (RTH 09:30-15:59; a change counts
    only between two consecutive minutes of the same day; RTH-only series shifted one bar, min 1,000 obs).
  * Shock (n in {3,5}): bars k-n..k contiguous minutes of one day AND |c[k]-c[k-n]| >= z*sigma1*sqrt(n)
      AND activity A_n(k) >= 3 x mean of A_n at the same NY minute over the prior 20 RTH days (today excluded, min 15)
      AND c[k] is a 30-bar closing extreme in the move direction (up: c[k] > max c[k-30..k-1]; down: < min).
    DATA ISSUE: the IC Markets CFD tick volume is a CAPPED quote count (NQ max ~580/min, 2025 median 410/min, so a 3x
    same-minute spike is arithmetically impossible -> the literal rule gives ~0 NQ trades; ES saturates at ~525 in
    2025-26). PRIMARY activity A_n = sum of bar ranges (h-l) over k-n+1..k (true-range spike = the observable part of a
    stop cascade). The literal tick-volume version (A_n = sum vol) is run as a diagnostic for the chosen cell.
  * d = -sign(c[k]-c[k-n]) (fade), market entry at the open of k+1.
  * Cooldown: a shock within 30 bars of the previous accepted shock (k-k_prev <= 30) is ignored; applied to shock
    events before the regime / VWAP filters (so regime-filtered sets are pure subsets).
  * stop_pts = max(0.12*ATR, 20 NQ / 4 ES, |shock extreme - c[k]| + 0.05*ATR); extreme = max h[k-n+1..k] for an
    up-shock (short) / min l for a down-shock (long). ATR = cache 'atr' (14-day mean RTH range of previous days).
  * Target RET: tgt_pts = 0.5*|c[k]-c[k-n]| from the fill.
    Target VWAP: tgt_px = vwap[k] (absolute, as specified) only if on the profitable side of c[k] and >= 0.08*ATR away,
    else skip.  (The engine then gives no opposite/random-direction control -> own random-time control below.)
  * maxbars = 30.
  * Regime CALM: today's ATR14 <= median of the daily ATR14 values of the prior 60 RTH days (min 40) AND trend_d == +1.
Grid (24): z {4,5,6} x n {3,5} x target {RET, VWAP} x regime {none, CALM}; v_mult fixed 3.
Selection (fixed before running, IS only): cells need >= 30 IS trades on both markets; score = min(IS PF NQ, IS PF ES);
plateau score = mean score of the cell and its z-neighbours (same n/target/regime); choose the max plateau score.
"""
import sys, json
sys.path.insert(0, '/home/user/TRADING/scalp')
import numpy as np
import pandas as pd
from engine import load, signals_frame, simulate, full_report, headline, pf, IS_END

MINSTOP = {'NQ': 20.0, 'ES': 4.0}
TICKFLAG = {'NQ': 10.0, 'ES': 2.0}


def features(D, n, act='range'):
    """Per-bar arrays for shock size n. Everything causal."""
    t, c, h, l, mod, date, dayi, vol = (D[x] for x in ('t', 'c', 'h', 'l', 'mod', 'date', 'dayi', 'vol'))
    N = len(c)
    rth = (mod >= 570) & (mod < 960)
    # sigma1 on RTH-only series of consecutive-minute changes, shifted one RTH bar
    dc = np.full(N, np.nan); ok1 = (np.diff(t) == 1) & (np.diff(dayi) == 0); dc[1:][ok1] = np.diff(c)[ok1]
    s = pd.Series(dc[rth]).rolling(1950, min_periods=1000).std().shift(1).to_numpy()
    sig1 = np.full(N, np.nan); sig1[rth] = s
    # n-bar move, contiguity
    mv = np.full(N, np.nan); okn = np.zeros(N, bool)
    okn[n:] = (t[n:] - t[:-n] == n) & (dayi[n:] == dayi[:-n])
    mv[n:] = c[n:] - c[:-n]; mv[~okn] = np.nan
    a = (h - l) if act == 'range' else vol
    A = pd.Series(a).rolling(n).sum().to_numpy()
    # same-minute mean over prior 20 RTH days
    df = pd.DataFrame({'date': date[rth], 'mod': mod[rth], 'A': A[rth]})
    pv = df.pivot_table(index='date', columns='mod', values='A', aggfunc='last')
    base = pv.rolling(20, min_periods=15).mean().shift(1)
    base = base.reindex(columns=np.arange(570, 960))
    ri = pd.Index(base.index).get_indexer(date[rth])
    Abase = np.full(N, np.nan)
    Abase[rth] = base.to_numpy()[ri, mod[rth] - 570]
    # extreme shock price and 30-bar closing extreme
    ext_hi = pd.Series(h).rolling(n).max().to_numpy(); ext_lo = pd.Series(l).rolling(n).min().to_numpy()
    cmax30 = pd.Series(c).rolling(30).max().shift(1).to_numpy(); cmin30 = pd.Series(c).rolling(30).min().shift(1).to_numpy()
    return dict(sig1=sig1, mv=mv, A=A, Abase=Abase, ext_hi=ext_hi, ext_lo=ext_lo, cmax30=cmax30, cmin30=cmin30)


def calm_flag(D):
    dayi, mod, atr, trend = D['dayi'], D['mod'], D['atr'], D['trend_d']
    rthd = np.unique(dayi[(mod >= 570) & (mod < 960)])
    first = pd.Series(np.arange(len(dayi))).groupby(dayi).first()
    a = pd.Series(atr[first.loc[rthd].to_numpy()], index=rthd)
    med = a.rolling(60, min_periods=40).median().shift(1)
    calm_day = (a <= med) & (pd.Series(trend[first.loc[rthd].to_numpy()], index=rthd) == 1)
    out = np.zeros(len(dayi), bool)
    m = pd.Series(calm_day).reindex(np.arange(dayi.max() + 1)).fillna(False).to_numpy().astype(bool)
    return m[dayi]


def shocks(D, F, z, n, mk):
    c, mod, atr = D['c'], D['mod'], D['atr']
    win = (mod >= 600) & (mod <= 900) & ~((mod >= 600) & (mod <= 604)) & ~((mod >= 840) & (mod <= 849))
    mv = F['mv']
    with np.errstate(invalid='ignore'):
        cond = win & (np.abs(mv) >= z * F['sig1'] * np.sqrt(n)) & (F['A'] >= 3.0 * F['Abase']) & ~np.isnan(atr)
        up = cond & (mv > 0) & (c > F['cmax30']); dn = cond & (mv < 0) & (c < F['cmin30'])
    ks = np.where(up | dn)[0]
    keep = []; last = -10**9
    for k in ks:
        if k - last > 30:
            keep.append(k); last = k
    return np.array(keep, dtype=np.int64)


def build(D, F, CALM, mk, z, n, tgt, reg, ks=None):
    c, atr, vwap = D['c'], D['atr'], D['vwap']
    if ks is None:
        ks = shocks(D, F, z, n, mk)
    if reg == 'CALM':
        ks = ks[CALM[ks]]
    mv = F['mv'][ks]; d = -np.sign(mv).astype(np.int64)
    ext = np.where(d == -1, F['ext_hi'][ks], F['ext_lo'][ks])
    stop = np.maximum.reduce([0.12 * atr[ks], np.full(len(ks), MINSTOP[mk]), np.abs(ext - c[ks]) + 0.05 * atr[ks]])
    if tgt == 'RET':
        return signals_frame(ks, d, stop_pts=stop, tgt_pts=0.5 * np.abs(mv), maxbars=30)
    dist = d * (vwap[ks] - c[ks])
    ok = dist >= 0.08 * atr[ks]
    return signals_frame(ks[ok], d[ok], stop_pts=stop[ok], tgt_px=vwap[ks][ok], maxbars=30)


def random_time_control(mk, D, sig, reps=30, seed=1):
    """Same days, same direction rule relative to the actual price path is not reproducible at random times, so:
    each signal is moved to a random allowed minute of the SAME day, keeping d, stop_pts and the target DISTANCE
    (VWAP target converted to points from c[k]). Median PF over reps."""
    rng = np.random.default_rng(seed); mod, dayi, c = D['mod'], D['dayi'], D['c']
    allowed = np.where((mod >= 605) & (mod <= 900) & ~((mod >= 840) & (mod <= 849)))[0]
    by_day = pd.Series(allowed).groupby(dayi[allowed]).apply(lambda x: x.to_numpy())
    tp = sig.tgt_pts.to_numpy().copy()
    m = sig.tgt_px.notna().to_numpy(); tp[m] = np.abs(sig.tgt_px.to_numpy()[m] - c[sig.k.to_numpy()[m]])
    out = []
    for _ in range(reps):
        kk = np.array([rng.choice(by_day[dayi[k]]) for k in sig.k])
        x = signals_frame(kk, sig.d.to_numpy(), stop_pts=sig.stop_pts.to_numpy(), tgt_pts=tp, maxbars=30)
        out.append(pf(simulate(mk, x).R))
    return round(float(np.nanmedian(out)), 3), round(float(np.nanpercentile(out, 95)), 3)


def lookahead_check(mk, z, n, tgt, reg):
    """Signals computed on data truncated at T must equal the full-data signals with k < T."""
    D = load(mk); res = []
    full = build(D, features(D, n), calm_flag(D), mk, z, n, tgt, reg)
    for frac in (0.45, 0.8):
        T = int(len(D['c']) * frac)
        Dt = {k: v[:T] for k, v in D.items()}
        st = build(Dt, features(Dt, n), calm_flag(Dt), mk, z, n, tgt, reg)
        a = full[full.k < T - 1].reset_index(drop=True); b = st[st.k < T - 1].reset_index(drop=True)
        res.append(bool(len(a) == len(b) and np.allclose(a.drop(columns=['k', 'd']).fillna(-1).to_numpy(),
                                                         b.drop(columns=['k', 'd']).fillna(-1).to_numpy())
                        and (a.k.to_numpy() == b.k.to_numpy()).all()))
    return res


if __name__ == '__main__':
    Z, NS, TG, RG = (4.0, 5.0, 6.0), (3, 5), ('RET', 'VWAP'), ('none', 'CALM')
    DD = {mk: load(mk) for mk in ('NQ', 'ES')}
    FF = {(mk, n): features(DD[mk], n) for mk in DD for n in NS}
    CC = {mk: calm_flag(DD[mk]) for mk in DD}
    grid = []
    for z in Z:
        for n in NS:
            for tg in TG:
                for rg in RG:
                    row = dict(z=z, n=n, tgt=tg, reg=rg)
                    for mk in DD:
                        sig = build(DD[mk], FF[(mk, n)], CC[mk], mk, z, n, tg, rg)
                        tr = simulate(mk, sig)
                        i = tr[tr.entry_t < IS_END]; o = tr[tr.entry_t >= IS_END]
                        tgtd = np.abs(tr.target - tr.entry)
                        row.update({f'{mk}_n': len(tr), f'{mk}_pw': round(len(tr) / 156.4, 2), f'{mk}_nIS': len(i),
                                    f'{mk}_IS': round(pf(i.R), 3), f'{mk}_OOS': round(pf(o.R), 3),
                                    f'{mk}_medtgt': round(float(tgtd.median()), 2) if len(tr) else None,
                                    f'{mk}_medstop': round(float(tr.risk.median()), 2) if len(tr) else None})
                    grid.append(row)
    G = pd.DataFrame(grid)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    print(G.drop(columns=['NQ_medstop', 'ES_medstop']).to_string())
    G['score'] = np.where((G.NQ_nIS >= 30) & (G.ES_nIS >= 30), np.minimum(G.NQ_IS, G.ES_IS), np.nan)
    pl = []
    for _, r in G.iterrows():
        nb = G[(G.n == r.n) & (G.tgt == r.tgt) & (G.reg == r.reg) & (np.abs(G.z - r.z) <= 1)]
        pl.append(nb.score.mean() if not np.isnan(r.score) else np.nan)
    G['plateau'] = pl
    best = G.loc[G.plateau.idxmax()] if G.plateau.notna().any() else G.loc[G.score.idxmax()]
    ch = dict(z=float(best.z), n=int(best.n), tgt=best.tgt, reg=best.reg)
    print('median IS PF all variants: NQ', G.NQ_IS.median(), 'ES', G.ES_IS.median(), '| chosen', ch)
    res = dict(slug='liq_shock_fade', chosen=ch, variants_tested=len(G),
               median_IS_pf=dict(NQ=round(float(G.NQ_IS.median()), 3), ES=round(float(G.ES_IS.median()), 3)),
               grid=G.round(3).to_dict('records'))
    for mk in DD:
        sig = build(DD[mk], FF[(mk, ch['n'])], CC[mk], mk, ch['z'], ch['n'], ch['tgt'], ch['reg'])
        rep = full_report(mk, sig)
        rep['random_time_same_day_pf_med_95'] = random_time_control(mk, DD[mk], sig)
        tr = simulate(mk, sig)
        rep['med_target_pts'] = round(float(np.abs(tr.target - tr.entry).median()), 2)
        rep['needs_tick_data'] = bool(rep['med_target_pts'] < TICKFLAG[mk])
        rep['exit_reasons'] = {int(k): int(v) for k, v in tr.reason.value_counts().items()}
        rep['lookahead_truncation_ok'] = lookahead_check(mk, ch['z'], ch['n'], ch['tgt'], ch['reg'])
        # diagnostic: literal capped tick-volume rule (not used for selection)
        Fv = features(DD[mk], ch['n'], act='vol')
        tv = simulate(mk, build(DD[mk], Fv, CC[mk], mk, ch['z'], ch['n'], ch['tgt'], ch['reg']))
        rep['diag_tickvol_rule'] = dict(n=len(tv), pf=round(pf(tv.R), 3) if len(tv) else None)
        res[mk] = rep
        print(mk, json.dumps(rep, default=str))
    json.dump(res, open('/home/user/TRADING/scalp/results/liq_shock_fade.json', 'w'), default=str, indent=1)
