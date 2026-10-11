"""liq_shock_fade -- Liquidity-shock fade: extreme 3-5 minute move with a relative-volume spike, partial-retrace
target, optional positive-gamma (CALM) regime.

Hypothesis: an extreme short-horizon move on abnormal volume is mostly forced liquidity demand (stop cascades,
liquidations) rather than information, so liquidity providers are paid by a partial reversal. Dealer long gamma
(common in the 0DTE era, especially in calm up-markets; Dim-Eraker-Vilkov) should strengthen the reversal.

Mechanical rules (as implemented; all features at bar k use data <= close of bar k):
  * Signal bars: NY minute-of-day mod in [600, 900] (10:00-15:00), excluding 600-604 (10:00 data releases) and
    840-849 (14:00 FOMC / minutes).
  * sigma1 = standard deviation of 1-minute close-to-close changes over the previous 1,950 RTH bars (RTH = 09:30-15:59,
    only changes between two CONSECUTIVE 1-minute bars of the same day; series shifted by one RTH bar so bar k itself
    is not in its own window; min 1,000 observations).
  * Shock at k (n in {3, 5}):
      |c[k] - c[k-n]| >= z * sigma1 * sqrt(n)            (bars k-n..k must be contiguous minutes of the same day)
      AND V_n(k) = sum vol[k-n+1..k] >= 3 x mean of V_n at the SAME NY minute over the prior 20 RTH days
          (mean over the available values of those 20 days, min 15; current day excluded).
          DATA ISSUE -> 'vol' = bar range (h-l) as the activity proxy (PRIMARY, decided before any PF was seen):
          the IC Markets CFD tick volume is a capped quote count (NQ saturates at ~550-580 per minute, 2025 RTH
          median 414 -> a 3x spike is impossible: the literal rule gives ZERO NQ signals; ES saturates at ~525).
          The literal tick-volume rule (own market, and ES volume for both) is printed as a diagnostic only.
      AND c[k] is a new 30-bar closing extreme in the move direction: up move -> c[k] > max(c[k-30..k-1]),
          down move -> c[k] < min(c[k-30..k-1]).
  * Direction d = -sign(c[k] - c[k-n]) (fade), market entry at the open of k+1 (etype 0).
  * Cooldown: a shock is ignored if it is within 30 bars of the previous accepted shock (k - k_prev <= 30); the
    cooldown is applied to shock events (before the regime / VWAP-target filters, which are day- or bar-level skips).
  * Stop (points from the fill) = max(0.12*ATR, 20 NQ / 4 ES, |shock extreme - c[k]| + 0.05*ATR), where the shock
    extreme = max(h[k-n+1..k]) for an up-shock (short) / min(l[k-n+1..k]) for a down-shock (long).
  * Target RETRACE: tgt_pts = 0.5 * |c[k] - c[k-n]| from the fill.
    Target VWAP: running RTH VWAP at bar k (cache 'vwap'); taken only if it is on the profitable side of c[k] and
    >= 0.08*ATR away, else the shock is skipped. Implemented in POINTS: tgt_pts = d*(vwap[k] - c[k]) from the fill
    (keeps the engine's opposite / random-direction controls valid; differs from an absolute price only by the
    c[k] -> open[k+1] gap).
  * maxbars = 30 (time exit at the open of the 30th bar after the fill).
  * Regime CALM: today's ATR14 (previous 14 RTH days) <= median of the daily ATR14 values of the prior 60 RTH days
    (min 40) AND trend_d == +1 (previous RTH close above its 20-day mean).
  * ATR = cache 'atr' = 14-day mean RTH range of PREVIOUS days.

Grid (24 variants = full selection budget): z {4,5,6} x n {3,5} x target {RET, VWAP} x regime {none, CALM}; v_mult 3.
Selection rule (fixed before running, IS only): among cells with >= 30 IS trades on BOTH markets,
score = mean over the cell and its z-neighbours (same n/target/regime, adjacent z) of min(IS PF NQ, IS PF ES);
pick the max score (plateau, both markets), tie-break min IS PF of the cell itself.
Diagnostics (not used for selection): same-day random-time control, sub-condition splits (partial edges).
"""
import sys, json
sys.path.insert(0, '/home/user/TRADING/scalp')
import numpy as np
import pandas as pd
from engine import load, signals_frame, simulate, headline, full_report, pf, IS_END

MINSTOP = {'NQ': 20.0, 'ES': 4.0}
TICKFLAG = {'NQ': 10.0, 'ES': 2.0}
ZS = (4.0, 5.0, 6.0)
NS = (3, 5)
TGTS = ('RET', 'VWAP')
REGS = ('none', 'CALM')
VMULT = 3.0
COOLDOWN = 30
VSRC = 'range'   # primary volume proxy (see vol_features)
MAXBARS = 30
OUT = '/home/user/TRADING/scalp/results/liq_shock_fade.json'
IS_WEEKS = (np.datetime64('2025-04-01') - np.datetime64('2023-10-01')).astype(int) / 7.0
_FEAT = {}


def base_features(mk):
    """Per-bar features (no lookahead): sigma1, contiguity, CALM flag, signal-window mask."""
    if mk in _FEAT:
        return _FEAT[mk]
    D = load(mk)
    t, c, mod, date = D['t'], D['c'], D['mod'], D['date']
    N = len(t)
    rth = (mod >= 570) & (mod < 960)
    # ---- sigma1: std of consecutive-minute same-day RTH close changes, previous 1,950 RTH bars (shifted)
    dc = np.full(N, np.nan)
    cont1 = np.zeros(N, bool); cont1[1:] = (t[1:] - t[:-1] == 1) & (date[1:] == date[:-1])
    dc[1:] = c[1:] - c[:-1]
    dc[~cont1] = np.nan
    ri = np.where(rth)[0]
    s = pd.Series(dc[ri]).rolling(1950, min_periods=1000).std().shift(1).to_numpy()
    sigma1 = np.full(N, np.nan); sigma1[ri] = s
    # ---- CALM regime: day-level (atr / trend_d are from previous days)
    rd = pd.DataFrame({'date': date[ri], 'atr': D['atr'][ri], 'trend': D['trend_d'][ri]}).groupby('date').first()
    rd['atr_med60'] = rd.atr.rolling(60, min_periods=40).median().shift(1)
    rd['calm'] = (rd.atr <= rd.atr_med60) & (rd.trend == 1)
    calm = pd.Series(rd.calm.to_numpy(), index=rd.index).reindex(date).fillna(False).to_numpy(bool)
    win = (mod >= 600) & (mod <= 900) & ~((mod >= 600) & (mod <= 604)) & ~((mod >= 840) & (mod <= 849))
    # ---- 30-bar closing extremes of the PREVIOUS 30 bars
    cs = pd.Series(c)
    prev_hi30 = cs.rolling(30).max().shift(1).to_numpy(); prev_lo30 = cs.rolling(30).min().shift(1).to_numpy()
    _FEAT[mk] = dict(D=D, rth=rth, sigma1=sigma1, calm=calm, win=win, prev_hi30=prev_hi30, prev_lo30=prev_lo30, ri=ri)
    return _FEAT[mk]


def vol_features(mk, n, vsrc=None):
    """Relative 'volume' of the n-bar window vs the same NY minute over the prior 20 RTH days.
    vsrc 'range' (PRIMARY): activity = sum of bar ranges (h-l) -- used because the CFD tick volume is capped
         (NQ quote-count saturates at ~550-580/min: 2025 RTH median 414, 99.9th pct 553, so a 3x spike is impossible;
         ES saturates at ~525/min in 2025-26).
    vsrc 'tick': literal tick volume of the market itself (diagnostic only).
    vsrc 'tickES': ES tick volume used for both markets (diagnostic only)."""
    vsrc = vsrc or VSRC
    key = (mk, 'vol', n, vsrc)
    if key in _FEAT:
        return _FEAT[key]
    F = base_features(mk); D = F['D']
    t, mod, date, rth, ri = D['t'], D['mod'], D['date'], F['rth'], F['ri']
    if vsrc == 'range':
        vol = D['h'] - D['l']
    elif vsrc == 'tick':
        vol = D['vol']
    else:   # ES tick volume aligned to this market's bars by NY minute timestamp
        E = load('ES'); vol = pd.Series(E['vol'], index=E['t']).reindex(t).to_numpy()
    N = len(t)
    contn = np.zeros(N, bool); contn[n:] = (t[n:] - t[:-n] == n) & (date[n:] == date[:-n])   # bars k-n..k contiguous
    vs = pd.Series(vol).rolling(n).sum().to_numpy()                                         # sum vol[k-n+1..k]
    vs_ok = np.zeros(N, bool); vs_ok[n - 1:] = (t[n - 1:] - t[:N - n + 1] == n - 1)
    vs = np.where(vs_ok, vs, np.nan)
    # same-minute baseline over the prior 20 RTH days (pivot RTH dates x minute; shift(1) excludes today)
    piv = pd.DataFrame({'date': date[ri], 'mod': mod[ri], 'v': vs[ri]}).pivot_table(index='date', columns='mod', values='v',
                                                                                      aggfunc='first', dropna=False)
    base = piv.rolling(20, min_periods=15).mean().shift(1)
    ridx = base.index.get_indexer(date[ri]); cidx = base.columns.get_indexer(mod[ri])
    vbase = np.full(N, np.nan); vbase[ri] = base.to_numpy()[ridx, cidx]
    _FEAT[key] = dict(contn=contn, vs=vs, vbase=vbase)
    return _FEAT[key]


def shocks(mk, z, n, vsrc=None):
    """Shock events after the cooldown: arrays k, move (c[k]-c[k-n]), extreme."""
    F = base_features(mk); V = vol_features(mk, n, vsrc); D = F['D']
    c, h, l = D['c'], D['h'], D['l']
    N = len(c)
    mv = np.full(N, np.nan); mv[n:] = c[n:] - c[:-n]
    with np.errstate(invalid='ignore'):
        cond = F['win'] & V['contn'] & np.isfinite(F['sigma1']) & np.isfinite(V['vbase']) & np.isfinite(D['atr'])
        cond &= np.abs(mv) >= z * F['sigma1'] * np.sqrt(n)
        cond &= V['vs'] >= VMULT * V['vbase']
        up = (mv > 0) & (c > F['prev_hi30']); dn = (mv < 0) & (c < F['prev_lo30'])
        cond &= (up | dn)
    cand = np.where(cond)[0]
    keep = []; last = -10 ** 9
    for k in cand:
        if k - last > COOLDOWN:
            keep.append(k); last = k
    k = np.array(keep, dtype=np.int64)
    hmax = pd.Series(h).rolling(n).max().to_numpy()[k]; lmin = pd.Series(l).rolling(n).min().to_numpy()[k]
    extreme = np.where(mv[k] > 0, hmax, lmin)
    return k, mv[k], extreme


def signals(mk, z=5.0, n=3, tgt='RET', reg='none', extra_mask=None, vsrc=None):
    F = base_features(mk); D = F['D']
    k, mv, ext = shocks(mk, z, n, vsrc)
    c, atr, vwap = D['c'][k], D['atr'][k], D['vwap'][k]
    d = -np.sign(mv).astype(np.int64)
    stop = np.maximum.reduce([0.12 * atr, np.full(len(k), MINSTOP[mk]), np.abs(ext - c) + 0.05 * atr])
    if tgt == 'RET':
        tp = 0.5 * np.abs(mv); ok = np.ones(len(k), bool)
    else:
        tp = d * (vwap - c)
        ok = np.isfinite(tp) & (tp >= 0.08 * atr)
    if reg == 'CALM':
        ok &= F['calm'][k]
    if extra_mask is not None:
        ok &= extra_mask(D, k, d)
    sig = signals_frame(k[ok], d[ok], etype=0, stop_pts=stop[ok], tgt_pts=tp[ok], maxbars=MAXBARS)
    return sig


def grid():
    rows = []
    for z in ZS:
        for n in NS:
            for tgt in TGTS:
                for reg in REGS:
                    r = dict(z=z, n=n, tgt=tgt, reg=reg)
                    for mk in ('NQ', 'ES'):
                        sig = signals(mk, z, n, tgt, reg)
                        hd = headline(simulate(mk, sig))
                        r[f'{mk}_n'] = hd.get('n', 0); r[f'{mk}_pw'] = hd.get('per_week', 0)
                        r[f'{mk}_nIS'] = hd.get('n_IS', 0)
                        r[f'{mk}_IS'] = hd.get('pf_IS', np.nan); r[f'{mk}_OOS'] = hd.get('pf_OOS', np.nan); r[f'{mk}_pf'] = hd.get('pf', np.nan)
                        r[f'{mk}_medtgt'] = round(float(np.median(sig.tgt_pts)), 2) if len(sig) else np.nan
                        r[f'{mk}_medstop'] = round(float(np.median(sig.stop_pts)), 2) if len(sig) else np.nan
                    rows.append(r)
    return pd.DataFrame(rows)


def select(g):
    m = g.copy()
    m['minIS'] = m[['NQ_IS', 'ES_IS']].min(axis=1)
    score = []
    for _, r in m.iterrows():
        zi = ZS.index(r.z)
        nb = [ZS[j] for j in (zi - 1, zi, zi + 1) if 0 <= j < len(ZS)]
        score.append(m[(m.n == r.n) & (m.tgt == r.tgt) & (m.reg == r.reg) & (m.z.isin(nb))].minIS.mean())
    m['plateau'] = score
    elig = m[(m.NQ_nIS >= 30) & (m.ES_nIS >= 30)]
    if len(elig) == 0:
        elig = m
    b = elig.sort_values(['plateau', 'minIS'], ascending=False).iloc[0]
    return m, dict(z=float(b.z), n=int(b.n), tgt=b.tgt, reg=b.reg)


def random_time_control(mk, ch, n_draw=30, seed=7):
    """Same days, same count, random bar inside the allowed signal window; direction = fade of the last n-bar move
    at that bar; stop/target sizes copied from the real signal. Median PF IS/OOS."""
    rng = np.random.default_rng(seed)
    F = base_features(mk); D = F['D']; c = D['c']; n = ch['n']
    sig = signals(mk, **ch)
    pool = {}
    wi = np.where(F['win'])[0]
    for dd, g in pd.Series(wi).groupby(D['dayi'][wi]):
        pool[dd] = g.to_numpy()
    out_is, out_oos, out_all = [], [], []
    for _ in range(n_draw):
        ks = []; ds = []
        for k in sig.k.to_numpy():
            p = pool[D['dayi'][k]]; kk = int(rng.choice(p))
            mvr = c[kk] - c[kk - n]; ks.append(kk); ds.append(-1 if mvr > 0 else 1)
        x = signals_frame(ks, ds, etype=0, stop_pts=sig.stop_pts.to_numpy(), tgt_pts=sig.tgt_pts.to_numpy(), maxbars=MAXBARS)
        hd = headline(simulate(mk, x))
        out_is.append(hd['pf_IS']); out_oos.append(hd['pf_OOS']); out_all.append(hd['pf'])
    return dict(pf_med=round(float(np.nanmedian(out_all)), 3), pf_IS_med=round(float(np.nanmedian(out_is)), 3),
                pf_OOS_med=round(float(np.nanmedian(out_oos)), 3))


def splits(mk, ch):
    """Sub-condition PFs IS/OOS for the chosen cell, and median over all 24 cells (diagnostic only)."""
    D = load(mk); F = base_features(mk)
    tr = simulate(mk, signals(mk, **ch)); kk = tr.k.to_numpy()
    mod = D['mod'][kk]
    seg = np.where(tr.entry_t < IS_END, 'IS', 'OOS')
    groups = {
        'long': tr.d == 1, 'short': tr.d == -1,
        '10-12': mod < 720, '12-14': (mod >= 720) & (mod < 840), '14-15': mod >= 840,
        'calm': F['calm'][kk], 'not_calm': ~F['calm'][kk],
        'trend_up': D['trend_d'][kk] == 1, 'trend_dn': D['trend_d'][kk] != 1,
        'fade_with_trend': tr.d.to_numpy() == D['trend_d'][kk], 'fade_against_trend': tr.d.to_numpy() != D['trend_d'][kk],
    }
    out = {}
    for name, m in groups.items():
        m = np.asarray(m)
        a = tr[m & (seg == 'IS')]; b = tr[m & (seg == 'OOS')]
        out[name] = dict(n_IS=len(a), pf_IS=round(pf(a.R), 2) if len(a) else None, n_OOS=len(b), pf_OOS=round(pf(b.R), 2) if len(b) else None)
    return out


def all_cell_splits():
    """For every (cell, market, sub-condition): IS and OOS PF -> count cells where both > 1."""
    rows = []
    for z in ZS:
        for n in NS:
            for tgt in TGTS:
                for reg in REGS:
                    for mk in ('NQ', 'ES'):
                        sp = splits(mk, dict(z=z, n=n, tgt=tgt, reg=reg))
                        for name, v in sp.items():
                            rows.append(dict(z=z, n=n, tgt=tgt, reg=reg, mk=mk, cond=name, **v))
    return pd.DataFrame(rows)


def by_year_str(rep):
    return ' '.join(f"{y}:{v['pf']}({v['n']})" for y, v in rep.get('by_year', {}).items())


if __name__ == '__main__':
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 400)
    g = grid()
    m, ch = select(g)
    print(m[['z', 'n', 'tgt', 'reg', 'NQ_n', 'NQ_pw', 'NQ_IS', 'NQ_OOS', 'NQ_medtgt', 'NQ_medstop', 'ES_n', 'ES_pw', 'ES_IS', 'ES_OOS',
             'ES_medtgt', 'ES_medstop', 'minIS', 'plateau']].to_string())
    med_is = {mk: round(float(np.nanmedian(g[f'{mk}_IS'])), 3) for mk in ('NQ', 'ES')}
    med_oos = {mk: round(float(np.nanmedian(g[f'{mk}_OOS'])), 3) for mk in ('NQ', 'ES')}
    print('chosen (IS-only plateau):', ch, '| median IS PF all 24 cells:', med_is, '| median OOS PF (info):', med_oos)
    res = dict(slug='liq_shock_fade', chosen=ch, variants_tested=int(len(g)), median_IS_pf=med_is, median_OOS_pf_info=med_oos,
               grid=g.to_dict(orient='records'))
    for mk in ('NQ', 'ES'):
        sig = signals(mk, **ch)
        rep = full_report(mk, sig)
        rep['med_tgt_pts'] = round(float(np.median(sig.tgt_pts)), 2)
        rep['med_stop_pts_signal'] = round(float(np.median(sig.stop_pts)), 2)
        rep['needs_tick_data_flag'] = bool(rep['med_tgt_pts'] < TICKFLAG[mk])
        rep['random_time_control'] = random_time_control(mk, ch)
        rep['by_year_str'] = by_year_str(rep)
        rep['splits'] = splits(mk, ch)
        res[mk] = rep
        print(mk, json.dumps({k: v for k, v in rep.items() if k != 'splits'}, default=str))
        print(mk, 'splits', json.dumps(rep['splits']))
    # ---- literal tick-volume rule for the chosen cell (diagnostic only; not used for selection)
    lit = {}
    for vs in ('tick', 'tickES'):
        for mk in ('NQ', 'ES'):
            hd = headline(simulate(mk, signals(mk, **ch, vsrc=vs)))
            lit[f'{vs}_{mk}'] = {k2: hd.get(k2) for k2 in ('n', 'per_week', 'pf', 'pf_IS', 'pf_OOS')}
    print('literal tick-volume diagnostic (chosen cell):', json.dumps(lit))
    res['literal_tick_volume_diag'] = lit
    acs = all_cell_splits()
    acs['both_gt1'] = (acs.pf_IS > 1) & (acs.pf_OOS > 1)
    summ = acs.groupby(['mk', 'cond']).agg(cells_both_gt1=('both_gt1', 'sum'), med_IS=('pf_IS', 'median'), med_OOS=('pf_OOS', 'median'),
                                           med_nIS=('n_IS', 'median'), med_nOOS=('n_OOS', 'median')).reset_index()
    print(summ.to_string())
    res['partial_edge_scan'] = summ.to_dict(orient='records')
    json.dump(res, open(OUT, 'w'), default=str, indent=1)
    print('saved', OUT)
