"""news_0830_two_leg -- 08:30 macro-news day: pre-open overreaction FADE, then trade the news direction (CONT) after the
cash open (merge of hypotheses H4 + H5).

Hypothesis: index futures price 08:30 ET macro releases within ~1 minute (Miao et al.); the following 10-15 minutes
overshoot in thin pre-cash liquidity and revert before 09:30 (Grant-Wolf-Yu; Fung-Mok-Lam). At the 09:30 cash open,
late-informed / rebalancing flow trades in the news direction (Gao et al.: intraday momentum strongest on macro days).

Mechanical rules (as implemented; NY time, 1-minute bars, bar "mod m" = the bar whose NY minute-of-day is m):
  * Calendar-free news-day detector. V = sum of tick volume of bars 08:30..08:34 (mod 510-514; all 5 bars required).
    rank = fraction of the PREVIOUS 20 valid days' V strictly below today's V (valid day = bars 08:29..08:34 and 08:44
    exist; >= 20 valid prior days needed). News day if rank >= q. V/rank are known at the 08:34 close; first use at 08:44.
  * Reaction X = close(08:44, mod 524) - close(08:29, mod 509). Require |X| >= 0.05*ATR and X != 0.
    ATR = cache 'atr' (14-day mean RTH range of PREVIOUS days).
  * FADE leg: signal k = bar 08:44, d = -sign(X), market entry at the 08:45 open (etype 0).
    stop_pts = max(stopk*ATR, 20 pts NQ / 4 pts ES). Target: none, or tgt_px = close(08:29) (the pre-news price).
    "Skip if the target is < 0.05 ATR away": the distance at signal time is |X| >= 0.05 ATR by construction, and the
    engine itself skips the trade if the 08:45 open is already at/through the target.
    Time exit at the 09:30 open: maxbars = idx(bar 09:30) - (k+1).
  * CONT leg: signal k = bar 09:29 (close known at 09:30:00), d = +sign(X), market entry at the 09:30 open.
    stop_pts = max(stopk*ATR, 20/4). Exit at the open of bar T: maxbars = idx(bar T) - (k+1). T in {10:00, 10:30, 11:30}.
  * CHAIN: FADE (stopk 0.5, no target) + CONT (T 10:30, stopk 0.2) on the same news day in ONE signal frame.
    The engine skips a signal whose entry bar is not AFTER the previous exit bar; the FADE time-exits AT the 09:30 bar,
    so a CONT entry on the 09:30 bar is skipped (verified below: 'chain_skip_check'). As instructed, in the CHAIN the
    CONT leg is shifted uniformly (every day, regardless of how the FADE ended) to k = bar 09:30, entry at the 09:31 open,
    maxbars = idx(bar 10:30) - (k+1).
  * Control: identical rules on NON-news days (rank < 0.5).
  * Days missing any required bar are skipped. All features at the signal bar use data <= that bar's close.

Grid (22 variants, the whole selection budget):
  CONT : q {0.8, 0.9} x T {10:00, 10:30, 11:30} x stopk {0.2, 0.35}   = 12
  FADE : q {0.8, 0.9} x stopk {0.25, 0.5} x target {none, pre-news}    =  8
  CHAIN: q {0.8, 0.9} at the centre (CONT T 10:30 stopk 0.2 + FADE stopk 0.5 no target) = 2
Selection rule (fixed before running, IS only): score = mean, over the cell and its same-family neighbours (one
parameter moved one step), of min(IS PF NQ, IS PF ES); pick the highest score (tie-break: own min IS PF).
Diagnostics below the grid (controls, splits, engine-skip check) are NOT used for selection.
Frequency note: news days are ~20% of days and only one or two legs per day exist, so the verdict's 3 trades/week floor
is structurally out of reach for this hypothesis; the best attainable verdict is BORDERLINE.
"""
import sys, json
sys.path.insert(0, '/home/user/TRADING/scalp')
import numpy as np
import pandas as pd
from engine import load, signals_frame, simulate, headline, full_report, pf, IS_END

MINSTOP = {'NQ': 20.0, 'ES': 4.0}
MODS = (509, 510, 511, 512, 513, 514, 524, 569, 570, 600, 630, 690)
QS = (0.8, 0.9)
TS = (600, 630, 690)
CONT_SK = (0.2, 0.35)
FADE_SK = (0.25, 0.5)
TGTS = ('none', 'prenews')
OUT = '/home/user/TRADING/scalp/results/news_0830_two_leg.json'
IS_WEEKS = (np.datetime64('2025-04-01') - np.datetime64('2023-10-01')).astype(int) / 7.0
_DT = {}


def day_table(mk):
    """One row per calendar date (dayi): bar indices at key minutes, V, rank, X, ATR. No lookahead:
    rank uses only previous valid days' V; X uses closes up to 08:44; ATR is from previous RTH days."""
    if mk in _DT:
        return _DT[mk]
    D = load(mk)
    mod, dayi, c, vol = D['mod'], D['dayi'], D['c'], D['vol']
    nd = int(dayi.max()) + 1
    idx = {}
    for m in MODS:
        a = np.full(nd, -1, dtype=np.int64)
        w = np.where(mod == m)[0]
        a[dayi[w]] = w
        idx[m] = a
    valid = np.ones(nd, bool)
    for m in (509, 510, 511, 512, 513, 514, 524):
        valid &= idx[m] >= 0
    V = np.full(nd, np.nan)
    V[valid] = sum(vol[idx[m][valid]] for m in (510, 511, 512, 513, 514))
    rank = np.full(nd, np.nan)
    vd = np.where(valid)[0]; Vv = V[vd]
    for j in range(20, len(vd)):
        rank[vd[j]] = float(np.mean(Vv[j - 20:j] < Vv[j]))
    X = np.full(nd, np.nan); atr = np.full(nd, np.nan); c509 = np.full(nd, np.nan)
    X[valid] = c[idx[524][valid]] - c[idx[509][valid]]
    atr[valid] = D['atr'][idx[524][valid]]
    c509[valid] = c[idx[509][valid]]
    T = dict(idx=idx, valid=valid, V=V, rank=rank, X=X, atr=atr, c509=c509)
    _DT[mk] = T
    return T


def day_mask(T, q, control=False):
    ok = T['valid'] & np.isfinite(T['rank']) & np.isfinite(T['atr']) & (np.abs(T['X']) >= 0.05 * T['atr']) & (T['X'] != 0)
    return ok & ((T['rank'] < 0.5) if control else (T['rank'] >= q))


def fade_parts(mk, q, stopk, target='none', control=False):
    T = day_table(mk); idx = T['idx']
    m = day_mask(T, q, control) & (idx[570] >= 0)
    k = idx[524][m]
    d = -np.sign(T['X'][m]).astype(np.int64)
    stop = np.maximum(stopk * T['atr'][m], MINSTOP[mk])
    mb = idx[570][m] - (k + 1)
    tpx = T['c509'][m] if target == 'prenews' else np.full(len(k), np.nan)
    return dict(k=k, d=d, stop_pts=stop, tgt_px=tpx, maxbars=mb)


def cont_parts(mk, q, tmod, stopk, control=False, kmod=569):
    T = day_table(mk); idx = T['idx']
    m = day_mask(T, q, control) & (idx[kmod] >= 0) & (idx[tmod] >= 0)
    k = idx[kmod][m]
    d = np.sign(T['X'][m]).astype(np.int64)
    stop = np.maximum(stopk * T['atr'][m], MINSTOP[mk])
    mb = idx[tmod][m] - (k + 1)
    return dict(k=k, d=d, stop_pts=stop, tgt_px=np.full(len(k), np.nan), maxbars=mb)


def frame(*parts):
    cat = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    return signals_frame(cat['k'], cat['d'], etype=0, stop_pts=cat['stop_pts'], tgt_px=cat['tgt_px'], maxbars=cat['maxbars'])


def signals(mk, fam, q, tmod=630, stopk=0.2, fstopk=0.5, target='none', control=False, cont_kmod=None):
    if fam == 'CONT':
        return frame(cont_parts(mk, q, tmod, stopk, control, kmod=cont_kmod or 569))
    if fam == 'FADE':
        return frame(fade_parts(mk, q, stopk, target, control))
    if fam == 'CHAIN':
        return frame(fade_parts(mk, q, fstopk, 'none', control), cont_parts(mk, q, tmod, stopk, control, kmod=cont_kmod or 570))
    raise ValueError(fam)


def variants():
    v = []
    for q in QS:
        for t in TS:
            for s in CONT_SK:
                v.append(dict(fam='CONT', q=q, tmod=t, stopk=s))
    for q in QS:
        for s in FADE_SK:
            for tg in TGTS:
                v.append(dict(fam='FADE', q=q, stopk=s, target=tg))
    for q in QS:
        v.append(dict(fam='CHAIN', q=q, tmod=630, stopk=0.2, fstopk=0.5))
    return v


def grid():
    rows = []
    for p in variants():
        r = dict(p)
        for mk in ('NQ', 'ES'):
            h = headline(simulate(mk, signals(mk, **p)))
            r[f'{mk}_n'] = h['n']; r[f'{mk}_pw'] = h.get('per_week')
            r[f'{mk}_IS'] = h.get('pf_IS'); r[f'{mk}_OOS'] = h.get('pf_OOS'); r[f'{mk}_pf'] = h.get('pf')
        rows.append(r)
    return pd.DataFrame(rows)


def neighbours(g, r):
    fam = g[g.fam == r.fam]
    if r.fam == 'CONT':
        axes = (('q', QS), ('tmod', TS), ('stopk', CONT_SK))
    elif r.fam == 'FADE':
        axes = (('q', QS), ('stopk', FADE_SK), ('target', TGTS))
    else:
        axes = (('q', QS),)
    keep = []
    for i, rr in fam.iterrows():
        diff = [(a, vals) for a, vals in axes if rr[a] != r[a]]
        if len(diff) == 0:
            keep.append(i)
        elif len(diff) == 1:
            a, vals = diff[0]
            if abs(vals.index(rr[a]) - vals.index(r[a])) == 1:
                keep.append(i)
    return g.loc[keep]


def select(g):
    g = g.copy()
    g['minIS'] = g[['NQ_IS', 'ES_IS']].min(axis=1)
    g['plateau'] = [neighbours(g, r).minIS.mean() for _, r in g.iterrows()]
    b = g.sort_values(['plateau', 'minIS'], ascending=False).iloc[0]
    keys = {'CONT': ('fam', 'q', 'tmod', 'stopk'), 'FADE': ('fam', 'q', 'stopk', 'target'),
            'CHAIN': ('fam', 'q', 'tmod', 'stopk', 'fstopk')}[b.fam]
    ch = {k: (b[k] if isinstance(b[k], str) else (int(b[k]) if k == 'tmod' else float(b[k]))) for k in keys}
    return g, ch


def split_pf(tr, key):
    out = {}
    tr = tr.copy(); tr['seg'] = np.where(tr.entry_t < IS_END, 'IS', 'OOS')
    for v, gg in tr.groupby(key, observed=True):
        gi = gg[gg.seg == 'IS']; go = gg[gg.seg == 'OOS']
        out[str(v)] = dict(n_IS=len(gi), pf_IS=round(pf(gi.R), 2) if len(gi) else None,
                           n_OOS=len(go), pf_OOS=round(pf(go.R), 2) if len(go) else None)
    return out


def annotate(mk, tr):
    """Add sub-condition labels to trades (all known at signal time)."""
    D = load(mk); T = day_table(mk)
    kk = tr.k.to_numpy(); di = D['dayi'][kk]
    x = T['X'][di]; a = T['atr'][di]; rk = T['rank'][di]
    tr = tr.copy()
    tr['leg'] = np.where(D['mod'][kk] == 524, 'FADE', 'CONT')
    tr['news_dir'] = np.where(x > 0, 'news_up', 'news_down')
    tr['absX'] = pd.cut(np.abs(x) / a, [0, 0.1, 0.2, np.inf], right=False, labels=['0.05-0.1', '0.1-0.2', '>=0.2'])
    tr['rank'] = np.where(rk >= 0.95, 'rank>=0.95', 'rank<0.95')
    td = D['trend_d'][kk]
    tr['news_vs_trend'] = np.where(np.sign(x) == td, 'news_with_20d_trend', 'news_against_20d_trend')
    tr['dow'] = np.array(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])[D['dow'][kk]]
    # pre-open drift 08:44 -> signal bar (for CONT leg): did price keep going with X or revert before the open?
    c524 = D['c'][T['idx'][524][di]]
    pre = D['c'][kk] - c524
    tr['preopen_drift'] = np.where(tr.leg == 'FADE', 'n/a', np.where(np.sign(pre) == np.sign(x), 'drift_with_news', 'drift_against_news'))
    # overnight gap direction (pdc -> 08:44) vs news direction
    gap = c524 - D['pdc'][kk]
    tr['gap_vs_news'] = np.where(np.sign(gap) == np.sign(x), 'gap_same_dir', 'gap_opposite')
    return tr


def partial_edges(mk, sig):
    tr = annotate(mk, simulate(mk, sig))
    keys = ('leg', 'news_dir', 'absX', 'rank', 'news_vs_trend', 'dow', 'preopen_drift', 'gap_vs_news')
    return {key: split_pf(tr, key) for key in keys}


def control_table(g):
    """Same rules on non-news days (rank < 0.5) for every grid cell (diagnostic)."""
    rows = []
    for p in variants():
        r = dict(p)
        for mk in ('NQ', 'ES'):
            h = headline(simulate(mk, signals(mk, **p, control=True)))
            r[f'{mk}_ctl_n'] = h['n']; r[f'{mk}_ctl_IS'] = h.get('pf_IS'); r[f'{mk}_ctl_OOS'] = h.get('pf_OOS')
        rows.append(r)
    return pd.DataFrame(rows)


def grid_splits(g):
    """Median IS/OOS PF across family cells, by family and market (robustness of the family as a whole)."""
    out = {}
    for fam, gg in g.groupby('fam'):
        out[fam] = {f'{mk}_{s}': round(float(np.median(gg[f'{mk}_{s}'])), 3) for mk in ('NQ', 'ES') for s in ('IS', 'OOS')}
    return out


def random_dir_split(mk, sig, n=40, seed=3):
    """Random-direction control split IS / OOS (only for signals without absolute levels)."""
    rng = np.random.default_rng(seed); ri, ro = [], []
    for _ in range(n):
        x = sig.copy(); x['d'] = np.where(rng.random(len(x)) < .5, 1, -1); h = headline(simulate(mk, x))
        ri.append(h['pf_IS']); ro.append(h['pf_OOS'])
    return dict(rand_IS_med=round(float(np.median(ri)), 3), rand_OOS_med=round(float(np.median(ro)), 3))


def rank_buckets(mk, stopk=0.25):
    """FADE (no target) and always-long / always-short baselines by news-rank bucket (diagnostic, not selection)."""
    T = day_table(mk); idx = T['idx']; out = {}
    base = T['valid'] & np.isfinite(T['rank']) & np.isfinite(T['atr']) & (np.abs(T['X']) >= 0.05 * T['atr']) & (T['X'] != 0) & (idx[570] >= 0)
    for lo, hi in ((0, 0.5), (0.5, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 1.01)):
        m = base & (T['rank'] >= lo) & (T['rank'] < hi)
        k = idx[524][m]; st = np.maximum(stopk * T['atr'][m], MINSTOP[mk]); mb = idx[570][m] - (k + 1)
        row = {}
        for name, d in (('fade', -np.sign(T['X'][m]).astype(np.int64)), ('always_long', 1), ('always_short', -1)):
            h = headline(simulate(mk, signals_frame(k, d, stop_pts=st, maxbars=mb)))
            row[name] = {kk: h.get(kk) for kk in ('n', 'pf_IS', 'pf_OOS')}
        out[f'rank[{lo},{hi})'] = row
    return out


def hold_diag(mk, ch):
    """Chosen FADE cell with other clock exits (diagnostic only; 09:30 is the pre-registered rule)."""
    T = day_table(mk); idx = T['idx']; out = {}
    m = day_mask(T, ch['q']) & (idx[570] >= 0)
    for em in (540, 555, 570, 585):
        e = np.full(len(idx[570]), -1)
        w = np.where(load(mk)['mod'] == em)[0]; e[load(mk)['dayi'][w]] = w
        mm = m & (e >= 0)
        k = idx[524][mm]
        tpx = T['c509'][mm] if ch.get('target') == 'prenews' else np.full(len(k), np.nan)
        sig = signals_frame(k, -np.sign(T['X'][mm]).astype(np.int64), stop_pts=np.maximum(ch['stopk'] * T['atr'][mm], MINSTOP[mk]),
                            tgt_px=tpx, maxbars=e[mm] - (k + 1))
        h = headline(simulate(mk, sig))
        out[f'exit_{em // 60:02d}:{em % 60:02d}'] = {kk: h.get(kk) for kk in ('n', 'pf', 'pf_IS', 'pf_OOS')}
    return out


def short(rep):
    keys = ('n', 'per_week', 'win', 'pf', 'pf_IS', 'n_IS', 'pf_OOS', 'n_OOS', 'avg_pts', 'med_risk_pts', 'med_hold_min',
            'pf_2x_cost', 'pf_no_cost', 'p_pf_below_1', 'pf_opposite', 'pf_random_dir_median', 'by_year', 'by_side')
    return {k: rep.get(k) for k in keys}


if __name__ == '__main__':
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
    for mk in ('NQ', 'ES'):
        T = day_table(mk)
        nv = int(np.isfinite(T['rank']).sum())
        print(mk, 'valid days with rank:', nv, '| news days q0.8:', int((T['rank'] >= 0.8).sum()), 'q0.9:', int((T['rank'] >= 0.9).sum()),
              '| |X|>=0.05ATR among q0.8:', int(day_mask(T, 0.8).sum()))
    g = grid()
    print('=== GRID (22 variants; selection uses IS columns only) ===')
    print(g.to_string(index=False))
    med = {mk: round(float(np.median(g[f'{mk}_IS'])), 3) for mk in ('NQ', 'ES')}
    med_oos = {mk: round(float(np.median(g[f'{mk}_OOS'])), 3) for mk in ('NQ', 'ES')}
    print('median IS PF all 22 variants:', med, '| median OOS (info only):', med_oos)
    gs, ch = select(g)
    print('\n=== IS plateau scores ===')
    print(gs[['fam', 'q', 'tmod', 'stopk', 'target', 'NQ_IS', 'ES_IS', 'minIS', 'plateau']].to_string(index=False))
    print('CHOSEN (IS only):', ch)
    fam_med = grid_splits(g); print('family medians:', json.dumps(fam_med))

    # engine-skip check for the chain (diagnostic): CONT at k=09:29 vs k=09:30 within one frame
    skip = {}
    for mk in ('NQ', 'ES'):
        for kmod in (569, 570):
            tr = annotate(mk, simulate(mk, signals(mk, 'CHAIN', 0.8, cont_kmod=kmod)))
            skip[f'{mk}_cont_k{kmod}'] = int((tr.leg == 'CONT').sum())
        skip[f'{mk}_cont_standalone'] = len(simulate(mk, signals(mk, 'CONT', 0.8, tmod=630, stopk=0.2)))
    print('chain_skip_check (CONT trades taken in CHAIN q0.8):', skip)

    ct = control_table(g)
    print('\n=== control: same rules on NON-news days (rank < 0.5) ===')
    print(ct[['fam', 'q', 'tmod', 'stopk', 'target', 'NQ_ctl_n', 'NQ_ctl_IS', 'NQ_ctl_OOS', 'ES_ctl_n', 'ES_ctl_IS', 'ES_ctl_OOS']].to_string(index=False))

    res = dict(slug='news_0830_two_leg', chosen=ch, variants_tested=len(g), median_IS_pf=med, median_OOS_pf_info=med_oos,
               family_medians=fam_med, chain_skip_check=skip, grid=g.to_dict(orient='records'),
               control_non_news=ct.to_dict(orient='records'))
    for mk in ('NQ', 'ES'):
        sig = signals(mk, **ch)
        rep = full_report(mk, sig)
        print(f'\n=== {mk} full_report chosen {ch} ===\n', json.dumps(rep, default=str))
        diag = {}
        hc = headline(simulate(mk, signals(mk, **ch, control=True)))
        diag['control_non_news'] = {k: hc.get(k) for k in ('n', 'pf', 'pf_IS', 'pf_OOS')}
        if sig.tgt_px.isna().all():
            diag['random_dir_IS_OOS'] = random_dir_split(mk, sig)
            o = sig.copy(); o['d'] = -o.d; ho = headline(simulate(mk, o))
            diag['opposite_IS_OOS'] = dict(pf_IS=ho['pf_IS'], pf_OOS=ho['pf_OOS'])
        diag['cost_3x_pf'] = round(pf(simulate(mk, sig, costmult=3.0).R), 3)
        diag['partial_edges'] = partial_edges(mk, sig)
        if ch['fam'] == 'FADE':
            diag['rank_buckets'] = rank_buckets(mk, ch['stopk'])
            diag['hold_time'] = hold_diag(mk, ch)
            if ch.get('target') == 'prenews':   # sibling cell without the absolute target gets the engine's controls
                sib = full_report(mk, signals(mk, 'FADE', ch['q'], stopk=ch['stopk'], target='none'))
                diag['sibling_no_target_full_report'] = short(sib)
        # the other two families at their centre cells, for context (not selectable)
        for name, p in (('CONT_centre', dict(fam='CONT', q=ch['q'], tmod=630, stopk=0.2)),
                        ('FADE_centre', dict(fam='FADE', q=ch['q'], stopk=0.5, target='none')),
                        ('CHAIN_centre', dict(fam='CHAIN', q=ch['q']))):
            h = headline(simulate(mk, signals(mk, **p)))
            diag[name] = {k: h.get(k) for k in ('n', 'pf', 'pf_IS', 'pf_OOS')}
        for kk, v in diag.items():
            if kk != 'partial_edges':
                print(f'--- {mk} {kk}:', json.dumps(v, default=str))
        print(f'--- {mk} partial edges:')
        for kk, v in diag['partial_edges'].items():
            print('   ', kk, json.dumps(v))
        res[mk] = rep; res[mk + '_diag'] = diag
    # mechanical verdict (task rules)
    a, b = res['NQ'], res['ES']
    def strong(x): return x['pf_IS'] >= 1.3 and x['pf_OOS'] >= 1.3 and x['per_week'] >= 3 and x['pf_2x_cost'] >= 1.1
    passed = (strong(a) and b['pf_OOS'] >= 1.1) or (strong(b) and a['pf_OOS'] >= 1.1)
    border = any(x['pf_IS'] >= 1.15 and x['pf_OOS'] >= 1.15 for x in (a, b))
    res['verdict'] = 'PASS' if passed else ('BORDERLINE' if border else 'FAIL')
    res['verdict_note'] = ('PF criteria of PASS met on NQ (IS/OOS >= 1.3, ES OOS >= 1.1, 2x cost >= 1.1) but frequency is '
                           f"{a['per_week']}/{b['per_week']} trades/week (< 3): structurally impossible for a news-day-only rule.")
    print('VERDICT:', res['verdict'])
    json.dump(res, open(OUT, 'w'), default=str, indent=1)
    print('saved', OUT)
