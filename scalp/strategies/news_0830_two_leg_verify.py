"""Independent skeptic re-implementation of news_0830_two_leg (FADE, q0.9, stopk 0.25, target = pre-news close).
Written from the rule text only. Same engine. Adds a neighbour family and random-time / random-direction controls.
"""
import sys, json
sys.path.insert(0, '/home/user/TRADING/scalp')
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv
from engine import load, signals_frame, simulate, headline, pf

MINSTOP = {'NQ': 20.0, 'ES': 4.0}
BASE = dict(q=0.9, stopk=0.25, xthr=0.05, look=20, vb=(510, 515), kmod=524, pre=509, exitmod=570, target='prenews', ms=1.0)


def bars_by_minute(mk):
    """DataFrame index=dayi, columns=mod, value=bar index (only 08:00-10:00 for speed)."""
    D = load(mk)
    w = np.where((D['mod'] >= 480) & (D['mod'] <= 600))[0]
    return pd.DataFrame({'di': D['dayi'][w], 'mod': D['mod'][w], 'i': w}).pivot(index='di', columns='mod', values='i')


_P = {}


def days(mk, p):
    D = load(mk)
    if mk not in _P:
        _P[mk] = bars_by_minute(mk)
    P = _P[mk]
    vmods = list(range(*p['vb']))
    need = sorted(set(vmods + [p['pre'], p['kmod']]))
    ok = P[need].notna().all(axis=1)
    Q = P[ok]
    V = sum(D['vol'][Q[m].to_numpy(np.int64)] for m in vmods)
    L = p['look']
    rank = np.full(len(V), np.nan)
    if len(V) > L:
        win = swv(V, L)[:-1]                     # windows of the L valid days BEFORE day j (j = L..)
        rank[L:] = (win < V[L:, None]).mean(axis=1)
    k = Q[p['kmod']].to_numpy(np.int64)
    c0 = D['c'][Q[p['pre']].to_numpy(np.int64)]
    X = D['c'][k] - c0
    atr = D['atr'][k]
    # time exit: first bar of the day with mod >= exitmod (index-based so gaps do not shift it)
    ex = P.loc[ok, [m for m in P.columns if m >= p['exitmod']]].min(axis=1).to_numpy()
    return pd.DataFrame(dict(di=Q.index.to_numpy(), k=k, c0=c0, X=X, atr=atr, rank=rank, ex=ex, V=V))


def strat(mk, p=BASE, ret_days=False):
    t = days(mk, p)
    m = (t['rank'] >= p['q'] - 1e-12) & np.isfinite(t.atr) & (t.X.abs() >= p['xthr'] * t.atr) & (t.X != 0) & t.ex.notna()
    t = t[m]
    d = -np.sign(t.X).astype(np.int64)
    stop = np.maximum(p['stopk'] * t.atr, MINSTOP[mk] * p['ms'])
    tgt = t.c0.to_numpy() if p['target'] == 'prenews' else np.nan
    sig = signals_frame(t.k.to_numpy(), d.to_numpy(), stop_pts=stop.to_numpy(), tgt_px=tgt,
                        maxbars=(t.ex.to_numpy(np.int64) - (t.k.to_numpy() + 1)))
    return (sig, t) if ret_days else sig


def hl(tr):
    h = headline(tr)
    return {k: h.get(k) for k in ('n', 'per_week', 'pf', 'pf_IS', 'n_IS', 'pf_OOS', 'n_OOS', 'avg_pts')}


def neighbours():
    v = {}
    for key, vals in (('q', (0.8, 0.85, 0.95)), ('stopk', (0.2, 0.3)), ('xthr', (0.025, 0.1)), ('look', (15, 30)),
                      ('vb', ((510, 513), (510, 520))), ('kmod', (519, 529)), ('exitmod', (555, 585)),
                      ('ms', (0.75, 1.5)), ('target', ('none',))):
        for x in vals:
            p = dict(BASE); p[key] = x
            if key == 'kmod':
                p['kmod'] = x
            v[f'{key}={x}'] = p
    return v


def controls(mk, n_rep=20, seed=7):
    """Executed strategy trades -> (A) same days, random entry minute 08:30..09:14 signal bar (entry 08:31..09:15),
    exit 09:30 open, shuffled direction mix, same stop pts, same target distance in pts.
    (B) same days & same 08:44 signal, shuffled directions, same stop/target pts.
    (C) same days, random entry minute as in A, but the strategy's own direction (timing-only control)."""
    D = load(mk); rng = np.random.default_rng(seed)
    sig = strat(mk)
    tr = simulate(mk, sig)
    s = sig.set_index('k').loc[tr.k]
    stop = s.stop_pts.to_numpy(); tpts = np.abs(s.tgt_px.to_numpy() - tr.entry.to_numpy())
    # strategy expressed with point targets (must equal the absolute-target version)
    sp = signals_frame(tr.k.to_numpy(), tr.d.to_numpy(), stop_pts=stop, tgt_pts=tpts, maxbars=s.maxbars.to_numpy())
    pf_pts = headline(simulate(mk, sp))['pf']
    P = _P[mk]; di = D['dayi'][tr.k.to_numpy()]
    ex = P.loc[di, [m for m in P.columns if m >= 570]].min(axis=1).to_numpy(np.int64)
    out = {'A': [], 'B': [], 'C': []}
    for r in range(n_rep):
        dperm = rng.permutation(tr.d.to_numpy())
        rk = []
        for j, dd in enumerate(di):
            row = P.loc[dd, 510:554].dropna().to_numpy(np.int64)
            rk.append(rng.choice(row))
        rk = np.array(rk)
        A = signals_frame(rk, dperm, stop_pts=stop, tgt_pts=tpts, maxbars=ex - (rk + 1))
        B = signals_frame(tr.k.to_numpy(), dperm, stop_pts=stop, tgt_pts=tpts, maxbars=s.maxbars.to_numpy())
        C = signals_frame(rk, tr.d.to_numpy(), stop_pts=stop, tgt_pts=tpts, maxbars=ex - (rk + 1))
        for nm, x in (('A', A), ('B', B), ('C', C)):
            out[nm].append(pf(simulate(mk, x).R))
    res = {'strategy_pf_abs_target': headline(tr)['pf'], 'strategy_pf_pts_target': pf_pts}
    for nm, v in out.items():
        v = np.array(v)
        res[nm] = dict(median=round(float(np.nanmedian(v)), 3), p95=round(float(np.nanpercentile(v, 95)), 3),
                       max=round(float(np.nanmax(v)), 3), frac_ge_strategy=round(float(np.mean(v >= res['strategy_pf_abs_target'])), 2))
    return res


def broad_fade(mk, kmods=(524,), only_news=False):
    """Diagnostic: the same fade-to-pre-08:29-price rule on ALL valid days (no news filter)."""
    p = dict(BASE, q=-1.0)
    return hl(simulate(mk, strat(mk, p)))


if __name__ == '__main__':
    R = {}
    for mk in ('NQ', 'ES'):
        sig, t = strat(mk, ret_days=True)
        tr = simulate(mk, sig)
        base = hl(tr)
        base['pf_2x'] = round(pf(simulate(mk, sig, costmult=2).R), 3)
        base['pf_nocost'] = round(pf(simulate(mk, sig, cost=0).R), 3)
        base['by_year'] = {int(y): (len(g), round(pf(g.R), 2)) for y, g in tr.groupby(tr.entry_t.dt.year)}
        base['by_side'] = {int(s): (len(g), round(pf(g.R), 2)) for s, g in tr.groupby('d')}
        base['exit_reasons'] = tr.reason.value_counts().to_dict()
        print(mk, 'BASE', json.dumps(base))
        nb = {}
        for name, p in neighbours().items():
            h = hl(simulate(mk, strat(mk, p)))
            nb[name] = {k: h[k] for k in ('n', 'pf', 'pf_IS', 'pf_OOS')}
            print(f'  {mk} {name:22s}', nb[name])
        pfs = [v['pf'] for v in nb.values()]
        fam = dict(median_pf=round(float(np.median(pfs)), 3), n_variants=len(pfs), frac_gt_1_1=round(float(np.mean(np.array(pfs) > 1.1)), 2),
                   median_IS=round(float(np.median([v['pf_IS'] for v in nb.values()])), 3),
                   median_OOS=round(float(np.median([v['pf_OOS'] for v in nb.values()])), 3))
        print(mk, 'NEIGHBOUR FAMILY', fam)
        ctl = controls(mk)
        print(mk, 'CONTROLS', json.dumps(ctl))
        ctl200 = controls(mk, n_rep=200, seed=11)
        print(mk, 'CONTROLS 200 reps', json.dumps(ctl200))
        allday = broad_fade(mk)
        print(mk, 'same fade rule on ALL days (no news filter)', allday)
        # same rule by volume-rank bucket, IS vs OOS (is the edge news-specific and monotonic?)
        asig, at = strat(mk, dict(BASE, q=-1.0), ret_days=True); atr_ = simulate(mk, asig)
        atr_['b'] = pd.cut(at.set_index('k').loc[atr_.k, 'rank'].to_numpy(), [0, .5, .8, .9, .95, 1.01], right=False)
        atr_['seg'] = np.where(atr_.entry_t < np.datetime64('2025-04-01'), 'IS', 'OOS')
        buckets = {str(b): {s: (len(g), round(pf(g.R), 2)) for s, g in gg.groupby('seg')} for b, gg in atr_.groupby('b', observed=True)}
        print(mk, 'rank buckets', buckets)
        Rs = np.sort(tr.R.to_numpy())[::-1]
        conc = dict(drop_top3=round(pf(Rs[3:]), 3), drop_top5=round(pf(Rs[5:]), 3))
        print(mk, 'concentration', conc)
        R[mk] = dict(base=base, neighbours=nb, family=fam, controls=ctl, controls_200=ctl200, all_days=allday,
                     rank_buckets=buckets, concentration=conc)
    json.dump(R, open('/home/user/TRADING/scalp/results/news_0830_two_leg_verify.json', 'w'), indent=1, default=str)
