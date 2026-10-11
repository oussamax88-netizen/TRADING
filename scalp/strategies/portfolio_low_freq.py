"""portfolio_low_freq -- Low-frequency survivors portfolio: pre-market fades + ORG gap fill (no re-tuning).

Components (parameters FIXED from the round-1 files; each round-1 signals() is called exactly as written):
  PSR  premarket_selloff_rebound.signals(mk, thr=0.4, T=570, stopk=0.5, side='long')   k = 02:59 bar, entry 03:00
  LOF  london_open_fade.signals(mk, W=209, thr=0.15, X=570, sl=0.35)                  k = 03:29 bar, entry 03:30
  NEWS news_0830_two_leg.signals(mk, 'FADE', 0.9, stopk=0.25, target='prenews')       k = 08:44 bar, entry 08:45
  OGR  org_gap_rejection.signals(mk, gmin=0.3, lev='FULL', entry='A', ex='close')      k = 09:59 bar, entry 10:00
  CF   close_30m_fade.signals(mk, ref='vwap', thr=0.5, s=0.15, kmod=929)  (V3 only, info)  k = 15:29, entry 15:30
Combination (as implemented): per market the component signal frames are concatenated (plus a 'comp' tag), sorted by k
(stable) and run through ONE engine.simulate. The engine holds one position at a time and skips a signal whose entry
bar is not after the previous exit bar -> chronological first-come priority PSR > LOF > NEWS > OGR (> CF).
Signal bars are distinct minutes per component, so k is unique and trades are mapped back to a component via k.
Every trade is 1R (R = pts/risk). The two-market book = union of NQ and ES trades (separate accounts per market).
Variants (pre-declared, no parameter search): V1 PRIMARY = PSR+LOF+NEWS+OGR first-come; V2 = PSR+LOF+NEWS;
V3 = V1 + CF (info, CF ES OOS 0.95 does not qualify); V4 = V1 as independent sleeves, overlap allowed (union of the
standalone trades; info, needs a hedging account).
Interpretations: 'trading days' for daily-R correlations = NY dates with a 09:30 bar in 2023-10-01..2026-09-30;
daily R = sum of R of trades ENTERED that date (all trades are intraday), 0 on no-trade days. Correlation criterion
uses the all-trading-days Pearson; co-trade-day Pearson is reported as info. AZL exit time is unknown in the pickle ->
treated as held to 15:59 for the time-overlap count (it is a hold-to-close strategy).
Own control (NEWS uses tgt_px so full_report skips random direction): NEWS target converted to the same point
distance from the 08:45 fill (|c[08:29] - o[08:45]|; signals whose target is already behind the fill are dropped,
the engine skips them anyway; the converted frame reproduces the strategy exactly), then 200 reps of a random +-1 per
signal through the same one-position engine (same stops, time exits, target distance). Opposite direction likewise.
Lookahead: the round-1 signal functions were audited in round 1; here every trade is asserted to enter at k+1 on the
same date as k, exit the same date, at the component's signal minute, with stop >= 20 pts NQ / 4 pts ES.
"""
import sys, json
sys.path.insert(0, '/home/user/TRADING/scalp'); sys.path.insert(0, '/home/user/TRADING/scalp/strategies')
import numpy as np
import pandas as pd
from engine import load, simulate, headline, full_report, pf, IS_END, START, END, WEEKS
import premarket_selloff_rebound as m_psr, london_open_fade as m_lof, news_0830_two_leg as m_news
import org_gap_rejection as m_ogr, close_30m_fade as m_cf, gap_go_midnight as m_gg

RES = '/home/user/TRADING/scalp/results'
OUT = f'{RES}/portfolio_low_freq.json'
AZL = '/tmp/claude-0/-home-user-TRADING/b9cb08f9-c0e5-5011-839c-6894039a978b/scratchpad/AZL_trades.pkl'
MKS = ('NQ', 'ES')
FLOOR = {'NQ': 20.0, 'ES': 4.0}
COMP = {'PSR': lambda mk: m_psr.signals(mk, thr=0.4, T=570, stopk=0.5, side='long'),
        'LOF': lambda mk: m_lof.signals(mk, W=209, thr=0.15, X=570, sl=0.35),
        'NEWS': lambda mk: m_news.signals(mk, 'FADE', 0.9, stopk=0.25, target='prenews'),
        'OGR': lambda mk: m_ogr.signals(mk, gmin=0.3, lev='FULL', entry='A', ex='close'),
        'CF': lambda mk: m_cf.signals(mk, ref='vwap', thr=0.5, s=0.15, kmod=929)}
KMOD = {'PSR': (179, 179), 'LOF': (181, 209), 'NEWS': (524, 524), 'OGR': (599, 599), 'CF': (929, 929)}
REF = {'PSR': ('premarket_selloff_rebound.json', None), 'LOF': ('london_open_fade.json', 'report'),
       'NEWS': ('news_0830_two_leg.json', None), 'OGR': ('org_gap_rejection.json', None), 'CF': ('close_30m_fade.json', None)}
V1 = ('PSR', 'LOF', 'NEWS', 'OGR')
VARS = {'V1': V1, 'V2': ('PSR', 'LOF', 'NEWS'), 'V3': V1 + ('CF',), 'V4': V1}
D0, D1 = int(START.astype('datetime64[D]').astype(int)), int(END.astype('datetime64[D]').astype(int))
_S = {}


def r3(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 3)


def sig_c(mk, c):
    if (mk, c) not in _S:
        _S[(mk, c)] = COMP[c](mk).assign(comp=c)
    return _S[(mk, c)]


def combo(mk, comps):
    s = pd.concat([sig_c(mk, c) for c in comps], ignore_index=True).sort_values('k', kind='stable').reset_index(drop=True)
    assert s.k.is_unique
    return s


def sim(mk, sig, **kw):
    return simulate(mk, sig, **kw).merge(sig[['k', 'comp']], on='k', how='left', validate='one_to_one')


def sleeves(mk, comps, **kw):
    return pd.concat([sim(mk, sig_c(mk, c), **kw) for c in comps]).sort_values(['entry_t', 'k']).reset_index(drop=True)


def seg(tr):
    i, o = tr[tr.entry_t < IS_END], tr[tr.entry_t >= IS_END]
    return dict(n=len(tr), per_week=round(len(tr) / WEEKS, 2), pf=r3(pf(tr.R)), n_IS=len(i), pf_IS=r3(pf(i.R)),
                n_OOS=len(o), pf_OOS=r3(pf(o.R)))


def ystr(tr):
    return ' '.join(f'{y}:{pf(g.R):.2f}({len(g)})' for y, g in tr.groupby(tr.entry_t.dt.year))


def maxdd(R):
    eq = np.cumsum(np.asarray(R, float))
    return round(float((eq - np.maximum.accumulate(eq)).min()), 1) if len(eq) else None


def tdays(mk):
    D = load(mk); d = np.unique(D['t'][D['mod'] == 570] // 1440)
    return d[(d >= D0) & (d < D1)]


def daily(mk, tr, days):
    day = load(mk)['t'][tr.entry_k.to_numpy()] // 1440
    g = pd.DataFrame({'day': day, 'R': tr.R.to_numpy(), 'd': tr.d.to_numpy()}).groupby('day')
    return g.R.sum().reindex(days, fill_value=0.0), g.d.first()


def corr(a, b):
    return r3(np.corrcoef(a, b)[0, 1]) if len(a) > 4 and np.std(a) > 0 and np.std(b) > 0 else None


def pair_stats(A, dA, B, dB):
    co = dA.index.intersection(dB.index)
    return dict(corr_all_days=corr(A.to_numpy(), B.to_numpy()), co_days=len(co),
                corr_co_days=corr(A.loc[co].to_numpy(), B.loc[co].to_numpy()) if len(co) else None,
                same_dir_share=r3(float((dA.loc[co] == dB.loc[co]).mean())) if len(co) else None)


def pts_frame(mk, sig):
    """NEWS absolute target -> same point distance from the k+1 fill; drop rows the engine would skip anyway."""
    D = load(mk); s = sig.copy()
    m = s.tgt_px.notna().to_numpy(); k1 = np.minimum(s.k.to_numpy() + 1, len(D['o']) - 1)
    px = D['o'][k1]; dist = s.d.to_numpy() * (s.tgt_px.to_numpy() - px)
    s.loc[m, 'tgt_pts'] = np.abs(s.tgt_px.to_numpy() - px)[m]; s['tgt_px'] = np.nan
    return s[~(m & ~(dist > 0))].reset_index(drop=True)


def dir_controls(mk, sig, n=200, seed=11, perm=False, keep=None):
    """perm=False: random +-1 per signal; perm=True: shuffle the signals' own directions (keeps the long/short mix,
    i.e. controls for the 2023-26 long drift). keep: list that receives each rep's trades (for the two-market book)."""
    sp = pts_frame(mk, sig)
    a, b = sim(mk, sig), sim(mk, sp)
    assert len(a) == len(b) and np.allclose(a.R.to_numpy(), b.R.to_numpy()), 'pts-target frame does not reproduce'
    base = seg(a)
    op = sp.copy(); op['d'] = -op.d; opp = seg(simulate(mk, op))
    rng = np.random.default_rng(seed); rr = []
    for _ in range(n):
        x = sp.copy(); x['d'] = rng.permutation(sp.d.to_numpy()) if perm else np.where(rng.random(len(x)) < 0.5, 1, -1)
        t = simulate(mk, x); rr.append(seg(t))
        if keep is not None:
            keep.append(t[['entry_t', 'R']])
    out = dict(pf_opposite=opp['pf'], pf_opposite_IS=opp['pf_IS'], pf_opposite_OOS=opp['pf_OOS'], reps=n)
    for key in ('pf', 'pf_IS', 'pf_OOS'):
        v = np.array([r[key] if r[key] is not None else np.nan for r in rr], float)
        out[f'rand_{key}_median'] = r3(np.nanmedian(v)); out[f'rand_{key}_p95'] = r3(np.nanpercentile(v, 95))
        out[f'share_beaten_{key}'] = r3(np.mean(v < base[key]))
    return out


def selfcheck(mk, tr):
    D = load(mk); k, e, x = tr.k.to_numpy(), tr.entry_k.to_numpy(), tr.exit_k.to_numpy()
    assert (e == k + 1).all() and (D['dayi'][e] == D['dayi'][k]).all() and (D['dayi'][x] == D['dayi'][e]).all()
    for c, (lo, hi) in KMOD.items():
        mm = D['mod'][k[tr.comp.to_numpy() == c]]
        assert ((mm >= lo) & (mm <= hi)).all(), c
    em = D['mod'][e]
    assert ((em >= 180) & (em <= 930)).all() and (tr.risk >= FLOOR[mk] - 1e-9).all()
    return dict(entry_is_k_plus_1=True, same_day=True, signal_minutes_ok=True, entry_window_ok=True,
                min_risk_pts=r3(tr.risk.min()))


def ref_check(mk, c, st):
    f, sub = REF[c]; j = json.load(open(f'{RES}/{f}'))[mk]; j = j[sub] if sub else j
    h = headline(st); keys = ('n', 'pf', 'pf_IS', 'pf_OOS')
    return dict(match=all(h[q] == j[q] for q in keys), **{q: h[q] for q in keys}, json={q: j[q] for q in keys})


def conflicts(mk, sig, port):
    pa = sim(mk, sig, period=False).sort_values('entry_k').reset_index(drop=True)
    pe = pa.entry_k.to_numpy(); out = {}
    for c in sorted(set(sig.comp), key=list(COMP).index):
        st = sim(mk, sig_c(mk, c)); tk = st.k.isin(port.k)
        a = st[tk].merge(port[['k', 'R']], on='k', suffixes=('', '_p'))
        assert np.allclose(a.R, a.R_p)              # a taken signal trades exactly as standalone
        sk = st[~tk]; j = np.searchsorted(pe, sk.k.to_numpy(), side='right') - 1
        assert (j >= 0).all() and (pa.exit_k.to_numpy()[j] >= sk.k.to_numpy() + 1).all()
        by = pd.Series(pa.comp.to_numpy()[j]).value_counts().to_dict()
        out[c] = dict(standalone=len(st), taken=int(tk.sum()), lost=int((~tk).sum()), blocked_by=by,
                      taken_stats=seg(st[tk]), skipped_stats=seg(sk))
    return out


def corr_matrix(mk, std):
    days = tdays(mk); ser = {c: daily(mk, t, days) for c, t in std.items()}; out = {}
    names = list(std)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            out[f'{a}-{b}'] = pair_stats(ser[a][0], ser[a][1], ser[b][0], ser[b][1])
    return out


def known_edges(mk, items):
    """Overlap + daily-R correlation with the TDH proxy and AZL (trades of the known edges)."""
    D = load(mk); days = tdays(mk)
    tdh = simulate(mk, m_gg.tdh_signals(mk))
    tday = D['t'][tdh.entry_k.to_numpy()] // 1440
    T = pd.DataFrame({'day': tday, 'R': tdh.R.to_numpy(), 'd': tdh.d.to_numpy(), 'a': D['t'][tdh.entry_k.to_numpy()],
                      'b': D['t'][tdh.exit_k.to_numpy()]})
    A = pd.read_pickle(AZL); A = A[A.mk == mk]
    at = ((A.entry_t.to_numpy().astype('datetime64[m]').astype(np.int64)))
    Z = pd.DataFrame({'day': at // 1440, 'R': A.R.to_numpy(), 'd': A.side.to_numpy(), 'a': at, 'b': (at // 1440) * 1440 + 959})
    Z = Z[(Z.day >= D0) & (Z.day < D1)]
    out = dict(TDH=dict(n=len(tdh), pf=r3(pf(tdh.R))), AZL=dict(n=len(Z), pf=r3(pf(Z.R))))
    for nm, E in (('TDH', T), ('AZL', Z)):
        eg = E.groupby('day'); eR = eg.R.sum().reindex(days, fill_value=0.0); ed = eg.d.first()
        for lab, tr in items.items():
            R, dd = daily(mk, tr, days); s = pair_stats(R, dd, eR, ed)
            ta, tb = D['t'][tr.entry_k.to_numpy()], D['t'][tr.exit_k.to_numpy()]
            ov = 0
            for a_, b_, dy in zip(ta, tb, ta // 1440):
                q = E[E.day == dy]
                ov += int(((q.a <= b_) & (a_ <= q.b)).any())
            s['time_overlap_trades'] = ov
            out[f'{lab}~{nm}'] = s
    return out


def book(tr_by_mk):
    P = pd.concat([t.assign(mk=mk) for mk, t in tr_by_mk.items()]).sort_values('entry_t').reset_index(drop=True)
    return P, dict(**seg(P), maxDD_R=maxdd(P.R), netR=round(float(P.R.sum()), 1), by_year=ystr(P))


def main():
    pd.set_option('display.width', 220)
    res = dict(slug='portfolio_low_freq', name='Low-frequency survivors portfolio (pre-market fades + ORG gap fill)',
               primary='V1', variants_tested=4, markets={}, variants={}, book={})
    trades = {v: {} for v in VARS}; trades2x = {v: {} for v in VARS}; reps = {}
    for mk in MKS:
        R = res['markets'][mk] = {}
        std = {c: sim(mk, sig_c(mk, c)) for c in COMP}
        R['standalone'] = {c: dict(**seg(t), pf_2x_cost=r3(pf(sim(mk, sig_c(mk, c), costmult=2.0).R)), by_year=ystr(t),
                                   ref_json=ref_check(mk, c, t)) for c, t in std.items()}
        for v, comps in VARS.items():
            if v == 'V4':
                t, t2 = sleeves(mk, comps), sleeves(mk, comps, costmult=2.0)
            else:
                s = combo(mk, comps); t, t2 = sim(mk, s), sim(mk, s, costmult=2.0)
            trades[v][mk], trades2x[v][mk] = t, t2
            res['variants'].setdefault(v, {})[mk] = dict(**seg(t), pf_2x_cost=r3(pf(t2.R)), maxDD_R=maxdd(t.R), by_year=ystr(t))
        sig = combo(mk, V1); port = trades['V1'][mk]
        rep = full_report(mk, sig); rep['by_year_str'] = ystr(port)
        rep['pf_3x_cost'] = r3(pf(simulate(mk, sig, costmult=3.0).R))
        R['V1_full_report'] = rep
        R['V1_selfcheck'] = selfcheck(mk, port)
        R['V1_by_component'] = {c: dict(**seg(g), by_year=ystr(g)) for c, g in port.groupby('comp')}
        R['V1_by_side'] = {('long' if d == 1 else 'short'): seg(g) for d, g in port.groupby('d')}
        R['V1_conflicts'] = conflicts(mk, sig, port)
        R['V1_dir_control_200'] = dir_controls(mk, sig, keep=reps.setdefault(('rand', mk), []))
        R['V1_perm_control_200'] = dir_controls(mk, sig, perm=True, keep=reps.setdefault(('perm', mk), []))
        R['component_dir_control_200'] = {c: dir_controls(mk, sig_c(mk, c)) for c in V1}
        R['daily_R_corr_components'] = corr_matrix(mk, {c: std[c] for c in V1})
        R['daily_R_corr_with_CF_info'] = corr_matrix(mk, {c: std[c] for c in V1 + ('CF',)})
        R['known_edges'] = known_edges(mk, dict(BOOK_V1=port, **{c: std[c] for c in V1}))
        print(f'\n######## {mk}')
        print('standalone:', json.dumps({c: {q: v[q] for q in ('n', 'pf', 'pf_IS', 'pf_OOS', 'pf_2x_cost')} | {'match': v['ref_json']['match']}
                                         for c, v in R['standalone'].items()}))
        print('V1 full_report:', json.dumps({q: rep.get(q) for q in ('n', 'per_week', 'win', 'pf', 'pf_IS', 'n_IS', 'pf_OOS', 'n_OOS',
               'maxDD_R', 'med_risk_pts', 'med_hold_min', 'pf_2x_cost', 'pf_no_cost', 'pf_3x_cost', 'boot_pf_5_50_95', 'p_pf_below_1',
               'by_year_str')}))
        print('V1 by component:', json.dumps(R['V1_by_component']))
        print('V1 by side:', json.dumps(R['V1_by_side']))
        print('conflicts:', json.dumps({c: {q: v[q] for q in ('standalone', 'taken', 'lost', 'blocked_by')} |
                                        {'pf_taken': v['taken_stats']['pf'], 'pf_skipped': v['skipped_stats']['pf']}
                                        for c, v in R['V1_conflicts'].items()}, default=int))
        print('V1 dir control:', json.dumps(R['V1_dir_control_200']))
        print('V1 permutation control:', json.dumps(R['V1_perm_control_200']))
        print('component dir controls:', json.dumps({c: {q: v[q] for q in ('pf_opposite', 'rand_pf_median', 'share_beaten_pf',
              'share_beaten_pf_IS', 'share_beaten_pf_OOS')} for c, v in R['component_dir_control_200'].items()}))
        print('corr matrix:', json.dumps(R['daily_R_corr_components']))
        print('known edges:', json.dumps(R['known_edges']))
    # books
    for v in VARS:
        _, b = book(trades[v]); _, b2 = book(trades2x[v]); b['pf_2x_cost'] = b2['pf']
        res['book'][v] = b
    P1, b1 = book(trades['V1']); Rb = P1.R.to_numpy(); rng = np.random.default_rng(3)
    bs = np.array([pf(rng.choice(Rb, len(Rb))) for _ in range(2000)])
    res['book']['V1'].update(boot_pf_5_50_95=[r3(q) for q in np.percentile(bs, [5, 50, 95])], p_pf_below_1=r3(np.mean(bs < 1)),
                             days_with_exposure_per_week=round(len(np.unique(P1.entry_t.dt.normalize())) / WEEKS, 2))
    for kind in ('rand', 'perm'):
        rb = [seg(pd.concat([a, b])) for a, b in zip(reps[(kind, 'NQ')], reps[(kind, 'ES')])]
        res['book']['V1'][f'{kind}_control_200'] = {
            key: dict(median=r3(np.nanmedian([r[key] for r in rb])), p95=r3(np.nanpercentile([r[key] for r in rb], 95)),
                      share_beaten=r3(np.mean([r[key] < b1[key] for r in rb]))) for key in ('pf', 'pf_IS', 'pf_OOS')}
    days = np.intersect1d(tdays('NQ'), tdays('ES'))
    a, ad = daily('NQ', trades['V1']['NQ'], days); b_, bd = daily('ES', trades['V1']['ES'], days)
    res['book_NQ_vs_ES_daily_R'] = pair_stats(a, ad, b_, bd)
    res['median_IS_pf_all_variants'] = {mk: r3(np.median([res['variants'][v][mk]['pf_IS'] for v in VARS])) for mk in MKS}
    print('\n######## VARIANTS (per market) and BOOK')
    for v in VARS:
        print(v, json.dumps({mk: {q: res['variants'][v][mk][q] for q in ('n', 'per_week', 'pf', 'pf_IS', 'pf_OOS', 'pf_2x_cost', 'maxDD_R')}
                             for mk in MKS}), '| book:', json.dumps(res['book'][v]))
    print('NQ vs ES daily R (V1 book):', json.dumps(res['book_NQ_vs_ES_daily_R']))
    print('median IS PF of the 4 variants:', res['median_IS_pf_all_variants'])
    # ---------------- verdicts (V1 only)
    M = {mk: res['markets'][mk]['V1_full_report'] for mk in MKS}; bk = res['book']['V1']
    maxc = max(v['corr_all_days'] for mk in MKS for v in res['markets'][mk]['daily_R_corr_components'].values())
    maxc_co = max((v['corr_co_days'] or -1) for mk in MKS for v in res['markets'][mk]['daily_R_corr_components'].values())
    pre = dict(book_ge_3_per_week=bk['per_week'] >= 3,
               IS_OOS_ge_1_2_each_market=all(M[mk]['pf_IS'] >= 1.2 and M[mk]['pf_OOS'] >= 1.2 for mk in MKS),
               cost2x_ge_1_1=all(M[mk]['pf_2x_cost'] >= 1.1 for mk in MKS) and bk['pf_2x_cost'] >= 1.1,
               max_pair_corr_all_days_lt_0_3=maxc < 0.3)
    pre = {q: bool(v) for q, v in pre.items()}
    def gp(a, b):
        return M[a]['pf_IS'] >= 1.3 and M[a]['pf_OOS'] >= 1.3 and M[b]['pf_OOS'] >= 1.1 and M[a]['pf_2x_cost'] >= 1.1
    gen_pass = (gp('NQ', 'ES') or gp('ES', 'NQ')) and bk['per_week'] >= 3
    border = any(M[mk]['pf_IS'] >= 1.15 and M[mk]['pf_OOS'] >= 1.15 for mk in MKS)
    verdict = 'PASS' if (gen_pass and all(pre.values())) else ('BORDERLINE' if border else 'FAIL')
    res.update(predeclared_criteria=pre, max_pair_corr_all_days=maxc, max_pair_corr_co_days_info=maxc_co,
               generic_pass=bool(gen_pass), generic_pass_note='trades/week measured on the two-market book (portfolio by design); '
               'each market alone is ~2.1-2.3/wk', verdict=verdict)
    print('pre-declared criteria:', pre, '| max pair corr all-days', maxc, '| co-days', maxc_co)
    print('generic PASS:', gen_pass, '| VERDICT:', verdict)
    res['caveats'] = (
        'OOS is NOT clean: the 4 components were admitted because their round-1 PF was > 1 in BOTH halves on both markets '
        '(survivor selection among ~15 round-1/2 ideas), so the book OOS PF is optimistically biased; only data after '
        '2026-09-30 / live forward trading is a true test. NQ and ES books are strongly correlated (daily-R corr 0.55 all days, '
        '0.73 and 92% same direction on co-trade days) -> 4.46 trades/wk but only ~2.4 distinct exposure days/wk. '
        'Component all-day correlations are near zero partly because sparse daily series are mostly zeros; co-trade-day '
        'correlations reach 0.6-0.75 on 5-9 days (PSR-OGR, LOF-OGR NQ). ~73% of trades are long (PSR long-only); '
        'direction-permutation control (keeps the long share) is still beaten 100% full-sample / 94-97% OOS. '
        '2026 is the weakest year (book PF 1.17). Median hold 360 min: not a scalp. ES OGR is 22 trades; '
        'random-direction ES OGR median PF is 2.4 (tight stop + hold to close convexity), so its 5.1 PF is mostly structure. '
        'Conflict effect is noise: LOF skipped-vs-taken flips sign between markets (NQ 0.90 vs 1.70, ES 1.62 vs 1.12).')
    json.dump(res, open(OUT, 'w'), indent=1, default=lambda x: x.item() if hasattr(x, 'item') else str(x))
    print('saved', OUT)


if __name__ == '__main__':
    main()
