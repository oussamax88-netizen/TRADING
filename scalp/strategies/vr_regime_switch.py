"""Variance-ratio regime switch on gamma's VWAP 2-sigma band breaks.  Slug: vr_regime_switch.  NQ + ES, same rules.

Hypothesis: whether following or fading a VWAP-band break pays depends on the realized autocorrelation of intraday
returns, not on the volatility level.  A causal Lo-MacKinlay variance ratio of recent RTH 5-min returns measures it:
VR > 1 (positive autocorrelation) -> follow the break; VR < 1 (negative autocorrelation) -> fade it.

Mechanical rules as implemented
-------------------------------
Full RTH day: an RTH date (calendar date, NY) with a 09:30 bar (mod 570) and a last RTH bar at mod >= 955 (half days
   ending 13:00-13:15 are excluded from the VR pool; they can still be traded).
5-min marks: P0 = open of the 09:30 bar; P_j (j = 1..78) = close of the last RTH bar with mod <= 570 + 5j - 1 (i.e. the
   last bar CLOSING at or before 09:30 + 5j min; bars are stamped by their open minute).  r_j = ln(P_j / P_{j-1}).
Per full day i: s1_i = sum_j r_j^2 (78 terms); sq_i = sum over the 78/q non-overlapping blocks of (block sum)^2.
Trailing VR for date D: pool the N full days strictly earlier than D (no today data):
   sigma1^2 = sum s1 / (78N); sigmaq^2 = sum sq / (78N/q); VR = sigmaq^2 / (q sigma1^2) = sum sq / sum s1.
   Fewer than N prior full days -> no regime.  (Raw, non-demeaned returns; the drift term is negligible at 5 min.)
Regime code per bar (from its date): -1 TREND if VR >= 1+delta; +1 REVERT if VR <= 1-delta; 0 NONE (no trade).
Signals = gamma_regime_vwap.triggers(mk, K=2.0) (first close beyond VWAP +/- 2 sd after a close inside, 10:30-15:00).
REVERT -> gamma fade arm  (d=-s, tgt |c-vwap|, stop max(0.12 ATR, 20/4, 1.0 sd), maxbars 60).
TREND  -> gamma follow arm (d=+s, stop max(0.12 ATR, 20/4, |c-vwap|), exit fexit '1559' or 'TRAIL' = first close back
          across VWAP).  The code is written into gamma prep()['reg'][key]; candidates(fade_on=1, follow_on=-1) -> cap2.
Grid (24): N {10,20,40} x q {3,6} x delta {0, 0.1} x fexit {1559, TRAIL}.  K = 2.0 fixed.
Selection (IS only): eligible if >= 100 IS trades on both markets; score = min(NQ IS PF, ES IS PF); plateau = mean score of
   the variant and its N-neighbours (same q/delta/fexit); max plateau wins, tie -> own score.
Controls (never selectable): inverted switch, gamma both1559 (ATR regime) reproduction, always-fade / always-follow,
   single arms in each regime, IS persistence of VR, arm split, 2x cost, by year, random-time control.
"""
import sys, json
sys.path.insert(0, '/home/user/TRADING/scalp'); sys.path.insert(0, '/home/user/TRADING/scalp/strategies')
import numpy as np
import pandas as pd
from engine import load, simulate, full_report, pf, IS_END, START
import gamma_regime_vwap as G

OUT = '/home/user/TRADING/scalp/results/vr_regime_switch.json'
K = 2.0
NS, QS, DELTAS, FEXITS = (10, 20, 40), (3, 6), (0.0, 0.1), ('1559', 'TRAIL')
IS_WEEKS = (IS_END - START).astype(int) / 7.0
_R = {}


def day_returns(mk):
    """Full RTH dates and their 78 five-minute log returns (n_days x 78)."""
    if mk in _R:
        return _R[mk]
    D = load(mk); mod, date, o, c = D['mod'], D['date'], D['o'], D['c']
    rth = np.where((mod >= 570) & (mod < 960))[0]
    dts, st = np.unique(date[rth], return_index=True)
    en = np.r_[st[1:], len(rth)]
    fd, rets = [], []
    for d, a, b in zip(dts, st, en):
        idx = rth[a:b]; m = mod[idx] - 570
        if m[0] != 0 or m[-1] < 385:
            continue
        slot = np.full(390, np.nan); slot[m] = c[idx]
        slot = pd.Series(slot).ffill().to_numpy()
        P = np.r_[o[idx[0]], slot[4::5]]            # slot 5j-1 = bar 09:30+5j-1 closes at the mark
        fd.append(d); rets.append(np.diff(np.log(P)))
    _R[mk] = (np.array(fd), np.array(rets))
    return _R[mk]


def day_stats(mk, q):
    fd, r = day_returns(mk)
    s1 = (r ** 2).sum(1)
    sq = (r.reshape(len(r), 78 // q, q).sum(2) ** 2).sum(1)
    return fd, s1, sq


def trailing_vr(mk, N, q, dates):
    """VR for each date in `dates` from the N full days strictly before it (NaN if fewer)."""
    fd, s1, sq = day_stats(mk, q)
    c1 = np.r_[0, np.cumsum(s1)]; cq = np.r_[0, np.cumsum(sq)]
    i = np.searchsorted(fd, dates, side='left')             # number of full days strictly earlier
    ok = i >= N; j = np.where(ok, i, N)
    return np.where(ok, (cq[j] - cq[j - N]) / (c1[j] - c1[j - N]), np.nan)


def regime_key(mk, N, q, delta):
    key = f'VR_N{N}_q{q}_d{delta}'
    P = G.prep(mk)
    if key not in P['reg']:
        D = P['D']; ud, inv = np.unique(D['date'], return_inverse=True)
        vr = trailing_vr(mk, N, q, ud)[inv]
        P['reg'][key] = np.where(vr >= 1 + delta, -1, np.where(vr <= 1 - delta, 1, 0)).astype(int)
        P['reg'][key + '_ALL'] = np.where(np.isfinite(vr), 1, 0).astype(int)   # every day with a defined VR
    return key


def cands(mk, v, fade_on=1, follow_on=-1, fexit=None, allday=False):
    key = regime_key(mk, v['N'], v['q'], v['delta']) + ('_ALL' if allday else '')
    return G.candidates(mk, K=K, regdef=key, fade_on=fade_on, follow_on=follow_on, fexit=fexit or v['fexit'])


def run(mk, cand):
    tr = simulate(mk, G.to_sig(cand))
    if len(cand):
        m = cand.set_index('k'); tr['arm'] = m.loc[tr.k, 'arm'].to_numpy()
    tr['IS'] = tr.entry_t < IS_END
    return tr


def isoos(tr):
    a = tr[tr.IS]; b = tr[~tr.IS]
    return dict(n_IS=len(a), pf_IS=round(pf(a.R), 3), n_OOS=len(b), pf_OOS=round(pf(b.R), 3))


VARIANTS = [dict(N=N, q=q, delta=d, fexit=f) for N in NS for q in QS for d in DELTAS for f in FEXITS]
vname = lambda v: f"N{v['N']}_q{v['q']}_d{v['delta']}_{v['fexit']}"


def run_grid(fade_on=1, follow_on=-1):
    rows = []
    for v in VARIANTS:
        r = dict(name=vname(v), **v)
        for mk in ('NQ', 'ES'):
            x = isoos(run(mk, cands(mk, v, fade_on, follow_on)))
            r.update({f'{mk}_{k}': val for k, val in x.items()})
        rows.append(r)
    g = pd.DataFrame(rows)
    g['score'] = g[['NQ_pf_IS', 'ES_pf_IS']].min(axis=1)
    pl = []
    for _, r in g.iterrows():
        i = NS.index(r.N); nb = [NS[j] for j in (i - 1, i, i + 1) if 0 <= j < len(NS)]
        s = g[(g.q == r.q) & (g.delta == r.delta) & (g.fexit == r.fexit) & g.N.isin(nb)]
        pl.append(s.score.mean())
    g['plateau'] = np.round(pl, 3)
    return g


def persistence(mk, delta):
    """IS only (2023-10-01 .. 2025-03-31): corr(trailing VR, same-day realized VR); regime persistence."""
    out = {}
    for N in NS:
        for q in QS:
            fd, s1, sq = day_stats(mk, q)
            rv = sq / s1; tv = trailing_vr(mk, N, q, fd)
            m = (fd >= START.astype(int)) & (fd < IS_END.astype(int)) & np.isfinite(tv)
            code = lambda x: np.where(x >= 1 + delta, -1, np.where(x <= 1 - delta, 1, 0))
            rc, tc = code(rv[m]), code(tv[m])
            p = np.array([(rc == z).mean() for z in (-1, 0, 1)])
            out[f'N{N}_q{q}'] = dict(
                days=int(m.sum()), corr_trailVR_vs_sameday=round(float(np.corrcoef(tv[m], rv[m])[0, 1]), 3),
                spearman=round(float(pd.Series(tv[m]).corr(pd.Series(rv[m]), method='spearman')), 3),
                mean_sameday_VR=round(float(rv[m].mean()), 3), mean_trail_VR=round(float(tv[m].mean()), 3),
                share_TREND_trail=round(float((tc == -1).mean()), 3),
                realized_regime_eq_prev_day=round(float((rc[1:] == rc[:-1]).mean()), 3),
                expected_if_indep=round(float((p ** 2).sum()), 3),
                trail_regime_eq_sameday_regime=round(float((tc[tc != 0] == rc[tc != 0]).mean()), 3),
                lag1_corr_sameday_VR=round(float(np.corrcoef(rv[m][1:], rv[m][:-1])[0, 1]), 3))
    return out


def vr_lookahead_check(mk, v, ncheck=60, seed=5):
    """Recompute the regime for random dates from bars strictly BEFORE that date only and compare."""
    D = load(mk); P = G.prep(mk); key = regime_key(mk, v['N'], v['q'], v['delta'])
    rng = np.random.default_rng(seed); ud = np.unique(D['date'])
    for d in rng.choice(ud[60:], ncheck, replace=False):
        cut = np.searchsorted(D['date'], d)                 # first bar of date d
        mod, date, o, c = D['mod'][:cut], D['date'][:cut], D['o'][:cut], D['c'][:cut]
        rd = [x for x in np.unique(date[(mod >= 570) & (mod < 960)])][::-1]
        s1 = sq = 0.0; got = 0
        for x in rd:
            idx = np.where((date == x) & (mod >= 570) & (mod < 960))[0]
            if mod[idx[0]] != 570 or mod[idx[-1]] < 955:
                continue
            P_ = [o[idx[0]]] + [c[idx[mod[idx] <= 570 + 5 * j - 1][-1]] for j in range(1, 79)]
            r = np.diff(np.log(P_)); s1 += (r ** 2).sum(); sq += (r.reshape(-1, v['q']).sum(1) ** 2).sum(); got += 1
            if got == v['N']:
                break
        exp = 0 if got < v['N'] else (-1 if sq / s1 >= 1 + v['delta'] else (1 if sq / s1 <= 1 - v['delta'] else 0))
        assert P['reg'][key][cut] == exp, (mk, d, P['reg'][key][cut], exp)
    return True


def bys(rep):
    return ' '.join(f"{y}:{x['pf']}({x['n']})" for y, x in rep['by_year'].items())


if __name__ == '__main__':
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
    res = dict(slug='vr_regime_switch', K=K, variants_tested=len(VARIANTS))
    g = run_grid()
    cols = ['name', 'NQ_n_IS', 'NQ_pf_IS', 'ES_n_IS', 'ES_pf_IS', 'score', 'plateau']
    print('=== GRID: switch (IS used for selection) ===\n' + g[cols].to_string(index=False))
    med = {mk: round(float(g[f'{mk}_pf_IS'].median()), 3) for mk in ('NQ', 'ES')}
    print('median IS PF of all 24:', med)
    elig = g[(g.NQ_n_IS >= 100) & (g.ES_n_IS >= 100)]
    best = elig.sort_values(['plateau', 'score'], ascending=False).iloc[0]
    ch = dict(N=int(best.N), q=int(best.q), delta=float(best.delta), fexit=best.fexit)
    print('CHOSEN on IS:', vname(ch), 'plateau', best.plateau, 'score', best.score)
    res.update(chosen=vname(ch), median_IS_pf_all24=med, grid=g.to_dict('records'))

    gi = run_grid(fade_on=-1, follow_on=1)          # (a) inverted switch, whole grid (placebo, informational)
    cmp = g[['name']].copy()
    for mk in ('NQ', 'ES'):
        for h in ('IS', 'OOS'):
            cmp[f'{mk}_{h}_sw'] = g[f'{mk}_pf_{h}']; cmp[f'{mk}_{h}_inv'] = gi[f'{mk}_pf_{h}']
    cmp['sw_beats_inv_all4'] = np.all([cmp[f'{mk}_{h}_sw'] > cmp[f'{mk}_{h}_inv'] for mk in ('NQ', 'ES') for h in ('IS', 'OOS')], axis=0)
    print('\n=== switch vs INVERTED, all 24 (OOS informational) ===\n' + cmp.to_string(index=False))
    res['grid_inverted_cmp'] = cmp.to_dict('records')
    res['n_variants_switch_beats_inverted_all4'] = int(cmp.sw_beats_inv_all4.sum())

    G.MODES['VRSW'] = dict(fade_on=1, follow_on=-1, fexit=ch['fexit'])
    key = regime_key('NQ', ch['N'], ch['q'], ch['delta']); regime_key('ES', ch['N'], ch['q'], ch['delta'])
    ctl = {}
    for mk in ('NQ', 'ES'):
        G.lookahead_check(mk, dict(K=K)); vr_lookahead_check(mk, ch)
        cand = cands(mk, ch); sig = G.to_sig(cand)
        rep = full_report(mk, sig); rep['by_year_str'] = bys(rep)
        rep['random_time_control_med_95'] = G.random_time_control(mk, dict(K=K, regdef=key, mode='VRSW'))
        tr = run(mk, cand)
        rep['arm_split'] = {a: isoos(tr[tr.arm == a]) for a in ('fade', 'follow')}
        res[mk] = rep
        print(f'\n=== {mk} full_report {vname(ch)} ===\n' + json.dumps(rep, default=str))
        c = {'switch': isoos(tr),
             'INVERTED(fade TREND, follow REVERT)': isoos(run(mk, cands(mk, ch, -1, 1))),
             'follow_only_TREND': isoos(run(mk, cands(mk, ch, None, -1))),
             'fade_only_REVERT': isoos(run(mk, cands(mk, ch, 1, None))),
             'follow_only_REVERT': isoos(run(mk, cands(mk, ch, None, 1))),
             'fade_only_TREND': isoos(run(mk, cands(mk, ch, -1, None))),
             'always_fade(all VR days)': isoos(run(mk, cands(mk, ch, 1, None, allday=True))),
             'always_follow(all VR days)': isoos(run(mk, cands(mk, ch, None, 1, allday=True))),
             'gamma_both1559_ATR_repro': isoos(run(mk, G.candidates(mk, K=K, regdef='ATR', fade_on=1, follow_on=-1, fexit='1559'))),
             'switch_2x_cost': isoos(simulate(mk, sig, costmult=2.0).assign(IS=lambda x: x.entry_t < IS_END))}
        ctl[mk] = c
        print(f'--- {mk} controls ---')
        for k_, v_ in c.items():
            print(f'  {k_:38s} IS {v_["pf_IS"]}({v_["n_IS"]})  OOS {v_["pf_OOS"]}({v_["n_OOS"]})')
    res['controls'] = ctl
    beat = {mk: {h: ctl[mk]['switch'][f'pf_{h}'] > ctl[mk]['INVERTED(fade TREND, follow REVERT)'][f'pf_{h}'] for h in ('IS', 'OOS')} for mk in ('NQ', 'ES')}
    res['switch_beats_inverted'] = beat; print('switch beats inverted:', beat)
    res['persistence_IS'] = {mk: persistence(mk, ch['delta']) for mk in ('NQ', 'ES')}
    print('\n=== IS persistence (delta %.1f) ===' % ch['delta'])
    for mk in ('NQ', 'ES'):
        for k_, v_ in res['persistence_IS'][mk].items():
            print(' ', mk, k_, v_)

    # partial edges: subsets fixed in advance; reported only if PF >= 1.1 in BOTH halves with >= 30 trades each
    pe = {}
    for mk in ('NQ', 'ES'):
        D = load(mk); tr = run(mk, cands(mk, ch)); tr['mod'] = D['mod'][tr.k]; tr['trend'] = D['trend_d'][tr.k]
        tr['nth'] = tr.groupby(D['dayi'][tr.k]).cumcount() + 1
        subs = {'fade_arm': tr.arm == 'fade', 'follow_arm': tr.arm == 'follow', 'long': tr.d == 1, 'short': tr.d == -1,
                'fade_long': (tr.arm == 'fade') & (tr.d == 1), 'fade_short': (tr.arm == 'fade') & (tr.d == -1),
                'follow_long': (tr.arm == 'follow') & (tr.d == 1), 'follow_short': (tr.arm == 'follow') & (tr.d == -1),
                '10:30-12:00': tr['mod'] < 720, '12:00-13:30': (tr['mod'] >= 720) & (tr['mod'] < 810), '13:30-15:00': tr['mod'] >= 810,
                'first_of_day': tr.nth == 1, 'second_of_day': tr.nth == 2,
                'with_20d_trend': tr.d == tr.trend, 'against_20d_trend': tr.d == -tr.trend}
        pe[mk] = {n: isoos(tr[m]) for n, m in subs.items()}
    res['partial_edges_table'] = pe
    pos = [f'{mk}:{n} IS {x["pf_IS"]}({x["n_IS"]}) OOS {x["pf_OOS"]}({x["n_OOS"]})' for mk in pe for n, x in pe[mk].items()
           if x['n_IS'] >= 30 and x['n_OOS'] >= 30 and x['pf_IS'] >= 1.1 and x['pf_OOS'] >= 1.1]
    gpos = [r['name'] for r in res['grid'] if all(r[f'{mk}_pf_{h}'] >= 1.1 for mk in ('NQ', 'ES') for h in ('IS', 'OOS'))]
    res['partial_edges_positive_both_halves'] = pos; res['grid_variants_IS_OOS_ge_1.1_both_markets'] = gpos
    print('\npartial edges >=1.1 both halves:', pos, '\ngrid variants >=1.1 IS&OOS both mkts:', gpos)

    ok13 = lambda r: r['pf_IS'] >= 1.3 and r['pf_OOS'] >= 1.3
    passed = any(ok13(res[a]) and res[b]['pf_OOS'] >= 1.1 and res[a]['per_week'] >= 3 and res[a]['pf_2x_cost'] >= 1.1
                 for a, b in (('NQ', 'ES'), ('ES', 'NQ')))
    border = any(res[m]['pf_IS'] >= 1.15 and res[m]['pf_OOS'] >= 1.15 for m in ('NQ', 'ES'))
    res['verdict'] = 'PASS' if passed else ('BORDERLINE' if border else 'FAIL')
    print('VERDICT:', res['verdict'])
    json.dump(res, open(OUT, 'w'), default=str, indent=1)
    print('saved', OUT)
