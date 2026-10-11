"""lunch_effect_drift -- Lunch effect: post-lunch upward drift after a weak or flat morning
(Vojtko & Dujava 2024, SSRN 4934614 / Quantpedia "Lunch effect in the U.S. stock market indices").

Hypothesis: volume/volatility trough around 12:00-13:00; morning selling pauses and VWAP/TWAP execution and buy-side
flow resume into the afternoon -> upward drift from late morning into early/mid afternoon, strongest after a weak or
flat morning.

Mechanical rules (as implemented):
  * Signal bar k = the 1-minute bar with NY minute-of-day == T_in-1 (T_in 690 = 11:30 or 720 = 12:00); its close is
    known at T_in. Bar k+1 must be on the same date and start no later than T_in+2.
  * M = (c[k] - rth_open[k]) / atr[k]  (RTH 09:30 open of today; atr = 14-day mean RTH range of PREVIOUS days).
  * Condition m_thr: 'none' (all days), 'le0' (M <= 0), 'le-0.2' (M <= -0.2).
  * d = +1 (long), market entry at the open of k+1 (etype 0). stop_pts = max(0.15*ATR, 20 pts NQ / 4 pts ES). No target.
  * Time exit at the OPEN of bar T_out (810 13:30 / 840 14:00 / 870 14:30): j = first bar of the same date with
    mod >= T_out, required mod[j] <= T_out+5 (half-days / data holes skipped); maxbars = j - (k+1).
    Using the bar index of a clock time is not price lookahead (the exit time is fixed in advance).
Grid = T_in {11:30, 12:00} x T_out {13:30, 14:00, 14:30} x m_thr {none, <=0, <=-0.2} = 18 variants (budget 24).
Selection (pre-registered, IS only): among the 12 CONDITIONAL cells, score = mean over the cell and its T_out
neighbours (same T_in, same m_thr) of min(IS PF NQ, IS PF ES); highest score wins, tie-break min IS PF.
Controls: (a) unconditional cell with same T_in/T_out (long-drift baseline);
          (b) SHORT the same window when M >= +0.2 ATR;
          (c) random long windows between 11:00 and 15:00 on the SAME days as the chosen cell (2-hour, as specified,
              and matched length), same stop rule, 40 draws, median PF IS / OOS.
Edge counts only if the chosen conditional cell beats (a) and (c) on BOTH markets in IS and OOS.
"""
import sys, json
sys.path.insert(0, '/home/user/TRADING/scalp')
import numpy as np
import pandas as pd
from engine import load, signals_frame, simulate, headline, full_report, pf, IS_END

MINSTOP = {'NQ': 20.0, 'ES': 4.0}
TINS = (690, 720); TOUTS = (810, 840, 870); MTHR = ('none', 'le0', 'le-0.2')
OUT = '/home/user/TRADING/scalp/results/lunch_effect_drift.json'
IS_WEEKS = (np.datetime64('2025-04-01') - np.datetime64('2023-10-01')).astype(int) / 7.0
hm = lambda m: f'{m // 60:02d}:{m % 60:02d}'


def exit_idx(D, k, t_out):
    """Index of the first bar on the same date with mod >= t_out (only clock time, no prices). -1 if missing."""
    tgt = D['t'][k] - D['mod'][k] + t_out
    j = np.searchsorted(D['t'], tgt)
    j = np.minimum(j, len(D['t']) - 1)
    ok = (D['dayi'][j] == D['dayi'][k]) & (D['mod'][j] >= t_out) & (D['mod'][j] <= t_out + 5)
    return np.where(ok, j, -1)


def base(mk, t_in):
    D = load(mk)
    k = np.where(D['mod'] == t_in - 1)[0]
    k = k[k + 1 < len(D['mod'])]
    k = k[(D['dayi'][k + 1] == D['dayi'][k]) & (D['mod'][k + 1] <= t_in + 2)]
    M = (D['c'][k] - D['rth_open'][k]) / D['atr'][k]
    ok = np.isfinite(M)
    return D, k[ok], M[ok]


def cond(M, m_thr):
    return {'none': np.ones(len(M), bool), 'le0': M <= 0, 'le-0.2': M <= -0.2, 'ge0.2': M >= 0.2}[m_thr]


def signals(mk, t_in=720, t_out=840, m_thr='le0', d=1, days_only=False, s=0.15):
    D, k, M = base(mk, t_in)
    j = exit_idx(D, k, t_out)
    m = cond(M, m_thr) & (j > k + 1)
    k, j = k[m], j[m]
    if days_only:
        return D['dayi'][k]
    stop = np.maximum(s * D['atr'][k], MINSTOP[mk])
    return signals_frame(k, d, etype=0, stop_pts=stop, maxbars=j - (k + 1))


def random_windows(mk, days, L=120, lo=660, hi=900, n=40, seed=7):
    """Control (c): long windows of length L starting at a random minute in [lo, hi-L] on the given days."""
    D = load(mk); rng = np.random.default_rng(seed)
    first = {}
    # first index of each dayi
    u, fi = np.unique(D['dayi'], return_index=True); first = dict(zip(u, fi))
    d0 = np.array([first[x] for x in days]); midnight = D['t'][d0] - D['mod'][d0]
    res_is, res_oos, res_all = [], [], []
    for _ in range(n):
        s = rng.integers(lo, hi - L + 1, len(days))
        k = np.searchsorted(D['t'], midnight + s - 1); k = np.minimum(k, len(D['t']) - 2)
        okk = (D['dayi'][k] == days) & (D['mod'][k] == s - 1) & (D['dayi'][k + 1] == days)
        j = exit_idx(D, k, s + L)
        ok = okk & (j > k + 1) & np.isfinite(D['atr'][k])
        kk, jj = k[ok], j[ok]
        sig = signals_frame(kk, 1, stop_pts=np.maximum(0.15 * D['atr'][kk], MINSTOP[mk]), maxbars=jj - (kk + 1))
        h = headline(simulate(mk, sig)); res_is.append(h['pf_IS']); res_oos.append(h['pf_OOS']); res_all.append(h['pf'])
    return dict(L=L, rand_pf_med=round(float(np.median(res_all)), 3), rand_IS_med=round(float(np.median(res_is)), 3),
                rand_OOS_med=round(float(np.median(res_oos)), 3),
                rand_IS_95=round(float(np.percentile(res_is, 95)), 3), rand_OOS_95=round(float(np.percentile(res_oos, 95)), 3))


def grid():
    rows = []
    for t_in in TINS:
        for t_out in TOUTS:
            for mt in MTHR:
                r = dict(t_in=t_in, t_out=t_out, m_thr=mt)
                for mk in ('NQ', 'ES'):
                    h = headline(simulate(mk, signals(mk, t_in, t_out, mt)))
                    r[f'{mk}_n'] = h['n']; r[f'{mk}_pwIS'] = round(h['n_IS'] / IS_WEEKS, 2)
                    r[f'{mk}_IS'] = h['pf_IS']; r[f'{mk}_OOS'] = h['pf_OOS']
                rows.append(r)
    return pd.DataFrame(rows)


def select(g):
    m = g[g.m_thr != 'none'].copy()
    m['minIS'] = m[['NQ_IS', 'ES_IS']].min(axis=1)
    sc = []
    for _, r in m.iterrows():
        i = TOUTS.index(r.t_out); nb = [TOUTS[x] for x in (i - 1, i, i + 1) if 0 <= x < len(TOUTS)]
        sc.append(m[(m.t_in == r.t_in) & (m.m_thr == r.m_thr) & m.t_out.isin(nb)].minIS.mean())
    m['plateau'] = np.round(sc, 3)
    b = m.sort_values(['plateau', 'minIS'], ascending=False).iloc[0]
    return m, dict(t_in=int(b.t_in), t_out=int(b.t_out), m_thr=b.m_thr)


def split_pf(tr, lab):
    out = {}
    seg = np.where(tr.entry_t < IS_END, 'IS', 'OOS')
    for v in pd.unique(lab):
        a = (lab == v) & (seg == 'IS'); b = (lab == v) & (seg == 'OOS')
        out[str(v)] = dict(n_IS=int(a.sum()), pf_IS=round(pf(tr.R[a]), 2), n_OOS=int(b.sum()), pf_OOS=round(pf(tr.R[b]), 2))
    return out


def labels(mk, tr, t_in):
    D = load(mk); kk = tr.k.to_numpy()
    M = (D['c'][kk] - D['rth_open'][kk]) / D['atr'][kk]
    td = D['trend_d'][kk]
    gap = D['rth_open'][kk] - D['pdc'][kk]
    rel = D['atr'][kk] / D['c'][kk]; med = np.median(rel[(tr.entry_t < IS_END).to_numpy()])
    vw = D['c'][kk] - D['vwap'][kk]
    return {'trend20': np.where(td > 0, 'up20d', np.where(td < 0, 'down20d', 'flat')),
            'M_bucket': np.asarray(pd.cut(M, [-np.inf, -0.5, -0.2, 0, 0.2, np.inf], labels=['<-0.5', '-0.5..-0.2', '-0.2..0', '0..0.2', '>0.2']).astype(str)),
            'gap': np.where(gap > 0, 'gap_up', 'gap_down'),
            'vol': np.where(rel >= med, 'high_vol', 'low_vol'),
            'vs_vwap': np.where(vw > 0, 'above_vwap', 'below_vwap'),
            'dow': np.array(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])[D['dow'][kk]]}


def partial_edges(ch):
    """Splits of the CHOSEN cell (not used for selection). Positive = PF>1 in IS and OOS on both markets."""
    sp = {}
    for mk in ('NQ', 'ES'):
        tr = simulate(mk, signals(mk, **ch)); L = labels(mk, tr, ch['t_in'])
        sp[mk] = {key: split_pf(tr, lab) for key, lab in L.items()}
    pos = []
    for key in sp['NQ']:
        for v in sp['NQ'][key]:
            a = sp['NQ'][key][v]; b = sp['ES'][key].get(v)
            if b and min(a['pf_IS'], a['pf_OOS'], b['pf_IS'], b['pf_OOS']) > 1.0:
                pos.append(f"{key}={v}: NQ {a['pf_IS']}/{a['pf_OOS']} ES {b['pf_IS']}/{b['pf_OOS']} (n IS/OOS NQ {a['n_IS']}/{a['n_OOS']})")
    return sp, pos


def grid_splits():
    """Median PF across all 18 cells by 20d-trend split (robustness of the trend partial edge)."""
    rows = []
    for mk in ('NQ', 'ES'):
        for t_in in TINS:
            for t_out in TOUTS:
                for mt in MTHR:
                    tr = simulate(mk, signals(mk, t_in, t_out, mt)); L = labels(mk, tr, t_in)
                    seg = np.where(tr.entry_t < IS_END, 'IS', 'OOS')
                    for key in ('trend20', 'vol', 'vs_vwap'):
                        for v in np.unique(L[key]):
                            for sg in ('IS', 'OOS'):
                                mm = (L[key] == v) & (seg == sg)
                                rows.append(dict(mk=mk, key=key, val=v, seg=sg, pf=pf(tr.R[mm])))
    df = pd.DataFrame(rows)
    return df.pivot_table(index=['mk', 'key', 'val'], columns='seg', values='pf', aggfunc='median').round(3)


def selfcheck(mk, ch):
    """Lookahead / timing check: entries at T_in, time exits at T_out, M recomputed from bars <= k."""
    D = load(mk); sig = signals(mk, **ch); tr = simulate(mk, sig)
    em = D['mod'][tr.entry_k]; xm = D['mod'][tr.exit_k[tr.reason == 4]]
    k = tr.k.to_numpy()[:200]
    Mc = D['c'][k] - D['o'][np.searchsorted(D['t'], D['t'][k] - D['mod'][k] + 570)]  # 09:30 open re-derived from raw bars
    Mr = D['c'][k] - D['rth_open'][k]
    return dict(entry_mod_ok=bool(np.all(em == ch['t_in'])) or f'{(em != ch["t_in"]).sum()} off',
                exit_mod_ok=f'{(xm == ch["t_out"]).mean():.3f} of time exits exactly at T_out',
                rth_open_rederived_maxdiff=float(np.nanmax(np.abs(Mc - Mr))),
                exit_reasons=tr.reason.value_counts().to_dict())


if __name__ == '__main__':
    pd.set_option('display.width', 200)
    g = grid(); print(g.to_string())
    med = {mk: round(float(g[f'{mk}_IS'].median()), 3) for mk in ('NQ', 'ES')}
    m, ch = select(g); print(m[['t_in', 't_out', 'm_thr', 'NQ_IS', 'ES_IS', 'minIS', 'plateau']].to_string())
    print('median IS PF all 18:', med, 'chosen:', ch)
    diag = []   # 4 extra variants (counted: 18 + 4 = 22 <= 24), NOT used for selection
    for nm, kw in (('chosen_wide_stop_0.5ATR', dict(ch, s=0.5)), ('uncond_wide_stop_0.5ATR', dict(ch, m_thr='none', s=0.5)),
                   ('paper_1100_1400_le0', dict(t_in=660, t_out=840, m_thr='le0')), ('paper_1100_1400_none', dict(t_in=660, t_out=840, m_thr='none'))):
        r = dict(name=nm)
        for mk in ('NQ', 'ES'):
            h = headline(simulate(mk, signals(mk, **kw))); r[f'{mk}_n'] = h['n']; r[f'{mk}_IS'] = h['pf_IS']; r[f'{mk}_OOS'] = h['pf_OOS']
        diag.append(r)
    dg = pd.DataFrame(diag); print(dg.to_string())
    med_all = {mk: round(float(np.median(np.r_[g[f'{mk}_IS'], dg[f'{mk}_IS']])), 3) for mk in ('NQ', 'ES')}
    print('median IS PF all 22 variants:', med_all)
    res = dict(rules=__doc__, grid=g.to_dict('records'), median_IS_pf_grid18=med, median_IS_pf_all22=med_all, chosen=ch,
               diagnostics=diag, variants_tested=22)
    ctrl = {}
    for mk in ('NQ', 'ES'):
        rep = full_report(mk, signals(mk, **ch)); res[mk] = rep
        print(mk, json.dumps(rep, default=str))
        ha = headline(simulate(mk, signals(mk, ch['t_in'], ch['t_out'], 'none')))
        hb = headline(simulate(mk, signals(mk, ch['t_in'], ch['t_out'], 'ge0.2', d=-1)))
        hbl = headline(simulate(mk, signals(mk, ch['t_in'], ch['t_out'], 'ge0.2', d=1)))
        days = signals(mk, **ch, days_only=True)
        alld = signals(mk, ch['t_in'], ch['t_out'], 'none', days_only=True)
        L = ch['t_out'] - ch['t_in']
        ctrl[mk] = dict(a_uncond=dict(n=ha['n'], IS=ha['pf_IS'], OOS=ha['pf_OOS']),
                        b_short_strong_morning=dict(n=hb['n'], IS=hb['pf_IS'], OOS=hb['pf_OOS']),
                        b2_long_strong_morning=dict(n=hbl['n'], IS=hbl['pf_IS'], OOS=hbl['pf_OOS']),
                        c_random_2h_same_days=random_windows(mk, days, 120),
                        c_random_matchedL_same_days=random_windows(mk, days, L),
                        c_random_2h_all_days=random_windows(mk, alld, 120),
                        c_random_2h_same_days_after_signal=random_windows(mk, days, 120, lo=ch['t_in']))
        print(mk, 'controls', json.dumps(ctrl[mk]))
        res[mk + '_selfcheck'] = selfcheck(mk, ch); print(mk, 'selfcheck', res[mk + '_selfcheck'])
    res['controls'] = ctrl
    sp, pos = partial_edges(ch); res['splits'] = sp; res['partial_edges_positive_both'] = pos
    print('partial edges (PF>1 IS&OOS both mkts):'); [print('  ', p) for p in pos]
    gs = grid_splits(); print(gs.to_string()); res['grid_splits_median'] = {str(k): v for k, v in gs.to_dict('index').items()}
    json.dump(res, open(OUT, 'w'), default=str, indent=1)
