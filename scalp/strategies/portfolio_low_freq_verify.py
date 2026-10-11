"""portfolio_low_freq_verify -- independent re-implementation (skeptic, robustness lens) of portfolio_low_freq V1.

Re-written from the rule text only (no import of the round-1 component modules), same engine:
  PSR  long if prev RTH (close-open) <= -thr*ATR; k = 02:59 bar; stop max(stopk*ATR, floor); exit open of first bar >= T.
  LOF  s = 03:00 bar, e = last bar with 180<mod<=W; r=(c[e]-o[s])/ATR; |r|>=thr -> fade at k=e; stop max(sl*ATR,floor);
       exit open of first bar >= X.
  NEWS valid day = bars 08:29..08:34 + 08:44 exist; V = vol 08:30..08:34; rank = share of previous 20 valid days' V < V;
       news if rank >= q; X = c[08:44]-c[08:29], |X| >= 0.05 ATR; fade at k = 08:44, stop max(stopk*ATR,floor),
       target c[08:29] (absolute) or none; exit 09:30 open.
  OGR  R = close of prev trading date's 16:14 bar (else pdc); g = o[09:30]-R; |g| >= gmin*ATR; d=-sign(g);
       LV = R (FULL) or R+g/2 (CE); traded through in 09:30..09:59 -> k = 09:59; stop = extreme-c[k]+0.02ATR floored
       at max(0.12ATR, floor); hold to 15:59 (or tgt R-g).
  Book: concat, sort by k, one simulate per market (first-come); market union for the book.
Robustness: neighbour family (one parameter one step), random-time control (20 reps).
"""
import sys, json
sys.path.insert(0, '/home/user/TRADING/scalp')
import numpy as np
import pandas as pd
from engine import load, signals_frame, simulate, headline, pf, IS_END

FL = {'NQ': 20.0, 'ES': 4.0}
BASE = dict(psr_thr=0.4, psr_T=570, psr_sk=0.5, lof_W=209, lof_thr=0.15, lof_X=570, lof_sl=0.35,
            nw_q=0.9, nw_sk=0.25, nw_tgt='prenews', og_g=0.3, og_lev='FULL', og_ex='close')
_C = {}


def ctx(mk):
    """Per calendar day: first index of each minute (dict), day bounds."""
    if mk in _C:
        return _C[mk]
    D = load(mk); mod, di = D['mod'], D['dayi']
    nd = int(di.max()) + 1
    st = np.searchsorted(di, np.arange(nd)); en = np.searchsorted(di, np.arange(nd), side='right')
    _C[mk] = (D, st, en, nd)
    return _C[mk]


def at(mk, m):
    """index of first bar with exact minute m per day (-1 if missing)."""
    D, st, en, nd = ctx(mk); a = np.full(nd, -1, np.int64)
    w = np.where(D['mod'] == m)[0][::-1]; a[D['dayi'][w]] = w      # reversed -> first occurrence wins
    return a


def ge(mk, m):
    """index of first bar of the day with minute >= m (-1 if none)."""
    D, st, en, nd = ctx(mk); out = np.full(nd, -1, np.int64)
    for d in range(nd):
        j = st[d] + np.searchsorted(D['mod'][st[d]:en[d]], m)
        if j < en[d]:
            out[d] = j
    return out


def rthdays(mk):
    D, st, en, nd = ctx(mk); mod = D['mod']
    has = np.zeros(nd, bool); op = np.full(nd, np.nan); cl = np.full(nd, np.nan)
    for d in range(nd):
        w = np.where((mod[st[d]:en[d]] >= 570) & (mod[st[d]:en[d]] < 960))[0]
        if len(w):
            has[d] = True; op[d] = D['o'][st[d] + w[0]]; cl[d] = D['c'][st[d] + w[-1]]
    return has, op, cl


def psr(mk, thr, T, sk):
    D, st, en, nd = ctx(mk); has, op, cl = rthdays(mk); k179 = at(mk, 179); xT = ge(mk, T)
    days = np.where(has)[0]; K, S, M = [], [], []
    for a, b in zip(days[:-1], days[1:]):
        k = k179[b]
        if k < 0 or k + 1 >= len(D['o']) or D['dayi'][k + 1] != b or xT[b] < 0:
            continue
        A = D['atr'][k]; P = cl[a] - op[a]
        if not np.isfinite(A) or not np.isfinite(P) or P > -thr * A:
            continue
        mb = xT[b] - (k + 1)
        if mb <= 0:
            continue
        K.append(k); S.append(max(sk * A, FL[mk])); M.append(mb)
    return signals_frame(K, 1, stop_pts=np.array(S), maxbars=np.array(M, np.int64)).assign(comp='PSR')


def lof(mk, W, thr, X, sl):
    D, st, en, nd = ctx(mk); mod = D['mod']; s180 = at(mk, 180); xX = ge(mk, X); K, Dd, S, M = [], [], [], []
    for d in range(nd):
        s = s180[d]
        if s < 0 or xX[d] < 0:
            continue
        w = np.where((mod[st[d]:en[d]] > 180) & (mod[st[d]:en[d]] <= W))[0]
        if not len(w):
            continue
        e = st[d] + w[-1]; A = D['atr'][s]
        if not (np.isfinite(A) and A > 0):
            continue
        r = (D['c'][e] - D['o'][s]) / A
        if abs(r) < thr or xX[d] - (e + 1) < 1:
            continue
        K.append(e); Dd.append(-int(np.sign(r))); S.append(max(sl * A, FL[mk])); M.append(xX[d] - (e + 1))
    return signals_frame(K, Dd, stop_pts=np.array(S), maxbars=np.array(M, np.int64)).assign(comp='LOF')


def news(mk, q, sk, tgt):
    D, st, en, nd = ctx(mk); I = {m: at(mk, m) for m in (509, 510, 511, 512, 513, 514, 524, 570)}
    ok = np.all([I[m] >= 0 for m in (509, 510, 511, 512, 513, 514, 524)], axis=0)
    vd = np.where(ok)[0]
    V = np.array([sum(D['vol'][I[m][d]] for m in range(510, 515)) for d in vd])
    K, Dd, S, M, TP = [], [], [], [], []
    for j in range(20, len(vd)):
        d = vd[j]
        if I[570][d] < 0 or (V[j - 20:j] < V[j]).mean() < q:
            continue
        k = I[524][d]; A = D['atr'][k]; Xr = D['c'][k] - D['c'][I[509][d]]
        if not np.isfinite(A) or Xr == 0 or abs(Xr) < 0.05 * A:
            continue
        K.append(k); Dd.append(-int(np.sign(Xr))); S.append(max(sk * A, FL[mk])); M.append(I[570][d] - (k + 1))
        TP.append(D['c'][I[509][d]] if tgt == 'prenews' else np.nan)
    return signals_frame(K, Dd, stop_pts=np.array(S), tgt_px=np.array(TP), maxbars=np.array(M, np.int64)).assign(comp='NEWS')


def ogr(mk, gmin, lev, ex):
    D, st, en, nd = ctx(mk); i570, i599, i974 = at(mk, 570), at(mk, 599), at(mk, 974)
    td = np.where(i570 >= 0)[0]; K, Dd, S, TP = [], [], [], []
    for p, d in zip(td[:-1], td[1:]):
        k0 = i570[d]; A = D['atr'][k0]
        R = D['c'][i974[p]] if i974[p] >= 0 else D['pdc'][k0]
        if not np.isfinite(A) or not np.isfinite(R):
            continue
        g = D['o'][k0] - R
        if g == 0 or abs(g) < gmin * A:
            continue
        k = i599[d]
        if k <= k0:
            continue
        hm, lm = D['h'][k0:k + 1].max(), D['l'][k0:k + 1].min(); LV = R if lev == 'FULL' else R + g / 2
        if not ((g > 0 and lm <= LV) or (g < 0 and hm >= LV)):
            continue
        dd = -int(np.sign(g)); raw = (hm - D['c'][k] if dd == -1 else D['c'][k] - lm) + 0.02 * A
        K.append(k); Dd.append(dd); S.append(max(raw, 0.12 * A, FL[mk])); TP.append(np.nan if ex == 'close' else R - g)
    return signals_frame(K, Dd, stop_pts=np.array(S), tgt_px=np.array(TP)).assign(comp='OGR')


def book_sig(mk, p, drop=()):
    parts = dict(PSR=lambda: psr(mk, p['psr_thr'], p['psr_T'], p['psr_sk']),
                 LOF=lambda: lof(mk, p['lof_W'], p['lof_thr'], p['lof_X'], p['lof_sl']),
                 NEWS=lambda: news(mk, p['nw_q'], p['nw_sk'], p['nw_tgt']),
                 OGR=lambda: ogr(mk, p['og_g'], p['og_lev'], p['og_ex']))
    s = pd.concat([f() for c, f in parts.items() if c not in drop], ignore_index=True)
    s = s.sort_values('k', kind='stable').reset_index(drop=True); assert s.k.is_unique
    return s


def run(mk, sig, **kw):
    return simulate(mk, sig, **kw).merge(sig[['k', 'comp']], on='k', how='left', validate='one_to_one')


def seg(t):
    i, o = t[t.entry_t < IS_END], t[t.entry_t >= IS_END]
    return dict(n=len(t), pf=round(pf(t.R), 3), pf_IS=round(pf(i.R), 3), pf_OOS=round(pf(o.R), 3), pw=round(len(t) / 156.4, 2))


def checks(mk, t):
    D = load(mk); k, e, x = t.k.to_numpy(), t.entry_k.to_numpy(), t.exit_k.to_numpy()
    assert (e == k + 1).all() and (D['dayi'][e] == D['dayi'][k]).all() and (D['dayi'][x] == D['dayi'][e]).all()
    em = D['mod'][e]; assert ((em >= 180) & (em <= 930)).all() and (t.risk >= FL[mk] - 1e-9).all()
    want = dict(PSR=(179, 179), LOF=(181, 209), NEWS=(524, 524), OGR=(599, 599))
    for c, (lo, hi) in want.items():
        mm = D['mod'][k[t.comp.to_numpy() == c]]; assert ((mm >= lo) & (mm <= hi)).all(), c
    return True


NEIGH = [('psr_thr', 0.3), ('psr_thr', 0.5), ('psr_sk', 0.3), ('psr_sk', 0.7), ('psr_T', 600),
         ('lof_W', 194), ('lof_W', 239), ('lof_thr', 0.05), ('lof_thr', 0.25), ('lof_sl', 0.2), ('lof_sl', 0.5), ('lof_X', 510),
         ('nw_q', 0.8), ('nw_q', 0.95), ('nw_sk', 0.15), ('nw_sk', 0.5), ('nw_tgt', 'none'),
         ('og_g', 0.2), ('og_g', 0.4), ('og_lev', 'CE'), ('og_ex', 'tgt')]


def to_pts(mk, sig):
    """absolute targets -> point distance from the k+1 open (engine-equivalent); drop rows the engine would skip."""
    D = load(mk); s = sig.copy(); px = D['o'][s.k.to_numpy() + 1]; m = s.tgt_px.notna().to_numpy()
    dist = s.d.to_numpy() * (s.tgt_px.to_numpy() - px)
    s.loc[m, 'tgt_pts'] = dist[m]; s['tgt_px'] = np.nan
    return s[~(m & ~(dist > 0))].reset_index(drop=True)


def random_time(mk, sig, reps=20, seed=5, mode='window'):
    """Same days, same d, same stop/target points (NEWS target -> points from the fill), same planned hold (maxbars;
    OGR = to close); only the signal minute is random on the same date.
    mode 'window': entry minute uniform in [03:00, 10:00) (the book's entry span). Biased slightly AGAINST the random
      trades: an entry before a LOF/NEWS signal carries a direction derived from a move that has not happened yet.
    mode 'post':   entry minute uniform in [signal minute+1, 10:00) (OGR: [10:00, 12:00)) -> only information the
      strategy already had; tests whether the exact entry moment matters."""
    D, st, en, nd = ctx(mk); rng = np.random.default_rng(seed); s0 = to_pts(mk, sig); mod = D['mod']; out = []
    for _ in range(reps):
        K, keep = [], []
        for i, r in enumerate(s0.itertuples()):
            d = D['dayi'][r.k]; mm = mod[st[d]:en[d]]
            a, b = (179, 599) if mode == 'window' else (mod[r.k], 719 if r.comp == 'OGR' else 599)
            cand = np.where((mm >= a) & (mm < b))[0]
            if len(cand):
                K.append(st[d] + cand[rng.integers(len(cand))]); keep.append(i)
        x = s0.iloc[keep].copy(); x['k'] = K
        out.append(simulate(mk, x.drop_duplicates('k').sort_values('k', kind='stable').reset_index(drop=True)))
    return out


def ctl(base, rr):
    return {key: dict(med=round(float(np.nanmedian([r[key] for r in rr])), 3), p95=round(float(np.nanpercentile([r[key] for r in rr], 95)), 3),
                      share_beaten=round(float(np.mean([r[key] < base[key] for r in rr])), 3)) for key in ('pf', 'pf_IS', 'pf_OOS')}


def main():
    ref = json.load(open('/home/user/TRADING/scalp/results/portfolio_low_freq.json'))
    res = dict(slug='portfolio_low_freq_verify', markets={})
    books = {}
    for mk in ('NQ', 'ES'):
        R = res['markets'][mk] = {}
        sig = book_sig(mk, BASE); t = run(mk, sig); checks(mk, t); books[mk] = t
        R['rerun'] = seg(t); R['by_comp'] = {c: seg(g) for c, g in t.groupby('comp')}
        R['standalone'] = {c: seg(run(mk, sig[sig.comp == c].reset_index(drop=True))) for c in ('PSR', 'LOF', 'NEWS', 'OGR')}
        R['ref'] = {q: ref['markets'][mk]['V1_full_report'][q] for q in ('n', 'pf', 'pf_IS', 'pf_OOS')}
        R['pf_2x'] = round(pf(simulate(mk, sig, costmult=2.0).R), 3)
        # outlier sensitivity: drop top 5 R trades
        Rs = np.sort(t.R.to_numpy()); R['pf_drop_top5'] = round(pf(Rs[:-5]), 3); R['pf_drop_top10'] = round(pf(Rs[:-10]), 3)
        R['by_year'] = {int(y): seg(g)['pf'] for y, g in t.groupby(t.entry_t.dt.year)}
        R['by_side'] = {int(d): seg(g) for d, g in t.groupby('d')}
        # neighbours
        nb = {}
        for p, v in NEIGH:
            q = dict(BASE); q[p] = v; nb[f'{p}={v}'] = seg(run(mk, book_sig(mk, q)))
        for c in ('PSR', 'LOF', 'NEWS', 'OGR'):
            nb[f'drop_{c}(info)'] = seg(run(mk, book_sig(mk, BASE, drop=(c,))))
        R['neighbours'] = nb
        fam = [v['pf'] for kk, v in nb.items() if 'drop' not in kk]
        R['neigh_median_pf'] = round(float(np.median(fam)), 3); R['neigh_share_gt_1_1'] = round(float(np.mean(np.array(fam) > 1.1)), 3)
        R['neigh_median_IS'] = round(float(np.median([v['pf_IS'] for kk, v in nb.items() if 'drop' not in kk])), 3)
        R['neigh_median_OOS'] = round(float(np.median([v['pf_OOS'] for kk, v in nb.items() if 'drop' not in kk])), 3)
        # random-time controls: 20 reps (as specified) and 100 reps (stability)
        for mode in ('window', 'post'):
            for reps in (20, 100):
                R[f'rt_{mode}_{reps}'] = ctl(R['rerun'], [seg(x) for x in random_time(mk, sig, reps=reps, seed=5 + reps, mode=mode)])
        print(mk, json.dumps({q: R[q] for q in ('rerun', 'ref', 'by_comp', 'pf_2x', 'pf_drop_top5', 'pf_drop_top10', 'by_year',
                                                  'neigh_median_pf', 'neigh_share_gt_1_1', 'neigh_median_IS', 'neigh_median_OOS',
                                                  'rt_window_20', 'rt_window_100', 'rt_post_20', 'rt_post_100')}))
        print('  neighbours:', json.dumps({kk: (v['n'], v['pf'], v['pf_IS'], v['pf_OOS']) for kk, v in nb.items()}))
    P = pd.concat(books.values()); res['book'] = seg(P)
    print('book', res['book'])
    json.dump(res, open('/home/user/TRADING/scalp/results/portfolio_low_freq_verify.json', 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
