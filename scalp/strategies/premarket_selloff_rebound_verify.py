"""Independent skeptic re-implementation of premarket_selloff_rebound (written from the rule text, not the code).

Rule: on every date that has an RTH session, signal bar = 02:59 bar (minute ent-1). P = previous RTH date's
(close of last RTH bar, minute < 960) - (open of first RTH bar, minute >= 570). LONG at open of next bar if
P <= -thr*ATR (cache atr at the signal bar). stop = max(stopk*ATR, 20 NQ / 4 ES). No target. Exit at the open of the
first bar of the same date with minute >= T. Chosen: thr 0.4, stopk 0.5, T 570, long only.
Tests: reproduction, single-step neighbours, random-time controls on the same days, random-day control,
April-2025 sensitivity, trade concentration.
"""
import sys, json
sys.path.insert(0, '/home/user/TRADING/scalp')
import numpy as np
import pandas as pd
from engine import load, signals_frame, simulate, headline, full_report, pf, IS_END

MINSTOP = {'NQ': 20.0, 'ES': 4.0}
BASE = dict(thr=0.4, stopk=0.5, T=570, ent=180, pdef='oc')
OUT = '/home/user/TRADING/scalp/results/premarket_selloff_rebound_verify.json'


def daily(mk):
    D = load(mk)
    mod, date = D['mod'], D['date']
    ri = np.flatnonzero((mod >= 570) & (mod < 960))
    dts, fi = np.unique(date[ri], return_index=True)
    li = np.r_[fi[1:], len(ri)] - 1
    op, cl = D['o'][ri[fi]], D['c'][ri[li]]
    return dts, op, cl


def build(mk, thr=0.4, stopk=0.5, T=570, ent=180, pdef='oc', days=None, all_days=False):
    """Return (k, d, stop, maxbars, date) for qualifying days. days: restrict to a set of dates."""
    D = load(mk); t, date, dayi = D['t'], D['date'], D['dayi']
    dts, op, cl = daily(mk)
    if pdef == 'oc':
        P = np.r_[np.nan, (cl - op)[:-1]]
    else:  # close-to-close of the previous RTH day (info neighbour)
        P = np.r_[np.nan, np.nan, (cl[1:] - cl[:-1])[:-1]]
    tk = dts * 1440 + (ent - 1)
    k = np.searchsorted(t, tk)
    ok = (k < len(t) - 1)
    k = np.where(ok, k, 0)
    ok &= (t[k] == tk) & (dayi[np.minimum(k + 1, len(t) - 1)] == dayi[k])
    iT = np.searchsorted(t, dts * 1440 + T)
    iT = np.minimum(iT, len(t) - 1)
    ok &= (date[iT] == dts) & (iT > k + 1)
    atr = D['atr'][k]
    ok &= np.isfinite(atr) & np.isfinite(P)
    if not all_days:
        ok &= P <= -thr * atr
    if days is not None:
        ok &= np.isin(dts, days)
    k, iT, atr, dd = k[ok], iT[ok], atr[ok], dts[ok]
    stop = np.maximum(stopk * atr, MINSTOP[mk])
    return k, np.ones(len(k), np.int64), stop, iT - (k + 1), dd


def run(mk, **kw):
    k, d, st, mb, _ = build(mk, **{**BASE, **kw})
    return simulate(mk, signals_frame(k, d, etype=0, stop_pts=st, maxbars=mb))


def h3(tr):
    h = headline(tr)
    return dict(n=h['n'], pf=h['pf'], IS=h['pf_IS'], OOS=h['pf_OOS'])


def neighbours(mk):
    var = [('thr', v) for v in (0.2, 0.3, 0.5, 0.6)] + [('stopk', v) for v in (0.3, 0.4, 0.6, 0.7)] + \
          [('T', v) for v in (540, 555, 585, 600)] + [('ent', v) for v in (195, 210)]
    rows = []
    for p, v in var:
        r = h3(run(mk, **{p: v})); r['var'] = f'{p}={v}'; rows.append(r)
    return pd.DataFrame(rows)


def random_time_controls(mk, tr, reps=20, seed=11):
    """Same trade days, long, same ATR stop. A: entry random 03:00..08:59, exit 09:30 open (same exit rule).
    B: entry random 03:00..09:29, same 390-bar hold (exit 09:30..15:59)."""
    D = load(mk); t, dayi = D['t'], D['dayi']
    rng = np.random.default_rng(seed)
    k0 = tr.k.to_numpy(); dts = D['date'][k0]; stop = (tr.risk.to_numpy())  # risk = stop pts at entry
    iT = np.searchsorted(t, dts * 1440 + 570)
    out = {}
    for name, lo, hi, fixed in (('A_rand_entry_exit0930', 180, 540, False), ('B_rand_entry_hold390', 180, 570, True)):
        pfs = []; pis = []; pos = []
        for _ in range(reps):
            m = rng.integers(lo, hi, len(k0))           # entry minute; signal bar = last bar with t < entry minute
            ke = np.searchsorted(t, dts * 1440 + m) - 1
            okk = (dayi[ke] == dayi[k0]) & (ke + 1 < iT if not fixed else True)
            mb = np.full(len(ke), 390) if fixed else iT - (ke + 1)
            s = signals_frame(ke[okk], 1, etype=0, stop_pts=stop[okk], maxbars=mb[okk])
            x = simulate(mk, s); pfs.append(pf(x.R)); pis.append(pf(x[x.entry_t < IS_END].R)); pos.append(pf(x[x.entry_t >= IS_END].R))
        out[name] = dict(pf_med=round(float(np.median(pfs)), 3), pf_95=round(float(np.percentile(pfs, 95)), 3),
                         IS_med=round(float(np.median(pis)), 3), IS_95=round(float(np.percentile(pis, 95)), 3),
                         OOS_med=round(float(np.median(pos)), 3), OOS_95=round(float(np.percentile(pos, 95)), 3))
    return out


def random_day_control(mk, tr, reps=20, seed=5):
    """Same 03:00->09:30 trade, same stop rule, long, on random RTH days (same count per IS/OOS segment)."""
    base = run(mk, all_days=True)
    rng = np.random.default_rng(seed)
    nI = int((tr.entry_t < IS_END).sum()); nO = len(tr) - nI
    bI = base[base.entry_t < IS_END].R.to_numpy(); bO = base[base.entry_t >= IS_END].R.to_numpy()
    sims = [pf(np.r_[rng.choice(bI, nI, replace=False), rng.choice(bO, nO, replace=False)]) for _ in range(reps)]
    big = [pf(np.r_[rng.choice(bI, nI, replace=False), rng.choice(bO, nO, replace=False)]) for _ in range(4000)]
    simO = [pf(rng.choice(bO, nO, replace=False)) for _ in range(4000)]
    return dict(all_days=h3(base), pf_med=round(float(np.median(sims)), 3), pf_95=round(float(np.percentile(sims, 95)), 3),
                p_full=round(float(np.mean(np.array(big) >= pf(tr.R))), 3),
                p_OOS=round(float(np.mean(np.array(simO) >= pf(tr[tr.entry_t >= IS_END].R))), 3))


def stress(tr):
    o = tr[tr.entry_t >= IS_END]
    apr = (o.entry_t >= '2025-04-01') & (o.entry_t < '2025-05-01')
    R = np.sort(tr.R.to_numpy())[::-1]
    return dict(OOS_ex_apr2025=round(pf(o[~apr].R), 3), n_apr2025=int(apr.sum()), apr2025_netR=round(float(o[apr].R.sum()), 2),
                pf_drop_top5=round(pf(R[5:]), 3), pf_drop_top10=round(pf(R[10:]), 3),
                OOS_drop_top3=round(pf(np.sort(o.R.to_numpy())[::-1][3:]), 3))


if __name__ == '__main__':
    res = {}
    for mk in ('NQ', 'ES'):
        D = load(mk)
        k, d, st, mb, dd = build(mk, **BASE)
        dts, op, cl = daily(mk)
        pc = dict(zip(dts[1:], cl[:-1])); pdc = D['pdc'][k]
        assert np.allclose([pc[x] for x in dd], pdc), 'prev close != pdc'
        assert (D['mod'][k] == 179).all() and (D['mod'][k + 1] >= 180).all() and (D['mod'][k + mb + 1] >= 570).all()
        assert (D['mod'][k + mb] < 570).all()
        sig = signals_frame(k, d, etype=0, stop_pts=st, maxbars=mb)
        rep = full_report(mk, sig)
        tr = simulate(mk, sig)
        r = dict(headline={x: rep.get(x) for x in ('n', 'per_week', 'pf', 'pf_IS', 'n_IS', 'pf_OOS', 'n_OOS', 'pf_2x_cost',
                                                    'pf_opposite', 'pf_random_dir_median', 'pf_random_dir_95', 'by_year')})
        print(mk, 'REPRO', json.dumps(r['headline']))
        nb = neighbours(mk); print(nb.to_string(index=False))
        r['neighbours'] = nb.to_dict(orient='records')
        r['neigh_median_pf'] = float(nb.pf.median()); r['neigh_frac_gt_1.1'] = float((nb.pf > 1.1).mean())
        r['neigh_median_IS'] = float(nb.IS.median()); r['neigh_median_OOS'] = float(nb.OOS.median())
        print(mk, 'neighbour median PF', r['neigh_median_pf'], 'frac>1.1', r['neigh_frac_gt_1.1'], 'IS', r['neigh_median_IS'], 'OOS', r['neigh_median_OOS'])
        r['ctrl_close_to_close_P'] = h3(run(mk, pdef='cc'))
        r['ctrl_random_time'] = random_time_controls(mk, tr)
        r['ctrl_random_day'] = random_day_control(mk, tr)
        r['stress'] = stress(tr)
        for x in ('ctrl_close_to_close_P', 'ctrl_random_time', 'ctrl_random_day', 'stress'):
            print(mk, x, json.dumps(r[x]))
        res[mk] = r
    json.dump(res, open(OUT, 'w'), default=str, indent=1)
    print('saved', OUT)
