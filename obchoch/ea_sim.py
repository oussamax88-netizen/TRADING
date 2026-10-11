"""Bar-by-bar simulation with exactly the structure of the MQL5 EA (AZL_OB_CHOCH.mq5).
Used to prove the streaming logic reproduces the research trade list before porting to MQL5."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'obchoch')
from data import load_mt5

LIFE = {30: 5, 60: 5, 240: 10, 1440: 20}
SWN = 5

class Builder:
    def __init__(self, tf):
        self.tf = tf; self.bucket = None; self.cur = None
        self.o, self.h, self.l, self.c = [], [], [], []
        self.sh = None; self.sl = None          # (abs index, level)
    def bid(self, t):
        return (t - 1080) // 1440 if self.tf == 1440 else t // self.tf
    def end_time(self):
        return self.bucket * 1440 + 1080 + 1440 if self.tf == 1440 else (self.bucket + 1) * self.tf
    def on_bar(self, t, o, h, l, c):
        """Returns list of zones (dir, lo, hi, created_min) created by a bar finalized before this M1 bar."""
        out = []
        b = self.bid(t)
        if self.bucket is not None and b != self.bucket:
            out = self.finalize()
        if self.bucket is None or b != self.bucket:
            self.bucket = b; self.cur = [o, h, l, c]
        else:
            self.cur[1] = max(self.cur[1], h); self.cur[2] = min(self.cur[2], l); self.cur[3] = c
        return out
    def finalize(self):
        o, h, l, c = self.cur
        self.o.append(o); self.h.append(h); self.l.append(l); self.c.append(c)
        i = len(self.c) - 1; out = []; end = self.end_time()
        O, H, L, C = self.o, self.h, self.l, self.c
        if self.sh is not None and C[i] > self.sh[1]:
            for j in range(i - 1, max(self.sh[0], i - 15) - 1, -1):
                if C[j] < O[j]:
                    out.append((1, L[j], H[j], end)); break
            self.sh = None
        if self.sl is not None and C[i] < self.sl[1]:
            for j in range(i - 1, max(self.sl[0], i - 15) - 1, -1):
                if C[j] > O[j]:
                    out.append((-1, L[j], H[j], end)); break
            self.sl = None
        p = i - 2
        if p >= 2:
            if H[p] > max(H[p - 2:p]) and H[p] >= max(H[p + 1:i + 1]): self.sh = (p, H[p])
            if L[p] < min(L[p - 2:p]) and L[p] <= min(L[p + 1:i + 1]): self.sl = (p, L[p])
        return out

def run(d, cost, exact5=False):
    t = (d.t.astype('datetime64[ns]').astype('int64') // 60_000_000_000).to_numpy()
    O, H, L, C = (d[x].to_numpy() for x in ('open', 'high', 'low', 'close'))
    n = len(t)
    nxt5 = np.r_[(t[1:] // 5) != (t[:-1] // 5), True]
    builders = [Builder(tf) for tf in (30, 60, 240, 1440)]
    zones = []
    dayidx = -1; curdate = None
    rth = None; rngs, closes = [], []; atr = np.nan; trend = 0
    shl = np.nan; sll = np.nan
    pos = None; trades = []; pending = None
    for k in range(n):
        tk = t[k]; mod = tk % 1440; date = tk // 1440
        # --- 0) open of bar k: execute pending entry, manage position at open
        if pos is not None:
            if date != pos['date']:
                pos['exit'] = C[k - 1]; trades.append(pos); pos = None
            elif mod >= 959:
                pos['exit'] = O[k]; trades.append(pos); pos = None
        if pending is not None:
            sd, px_bar, stop_dist, zinfo = pending; pending = None
            if pos is None and date == zinfo['date'] and mod < 959:
                px = O[k]
                pos = dict(side=sd, entry_k=k, entry=px, stop=px - sd * stop_dist, risk=stop_dist, date=date, tf=zinfo['tf'], t=tk)
        # --- 1) new NY date
        if date != curdate:
            if rth is not None:
                rngs.append(rth[0] - rth[1]); closes.append(rth[2])
            rth = None; curdate = date; dayidx += 1
            atr = np.mean(rngs[-14:]) if len(rngs) >= 14 else np.nan
            trend = int(np.sign(closes[-1] - np.mean(closes[-20:]))) if len(closes) >= 20 else 0
        if 570 <= mod < 960:
            rth = [H[k], L[k], C[k]] if rth is None else [max(rth[0], H[k]), min(rth[1], L[k]), C[k]]
        # --- 2) higher-timeframe bars; zones created by bars finalized now start at bar k
        for b in builders:
            for (zd, zlo, zhi, created) in b.on_bar(tk, O[k], H[k], L[k], C[k]):
                cm = created % 1440
                if cm >= 1080 or cm <= 180:
                    zones.append(dict(dir=zd, lo=zlo, hi=zhi, exp=dayidx + LIFE[b.tf], touch=-1, tday=-1, tf=b.tf))
        # --- 3) 1m swings (pivot k-5 confirmed at close of k)
        p = k - SWN
        if p - SWN >= 0:
            if H[p] > H[p - SWN:p].max() and H[p] >= H[p + 1:k + 1].max(): shl = H[p]
            if L[p] < L[p - SWN:p].min() and L[p] <= L[p + 1:k + 1].min(): sll = L[p]
        # --- 4) position: stop inside bar k (entry bar included)
        if pos is not None:
            sd = pos['side']
            if (sd == 1 and L[k] <= pos['stop']) or (sd == -1 and H[k] >= pos['stop']):
                gap = (sd == 1 and O[k] <= pos['stop']) or (sd == -1 and O[k] >= pos['stop'])
                pos['exit'] = O[k] if (gap and k != pos['entry_k']) else pos['stop']; trades.append(pos); pos = None
        # --- 5) zones
        is5 = nxt5[k] if exact5 else (tk % 5 == 4)
        keep = []
        for z in zones:
            sd = z['dir']; dead = False
            if dayidx > z['exp']:
                continue
            if z['touch'] < 0 and ((sd == 1 and L[k] <= z['hi']) or (sd == -1 and H[k] >= z['lo'])):
                if mod < 180 or mod >= 930:
                    continue
                z['touch'] = k; z['tday'] = dayidx
            if is5 and ((sd == 1 and C[k] < z['lo']) or (sd == -1 and C[k] > z['hi'])):
                continue
            if z['touch'] >= 0:
                if mod >= 930 or dayidx != z['tday']:
                    continue
                lvl = shl if sd == 1 else sll
                if not np.isnan(lvl) and ((sd == 1 and C[k] > lvl) or (sd == -1 and C[k] < lvl)):
                    # signal: zone consumed
                    if mod < 570 and trend == sd and not np.isnan(atr) and pending is None:
                        pending = (sd, k, 0.2 * atr, dict(date=date, tf=z['tf']))
                    continue
            keep.append(z)
        zones = keep
    T = pd.DataFrame(trades)
    T['entry_t'] = pd.to_datetime(T.t * 60_000_000_000)
    T['R'] = (T.side * (T.exit - T.entry) - cost) / T.risk
    return T[(T.entry_t >= '2023-10-01') & (T.entry_t < '2026-10-01')]

if __name__ == '__main__':
    S = sys.argv[1]
    from evalr import pf
    for mk, f, cost in (('NQ', f'{S}/x2/hada.csv', 1.5), ('ES', f'{S}/x1/US500 2023.csv', 0.6)):
        d = load_mt5(f)
        for ex in (True, False):
            T = run(d, cost, exact5=ex)
            u = pd.read_csv(f'{S}/audit/audit_{mk}.csv'); u['k'] = pd.to_datetime(u.entry_time_ny)
            j = T[['entry_t', 'R']].rename(columns={'entry_t': 'k'}).merge(u[['k', 'R']], on='k', how='outer', indicator=True, suffixes=('_ea', '_au'))
            print(mk, '5m rule', 'data-exact' if ex else 'minute%5==4 (EA)', '| trades', len(T), 'PF', round(pf(T.R), 3), 'netR', round(T.R.sum(), 1),
                  '| matched audit', (j._merge == 'both').sum(), 'only EA', (j._merge == 'left_only').sum(), 'only audit', (j._merge == 'right_only').sum(),
                  'R diffs', (abs(j.R_ea - j.R_au) > 0.01).sum())
            if not ex: T.to_csv(f'{S}/ea_sim_{mk}.csv', index=False)
