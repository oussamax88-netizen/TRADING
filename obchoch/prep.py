"""Cache data + arrays + zones per market."""
import sys, pickle, pandas as pd
sys.path.insert(0, 'obchoch')
from data import load_mt5, resample, resample_daily, daily_context
from zones import find_zones
from engine import build_arrays, prepare_zones

def prep(path, out):
    d = load_mt5(path)
    ctx = daily_context(d)
    a = build_arrays(d, ctx)
    z = pd.concat([find_zones(resample(d, 60), '1h'), find_zones(resample(d, 30), '30m'),
                   find_zones(resample(d, 240), '4h'), find_zones(resample_daily(d), '1d')], ignore_index=True)
    z = prepare_zones(z.sort_values('created'), a)
    pickle.dump((a, z), open(out, 'wb'))
    print(out, len(d), 'bars', len(z), 'zones', z.groupby(['tf', 'dir']).size().to_dict())

if __name__ == '__main__':
    prep(sys.argv[1], sys.argv[2])
