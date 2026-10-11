# Scalping research kit (read fully before writing a strategy)

Data: IC Markets 1-minute bars, **New York time**, NQ (USTEC CFD) and ES (US500 CFD), March 2023 – Sept 2026.
Results are reported only for **2023-10-01 → 2026-09-30**:
- **In-sample (IS)** = before 2025-04-01. Choose all parameters here only.
- **Out-of-sample (OOS)** = from 2025-04-01. Never use it to choose anything.

## Files
- `scalp/engine.py`: the ONLY simulator you may use. Do not modify it. Read its docstring.
- `scalp/build_cache.py`: lists the precomputed per-bar arrays: `t, o, h, l, c, vol, mod, date, dayi, dow, spread, atr, trend_d, pdh, pdl, pdc, rth_open, on_high, on_low, vwap, rth_high, rth_low`.
  - `mod` = NY minute of day (570 = 09:30).
  - `t` = NY minutes since epoch.
  - `dow` 0 = Monday.
  - Every array value at index k is known at the close of bar k.
- `scalp/strategies/example_orb.py`: a template to copy.

## Hard rules
1. **No lookahead.** The signal at bar k uses only arrays[:k+1]. Rolling or EMA features from pandas are fine if they only look back. Never use `.shift(-1)`, centred windows, a day's final high/low before the day ends, etc.
2. **Entry is at k+1 or later** (handled by the engine). Stop must be ≥ 1 point (NQ) / 0.25 (ES) from entry.
3. **Costs:** NQ 1.5 pts, ES 0.6 pts per round trip (engine default). Always report the 2× cost result as well.
4. **Stop-size bias.** On 1-minute bars the engine counts stop+target in the same bar as a STOP, and the target cannot fill on the entry bar. Random-entry baselines (gross, before costs) are:
   - **NQ:** stop 5 pt → PF 0.54 · 10 pt → 0.82 · 20 pt → 0.99 · 30+ pt → ~1.0
   - **ES:** stop 1 pt → 0.51 · 2 pt → 0.82 · 4 pt → 0.97 · 6+ pt → ~1.0

   So test **NQ stops ≥ 20 pts (or ≥ 0.12×ATR) and ES stops ≥ 4 pts (or ≥ 0.12×ATR)**. Tighter scalps cannot be judged fairly without tick data; flag them as "needs tick data" if your idea requires them.
5. **Selection-bias budget:** at most **24 parameter variants per strategy**. Pick the variant on IS only (prefer the middle of a plateau, not the single best). Report how many you tried and the **median IS PF of all variants**.
6. **Session:** entries only between 03:00 and 15:30 NY. The engine force-closes at 15:59. One position at a time per market (engine).
7. **Run both markets** with the same rules (parameters may scale by ATR). A real edge should work on both NQ and ES, or at least not be negative on the other.
8. **Report** using `full_report(mk, sig)`:
   - headline, pf_IS / pf_OOS, years, 2× cost, no cost, bootstrap, opposite direction, random direction.
   - Signals using limit/stop entries or absolute stop/target prices do not get the random-direction control. If possible, add your own control (e.g. same rules at random times on the same days).

## Targets (user)
- PF ≥ 1.7 after costs and 2–3+ trades/week is the dream.
- For scalping, PF ≥ 1.3 in BOTH IS and OOS with ≥ 5 trades/week is interesting.
- Anything with OOS PF < 1.1 is a fail.

## Writing results
Save your strategy to `scalp/strategies/<slug>.py`, runnable as `python3 -I scalp/strategies/<slug>.py`, printing the full reports. Also write a JSON summary to `scalp/results/<slug>.json`.
