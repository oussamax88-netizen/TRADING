# Scalping / short-hold edge hunt: NQ & ES, Oct 2023 – Sep 2026

## Method
- Shared 1-minute engine with the following rules:
  - entry at the next bar
  - stop wins when stop and target fall in the same bar
  - no target fill on the entry bar
  - costs of NQ 1.5 / ES 0.6 points
- Each idea had at most 24 variants. The variant was chosen on in-sample data (Oct 2023 – Mar 2025) only.
- Ideas were found by web research (academic papers, practitioner setups, ICT/SMC, time-of-day effects) and curated by a lead agent.
- Candidates went to two skeptic agents:
  - a lookahead/bug audit
  - an independent rebuild with neighbour settings and random-timing controls
- About **18 strategies and ~400 variants** were tested in total.

## Results (PF in-sample / out-of-sample, trades per week per market)
| Strategy | NQ | ES | Hold | Verdict |
|---|---|---|---|---|
| 5-min ORB scalp (baseline) | 1.11 / 0.97 (4.9) | 0.90 / 0.81 (4.9) | 40 min | FAIL |
| Liquidity-shock fade (true scalp) | 1.34 / 0.61 (0.5) | 0.99 / 0.56 (0.7) | 4 min | FAIL: gross PF < 1 |
| Zarattini noise-band momentum | 1.79 / 0.88 (1.9) | 1.77 / 0.63 (2.3) | 60 min | FAIL out of sample |
| Noise-band, volatility filter inverted | 1.11 / 0.95 (6.7) | 1.20 / 0.97 (6.1) | 70 min | FAIL |
| Variance-ratio regime switch | 0.90 / 0.88 (6.0) | 0.89 / 0.87 (6.8) | 50 min | FAIL |
| Gamma-regime VWAP fade/follow | 1.04 / 1.19 (2.8) | 1.06 / 1.15 (3.1) | 150 min | FAIL |
| Power-of-3 reclaim (ICT) | 1.19 / 0.90 (2.7) | 1.28 / 0.79 (2.7) | — | FAIL |
| Inverse Silver Bullet (ICT) | 1.61 / 1.07 (1.3) | 1.23 / 0.95 (1.3) | 80 min | FAIL |
| Silver Bullet failure + trend + volatility | 2.02 / 0.63 (0.6) | 1.29 / 0.39 (0.6) | 90 min | FAIL |
| Lunch-effect drift (Quantpedia 2024) | 0.97 / 0.93 (2.1) | 0.96 / 0.88 (2.2) | 70 min | FAIL |
| Gap-and-go from midnight | 1.80 / 0.84 (1.0) | 2.14 / 0.84 (0.9) | — | FAIL |
| Gap against trend, fill with trend | 1.31 / 1.29 (1.9) | 1.30 / 1.30 (1.9) | 150 min | REJECTED by skeptic (no better than random gap days) |
| Last-30-min fade | 2.25 / 1.32 (0.5) | 1.65 / 0.95 (0.7) | 30 min | BORDERLINE (too rare) |
| London-open fade | 1.81 / 1.11 (0.7) | 1.37 / 1.17 (0.7) | 6 h | BORDERLINE |
| London Judas sweep fade | 2.23 / 1.76 (0.4) | 1.62 / 1.64 (0.4) | 6 h | BORDERLINE (out of sample not clean, too rare) |
| Opening-range gap rejection | 2.49 / 1.05 (0.25) | 3.96 / 7.0 (0.14) | to close | too rare to judge |
| Premarket sell-off rebound | 1.39 / 1.31 (1.2) | 1.80 / 1.32 (1.1) | 6.5 h | WEAK: does not beat "long every morning" drift |
| 8:30 news fade (two-leg) | 1.33 / 1.34 (0.6) | 1.46 / 1.16 (0.6) | 45 min | WEAK: works as well on non-news days |
| **Low-frequency portfolio** (premarket rebound + London fade + news fade + gap rejection) | 1.68 / 1.32 (2.3) | 2.04 / 1.44 (2.1) | 6 h | **WEAK**: components picked after seeing out-of-sample; clean admission gives NQ 1.10–1.18, ES 0.93–1.17 out of sample |

## Conclusion
- **No true scalping edge (holds of minutes) survived.** The one real scalp (liquidity-shock fade, 4-minute hold) loses even before costs. Popular momentum/VWAP/ICT scalps are about PF 0.9–1.0 after costs.
- What looks good is **not scalping**. These are 1.5–6 hour holds that capture overnight long drift or trend days, and they don't clearly beat random-timing controls out of sample.
- The **proven edges remain TDH and AZL**, which are trend-day captures. Combining them is the honest way to reach 2–3 trades per week.
- True tight scalps (NQ stop < 20 pts, ES < 4 pts) **cannot be judged on 1-minute bars**. They need tick data.
