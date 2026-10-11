# AZL EA (Asia Zone → London CHOCH): install and test in MT5

## Install
1. In MT5, open **File → Open Data Folder** → `MQL5` → `Experts`.
2. Copy `AZL_OB_CHOCH.mq5` into that folder.
3. Open **MetaEditor** (F4), open the file, and press **Compile** (F7). You should see "0 errors". Send me a screenshot if there are errors.
4. Back in MT5, the EA appears in **Navigator → Expert Advisors** (right-click → Refresh if needed).

## Backtest in the Strategy Tester (Ctrl+R)
| Setting | Value |
|---|---|
| Expert | AZL_OB_CHOCH |
| Symbol | USTEC (Nasdaq), then a second run on US500 |
| Timeframe | M1 (the EA uses M1 data in every case) |
| Date | **2023.08.01 → 2026.10.01**. The first 2 months only warm up ATR, trend and zones. |
| Modelling | "Every tick based on real ticks" (best) or "1 minute OHLC" (fast) |
| Deposit | 10000 USD, leverage as on your account |
| Inputs | `InpRiskPercentMode = false`, `InpRiskMoney = 100`, `InpTradeFrom = 2023.10.01` |

Fixed-money risk ($100 per trade) makes the tester's profit factor directly comparable with the research numbers in R.

## Expected numbers (research, Oct 2023 – Sep 2026)
| | USTEC | US500 |
|---|---|---|
| Trades | 133 | 127 |
| Win rate | 31% | 34% |
| Profit factor (no cost) | 1.85 | 1.89 |
| Profit factor (typical spread: 1.0 / 0.5 pt) | ~1.81 | ~1.77 |
| Profit factor (research cost: 1.5 / 0.6 pt) | 1.79 | 1.75 |
| Net profit at $100 risk | ≈ +$7,000 | ≈ +$6,000 |
| Longs / shorts | 98 / 35 | 99 / 28 |

Trades per year: USTEC 2023 (Q4) 12 · 2024 43 · 2025 44 · 2026 34. US500: 8 · 35 · 49 · 35.

A difference of a few trades is normal: broker data differs slightly, ties, spread. **More than ~10% difference in trade count means something is wrong**, so send me the tester report and journal.
The full expected trade list is in `reports/AZL_expected_trades.csv`. Send me the tester "Deals" tab and I'll compare it line by line.

## Live use
- Attach the EA to a USTEC or US500 chart, any timeframe (it reads M1 itself). Enable Algo Trading.
- The **status box** (top-left) shows NY time, ATR, trend, active Asia zones and the last signal. It updates every 5 seconds, even when the market is closed.
- **Restarts are safe:** on start the EA replays the last 60 days of M1 history to rebuild zones, ATR and trend. It never acts twice on the same signal.
- `InpServerMinusNY = 7` is right for IC Markets: server time is GMT+2/+3 and switches with US daylight saving.
- `InpSkipDates`: US cash-market holidays when London still trades (Good Friday, special closures). Add new ones every year.
- Optional: `InpDDBrakePct = 25` halves the risk after a −25% drawdown from the peak. `InpNotify = true` sends phone notifications (set up the MetaQuotes ID in Tools → Options → Notifications).
