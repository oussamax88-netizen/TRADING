# Running TDH + AZL on a VPS ($1,000, high-risk plan)

## The bots
| File | Chart | Magic | What it does |
|---|---|---|---|
| `AZL_OB_CHOCH.mq5` | USTEC | 7302601 | Asia zone → London CHOCH, entries 03:00–09:30 NY |
| `AZL_OB_CHOCH.mq5` | US500 | 7302601 (another symbol, so it's fine) | same, on the S&P |
| `TDH_NQ.mq5` | USTEC | **7302602** | US-open trend-day breakout, decisions 09:40–15:30 NY |

All three close everything at 15:59 NY. Nothing is held overnight.

## Settings for your plan (5% of the current balance)
On each of the 3 charts:
- `InpRiskPercentMode = true`
- `InpRiskPercent = 5`
- `InpDDBrakePct = 25` (halves risk after −25% from peak) or `0` (no brake: higher typical result, deeper drawdowns)
- `InpNotify = true` (phone notifications: MT5 → Tools → Options → Notifications → enter your MetaQuotes ID from the MT5 phone app)
- Leave every strategy input at its default (tested values).

## 1. Test both bots in the Strategy Tester first (once)
Settings: Symbol USTEC, M1, **2023.08.01 → 2026.10.01**, real ticks or 1-minute OHLC, `InpRiskPercentMode = false`, `InpRiskMoney = 100`, `InpTradeFrom = 2023.10.01`.

| | TDH on USTEC | AZL on USTEC | AZL on US500 |
|---|---|---|---|
| Trades | **265** | **133** | **127** |
| Win rate | 40% | 31% | 34% |
| Profit factor | **~1.70–1.77** | **~1.79–1.85** | **~1.75–1.89** |
| Trades per year (2023 Q4 / 2024 / 2025 / 2026) | 22 / 85 / 91 / 67 | 12 / 43 / 44 / 34 | 8 / 35 / 49 / 35 |

**First TDH trades** (server time):

| Time | Direction | Entry | Result |
|---|---|---|---|
| 2023-10-09 16:49 | BUY | 14916.3 | −1R |
| 2023-10-16 17:13 | BUY | 15139.0 | −1R |
| 2023-10-18 17:20 | BUY | 15082.5 | −1R |
| 2023-10-23 16:41 | SELL | 14440.8 | −1R |
| 2023-10-25 16:42 | SELL | 14545.7 | **+3.25R** |
| 2023-10-26 16:49 | SELL | 14275.7 | **+3.04R** |

Full lists: `reports/TDH_expected_trades.csv` and `reports/AZL_expected_trades.csv`. Send me the tester report if the trade count is off by more than ~10%.

## 2. Before going live: check the minimum lot
At $1,000 and 5% you risk **$50 per trade**. The stop is about 50–60 points on USTEC and about 10–12 points on US500.
- In MT5, right-click USTEC → Specification. Check the **contract size, minimum volume and volume step**.
- The lot the bot needs is $50 ÷ (stop points × $ per point per lot). If that's below the minimum volume, the bot **skips the trade** and prints "risk too small for minimum lot" in the Experts tab.

## 3. VPS
- **Cheapest option:** check whether IC Markets gives you a free VPS (it usually depends on deposit or volume).
- **Easiest option:** the MetaQuotes VPS built into MT5 (right-click your account in Navigator → "Register a Virtual Server", about $10–15/month).
  - Attach the 3 charts with the bots on your PC, enable Algo Trading, then **Migrate** to the VPS.
  - After any change, migrate again.
- **Windows VPS from a provider** (about $5/month): install MT5, log in, attach the bots, enable Algo Trading, enable auto-start.
- A VPS fee of $10/month is **12% of a $1,000 account per year**. Keep it as cheap as possible.

## 4. "Forget it" maintenance (about 10 minutes, 4 times a year)
1. **Every 3 months (about 40 trades):** check the combined profit factor in MT5 History (right-click → Report).
   - PF > 1.3 → do nothing.
   - PF 1.0–1.3 → halve `InpRiskPercent`.
   - PF < 1.0 → stop the bots and send me the report.
2. **Every January:** add the new year's US holiday dates when London still trades (Good Friday etc.) to `InpSkipDates` on all 3 charts.
3. **If the broker changes its server time** (rare): `InpServerMinusNY` must still be 7. The status box shows NY time; check that it matches real New York time.
4. **Optional profit rule:** when the account first doubles to $2,000, withdraw your $1,000. From then on you are trading only the market's money.
