//+------------------------------------------------------------------+
//|                                                    TDH_NQ.mq5    |
//|  TDH = Trend Day Hunter. US-open momentum breakout with the      |
//|  20-day trend, hold to the New York close. Use on USTEC (NQ).    |
//+------------------------------------------------------------------+
//  RULES (all times New York; NY time = server time - InpServerMinusNY hours)
//  1. ATR = average 09:30-16:00 range of the previous 14 days.
//     Trend = previous 09:30-16:00 close vs the average of the last 20 such closes.
//  2. Skip flat opens: |09:30 open - previous 16:00 close| < 0.15 x ATR.
//  3. Opening range = 09:30-09:40 high/low.
//  4. After 09:40, the FIRST 1-minute close beyond the range that is also
//     >= 0.15 x ATR away from the 09:30 open decides the day (one decision per day,
//     until 15:30). Trade only if that direction agrees with the 20-day trend.
//  5. Entry: market order at the open of the next 1-minute bar.
//     Stop: 0.2 x ATR. No target. Forced close at 15:59 NY.
//  Research (IC Markets USTEC M1, Oct 2023 - Sep 2026, cost 1.5 pts):
//     265 trades, win 40%, PF 1.70 (longs 1.59, shorts 1.90). Not for ES/US500 (PF 0.93).
//  Can run on the same USTEC account as AZL_OB_CHOCH: use a different magic number.
//+------------------------------------------------------------------+
#property copyright "Quant coach research"
#property version   "1.00"

#include <Trade/Trade.mqh>

input group "Time"
input int      InpServerMinusNY   = 7;        // Server time minus New York time (hours)
input group "Risk"
input bool     InpRiskPercentMode = false;    // true = risk % of balance, false = fixed money
input double   InpRiskPercent     = 1.0;      // Risk per trade, % of balance
input double   InpRiskMoney       = 100.0;    // Risk per trade, account currency
input double   InpDDBrakePct      = 0.0;      // Halve risk when equity is this % below its peak (0 = off)
input group "Strategy (tested values)"
input double   InpStopATR         = 0.2;      // Stop distance in ATR
input double   InpTriggerATR      = 0.15;     // Breakout must be this far from the 09:30 open (ATR)
input double   InpFlatATR         = 0.15;     // Skip day if |open - prev close| below this (ATR)
input int      InpATRDays         = 14;       // ATR days (09:30-16:00 range)
input int      InpTrendDays       = 20;       // Trend average days
input int      InpORMinutes       = 10;       // Opening range length (minutes from 09:30)
input int      InpLastSignal      = 930;      // No decision after (NY minute, 930 = 15:30)
input int      InpForceClose      = 959;      // Forced close (NY minute, 15:59)
input bool     InpLongs           = true;     // Allow longs
input bool     InpShorts          = true;     // Allow shorts
input group "Filters"
input bool     InpMon = true;                 // Trade Monday
input bool     InpTue = true;                 // Trade Tuesday
input bool     InpWed = true;                 // Trade Wednesday
input bool     InpThu = true;                 // Trade Thursday
input bool     InpFri = true;                 // Trade Friday
input string   InpSkipDates = "2024.03.29,2025.01.09,2025.04.18,2026.04.03,2027.03.26"; // NY dates to skip (yyyy.mm.dd, comma separated)
input datetime InpTradeFrom = D'1970.01.01';  // No new trades before this server date (for tester comparison)
input group "Other"
input long     InpMagic           = 7302602;  // Magic number (must differ from AZL)
input int      InpWarmupDays      = 60;       // History replay on start (calendar days)
input bool     InpNotify          = false;    // Phone notification on entry/exit
input int      InpSlippagePoints  = 50;       // Max deviation for market orders (points)

CTrade   trade;
double   g_rng[], g_cls[]; int g_nd = 0;
bool     g_rthHas = false; double g_rthH, g_rthL, g_rthC;
long     g_curDate = -1;
double   g_atr = 0; bool g_atrOk = false; int g_trend = 0; double g_pdc = 0; bool g_pdcOk = false;
double   g_open = 0; bool g_openOk = false; bool g_flatOk = false;
double   g_orH = 0, g_orL = 0; bool g_orOk = false; bool g_decided = false;
string   g_dayStatus = "waiting for 09:30";
datetime g_lastProcessed = 0;
bool     g_pend = false; int g_pendDir = 0; double g_pendDist = 0; long g_pendDate = 0; datetime g_pendBar = 0;
long     g_skip[]; int g_nskip = 0;
string   g_gvLast, g_gvPeak;

long NYMin(datetime server) { return ((long)server - (long)InpServerMinusNY * 3600) / 60; }

void ParseSkipDates()
  {
   string parts[]; int k = StringSplit(InpSkipDates, ',', parts);
   ArrayResize(g_skip, MathMax(k, 0)); g_nskip = 0;
   for(int i = 0; i < k; i++)
     {
      string s = parts[i]; StringTrimLeft(s); StringTrimRight(s);
      if(StringLen(s) < 10) continue;
      g_skip[g_nskip++] = (long)StringToTime(s) / 86400;
     }
  }
bool IsSkipDate(long d) { for(int i = 0; i < g_nskip; i++) if(g_skip[i] == d) return true; return false; }
bool WeekdayAllowed(long d)
  {
   MqlDateTime t; TimeToStruct((datetime)(d * 86400), t);
   switch(t.day_of_week) { case 1: return InpMon; case 2: return InpTue; case 3: return InpWed; case 4: return InpThu; case 5: return InpFri; }
   return false;
  }

//+------------------------------------------------------------------+
//| Process one CLOSED M1 bar                                          |
//+------------------------------------------------------------------+
void ProcessBar(datetime bt, double o, double h, double l, double c)
  {
   long t = NYMin(bt), mod = t % 1440, date = t / 1440;
   g_pend = false;
   if(date != g_curDate)
     {
      if(g_rthHas)
        {
         ArrayResize(g_rng, g_nd + 1, 1024); ArrayResize(g_cls, g_nd + 1, 1024);
         g_rng[g_nd] = g_rthH - g_rthL; g_cls[g_nd] = g_rthC; g_nd++;
        }
      g_rthHas = false; g_curDate = date;
      g_atrOk = false; g_trend = 0; g_pdcOk = false;
      if(g_nd >= InpATRDays)
        { double s = 0; for(int i = g_nd - InpATRDays; i < g_nd; i++) s += g_rng[i]; g_atr = s / InpATRDays; g_atrOk = true; }
      if(g_nd >= InpTrendDays)
        {
         double s = 0; for(int i = g_nd - InpTrendDays; i < g_nd; i++) s += g_cls[i];
         double d = g_cls[g_nd - 1] - s / InpTrendDays; g_trend = d > 0 ? 1 : (d < 0 ? -1 : 0);
        }
      if(g_nd >= 1) { g_pdc = g_cls[g_nd - 1]; g_pdcOk = true; }
      g_openOk = false; g_orOk = false; g_decided = false; g_flatOk = false;
      g_dayStatus = "waiting for 09:30";
     }
   if(mod >= 570 && mod < 960)
     {
      if(!g_rthHas) { g_rthHas = true; g_rthH = h; g_rthL = l; }
      else { if(h > g_rthH) g_rthH = h; if(l < g_rthL) g_rthL = l; }
      g_rthC = c;
     }
   if(mod == 570)
     {
      g_open = o; g_openOk = true;
      g_flatOk = g_atrOk && g_pdcOk && MathAbs(g_open - g_pdc) >= InpFlatATR * g_atr;
      g_dayStatus = g_flatOk ? "building opening range" : "flat open or no history: no trade today";
     }
   if(g_openOk && mod >= 570 && mod < 570 + InpORMinutes)
     {
      if(!g_orOk) { g_orH = h; g_orL = l; g_orOk = true; }
      else { if(h > g_orH) g_orH = h; if(l < g_orL) g_orL = l; }
     }
   if(g_openOk && g_orOk && !g_decided && mod >= 570 + InpORMinutes && mod < InpLastSignal)
     {
      bool up = c > g_orH && c - g_open >= InpTriggerATR * g_atr;
      bool dn = c < g_orL && g_open - c >= InpTriggerATR * g_atr;
      if(up || dn)
        {
         g_decided = true;
         int d = up ? 1 : -1;
         bool allowed = (d == 1 && InpLongs) || (d == -1 && InpShorts);
         if(g_flatOk && d == g_trend && allowed)
           {
            g_pend = true; g_pendDir = d; g_pendDist = InpStopATR * g_atr; g_pendDate = date; g_pendBar = bt;
            g_dayStatus = StringFormat("%s signal at %s NY", d == 1 ? "LONG" : "SHORT", TimeToString((datetime)(t * 60), TIME_MINUTES));
           }
         else
            g_dayStatus = StringFormat("day decided %s but %s: no trade", d == 1 ? "UP" : "DOWN", !g_flatOk ? "flat open" : (d != g_trend ? "against 20-day trend" : "side disabled"));
        }
     }
  }

//+------------------------------------------------------------------+
bool HasPosition(ulong &ticket)
  {
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong tk = PositionGetTicket(i);
      if(tk == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) == _Symbol && PositionGetInteger(POSITION_MAGIC) == InpMagic) { ticket = tk; return true; }
     }
   return false;
  }

double CalcLots(double dist)
  {
   double risk = InpRiskPercentMode ? AccountInfoDouble(ACCOUNT_BALANCE) * InpRiskPercent / 100.0 : InpRiskMoney;
   if(InpDDBrakePct > 0)
     {
      double eq = AccountInfoDouble(ACCOUNT_EQUITY);
      double peak = GlobalVariableCheck(g_gvPeak) ? GlobalVariableGet(g_gvPeak) : eq;
      if(eq > peak) { peak = eq; GlobalVariableSet(g_gvPeak, peak); }
      if(eq < peak * (1.0 - InpDDBrakePct / 100.0)) risk *= 0.5;
     }
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK), loss1 = 0;
   if(!OrderCalcProfit(ORDER_TYPE_BUY, _Symbol, 1.0, ask, ask - dist, loss1) || loss1 >= 0)
     {
      double tv = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE), ts = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
      loss1 = -dist / ts * tv;
     }
   double lots = risk / MathAbs(loss1);
   double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP), vmin = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN), vmax = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   lots = MathFloor(lots / step) * step;
   if(lots < vmin) { Print("TDH: risk too small for minimum lot, trade skipped. lots=", lots); return 0; }
   return MathMin(lots, vmax);
  }

void Notify(string msg) { Print(msg); if(InpNotify && !MQLInfoInteger(MQL_TESTER)) SendNotification(msg); }

void OnBarOpen(datetime cur)
  {
   long t = NYMin(cur), mod = t % 1440, date = t / 1440;
   ulong tk;
   if(HasPosition(tk))
     {
      long pdate = NYMin((datetime)PositionGetInteger(POSITION_TIME)) / 1440;
      if(mod >= InpForceClose || date != pdate)
        {
         double pr = PositionGetDouble(POSITION_PROFIT);
         if(trade.PositionClose(tk)) Notify(StringFormat("TDH %s closed at %s NY, profit %.2f", _Symbol, mod >= InpForceClose ? "15:59" : "new day", pr));
        }
     }
   if(!g_pend) return;
   g_pend = false;
   if(date != g_pendDate || mod >= InpForceClose || cur < InpTradeFrom) return;
   if(IsSkipDate(date) || !WeekdayAllowed(date)) { Print("TDH: signal skipped (holiday/weekday filter)"); return; }
   if(HasPosition(tk)) return;
   if(GlobalVariableCheck(g_gvLast) && (datetime)GlobalVariableGet(g_gvLast) >= g_pendBar) return;   // restart protection
   double dist = g_pendDist, lots = CalcLots(dist);
   if(lots <= 0) return;
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID), ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   bool ok;
   if(g_pendDir == 1) ok = trade.Buy(lots, _Symbol, 0.0, NormalizeDouble(bid - dist, _Digits), 0.0, "TDH");
   else               ok = trade.Sell(lots, _Symbol, 0.0, NormalizeDouble(bid + dist + (ask - bid), _Digits), 0.0, "TDH");
   if(ok && (trade.ResultRetcode() == TRADE_RETCODE_DONE || trade.ResultRetcode() == TRADE_RETCODE_PLACED))
     {
      GlobalVariableSet(g_gvLast, (double)g_pendBar);
      Notify(StringFormat("TDH %s %s %.2f lots, stop %.1f pts (0.2 ATR)", _Symbol, g_pendDir == 1 ? "BUY" : "SELL", lots, dist));
     }
   else Print("TDH: order failed, retcode ", trade.ResultRetcode(), " ", trade.ResultRetcodeDescription());
  }

bool CatchUp()
  {
   datetime cur = iTime(_Symbol, PERIOD_M1, 0);
   if(cur == 0) return false;
   MqlRates r[];
   datetime from = g_lastProcessed > 0 ? g_lastProcessed + 60 : cur - (datetime)InpWarmupDays * 86400;
   if(from >= cur) return true;
   int n = CopyRates(_Symbol, PERIOD_M1, from, cur - 60, r);
   if(n < 0) return false;
   for(int i = 0; i < n; i++)
     {
      if(r[i].time <= g_lastProcessed || r[i].time >= cur) continue;
      ProcessBar(r[i].time, r[i].open, r[i].high, r[i].low, r[i].close);
      g_lastProcessed = r[i].time;
     }
   if(g_pend && iBarShift(_Symbol, PERIOD_M1, g_pendBar) != 1) g_pend = false;   // only act on the last closed bar
   return true;
  }

void StatusBox()
  {
   long t = NYMin(TimeCurrent()), mod = t % 1440;
   ulong tk; string ps = "flat";
   if(HasPosition(tk)) ps = StringFormat("%s %.2f lots, P/L %.2f", PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY ? "LONG" : "SHORT", PositionGetDouble(POSITION_VOLUME), PositionGetDouble(POSITION_PROFIT));
   string orTxt = g_orOk ? StringFormat("%s - %s", DoubleToString(g_orL, _Digits), DoubleToString(g_orH, _Digits)) : "n/a";
   Comment(StringFormat("TDH  Trend Day Hunter   [%s]\nNY time: %s   decisions 09:40-15:30 NY, close 15:59\nATR(14): %s   20d trend: %s   stop = %.1f\nPrev close: %s   09:30 open: %s   opening range: %s\nToday: %s\nPosition: %s\nBars processed up to: %s (server)",
                        _Symbol, TimeToString((datetime)(t * 60), TIME_DATE | TIME_MINUTES),
                        g_atrOk ? DoubleToString(g_atr, 1) : "warming up", g_trend == 1 ? "UP (longs only)" : g_trend == -1 ? "DOWN (shorts only)" : "none (no trades)",
                        g_atrOk ? InpStopATR * g_atr : 0.0, g_pdcOk ? DoubleToString(g_pdc, _Digits) : "n/a", g_openOk ? DoubleToString(g_open, _Digits) : "n/a",
                        orTxt, g_dayStatus, ps, TimeToString(g_lastProcessed, TIME_DATE | TIME_MINUTES)));
  }

int OnInit()
  {
   trade.SetExpertMagicNumber(InpMagic);
   trade.SetDeviationInPoints(InpSlippagePoints);
   trade.SetTypeFillingBySymbol(_Symbol);
   ParseSkipDates();
   g_gvLast = "TDH_" + _Symbol + "_" + IntegerToString(InpMagic) + "_last";
   g_gvPeak = "TDH_" + _Symbol + "_" + IntegerToString(InpMagic) + "_peak";
   if(MQLInfoInteger(MQL_TESTER)) { GlobalVariableDel(g_gvLast); GlobalVariableDel(g_gvPeak); }
   CatchUp();          // replay history silently: rebuilds ATR, trend and today's state after a restart
   g_pend = false;
   EventSetTimer(5);
   StatusBox();
   PrintFormat("TDH started on %s. Replayed up to %s. ATR %s, trend %d", _Symbol, TimeToString(g_lastProcessed), g_atrOk ? DoubleToString(g_atr, 1) : "n/a", g_trend);
   return INIT_SUCCEEDED;
  }

void OnDeinit(const int reason) { EventKillTimer(); Comment(""); }

void OnTick()
  {
   static datetime lastBar = 0;
   datetime cur = iTime(_Symbol, PERIOD_M1, 0);
   if(cur == 0 || cur == lastBar) return;
   lastBar = cur;
   if(!CatchUp()) return;
   OnBarOpen(cur);
   if(!MQLInfoInteger(MQL_TESTER) || MQLInfoInteger(MQL_VISUAL_MODE)) StatusBox();
  }

void OnTimer() { StatusBox(); }
//+------------------------------------------------------------------+
