//+------------------------------------------------------------------+
//|                                              AZL_OB_CHOCH.mq5    |
//|  AZL = Asia Zone -> London CHOCH  (order block + 1m change of    |
//|  character, with the 20-day trend, hold to the New York close)   |
//+------------------------------------------------------------------+
//  RULES (all times New York; NY time = server time - InpServerMinusNY hours)
//  1. Order blocks on 30m, 1h, 4h and daily bars that the EA builds itself
//     from M1 data in NY time (4h bars start 00:00/04:00/..., daily bars
//     start 18:00). Swing = fractal with 2 bars each side, known 2 bars later.
//     A close beyond the latest unbroken swing high (low) = break of
//     structure; the last bearish (bullish) candle between that swing and the
//     breaking bar (max 15 bars back) is the DEMAND (SUPPLY) zone, full candle
//     high..low. The zone exists from the end of the breaking bar.
//  2. Keep only zones created between 18:00 and 03:00 NY ("Asia zones").
//     Life: 5 NY dates (30m/1h), 10 (4h), 20 (daily).
//  3. First touch only. If the first touch is outside 03:00-15:30 the zone is
//     used up. Any 5-minute close beyond the far side of the zone kills it.
//  4. After the touch (same NY date, before 15:30): CHOCH = first M1 close
//     above the latest confirmed M1 swing high (5 bars each side) for demand,
//     below the latest M1 swing low for supply. The zone is then used up.
//  5. Trade only if the CHOCH bar is before 09:30 NY and the 20-day trend
//     agrees: previous RTH (09:30-16:00) close vs average of the last 20 RTH
//     closes. Demand -> long in uptrend, supply -> short in downtrend.
//  6. Entry: market order at the open of the next M1 bar.
//     Stop: 0.2 x ATR, ATR = average 09:30-16:00 range of the previous 14 days.
//     No target. Forced close at 15:59 NY. One position at a time per symbol.
//  Research (IC Markets M1, Oct 2023 - Sep 2026, after costs):
//     USTEC 134 trades, win 31%, PF 1.79 | US500 127 trades, win 34%, PF 1.75
//+------------------------------------------------------------------+
#property copyright "Quant coach research"
#property version   "1.00"

#include <Trade/Trade.mqh>

//--- inputs
input group "Time"
input int      InpServerMinusNY   = 7;        // Server time minus New York time (hours)
input group "Risk"
input bool     InpRiskPercentMode = false;    // true = risk % of balance, false = fixed money
input double   InpRiskPercent     = 1.0;      // Risk per trade, % of balance
input double   InpRiskMoney       = 100.0;    // Risk per trade, account currency
input double   InpDDBrakePct      = 0.0;      // Halve risk when equity is this % below its peak (0 = off)
input group "Strategy (tested values)"
input double   InpStopATR         = 0.2;      // Stop distance in ATR
input int      InpATRDays         = 14;       // ATR days (09:30-16:00 range)
input int      InpTrendDays       = 20;       // Trend average days
input int      InpSwingM1         = 5;        // M1 swing bars each side (CHOCH)
input int      InpZoneFrom        = 1080;     // Asia zone creation from (NY minute, 1080 = 18:00)
input int      InpZoneTo          = 180;      // Asia zone creation to (NY minute, 180 = 03:00)
input int      InpTouchStart      = 180;      // Touch window start (NY minute, 03:00)
input int      InpTouchEnd        = 930;      // Touch/CHOCH window end (NY minute, 15:30)
input int      InpEntryEnd        = 570;      // CHOCH must be before (NY minute, 09:30)
input int      InpForceClose      = 959;      // Forced close (NY minute, 15:59)
input bool     InpUse30m          = true;     // Use 30m zones
input bool     InpUse1h           = true;     // Use 1h zones
input bool     InpUse4h           = true;     // Use 4h zones
input bool     InpUseD1           = true;     // Use daily zones
input group "Filters"
input bool     InpMon = true;                 // Trade Monday
input bool     InpTue = true;                 // Trade Tuesday
input bool     InpWed = true;                 // Trade Wednesday
input bool     InpThu = true;                 // Trade Thursday
input bool     InpFri = true;                 // Trade Friday
input int      InpMaxTradesPerDay = 0;        // Max trades per NY day (0 = no limit, as tested)
input string   InpSkipDates = "2024.03.29,2025.01.09,2025.04.18,2026.04.03,2027.03.26"; // NY dates with US cash market closed (yyyy.mm.dd, comma separated)
input datetime InpTradeFrom = D'1970.01.01';  // No new trades before this server date (for tester comparison)
input group "Other"
input long     InpMagic           = 7302601;  // Magic number
input int      InpWarmupDays      = 60;       // History replay on start (calendar days)
input bool     InpDrawZones       = true;     // Draw Asia zones on the chart
input bool     InpNotify          = false;    // Phone notification on entry/exit
input int      InpSlippagePoints  = 50;       // Max deviation for market orders (points)

//--- types
struct HTF
  {
   int      tf;          // minutes (1440 = daily 18:00-18:00)
   bool     use;
   long     bucket;      // current bucket id
   bool     has;
   double   co, ch, cl, cc; // current bar OHLC
   double   o[], h[], l[], c[];
   int      n;
   int      shIdx; double shLvl; bool shOn;
   int      slIdx; double slLvl; bool slOn;
  };

struct Zone
  {
   int      dir;     // +1 demand, -1 supply
   double   lo, hi;
   int      exp;     // last allowed NY date index
   int      touch;   // 0 = not yet, 1 = touched
   int      tday;    // NY date index of touch
   int      tf;
   long     created; // NY minute
   int      obj;     // chart object id (-1 = none)
  };

//--- state
CTrade   trade;
HTF      g_htf[4];
Zone     g_z[];
int      g_nz = 0;
double   g_rng[], g_cls[];
int      g_nd = 0;
bool     g_rthHas = false; double g_rthH, g_rthL, g_rthC;
long     g_curDate = -1;
int      g_dayIdx = -1;
double   g_atr = 0; bool g_atrOk = false;
int      g_trend = 0;
double   g_m1h[], g_m1l[]; int g_m1n = 0;   // last 2*N+1 M1 bars (ring)
double   g_shl = 0, g_sll = 0; bool g_shOk = false, g_slOk = false;
datetime g_lastProcessed = 0;
bool     g_live = false;
// pending signal from the last processed bar
bool     g_pend = false; int g_pendDir = 0; double g_pendDist = 0; long g_pendDate = 0; datetime g_pendBar = 0; int g_pendTf = 0;
// stats for status box
string   g_lastSignal = "none";
int      g_tradesToday = 0; long g_tradesDate = -1;
long     g_skip[]; int g_nskip = 0;
int      g_objId = 0;
string   g_gvLast, g_gvPeak;

//+------------------------------------------------------------------+
long NYMin(datetime server) { return ((long)server - (long)InpServerMinusNY * 3600) / 60; }
datetime ServerFromNYMin(long nymin) { return (datetime)(nymin * 60 + (long)InpServerMinusNY * 3600); }
string TfName(int tf) { return tf == 30 ? "30m" : tf == 60 ? "1h" : tf == 240 ? "4h" : "D1"; }

//+------------------------------------------------------------------+
void ParseSkipDates()
  {
   string parts[];
   int k = StringSplit(InpSkipDates, ',', parts);
   ArrayResize(g_skip, MathMax(k, 0)); g_nskip = 0;
   for(int i = 0; i < k; i++)
     {
      string s = parts[i]; StringTrimLeft(s); StringTrimRight(s);
      if(StringLen(s) < 10) continue;
      datetime d = StringToTime(s);           // date as if UTC midnight -> NY date number
      g_skip[g_nskip++] = (long)d / 86400;
     }
  }

bool IsSkipDate(long nyDate)
  {
   for(int i = 0; i < g_nskip; i++) if(g_skip[i] == nyDate) return true;
   return false;
  }

bool WeekdayAllowed(long nyDate)
  {
   MqlDateTime t; TimeToStruct((datetime)(nyDate * 86400), t);
   switch(t.day_of_week)
     {
      case 1: return InpMon;
      case 2: return InpTue;
      case 3: return InpWed;
      case 4: return InpThu;
      case 5: return InpFri;
     }
   return false;
  }

//+------------------------------------------------------------------+
void HTFInit(HTF &b, int tf, bool use)
  {
   b.tf = tf; b.use = use; b.bucket = -1; b.has = false; b.n = 0;
   b.shOn = false; b.slOn = false;
   ArrayResize(b.o, 0, 4096); ArrayResize(b.h, 0, 4096); ArrayResize(b.l, 0, 4096); ArrayResize(b.c, 0, 4096);
  }

long HTFBucket(const HTF &b, long t) { return b.tf == 1440 ? (t - 1080) / 1440 : t / b.tf; }
long HTFEnd(const HTF &b) { return b.tf == 1440 ? b.bucket * 1440 + 1080 + 1440 : (b.bucket + 1) * b.tf; }

void AddZone(int dir, double lo, double hi, long created, int tf)
  {
   long cm = created % 1440;
   if(!(cm >= InpZoneFrom || cm <= InpZoneTo)) return;
   int life = (tf == 30 || tf == 60) ? 5 : (tf == 240 ? 10 : 20);
   ArrayResize(g_z, g_nz + 1, 256);
   g_z[g_nz].dir = dir; g_z[g_nz].lo = lo; g_z[g_nz].hi = hi; g_z[g_nz].exp = g_dayIdx + life;
   g_z[g_nz].touch = 0; g_z[g_nz].tday = -1; g_z[g_nz].tf = tf; g_z[g_nz].created = created; g_z[g_nz].obj = -1;
   if(InpDrawZones && g_live && !MQLInfoInteger(MQL_OPTIMIZATION))
     {
      int id = g_objId++;
      string nm = "AZL_Z_" + IntegerToString(id);
      datetime t1 = ServerFromNYMin(created), t2 = t1 + 6 * 3600;
      if(ObjectCreate(0, nm, OBJ_RECTANGLE, 0, t1, hi, t2, lo))
        {
         ObjectSetInteger(0, nm, OBJPROP_COLOR, dir == 1 ? clrDodgerBlue : clrDarkOrange);
         ObjectSetInteger(0, nm, OBJPROP_FILL, true);
         ObjectSetInteger(0, nm, OBJPROP_BACK, true);
         ObjectSetString(0, nm, OBJPROP_TOOLTIP, (dir == 1 ? "Demand " : "Supply ") + TfName(tf));
         g_z[g_nz].obj = id;
        }
     }
   g_nz++;
  }

void HTFFinalize(HTF &b)
  {
   int i = b.n;
   ArrayResize(b.o, i + 1, 4096); ArrayResize(b.h, i + 1, 4096); ArrayResize(b.l, i + 1, 4096); ArrayResize(b.c, i + 1, 4096);
   b.o[i] = b.co; b.h[i] = b.ch; b.l[i] = b.cl; b.c[i] = b.cc; b.n = i + 1;
   long end = HTFEnd(b);
   // 1) break of structure on bar i using swings confirmed before bar i
   if(b.shOn && b.c[i] > b.shLvl)
     {
      int lim = MathMax(b.shIdx, i - 15);
      for(int j = i - 1; j >= lim; j--)
         if(b.c[j] < b.o[j]) { if(b.use) AddZone(1, b.l[j], b.h[j], end, b.tf); break; }
      b.shOn = false;
     }
   if(b.slOn && b.c[i] < b.slLvl)
     {
      int lim = MathMax(b.slIdx, i - 15);
      for(int j = i - 1; j >= lim; j--)
         if(b.c[j] > b.o[j]) { if(b.use) AddZone(-1, b.l[j], b.h[j], end, b.tf); break; }
      b.slOn = false;
     }
   // 2) swing at p = i-2 confirmed now
   int p = i - 2;
   if(p >= 2)
     {
      if(b.h[p] > MathMax(b.h[p - 2], b.h[p - 1]) && b.h[p] >= MathMax(b.h[p + 1], b.h[p + 2])) { b.shOn = true; b.shIdx = p; b.shLvl = b.h[p]; }
      if(b.l[p] < MathMin(b.l[p - 2], b.l[p - 1]) && b.l[p] <= MathMin(b.l[p + 1], b.l[p + 2])) { b.slOn = true; b.slIdx = p; b.slLvl = b.l[p]; }
     }
  }

void HTFOnBar(HTF &b, long t, double o, double h, double l, double c)
  {
   long id = HTFBucket(b, t);
   if(b.has && id != b.bucket) HTFFinalize(b);
   if(!b.has || id != b.bucket) { b.bucket = id; b.has = true; b.co = o; b.ch = h; b.cl = l; b.cc = c; }
   else { if(h > b.ch) b.ch = h; if(l < b.cl) b.cl = l; b.cc = c; }
  }

//+------------------------------------------------------------------+
void KillZone(int i, long nymin)
  {
   if(g_z[i].obj >= 0) ObjectSetInteger(0, "AZL_Z_" + IntegerToString(g_z[i].obj), OBJPROP_TIME, 1, ServerFromNYMin(nymin));
   g_z[i].dir = 0;   // marked dead, compacted later
  }

void CompactZones()
  {
   int w = 0;
   for(int i = 0; i < g_nz; i++) if(g_z[i].dir != 0) { if(w != i) g_z[w] = g_z[i]; w++; }
   g_nz = w; ArrayResize(g_z, g_nz, 256);
  }

//+------------------------------------------------------------------+
//| Process one CLOSED M1 bar (same order as the research simulation) |
//+------------------------------------------------------------------+
void ProcessBar(datetime bt, double o, double h, double l, double c)
  {
   long t = NYMin(bt);
   long mod = t % 1440;
   long date = t / 1440;
   g_pend = false;
   // 1) new NY date: close the previous date's RTH stats, update ATR and trend
   if(date != g_curDate)
     {
      if(g_rthHas)
        {
         ArrayResize(g_rng, g_nd + 1, 1024); ArrayResize(g_cls, g_nd + 1, 1024);
         g_rng[g_nd] = g_rthH - g_rthL; g_cls[g_nd] = g_rthC; g_nd++;
        }
      g_rthHas = false; g_curDate = date; g_dayIdx++;
      g_atrOk = false; g_trend = 0;
      if(g_nd >= InpATRDays)
        {
         double s = 0; for(int i = g_nd - InpATRDays; i < g_nd; i++) s += g_rng[i];
         g_atr = s / InpATRDays; g_atrOk = true;
        }
      if(g_nd >= InpTrendDays)
        {
         double s = 0; for(int i = g_nd - InpTrendDays; i < g_nd; i++) s += g_cls[i];
         double d = g_cls[g_nd - 1] - s / InpTrendDays;
         g_trend = d > 0 ? 1 : (d < 0 ? -1 : 0);
        }
     }
   if(mod >= 570 && mod < 960)
     {
      if(!g_rthHas) { g_rthHas = true; g_rthH = h; g_rthL = l; }
      else { if(h > g_rthH) g_rthH = h; if(l < g_rthL) g_rthL = l; }
      g_rthC = c;
     }
   // 2) higher-timeframe bars (zones created here become active from this bar)
   for(int i = 0; i < 4; i++) HTFOnBar(g_htf[i], t, o, h, l, c);
   // 3) M1 swings: pivot k-N confirmed at the close of bar k
   int N = InpSwingM1, W = 2 * N + 1;
   for(int i = 0; i < W - 1; i++) { g_m1h[i] = g_m1h[i + 1]; g_m1l[i] = g_m1l[i + 1]; }
   g_m1h[W - 1] = h; g_m1l[W - 1] = l; if(g_m1n < W) g_m1n++;
   if(g_m1n == W)
     {
      double ph = g_m1h[N], pl = g_m1l[N];
      bool isH = true, isL = true;
      for(int i = 0; i < N; i++) { if(!(ph > g_m1h[i])) isH = false; if(!(pl < g_m1l[i])) isL = false; }
      for(int i = N + 1; i < W; i++) { if(!(ph >= g_m1h[i])) isH = false; if(!(pl <= g_m1l[i])) isL = false; }
      if(isH) { g_shl = ph; g_shOk = true; }
      if(isL) { g_sll = pl; g_slOk = true; }
     }
   // 5) zones
   bool is5 = (t % 5 == 4);
   for(int i = 0; i < g_nz; i++)
     {
      int sd = g_z[i].dir;
      if(sd == 0) continue;
      if(g_dayIdx > g_z[i].exp) { KillZone(i, t); continue; }
      if(g_z[i].touch == 0 && ((sd == 1 && l <= g_z[i].hi) || (sd == -1 && h >= g_z[i].lo)))
        {
         if(mod < InpTouchStart || mod >= InpTouchEnd) { KillZone(i, t); continue; }
         g_z[i].touch = 1; g_z[i].tday = g_dayIdx;
        }
      if(is5 && ((sd == 1 && c < g_z[i].lo) || (sd == -1 && c > g_z[i].hi))) { KillZone(i, t); continue; }
      if(g_z[i].touch == 1)
        {
         if(mod >= InpTouchEnd || g_dayIdx != g_z[i].tday) { KillZone(i, t); continue; }
         bool ok = (sd == 1) ? g_shOk : g_slOk;
         double lvl = (sd == 1) ? g_shl : g_sll;
         if(ok && ((sd == 1 && c > lvl) || (sd == -1 && c < lvl)))
           {
            if(mod < InpEntryEnd && g_trend == sd && g_atrOk && !g_pend)
              {
               g_pend = true; g_pendDir = sd; g_pendDist = InpStopATR * g_atr; g_pendDate = date; g_pendBar = bt; g_pendTf = g_z[i].tf;
              }
            g_lastSignal = StringFormat("%s CHOCH %s %s zone %s (%s)", TimeToString((datetime)(t * 60), TIME_DATE | TIME_MINUTES),
                                        sd == 1 ? "LONG" : "SHORT", TfName(g_z[i].tf), DoubleToString(g_z[i].lo, _Digits) + "-" + DoubleToString(g_z[i].hi, _Digits),
                                        (mod < InpEntryEnd && g_trend == sd && g_atrOk) ? "valid" : (g_trend != sd ? "against trend: skip" : "after 09:30: skip"));
            KillZone(i, t); continue;
           }
        }
     }
   CompactZones();
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
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double loss1 = 0;
   if(!OrderCalcProfit(ORDER_TYPE_BUY, _Symbol, 1.0, ask, ask - dist, loss1) || loss1 >= 0)
     {
      double tv = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE), ts = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
      loss1 = -dist / ts * tv;
     }
   double lots = risk / MathAbs(loss1);
   double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP), vmin = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN), vmax = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   lots = MathFloor(lots / step) * step;
   if(lots < vmin) { Print("AZL: risk too small for minimum lot, trade skipped. lots=", lots); return 0; }
   return MathMin(lots, vmax);
  }

void Notify(string msg) { Print(msg); if(InpNotify && g_live && !MQLInfoInteger(MQL_TESTER)) SendNotification(msg); }

//+------------------------------------------------------------------+
//| Actions at the open of the current (forming) bar                   |
//+------------------------------------------------------------------+
void OnBarOpen(datetime cur)
  {
   long t = NYMin(cur), mod = t % 1440, date = t / 1440;
   ulong tk;
   // forced close at 15:59 NY or if a new NY date started
   if(HasPosition(tk))
     {
      datetime pt = (datetime)PositionGetInteger(POSITION_TIME);
      long pdate = NYMin(pt) / 1440;
      if(mod >= InpForceClose || date != pdate)
        {
         double pr = PositionGetDouble(POSITION_PROFIT);
         if(trade.PositionClose(tk)) Notify(StringFormat("AZL %s closed at %s NY, profit %.2f", _Symbol, mod >= InpForceClose ? "15:59" : "new day", pr));
        }
     }
   if(!g_pend) return;
   g_pend = false;
   if(date != g_pendDate || mod >= InpForceClose) return;
   if(cur < InpTradeFrom) return;
   if(IsSkipDate(date) || !WeekdayAllowed(date)) { Print("AZL: signal skipped (holiday/weekday filter)"); return; }
   if(HasPosition(tk)) { Print("AZL: signal skipped, position already open"); return; }
   if(g_tradesDate != date) { g_tradesDate = date; g_tradesToday = 0; }
   if(InpMaxTradesPerDay > 0 && g_tradesToday >= InpMaxTradesPerDay) return;
   // restart protection: never act twice on the same signal bar
   if(GlobalVariableCheck(g_gvLast) && (datetime)GlobalVariableGet(g_gvLast) >= g_pendBar) return;
   double dist = g_pendDist;
   double lots = CalcLots(dist);
   if(lots <= 0) return;
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID), ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   bool ok;
   string cm = "AZL " + TfName(g_pendTf);
   if(g_pendDir == 1)
      ok = trade.Buy(lots, _Symbol, 0.0, NormalizeDouble(bid - dist, _Digits), 0.0, cm);
   else
      ok = trade.Sell(lots, _Symbol, 0.0, NormalizeDouble(bid + dist + (ask - bid), _Digits), 0.0, cm);   // stop on bid level + spread (ask triggers)
   if(ok && (trade.ResultRetcode() == TRADE_RETCODE_DONE || trade.ResultRetcode() == TRADE_RETCODE_PLACED))
     {
      GlobalVariableSet(g_gvLast, (double)g_pendBar);
      g_tradesToday++;
      Notify(StringFormat("AZL %s %s %.2f lots, stop %.1f pts (0.2 ATR), zone %s", _Symbol, g_pendDir == 1 ? "BUY" : "SELL", lots,
                          dist, TfName(g_pendTf)));
     }
   else
      Print("AZL: order failed, retcode ", trade.ResultRetcode(), " ", trade.ResultRetcodeDescription());
  }

//+------------------------------------------------------------------+
//| Process all closed M1 bars after g_lastProcessed                   |
//+------------------------------------------------------------------+
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
   // a signal is only actionable if it came from the last closed bar
   if(g_pend && iBarShift(_Symbol, PERIOD_M1, g_pendBar) != 1) g_pend = false;
   return true;
  }

//+------------------------------------------------------------------+
void StatusBox()
  {
   datetime now = TimeCurrent();
   long t = NYMin(now), mod = t % 1440;
   int nd = 0, ns = 0;
   string near = "";
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   for(int i = 0; i < g_nz; i++)
     {
      if(g_z[i].dir == 1) nd++; else if(g_z[i].dir == -1) ns++;
      if(g_z[i].dir != 0 && g_z[i].touch == 0 && StringLen(near) < 300)
         near += StringFormat("\n   %s %s %s-%s  %s", g_z[i].dir == 1 ? "DEMAND" : "SUPPLY", TfName(g_z[i].tf), DoubleToString(g_z[i].lo, _Digits), DoubleToString(g_z[i].hi, _Digits),
                              g_z[i].dir == g_trend ? "(with trend: tradable)" : "(against trend: ignored)");
     }
   ulong tk; string ps = "flat";
   if(HasPosition(tk)) ps = StringFormat("%s %.2f lots, P/L %.2f", PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY ? "LONG" : "SHORT", PositionGetDouble(POSITION_VOLUME), PositionGetDouble(POSITION_PROFIT));
   string win = (mod >= InpTouchStart && mod < InpEntryEnd) ? "ACTIVE (03:00-09:30 NY)" : "waiting (entries 03:00-09:30 NY)";
   Comment(StringFormat("AZL  Asia Zone -> London CHOCH   [%s]\nNY time: %s   window: %s\nATR(14): %s   20d trend: %s   stop = %.1f\nPosition: %s\nActive Asia zones: %d demand, %d supply%s\nLast signal: %s\nBars processed up to: %s (server)",
                        _Symbol, TimeToString((datetime)(t * 60), TIME_DATE | TIME_MINUTES), win,
                        g_atrOk ? DoubleToString(g_atr, 1) : "warming up", g_trend == 1 ? "UP (longs only)" : g_trend == -1 ? "DOWN (shorts only)" : "none (no trades)",
                        g_atrOk ? InpStopATR * g_atr : 0.0, ps, nd, ns, near, g_lastSignal, TimeToString(g_lastProcessed, TIME_DATE | TIME_MINUTES)));
  }

//+------------------------------------------------------------------+
int OnInit()
  {
   trade.SetExpertMagicNumber(InpMagic);
   trade.SetDeviationInPoints(InpSlippagePoints);
   trade.SetTypeFillingBySymbol(_Symbol);
   HTFInit(g_htf[0], 30, InpUse30m);
   HTFInit(g_htf[1], 60, InpUse1h);
   HTFInit(g_htf[2], 240, InpUse4h);
   HTFInit(g_htf[3], 1440, InpUseD1);
   ArrayResize(g_m1h, 2 * InpSwingM1 + 1); ArrayResize(g_m1l, 2 * InpSwingM1 + 1); g_m1n = 0;
   ArrayResize(g_z, 0, 256); g_nz = 0;
   ParseSkipDates();
   g_gvLast = "AZL_" + _Symbol + "_" + IntegerToString(InpMagic) + "_last";
   g_gvPeak = "AZL_" + _Symbol + "_" + IntegerToString(InpMagic) + "_peak";
   if(MQLInfoInteger(MQL_TESTER)) { GlobalVariableDel(g_gvLast); GlobalVariableDel(g_gvPeak); }
   // replay history silently (restart-safe: rebuilds zones, ATR and trend)
   g_live = false;
   CatchUp();
   g_pend = false;
   g_live = true;
   EventSetTimer(5);
   StatusBox();
   PrintFormat("AZL started on %s. Replayed history up to %s. ATR %s, trend %d, zones %d", _Symbol, TimeToString(g_lastProcessed),
               g_atrOk ? DoubleToString(g_atr, 1) : "n/a", g_trend, g_nz);
   return INIT_SUCCEEDED;
  }

void OnDeinit(const int reason)
  {
   EventKillTimer();
   Comment("");
   if(reason == REASON_REMOVE) ObjectsDeleteAll(0, "AZL_Z_");
  }

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
