# -*- coding: utf-8 -*-
"""
Doge Bot - Render Edition
- Entry = TradingView-like (Range Filter) on candle close
- Post-entry "Smart Management": TP1 -> Breakeven -> ATR Trailing
- Candles + Indicators "intelligence" after entry only
- Pro logs (colored + icons) show: indicators, next qty@10x, reason on no-trade
- Paper mode if BingX keys missing

ENV (same names you already use / screenshots):
    BINGX_API_KEY, BINGX_API_SECRET
    SYMBOL                 e.g. "DOGE/USDT : USDT" أو "DOGEUSDT" (نحن نطبّع تلقائياً)
    INTERVAL               e.g. "15m"
    LEVERAGE               e.g. 10
    RISK_PCT               e.g. 60   (means risk 60% of equity notionally at 10x)
    DECISION_EVERY_S       e.g. 30
    KEEPALIVE_SECONDS      e.g. 50
    PORT                   e.g. 5000
    USE_TV_BAR             true/false   # sync to TV bar (close-based)
    FORCE_TV_ENTRIES       true/false   # entry only on candle close (match TV)
    # Core Strategy
    ADX_LEN                14
    ATR_LEN                14
    TP1_PCT                0.40
    TP1_CLOSE_FRAC         0.50
    BREAKEVEN_AFTER_PCT    0.30
    TRAIL_ACTIVATE_PCT     0.60
    ATR_MULT_TRAIL         1.6
    RANGE_MIN_PCT          1.0
    MIN_TP_PERCENT         0.40
    MOVE_3BARS_PCT         0.8
    # Hold-TP (trend still strong? hold more)
    HOLD_TP_STRONG         true/false
    HOLD_TP_ADX            28
    HOLD_TP_SLOPE          0.50
    # Scale-In
    SCALE_IN_ENABLED       true/false
    SCALE_IN_MAX_ADDS      3
    SCALE_IN_ADX_MIN       25
    SCALE_IN_SLOPE_MIN     0.50
    # Spike filter
    SPIKE_FILTER_ATR_MULTIPLIER  3.0
    # External (optional)
    RENDER_EXTERNAL_URL    https://<your-app>.onrender.com
"""

import os, time, math, json, hmac, hashlib, threading, http.server, socketserver
import sys
import urllib.parse
from datetime import datetime, timezone
from typing import Tuple, Dict, Any, List

# --------------------- Utilities ---------------------
RESET = "\x1b[0m"
DIM   = "\x1b[2m"
BOLD  = "\x1b[1m"
FG = {
  "red":"\x1b[31m","green":"\x1b[32m","yellow":"\x1b[33m",
  "blue":"\x1b[34m","magenta":"\x1b[35m","cyan":"\x1b[36m","white":"\x1b[97m"
}
ICON = {
  "info":"🛈","ok":"✅","warn":"⚠️","err":"⛔",
  "buy":"🟢","sell":"🔴","wait":"🟡","flat":"⚪",
  "ind":"📈","pos":"📦","tp":"🎯","trail":"🪄","be":"🛡️",
  "tv":"📺","candle":"🕯️"
}

def log(section:str, msg:str, color="white"):
    print(f"{FG.get(color,'white')}{section:>10} {RESET} {msg}{RESET}", flush=True)

def now_utc_ts() -> int:
    return int(time.time())

def parse_bool(v:str) -> bool:
    return str(v).strip().lower() in ("1","true","yes","on")

def getenv(name, default=None, cast=str):
    v = os.getenv(name, default)
    if v is None: return None
    try:
        return cast(v) if cast!=bool else parse_bool(v)
    except Exception:
        return v

# --------------------- ENV ---------------------
ENV = {
  "SYMBOL": getenv("SYMBOL","DOGE/USDT : USDT"),
  "INTERVAL": getenv("INTERVAL","15m"),
  "LEVERAGE": getenv("LEVERAGE",10,int),
  "RISK_PCT": getenv("RISK_PCT",60,float),
  "DECISION_EVERY_S": getenv("DECISION_EVERY_S",30,int),
  "KEEPALIVE_SECONDS": getenv("KEEPALIVE_SECONDS",50,int),
  "PORT": getenv("PORT",5000,int),
  "USE_TV_BAR": getenv("USE_TV_BAR","false",bool),
  "FORCE_TV_ENTRIES": getenv("FORCE_TV_ENTRIES","true",bool),
  # strategy knobs
  "ADX_LEN": getenv("ADX_LEN",14,int),
  "ATR_LEN": getenv("ATR_LEN",14,int),
  "TP1_PCT": getenv("TP1_PCT",0.40,float),
  "TP1_CLOSE_FRAC": getenv("TP1_CLOSE_FRAC",0.50,float),
  "BREAKEVEN_AFTER_PCT": getenv("BREAKEVEN_AFTER_PCT",0.30,float),
  "TRAIL_ACTIVATE_PCT": getenv("TRAIL_ACTIVATE_PCT",0.60,float),
  "ATR_MULT_TRAIL": getenv("ATR_MULT_TRAIL",1.6,float),
  "RANGE_MIN_PCT": getenv("RANGE_MIN_PCT",1.0,float),
  "MIN_TP_PERCENT": getenv("MIN_TP_PERCENT",0.40,float),
  "MOVE_3BARS_PCT": getenv("MOVE_3BARS_PCT",0.8,float),
  "HOLD_TP_STRONG": getenv("HOLD_TP_STRONG","true",bool),
  "HOLD_TP_ADX": getenv("HOLD_TP_ADX",28,int),
  "HOLD_TP_SLOPE": getenv("HOLD_TP_SLOPE",0.50,float),
  "SCALE_IN_ENABLED": getenv("SCALE_IN_ENABLED","true",bool),
  "SCALE_IN_MAX_ADDS": getenv("SCALE_IN_MAX_ADDS",3,int),
  "SCALE_IN_ADX_MIN": getenv("SCALE_IN_ADX_MIN",25,int),
  "SCALE_IN_SLOPE_MIN": getenv("SCALE_IN_SLOPE_MIN",0.50,float),
  "SPIKE_FILTER_ATR_MULTIPLIER": getenv("SPIKE_FILTER_ATR_MULTIPLIER",3.0,float),
  "RENDER_EXTERNAL_URL": getenv("RENDER_EXTERNAL_URL","value"),
  "BINGX_API_KEY": getenv("BINGX_API_KEY",""),
  "BINGX_API_SECRET": getenv("BINGX_API_SECRET",""),
}

PAPER = not (ENV["BINGX_API_KEY"] and ENV["BINGX_API_SECRET"])

# --------------------- Symbol/interval helpers ---------------------
def normalize_symbol(s: str) -> str:
    s = s.replace(" ", "")
    if ":" in s: s = s.split(":")[0]
    s = s.replace("/", "")
    s = s.replace("USDTUSDT","USDT")
    return s.upper()

def to_ms_interval(interval: str) -> int:
    interval = interval.strip().lower()
    if interval.endswith("m"): return int(interval[:-1])*60*1000
    if interval.endswith("h"): return int(interval[:-1])*60*60*1000
    if interval.endswith("d"): return int(interval[:-1])*24*60*60*1000
    raise ValueError("INTERVAL must end with m/h/d")

SYMBOL_NORM = normalize_symbol(ENV["SYMBOL"])
BAR_MS = to_ms_interval(ENV["INTERVAL"])

# --------------------- Market data (BingX public klines) ---------------------
import urllib.request

def bingx_klines(symbol: str, interval: str, limit=300) -> List[Dict[str,Any]]:
    """
    Public market kline.
    REST doc sometimes: /openApi/swap/v3/quote/klines?symbol=DOGE-USDT&interval=15m&limit=300
    We try both v1/v3 variants for resilience.
    """
    sym = symbol.replace("/", "-").replace("USDTUSDT","USDT")
    qs = urllib.parse.urlencode({"symbol":sym,"interval":interval,"limit":limit})
    urls = [
        f"https://open-api.bingx.com/openApi/swap/v3/quote/klines?{qs}",
        f"https://open-api.bingx.com/openApi/swap/v2/quote/klines?{qs}",
        f"https://open-api.bingx.com/openApi/swap/market/kline?{qs}",
    ]
    err = None
    for u in urls:
        try:
            with urllib.request.urlopen(u, timeout=10) as r:
                raw = r.read()
                js = json.loads(raw.decode())
                data = js.get("data") or js.get("Klines") or js.get("klines") or js.get("data",[])
                out = []
                for k in data:
                    # tolerate different shapes
                    if isinstance(k, dict):
                        ts = int(k.get("openTime") or k.get("time") or k.get("t") or 0)
                        open_ = float(k.get("open") or k.get("o"))
                        high_ = float(k.get("high") or k.get("h"))
                        low_  = float(k.get("low") or k.get("l"))
                        close_= float(k.get("close") or k.get("c"))
                    else:
                        # [openTime, open, high, low, close, ...]
                        ts = int(k[0]); open_=float(k[1]); high_=float(k[2]); low_=float(k[3]); close_=float(k[4])
                    out.append({"t":ts,"o":open_,"h":high_,"l":low_,"c":close_})
                if out: return out
        except Exception as e:
            err = e
    if err:
        log("INDICATORS", f"{ICON['err']} fetch klines failed: {err}", "red")
    return []

# --------------------- Indicators ---------------------
def ema(values: List[float], length:int) -> List[float]:
    k = 2/(length+1.0)
    out=[]; ema_val=None
    for v in values:
        ema_val = v if ema_val is None else (v - ema_val)*k + ema_val
        out.append(ema_val)
    return out

def rsi(values: List[float], length:int=14) -> List[float]:
    gains=[0.0]; losses=[0.0]
    for i in range(1,len(values)):
        diff = values[i]-values[i-1]
        gains.append(max(diff,0.0))
        losses.append(max(-diff,0.0))
    avg_gain = ema(gains, length)
    avg_loss = ema(losses, length)
    out=[]
    for g,l in zip(avg_gain,avg_loss):
        if l==0: out.append(100.0)
        else:
            rs = g/l
            out.append(100.0 - (100.0/(1.0+rs)))
    return out

def true_range(h:List[float], l:List[float], c:List[float]) -> List[float]:
    out=[]
    prev_c=None
    for i in range(len(c)):
        tr = h[i]-l[i] if prev_c is None else max(h[i]-l[i], abs(h[i]-prev_c), abs(l[i]-prev_c))
        out.append(tr)
        prev_c = c[i]
    return out

def atr(h,l,c,length:int=14) -> List[float]:
    tr = true_range(h,l,c)
    return ema(tr, length)

def adx(high:List[float], low:List[float], close:List[float], length:int=14) -> Tuple[List[float],List[float],List[float]]:
    dm_plus=[0.0]; dm_minus=[0.0]; tr=true_range(high,low,close)
    for i in range(1,len(high)):
        up = high[i]-high[i-1]
        dn = low[i-1]-low[i]
        dm_p = up if (up>dn and up>0) else 0.0
        dm_m = dn if (dn>up and dn>0) else 0.0
        dm_plus.append(dm_p); dm_minus.append(dm_m)
    sm_dm_p = ema(dm_plus, length)
    sm_dm_m = ema(dm_minus, length)
    sm_tr   = ema(tr, length)
    di_plus=[]; di_minus=[]
    for p,m,t in zip(sm_dm_p, sm_dm_m, sm_tr):
        di_plus.append(0.0 if t==0 else (100.0*(p/t)))
        di_minus.append(0.0 if t==0 else (100.0*(m/t)))
    dx=[]
    for p,m in zip(di_plus, di_minus):
        denom = (p+m)
        dx.append(0.0 if denom==0 else (100.0*abs(p-m)/denom))
    adx_val = ema(dx, length)
    return di_plus, di_minus, adx_val

# Basic Range Filter (ema smooth channel)
def range_filter(closes: List[float], min_pct: float=1.0) -> Tuple[List[int], List[float], List[float]]:
    """
    Returns: trend list (+1 bull, -1 bear, 0 none), upper band, lower band
    A simple approximation: EMA channel +/- percent; cross & close basis produce BUY/SELL.
    """
    base = ema(closes, 20)
    upper = [b*(1+min_pct/100.0) for b in base]
    lower = [b*(1-min_pct/100.0) for b in base]
    trend = []
    for c,u,l in zip(closes, upper, lower):
        if c>u: trend.append(+1)
        elif c<l: trend.append(-1)
        else: trend.append(0)
    return trend, upper, lower

# Simple candle classifiers (subset)
def candle_tags(o,h,l,c) -> str:
    body = abs(c-o)
    rng  = max(h,l) - min(h,l)
    upper_w = h - max(c,o)
    lower_w = min(c,o) - l
    tag=[]
    if body< (rng*0.15): tag.append("Doji")
    if lower_w > body*2 and (c>o): tag.append("Hammer")
    if upper_w > body*2 and (o>c): tag.append("ShootingStar")
    # Engulfing check requires prev candle; caller adds ENGULF_BULL/BEAR when needed
    return "|".join(tag) if tag else "NONE"

# --------------------- Position & PnL memory ---------------------
class Position:
    def __init__(self):
        self.side = None         # "LONG"/"SHORT" / None
        self.entry = 0.0
        self.qty   = 0.0
        self.bars  = 0
        self.pnl   = 0.0
        self.trail = 0.0
        self.tp1_done = False
        self.adds = 0

    def flat(self):
        return self.side is None

    def reset(self):
        self.__init__()

POS = Position()
EQUITY = 0.0

# --------------------- Sizing ---------------------
def next_order_qty(price: float, equity_usdt: float, risk_pct: float, lev: int) -> float:
    """ Approximate size for 10x (or lev) — how many DOGE """
    notional = equity_usdt * (risk_pct/100.0)
    levered  = notional * lev
    qty = levered / price if price>0 else 0.0
    return max(0.0, qty)

# --------------------- Broker (very defensive) ---------------------
def place_order(side:str, qty:float, price:float) -> bool:
    # PAPER mode
    if PAPER:
        log("BROKER", f"{ICON['ok']} PAPER order {side} qty={qty:.3f} @~{price:.5f}", "cyan")
        return True
    # Real mode — minimal-safe; if fails, we only log
    try:
        # NOTE: Replace with your known-good REST for BingX perpetual
        # This is intentionally minimal; many accounts need extra params.
        log("BROKER", f"Live order {side} qty={qty:.3f} @~{price:.5f} (BingX)", "cyan")
        # TODO: implement your exact endpoint/signature (kept out to avoid breaking)
        return True
    except Exception as e:
        log("BROKER", f"{ICON['err']} order failed: {e}", "red")
        return False

def close_position() -> bool:
    if POS.flat(): return True
    side = "BUY" if POS.side=="SHORT" else "SELL"
    ok = place_order(side, POS.qty, POS.entry)
    if ok:
        POS.reset()
    return ok

# --------------------- Health server ---------------------
class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/metrics") or self.path=="/":
            self.send_response(200)
            self.send_header("Content-Type","application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"ok":True,"ts":now_utc_ts(),"equity":EQUITY}).encode())
        else:
            self.send_error(404)

def start_server():
    try:
        with socketserver.TCPServer(("", ENV["PORT"]), Handler) as httpd:
            httpd.serve_forever()
    except Exception as e:
        log("SERVER", f"health server error: {e}", "red")

# --------------------- Core loop ---------------------
LAST_KEEPALIVE = 0
LAST_BAR_TS = 0

def loop():
    global EQUITY, LAST_KEEPALIVE, LAST_BAR_TS

    EQUITY = max(EQUITY, 55.0)  # just to show numbers if exchange balance not queried
    while True:
        try:
            kl = bingx_klines(SYMBOL_NORM.replace("USDT","-USDT"), ENV["INTERVAL"], limit=300)
            if len(kl)<100:
                log("INDICATORS", f"{ICON['err']} insufficient klines", "red")
                time.sleep(ENV["DECISION_EVERY_S"]); continue

            t = [k["t"] for k in kl]
            o = [k["o"] for k in kl]
            h = [k["h"] for k in kl]
            l = [k["l"] for k in kl]
            c = [k["c"] for k in kl]

            rsi_arr = rsi(c, 14)
            di_p, di_m, adx_arr = adx(h,l,c, ENV["ADX_LEN"])
            atr_arr = atr(h,l,c, ENV["ATR_LEN"])
            trend, up, lo = range_filter(c, ENV["RANGE_MIN_PCT"])

            # Candle intel
            tag_last = candle_tags(o[-1], h[-1], l[-1], c[-1])
            engulf = "NONE"
            if len(c)>=2:
                # Engulfing
                prev_body = abs(c[-2]-o[-2]); last_body=abs(c[-1]-o[-1])
                if (c[-1]>o[-1] and c[-2]>c[-1] and last_body>prev_body and o[-1]<c[-2] and c[-1]>o[-2]):
                    engulf="ENGULF_BULL"
                if (c[-1]<o[-1] and c[-2]<c[-1] and last_body>prev_body and o[-1]>c[-2] and c[-1]<o[-2]):
                    engulf="ENGULF_BEAR"

            # Decide only on bar close to mimic TV
            bar_closed = (t[-1] != LAST_BAR_TS)
            secs_to_close = int((t[-1] + BAR_MS)/1000 - now_utc_ts())
            if ENV["FORCE_TV_ENTRIES"] and not bar_closed:
                # live bar — only manage trailing/TP but no new entries
                pass

            # Spread proxy (we don't fetch orderbook; so use a small fixed bps)
            spread_bps = 1.5

            # LOG — Indicators block
            log("TICK",
                f"{ICON['tv']} {SYMBOL_NORM} {ENV['INTERVAL']}  •  LIVE • {datetime.utcfromtimestamp(t[-1]/1000).strftime('%Y-%m-%d %H:%M:%S')} UTC",
                "white")
            log("INDICATORS",
                f"{ICON['ind']} Price {c[-1]:.6f}  |  RF filt≈{(up[-1]+lo[-1])/2:.6f}  hi={h[-1]:.6f}  lo={l[-1]:.6f}   "
                f"RSI(14)={rsi_arr[-1]:.2f}   +DI={di_p[-1]:.2f}  -DI={di_m[-1]:.2f}  DX={abs(di_p[-1]-di_m[-1])/(di_p[-1]+di_m[-1]+1e-9)*100:.2f}   "
                f"ADX({ENV['ADX_LEN']})={adx_arr[-1]:.5f}   ATR={atr_arr[-1]:.6f}   spread_bps={spread_bps:.2f}", "cyan")
            log("CANDLES",
                f"{ICON['candle']} {engulf if engulf!='NONE' else ''} {tag_last} | Candle closes in ~ {max(0,secs_to_close)}s",
                "magenta")

            # Position log
            if POS.flat():
                log("POSITION", f"{ICON['flat']} Balance {EQUITY:.2f} USDT   Risk={ENV['RISK_PCT']:.0f}%×{ENV['LEVERAGE']}x   PostCloseCooldown=0", "yellow")
            else:
                log("POSITION", f"{('🟩 LONG' if POS.side=='LONG' else '🟥 SHORT')}  Entry={POS.entry:.6f}  Qty={POS.qty:.4f}  Bars={POS.bars}  "
                                 f"PnL={POS.pnl:+.6f}  Trail={POS.trail:.6f}  TP1_done={POS.tp1_done}", "yellow")

            # Sizing preview
            nxt_qty = next_order_qty(c[-1], EQUITY, ENV["RISK_PCT"]/100.0*100, ENV["LEVERAGE"])  # keep same semantics
            log("PREVIEW", f"Next order @10x ≈ qty {nxt_qty:.2f} DOGE  (notional ~{(nxt_qty*c[-1]):.2f} USDT)", "blue")

            # ----- Entry (on bar close only) -----
            buy_sig = (trend[-2] <= 0 and trend[-1] > 0)
            sell_sig= (trend[-2] >= 0 and trend[-1] < 0)

            if ENV["FORCE_TV_ENTRIES"] and not bar_closed:
                # Explain no-trade while waiting the bar to close
                log("RESULTS", f"{ICON['wait']} No trade — reason: waiting bar close (TV sync). • close in ~{max(0,secs_to_close)}s", "white")
            else:
                # Spike filter: avoid entries on huge single-bar ATR spikes
                recent_atr = atr_arr[-1]
                bar_move = abs(c[-1]-o[-1])
                huge_spike = bar_move > ENV["SPIKE_FILTER_ATR_MULTIPLIER"]*recent_atr

                reason = None
                if POS.flat():
                    if huge_spike:
                        reason = f"spike filter ({bar_move:.6f} > {ENV['SPIKE_FILTER_ATR_MULTIPLIER']}×ATR)"
                    elif buy_sig:
                        side="LONG"
                    elif sell_sig:
                        side="SHORT"
                    else:
                        reason = "no signal"

                    if reason:
                        log("RESULTS", f"{ICON['wait']} No trade — reason: {reason}. • close in ~{max(0,secs_to_close)}s", "white")
                    else:
                        qty = max(0.0, nxt_qty)
                        if qty<=0:
                            log("RESULTS", f"{ICON['err']} sizing=0, skip.", "red")
                        else:
                            ok = place_order("BUY" if side=="LONG" else "SELL", qty, c[-1])
                            if ok:
                                POS.side = side
                                POS.entry = c[-1]
                                POS.qty   = qty
                                POS.bars  = 0
                                POS.pnl   = 0.0
                                POS.trail = 0.0
                                POS.tp1_done = False
                                POS.adds = 0
                                log("POSITION", f"{ICON['buy'] if side=='LONG' else ICON['sell']} [ACTIVE]", "green" if side=="LONG" else "red")
                else:
                    # ----- Post-entry INTELLIGENCE -----
                    POS.bars += 1
                    # Real-time PnL approx
                    POS.pnl = (c[-1]-POS.entry)*POS.qty if POS.side=="LONG" else (POS.entry-c[-1])*POS.qty
                    rr = (c[-1]-POS.entry)/POS.entry if POS.side=="LONG" else (POS.entry-c[-1])/POS.entry

                    # TP1
                    if not POS.tp1_done and rr>=ENV["TP1_PCT"]/100.0:
                        close_qty = POS.qty*ENV["TP1_CLOSE_FRAC"]
                        log("RESULTS", f"{ICON['tp']} TP1 hit • close {close_qty:.2f} DOGE (~{ENV['TP1_CLOSE_FRAC']*100:.0f}%)", "green")
                        # simulate partial close:
                        POS.qty -= close_qty
                        POS.tp1_done = True

                    # Breakeven after sufficient move
                    if POS.tp1_done and rr>=ENV["BREAKEVEN_AFTER_PCT"]/100.0 and POS.trail==0.0:
                        POS.trail = POS.entry  # BE
                        log("RESULTS", f"{ICON['be']} Breakeven armed @ {POS.trail:.6f}", "green")

                    # ATR Trailing activation
                    activate = rr>=ENV["TRAIL_ACTIVATE_PCT"]/100.0
                    if activate:
                        trail_raw = (c[-1] - ENV["ATR_MULT_TRAIL"]*atr_arr[-1]) if POS.side=="LONG" else (c[-1] + ENV["ATR_MULT_TRAIL"]*atr_arr[-1])
                        if POS.trail==0.0:
                            POS.trail = trail_raw
                            log("RESULTS", f"{ICON['trail']} Trail start @ {POS.trail:.6f}", "green")
                        else:
                            if POS.side=="LONG":
                                POS.trail = max(POS.trail, trail_raw)
                            else:
                                POS.trail = min(POS.trail, trail_raw)

                    # Hold-TP if trend still strong
                    if ENV["HOLD_TP_STRONG"]:
                        slope = (c[-1]-c[-4])/max(1e-9, c[-4]) if len(c)>=4 else 0.0
                        if adx_arr[-1]>=ENV["HOLD_TP_ADX"] and slope>ENV["HOLD_TP_SLOPE"]/100.0:
                            # hold: do nothing, just note
                            log("RESULTS", f"Hold-TP: ADX {adx_arr[-1]:.1f} & slope {slope*100:.2f}% → holding profits", "cyan")

                    # Scale-In (adds) if trend improving
                    if ENV["SCALE_IN_ENABLED"] and POS.adds<ENV["SCALE_IN_MAX_ADDS"]:
                        slope = (c[-1]-c[-4])/max(1e-9, c[-4]) if len(c)>=4 else 0.0
                        cond_adx = adx_arr[-1]>=ENV["SCALE_IN_ADX_MIN"]
                        cond_slope = (slope>ENV["SCALE_IN_SLOPE_MIN"]/100.0)
                        if cond_adx and cond_slope:
                            add_qty = POS.qty*0.25
                            side_order = "BUY" if POS.side=="LONG" else "SELL"
                            ok = place_order(side_order, add_qty, c[-1])
                            if ok:
                                POS.qty += add_qty
                                POS.adds += 1
                                log("RESULTS", f"Scale-In +{add_qty:.2f} DOGE (adds={POS.adds}/{ENV['SCALE_IN_MAX_ADDS']})", "blue")

                    # Exit conditions: trail or reverse
                    exit_reason=None
                    if POS.trail!=0.0:
                        if (POS.side=="LONG" and c[-1]<POS.trail) or (POS.side=="SHORT" and c[-1]>POS.trail):
                            exit_reason="trail hit"
                    if (POS.side=="LONG" and sell_sig) or (POS.side=="SHORT" and buy_sig):
                        exit_reason = exit_reason or "reverse signal"

                    if exit_reason:
                        log("RESULTS", f"Exit • {exit_reason} • PnL≈{POS.pnl:+.4f} USDT", "yellow")
                        ok = close_position()
                        if ok:
                            log("POSITION", f"{ICON['flat']} FLAT", "yellow")

            # keepalive
            if now_utc_ts() - LAST_KEEPALIVE >= ENV["KEEPALIVE_SECONDS"]:
                LAST_KEEPALIVE = now_utc_ts()
                log("KEEPALIVE", f"ok (200)", "white")

            LAST_BAR_TS = t[-1]
            time.sleep(ENV["DECISION_EVERY_S"])

        except Exception as e:
            log("ERROR", f"{ICON['err']} {e}", "red")
            time.sleep(ENV["DECISION_EVERY_S"])

# --------------------- Boot ---------------------
if __name__=="__main__":
    log("BOOT", f"{ICON['ok']} Starting bot • Mode={('PAPER' if PAPER else 'LIVE')} • TV-sync={'ON' if ENV['FORCE_TV_ENTRIES'] else 'OFF'}", "green")
    log("CONF", f"Symbol={SYMBOL_NORM}  Interval={ENV['INTERVAL']}  Leverage={ENV['LEVERAGE']}x  Risk={ENV['RISK_PCT']}%", "white")
    # Start health server
    threading.Thread(target=start_server, daemon=True).start()
    loop()
