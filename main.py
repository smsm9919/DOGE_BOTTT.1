# -*- coding: utf-8 -*-
"""
مشروع البوت — النسخة النهائية
RF Futures Bot — Smart Pro (BingX Perp, CCXT)
- Entries: TradingView Range Filter EXACT (BUY/SELL)
- Size: 60% balance × leverage (default 10x)
- Exit:
  • Opposite RF signal ALWAYS closes
  • Smart Profit: TP1 partial + move to breakeven + ATR trailing (trend-riding)
- Indicators (display-only): RSI/DI+/DI-/DX/ADX/ATR
- Keepalive + /metrics

Additions (non-invasive, signatures unchanged):
- BreakIntel: ذكاء اختراق/كسر (Resistance/Support) يوجّه جني الأرباح (HOLD/Skip TP1).
- Regime detection: TREND_UP / TREND_DOWN / RANGE + "micro" لأسواق هادئة.
- Trailing Ratchet + Give-Back: تشديد وقف ربح ديناميكي ومنع التنازل عن ربح كبير.
- Candles Intelligence: كشف الدوجي/الدبوس/الابتلاع + الكسر الوهمي، وتكييف الإدارة.
- HUD منظم: المؤشرات تحت بعضها + آيكونات + ألوان + زر الحالة 🟩🟥🟨.
"""

import os, time, math, threading, requests, traceback, random
import pandas as pd
import ccxt
from flask import Flask, jsonify
from datetime import datetime

# ------------ console colors ------------
try:
    from termcolor import colored
except Exception:
    def colored(t,*a,**k): return t

# ------------ ENV ------------
def getenv(k, d=None, typ=str):
    v = os.getenv(k, d)
    if v is None: return d
    if typ is bool:  return str(v).lower() in ("1","true","yes","y","on")
    if typ is int:   return int(float(v))
    if typ is float: return float(v)
    return v

API_KEY    = getenv("BINGX_API_KEY", "")
API_SECRET = getenv("BINGX_API_SECRET", "")
MODE_LIVE  = bool(API_KEY and API_SECRET)

SYMBOL     = getenv("SYMBOL", "DOGE/USDT:USDT")
INTERVAL   = getenv("INTERVAL", "15m")
LEVERAGE   = getenv("LEVERAGE", 10, int)
RISK_ALLOC = getenv("RISK_ALLOC", 0.60, float)

# Range Filter params (EXACT as TV script)
RF_SOURCE  = getenv("RF_SOURCE", "close").lower()   # close/open/high/low
RF_PERIOD  = getenv("RF_PERIOD", 20, int)
RF_MULT    = getenv("RF_MULT", 3.5, float)
USE_TV_BAR = getenv("USE_TV_BAR", False, bool)      # False => last CLOSED bar

# Indicators (display-only)
RSI_LEN = getenv("RSI_LEN", 14, int)
ADX_LEN = getenv("ADX_LEN", 14, int)
ATR_LEN = getenv("ATR_LEN", 14, int)

# Execution guards
SPREAD_GUARD_BPS = getenv("SPREAD_GUARD_BPS", 6, int)
COOLDOWN_AFTER_CLOSE_BARS = getenv("COOLDOWN_AFTER_CLOSE_BARS", 0, int)

# Strategy mode
STRATEGY = getenv("STRATEGY", "smart").lower()      # smart | pure
USE_SMART_EXIT = getenv("USE_SMART_EXIT", True, bool)

# Smart Profit params
TP1_PCT          = getenv("TP1_PCT", 0.40, float)        # partial at +0.40%
TP1_CLOSE_FRAC   = getenv("TP1_CLOSE_FRAC", 0.50, float) # close 50% at TP1
BREAKEVEN_AFTER  = getenv("BREAKEVEN_AFTER_PCT", 0.30, float) # BE after +0.30%
TRAIL_ACTIVATE   = getenv("TRAIL_ACTIVATE_PCT", 0.60, float)  # start trailing after +0.60%
ATR_MULT_TRAIL   = getenv("ATR_MULT_TRAIL", 1.6, float)       # trail distance

# pacing / keepalive
SLEEP_S  = getenv("DECISION_EVERY_S", 30, int)
SELF_URL = getenv("SELF_URL", "") or getenv("RENDER_EXTERNAL_URL","")
KEEPALIVE_SECONDS = getenv("KEEPALIVE_SECONDS", 50, int)
PORT     = getenv("PORT", 5000, int)

print(colored(f"MODE: {'LIVE' if MODE_LIVE else 'PAPER'} • SYMBOL={SYMBOL} • {INTERVAL}", "yellow"))
print(colored(f"STRATEGY: {STRATEGY.upper()} • SMART_EXIT={'ON' if USE_SMART_EXIT else 'OFF'}", "yellow"))
print(colored(f"KEEPALIVE: url={'SET' if SELF_URL else 'NOT SET'} • every {KEEPALIVE_SECONDS}s", "yellow"))

# ------------ Exchange ------------
def make_exchange():
    return ccxt.bingx({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "timeout": 20000,
        "options": {"defaultType": "swap"}
    })

ex = make_exchange()
try:
    ex.load_markets()
except Exception as e:
    print(colored(f"⚠️ load_markets: {e}", "yellow"))

# ------------ Helpers ------------
def fmt(v, d=6, na="N/A"):
    try:
        if v is None or (isinstance(v,float) and (math.isnan(v) or math.isinf(v))): return na
        return f"{float(v):.{d}f}"
    except Exception: return na

def with_retry(fn, attempts=3, base_wait=0.4):
    for i in range(attempts):
        try: return fn()
        except Exception:
            if i == attempts-1: raise
            time.sleep(base_wait*(2**i) + random.random()*0.2)

def fetch_ohlcv(limit=600):
    rows = with_retry(lambda: ex.fetch_ohlcv(SYMBOL, timeframe=INTERVAL, limit=limit, params={"type":"swap"}))
    return pd.DataFrame(rows, columns=["time","open","high","low","close","volume"])

def price_now():
    try:
        t = with_retry(lambda: ex.fetch_ticker(SYMBOL))
        return t.get("last") or t.get("close")
    except Exception as e:
        print(colored(f"❌ ticker: {e}", "red")); return None

def balance_usdt():
    if not MODE_LIVE: return 100.0
    try:
        b = with_retry(lambda: ex.fetch_balance(params={"type":"swap"}))
        return b.get("total",{}).get("USDT") or b.get("free",{}).get("USDT")
    except Exception as e:
        print(colored(f"❌ balance: {e}", "red")); return None

def orderbook_spread_bps():
    try:
        ob = with_retry(lambda: ex.fetch_order_book(SYMBOL, limit=5))
        bid = ob["bids"][0][0] if ob["bids"] else None
        ask = ob["asks"][0][0] if ob["asks"] else None
        if not (bid and ask): return None
        mid = (bid+ask)/2.0
        return ((ask-bid)/mid)*10000.0
    except Exception:
        return None

# ------------ Indicators (display-only) ------------
def wilder_ema(s: pd.Series, n: int): return s.ewm(alpha=1/n, adjust=False).mean()

def compute_indicators(df: pd.DataFrame):
    c, h, l = df["close"].astype(float), df["high"].astype(float), df["low"].astype(float)
    tr = pd.concat([(h-l).abs(), (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    atr = wilder_ema(tr, ATR_LEN)

    delta = c.diff()
    up = delta.clip(lower=0.0); dn = (-delta).clip(lower=0.0)
    rs = wilder_ema(up, RSI_LEN) / wilder_ema(dn, RSI_LEN).replace(0, 1e-12)
    rsi = 100 - (100 / (1+rs))

    up_move = h.diff(); down_move = l.shift(1) - l
    plus_dm  = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    plus_di  = 100 * (wilder_ema(plus_dm, ADX_LEN) / atr.replace(0,1e-12))
    minus_di = 100 * (wilder_ema(minus_dm, ADX_LEN) / atr.replace(0,1e-12))
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0,1e-12)).fillna(0.0)
    adx = wilder_ema(dx, ADX_LEN)

    i = len(df)-1 if USE_TV_BAR else len(df)-2
    return {
        "rsi": float(rsi.iloc[i]), "plus_di": float(plus_di.iloc[i]),
        "minus_di": float(minus_di.iloc[i]), "dx": float(dx.iloc[i]),
        "adx": float(adx.iloc[i]), "atr": float(atr.iloc[i])
    }

# ------------ Range Filter (EXACT) ------------
def _ema(s: pd.Series, n: int): return s.ewm(span=n, adjust=False).mean()
def _rng_size(src: pd.Series, qty: float, n: int) -> pd.Series:
    avrng = _ema((src - src.shift(1)).abs(), n); wper = (n*2)-1
    return _ema(avrng, wper) * qty
def _rng_filter(src: pd.Series, rsize: pd.Series):
    rf = [float(src.iloc[0])]
    for i in range(1, len(src)):
        prev = rf[-1]; x = float(src.iloc[i]); r = float(rsize.iloc[i]); cur = prev
        if x - r > prev: cur = x - r
        if x + r < prev: cur = x + r
        rf.append(cur)
    filt = pd.Series(rf, index=src.index, dtype="float64")
    return filt + rsize, filt - rsize, filt

def compute_tv_signals(df: pd.DataFrame):
    src = df[RF_SOURCE].astype(float)
    hi, lo, filt = _rng_filter(src, _rng_size(src, RF_MULT, RF_PERIOD))
    dfilt = filt - filt.shift(1)
    fdir = pd.Series(0.0, index=filt.index).mask(dfilt>0,1).mask(dfilt<0,-1).ffill().fillna(0.0)
    upward = (fdir==1).astype(int); downward=(fdir==-1).astype(int)
    src_gt_f=(src>filt); src_lt_f=(src<filt); src_gt_p=(src>src.shift(1)); src_lt_p=(src<src.shift(1))
    longCond=(src_gt_f&((src_gt_p)|(src_lt_p))&(upward>0))
    shortCond=(src_lt_f&((src_lt_p)|(src_gt_p))&(downward>0))
    CondIni=pd.Series(0,index=src.index)
    for i in range(1,len(src)):
        if bool(longCond.iloc[i]): CondIni.iloc[i]=1
        elif bool(shortCond.iloc[i]): CondIni.iloc[i]=-1
        else: CondIni.iloc[i]=CondIni.iloc[i-1]
    longSignal=longCond&(CondIni.shift(1)==-1)
    shortSignal=shortCond&(CondIni.shift(1)==1)
    i=len(df)-1 if USE_TV_BAR else len(df)-2
    return {
        "time": int(df["time"].iloc[i]), "price": float(df["close"].iloc[i]),
        "long": bool(longSignal.iloc[i]), "short": bool(shortSignal.iloc[i]),
        "filter": float(filt.iloc[i]), "hi": float(hi.iloc[i]), "lo": float(lo.iloc[i]),
        "fdir": float(fdir.iloc[i])
    }

# ------------ State & Sync ------------
state={
    "open":False,"side":None,"entry":None,"qty":0.0,"pnl":0.0,"bars":0,
    "trail":None,"tp1_done":False,"breakeven":None,
    "break_bias":"NEUTRAL","ui_wait":False,
    "regime":"RANGE","micro":False,"peak_rr":0.0,"activated_trail":False,
    "candle_note":"NONE","candle_fake":False
}
compound_pnl=0.0
last_signal_id=None
post_close_cooldown=0

def compute_size(balance, price):
    capital = balance*RISK_ALLOC*LEVERAGE
    return max(0.0, capital/max(price,1e-9))

def sync_from_exchange_once():
    try:
        poss = ex.fetch_positions(params={"type":"swap"})
        for p in poss:
            sym = p.get("symbol") or p.get("info",{}).get("symbol") or ""
            if SYMBOL.split(":")[0] not in sym: 
                continue
            qty = abs(float(p.get("contracts") or p.get("info",{}).get("positionAmt") or 0))
            if qty<=0: continue
            entry = float(p.get("entryPrice") or p.get("info",{}).get("avgEntryPrice") or 0)
            side = (p.get("side") or p.get("info",{}).get("positionSide") or "").lower()
            if side not in ("long","short"):
                cost = float(p.get("cost") or 0.0)
                side = "long" if cost>0 else "short"
            state.update({"open":True,"side":side,"entry":entry,"qty":qty,"pnl":0.0,"bars":0,"trail":None,"tp1_done":False,"breakeven":None,
                          "break_bias":"NEUTRAL","ui_wait":False,"regime":"RANGE","micro":False,"peak_rr":0.0,"activated_trail":False,
                          "candle_note":"NONE","candle_fake":False})
            print(colored(f"✅ Synced position ⇒ {side.upper()} qty={fmt(qty,4)} @ {fmt(entry)}","green"))
            return
        print(colored("↔️  Sync: no open position on exchange.","yellow"))
    except Exception as e:
        print(colored(f"❌ sync error: {e}","red"))

# ------------ Orders ------------
def open_market(side, qty, price):
    global state
    if qty<=0: 
        print(colored("❌ qty<=0 skip open","red")); return
    if MODE_LIVE:
        try: ex.set_leverage(LEVERAGE, SYMBOL, params={"side":"BOTH"})
        except Exception as e: print(colored(f"⚠️ set_leverage: {e}","yellow"))
        try: ex.create_order(SYMBOL,"market",side,qty,params={"reduceOnly":False})
        except Exception as e: print(colored(f"❌ open: {e}","red"))
    state={"open":True,"side":"long" if side=="buy" else "short","entry":price,"qty":qty,"pnl":0.0,"bars":0,"trail":None,"tp1_done":False,"breakeven":None,
           "break_bias":"NEUTRAL","ui_wait":False,"regime":"RANGE","micro":False,"peak_rr":0.0,"activated_trail":False,
           "candle_note":"NONE","candle_fake":False}
    print(colored(f"✅ OPEN {side.upper()} qty={fmt(qty,4)} @ {fmt(price)}","green" if side=="buy" else "red"))

def close_partial(frac, reason):
    """Close fraction of current position (smart TP1)."""
    global state, compound_pnl
    if not state["open"]: return
    qty_close = max(0.0, state["qty"]*min(max(frac,0.0),1.0))
    if qty_close<=0: return
    px = price_now() or state["entry"]
    side = "sell" if state["side"]=="long" else "buy"
    if MODE_LIVE:
        try: ex.create_order(SYMBOL,"market",side,qty_close,params={"reduceOnly":True})
        except Exception as e: print(colored(f"❌ partial close: {e}","red"))
    pnl=(px-state["entry"])*qty_close*(1 if state["side"]=="long" else -1)
    compound_pnl+=pnl
    state["qty"]-=qty_close
    print(colored(f"🔻 PARTIAL {reason} closed={fmt(qty_close,4)} pnl={fmt(pnl)} rem_qty={fmt(state['qty'],4)}","magenta"))
    if state["qty"]<=0:
        reset_after_full_close("fully_exited")

def reset_after_full_close(reason):
    global state, post_close_cooldown
    print(colored(f"🔚 CLOSE {reason} totalCompounded now={fmt(compound_pnl)}","magenta"))
    state={"open":False,"side":None,"entry":None,"qty":0.0,"pnl":0.0,"bars":0,"trail":None,"tp1_done":False,"breakeven":None,
           "break_bias":"NEUTRAL","ui_wait":False,"regime":"RANGE","micro":False,"peak_rr":0.0,"activated_trail":False,
           "candle_note":"NONE","candle_fake":False}
    post_close_cooldown = COOLDOWN_AFTER_CLOSE_BARS

def close_market(reason):
    global state, compound_pnl
    if not state["open"]: return
    px=price_now() or state["entry"]; qty=state["qty"]
    side="sell" if state["side"]=="long" else "buy"
    if MODE_LIVE:
        try: ex.create_order(SYMBOL,"market",side,qty,params={"reduceOnly":True})
        except Exception as e: print(colored(f"❌ close: {e}","red"))
    pnl=(px-state["entry"])*qty*(1 if state["side"]=="long" else -1)
    compound_pnl+=pnl
    print(colored(f"🔚 CLOSE {state['side']} reason={reason} pnl={fmt(pnl)} total={fmt(compound_pnl)}","magenta"))
    reset_after_full_close(reason)

# ------------ Candles Intelligence ------------
def _last(df, n=1):
    return df.iloc[-n]

def candle_stats(o,h,l,c):
    rng = max(h-l, 1e-12)
    body = abs(c-o)
    upper = h - max(c,o)
    lower = min(c,o) - l
    body_pct = (body/rng)*100.0
    upper_pct = (upper/rng)*100.0
    lower_pct = (lower/rng)*100.0
    dir_up = c >= o
    return {"range": rng, "body": body, "upper": upper, "lower": lower,
            "body_pct": body_pct, "upper_pct": upper_pct, "lower_pct": lower_pct,
            "up": dir_up}

def is_doji(stats, max_body_pct=15): return stats["body_pct"] <= max_body_pct
def is_pin_bull(stats, min_lower_pct=55, max_body_pct=30): return stats["lower_pct"] >= min_lower_pct and stats["body_pct"] <= max_body_pct and stats["up"]
def is_pin_bear(stats, min_upper_pct=55, max_body_pct=30): return stats["upper_pct"] >= min_upper_pct and stats["body_pct"] <= max_body_pct and (not stats["up"])
def engulfing_bull(prev_o, prev_c, o, c): return (prev_c < prev_o) and (c > o) and (o <= prev_c) and (c >= prev_o)
def engulfing_bear(prev_o, prev_c, o, c): return (prev_c > prev_o) and (c < o) and (o >= prev_c) and (c <= prev_o)

def fake_break(close, high, low, ref_level, side, min_close_buffer=0.05):
    buf = (close * min_close_buffer) / 100.0
    if side == "up":
        return close <= ref_level + buf and high > ref_level
    else:
        return close >= ref_level - buf and low < ref_level

def analyze_candles(df, prev_high, prev_low):
    c1 = _last(df,1); c2 = _last(df,2)
    o1,h1,l1,c1p = float(c1.open),float(c1.high),float(c1.low),float(c1.close)
    o2,h2,l2,c2p = float(c2.open),float(c2.high),float(c2.low),float(c2.close)

    s1 = candle_stats(o1,h1,l1,c1p)
    patt = []
    if is_doji(s1):      patt.append("DOJI")
    if is_pin_bull(s1):  patt.append("PIN_BULL")
    if is_pin_bear(s1):  patt.append("PIN_BEAR")
    if engulfing_bull(o2,c2p,o1,c1p): patt.append("ENGULF_BULL")
    if engulfing_bear(o2,c2p,o1,c1p): patt.append("ENGULF_BEAR")

    fb_up = fake_break(c1p,h1,l1,prev_high,"up")
    fb_dn = fake_break(c1p,h1,l1,prev_low,"down")
    if fb_up: patt.append("FAKE_BREAK_UP")
    if fb_dn: patt.append("FAKE_BREAK_DOWN")

    return {"text": "|".join(patt) if patt else "NONE",
            "is_fake_up": fb_up, "is_fake_down": fb_dn, "stats": s1}

# ------------ BreakIntel helpers ------------
def compute_prev_levels(df, lookback=20):
    ph = float(df["high"].iloc[-(lookback+1):-1].max())
    pl = float(df["low"].iloc[-(lookback+1):-1].min())
    return ph, pl

def break_intel_signal(price, prev_high, prev_low, ema50, ema200, adx, pdi, mdi, atr):
    trend_up = ema50 > ema200
    trend_dn = ema50 < ema200
    break_res = price > prev_high + 0.2*atr
    break_sup = price < prev_low  - 0.2*atr
    if break_res and trend_up and adx>28 and pdi>mdi: return "WAIT_UP"
    if break_sup and trend_dn and adx>28 and mdi>pdi: return "WAIT_DOWN"
    return "NEUTRAL"

def detect_regime(ema50, ema200, adx, pdi, mdi, atr, price):
    atr_pct = (atr/max(price,1e-9))*100.0
    up  = (ema50>ema200) and (adx>=24) and (pdi>mdi)
    dn  = (ema50<ema200) and (adx>=24) and (mdi>pdi)
    micro = (adx<22) or (atr_pct<0.12)
    if up:   return "TREND_UP", micro
    if dn:   return "TREND_DOWN", micro
    return "RANGE", micro

def status_pill():
    if state["open"]:
        return colored("🟩 [BUY ACTIVE]", "green") if state["side"]=="long" else colored("🟥 [SELL ACTIVE]", "red")
    return colored("🟨 [WAIT]", "yellow")

# ------------ Smart Profit (trend-aware) ------------
def smart_exit_check(info, ind):
    if not (STRATEGY=="smart" and USE_SMART_EXIT and state["open"]): return None
    px = info["price"]; e=state["entry"]; side=state["side"]
    rr = (px - e)/e * 100.0 * (1 if side=="long" else -1)
    atr = ind.get("atr") or 0.0

    if state["bars"] < 2:
        state["peak_rr"] = max(state.get("peak_rr",0.0), rr)
        return None

    state["peak_rr"] = max(state.get("peak_rr",0.0), rr)

    # Micro/Range scalp
    if state.get("micro", False) or state.get("regime")=="RANGE":
        micro_tp = 0.30
        if rr >= micro_tp and not state["tp1_done"]:
            close_partial(0.60, f"MICRO_TP@{micro_tp:.2f}%")
            state["tp1_done"] = True
            state["breakeven"] = e
        if state["bars"] >= 8 and abs(rr) < 0.20:
            close_partial(0.30, "MICRO_TIME_DERISK")
            if state["breakeven"] is None: state["breakeven"] = e

    # BreakIntel: منع TP1 المبكر
    bias = state.get("break_bias","NEUTRAL")
    hold_bias = (side=="long" and bias=="WAIT_UP") or (side=="short" and bias=="WAIT_DOWN")
    if (not state["tp1_done"]) and rr >= TP1_PCT:
        if hold_bias:
            if state["breakeven"] is None and rr >= BREAKEVEN_AFTER: state["breakeven"]=e
            print(colored(f"🧠 BreakIntel HOLD_TP: skip TP1 (bias={bias}, rr={rr:.2f}%)","cyan"))
        else:
            close_partial(TP1_CLOSE_FRAC, f"TP1@{TP1_PCT:.2f}%")
            state["tp1_done"]=True
            if state["breakeven"] is None and rr >= BREAKEVEN_AFTER: state["breakeven"]=e

    # Trailing Ratchet
    if (rr >= TRAIL_ACTIVATE) or state.get("activated_trail", False):
        state["activated_trail"] = True
        if rr >= 3.5:   mult = min(1.0, ATR_MULT_TRAIL)
        elif rr >= 2.0: mult = min(1.3, ATR_MULT_TRAIL)
        else:           mult = ATR_MULT_TRAIL
        gap = atr * mult
        if side=="long":
            new_trail = px - gap
            state["trail"] = max(state["trail"] or new_trail, new_trail)
            if state["breakeven"] is not None: state["trail"] = max(state["trail"], state["breakeven"])
            if px <= state["trail"]:
                close_market(f"TRAIL_ATR({mult:.2f}x)"); return True
        else:
            new_trail = px + gap
            state["trail"] = min(state["trail"] or new_trail, new_trail)
            if state["breakeven"] is not None: state["trail"] = min(state["trail"], state["breakeven"])
            if px >= state["trail"]:
                close_market(f"TRAIL_ATR({mult:.2f}x)"); return True

    # Candles management
    note = state.get("candle_note","NONE")
    fake = state.get("candle_fake", False)
    if side=="long":
        if ("ENGULF_BEAR" in note or "PIN_BEAR" in note) and rr > 0.25:
            close_partial(0.25, "CANDLE_REV_LONG")
            if state["breakeven"] is None: state["breakeven"]=e
        if fake and rr > 0.20 and state.get("trail") is not None:
            state["trail"] = max(state["trail"], e)
    else:
        if ("ENGULF_BULL" in note or "PIN_BULL" in note) and rr > 0.25:
            close_partial(0.25, "CANDLE_REV_SHORT")
            if state["breakeven"] is None: state["breakeven"]=e
        if fake and rr > 0.20 and state.get("trail") is not None:
            state["trail"] = min(state["trail"], e)

    # Give-back protection
    peak = state.get("peak_rr", 0.0)
    loss_from_peak = peak - rr
    if peak >= 1.2 and loss_from_peak >= 0.6:
        close_partial(0.25, "GIVEBACK_LOCK_0.6%")
        if state["breakeven"] is None: state["breakeven"]=e
    if peak >= 2.5 and loss_from_peak >= 1.0:
        close_partial(0.20, "GIVEBACK_LOCK_1.0%")

    # Breakeven enforcement
    if state["breakeven"] is not None:
        if (side == "long" and px <= state["breakeven"]) or (side == "short" and px >= state["breakeven"]):
            close_market("BREAKEVEN_HIT"); return True

    return None

# ------------ HUD ------------
def snapshot(bal,info,ind,spread_bps,reason=None):
    print(colored("─"*100,"cyan"))
    print(colored(f"📊 {SYMBOL} {INTERVAL} • {'LIVE' if MODE_LIVE else 'PAPER'} • {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC","cyan"))
    print(colored("─"*100,"cyan"))
    print(colored("📈 INDICATORS","cyan"))
    print(colored(f"   💲 Price = {fmt(info.get('price'))}","white"))
    print(colored(f"   📊 RF    = filt {fmt(info.get('filter'))} | hi {fmt(info.get('hi'))} | lo {fmt(info.get('lo'))}","blue"))
    print(colored(f"   🌀 RSI({RSI_LEN}) = {fmt(ind['rsi'])}","cyan"))
    print(colored(f"   ➕ +DI      = {fmt(ind['plus_di'])}","green"))
    print(colored(f"   ➖ -DI      = {fmt(ind['minus_di'])}","red"))
    print(colored(f"   📉 DX       = {fmt(ind['dx'])}","magenta"))
    print(colored(f"   📡 ADX({ADX_LEN}) = {fmt(ind['adx'])}","yellow"))
    print(colored(f"   📏 ATR      = {fmt(ind['atr'])}","blue"))
    print(colored(f"   🧠 BreakIntel = {state.get('break_bias','NEUTRAL')}  |  📈 Regime = {state.get('regime','RANGE')} (micro={state.get('micro',False)})","cyan"))
    print(colored(f"   🕯️ Candles = {state.get('candle_note','NONE')}","white"))
    print(colored(f"   🧮 spread_bps = {fmt(spread_bps,2)}   Mode={STRATEGY.upper()}   BUY={info['long']}  SELL={info['short']}","white"))
    print()
    print("🧭 POSITION")
    print(f"   💰 Balance {fmt(bal,2)} USDT   Risk={int(RISK_ALLOC*100)}%×{LEVERAGE}x   PostCloseCooldown={post_close_cooldown}")
    if state["open"]:
        print(f"   📌 {'🟩 LONG' if state['side']=='long' else '🟥 SHORT'}  Entry={fmt(state['entry'])}  Qty={fmt(state['qty'],4)}  Bars={state['bars']}  PnL={fmt(state['pnl'])}")
        print(f"   🎯 Trail={fmt(state['trail'])}  TP1_done={state['tp1_done']}  BE={fmt(state['breakeven'])}  PeakRR={fmt(state['peak_rr'],2)}%")
    else:
        print("   ⚪ FLAT")
    print(f"   {status_pill()}")
    if state.get("ui_wait"):
        print(colored("   ⏳ WAIT SIGNAL / CandleGuard / BreakIntel filter", "yellow"))
    print()
    print("📦 RESULTS")
    eff_eq = (bal or 0.0) + compound_pnl if MODE_LIVE else compound_pnl
    print(f"   🧮 CompoundPnL {fmt(compound_pnl)}   🚀 EffectiveEq {fmt(eff_eq)} USDT")
    if reason:
        print(colored(f"   ℹ️ No trade — reason: {reason}","yellow"))
    print(colored("─"*100,"cyan"))

# ------------ Decision Loop ------------
def trade_loop():
    global last_signal_id, state, post_close_cooldown
    sync_from_exchange_once()
    while True:
        try:
            bal=balance_usdt()
            px=price_now()
            df=fetch_ohlcv()
            info=compute_tv_signals(df)
            ind=compute_indicators(df)
            spread_bps = orderbook_spread_bps()

            if state["open"] and px:
                state["pnl"]=(px-state["entry"])*state["qty"] if state["side"]=="long" else (state["entry"]-px)*state["qty"]

            # ==== BreakIntel + Regime ====
            prev_high, prev_low = compute_prev_levels(df, lookback=20)
            ema50 = df["close"].ewm(span=50, adjust=False).mean().iloc[-2]
            ema200= df["close"].ewm(span=200,adjust=False).mean().iloc[-2]
            bias = break_intel_signal(px or info["price"], prev_high, prev_low, ema50, ema200, ind["adx"], ind["plus_di"], ind["minus_di"], ind["atr"])
            state["break_bias"] = bias
            regime, micro = detect_regime(ema50, ema200, ind["adx"], ind["plus_di"], ind["minus_di"], ind["atr"], px or info["price"])
            state["regime"] = regime; state["micro"]  = micro

            # ==== Candles ====
            candle = analyze_candles(df, prev_high, prev_low)
            state["candle_note"] = candle["text"]
            state["candle_fake"] = candle["is_fake_up"] or candle["is_fake_down"]

            # Smart profit (trend-aware + candles)
            smart_exit_check(info, ind)

            # Decide
            sig="buy" if info["long"] else ("sell" if info["short"] else None)
            reason=None
            if not sig:
                reason="no signal"
            elif spread_bps is not None and spread_bps>SPREAD_GUARD_BPS:
                reason=f"spread too high ({fmt(spread_bps,2)}bps > {SPREAD_GUARD_BPS})"
            elif post_close_cooldown>0:
                reason=f"cooldown {post_close_cooldown} bars"

            # CandleGuard: لا تدخل لو كسر وهمي ضد الإشارة
            if reason is None and state["candle_fake"]:
                if sig=="buy" and candle["is_fake_up"]:
                    reason="candle_guard: FAKE_BREAK_UP"
                if sig=="sell" and candle["is_fake_down"]:
                    reason="candle_guard: FAKE_BREAK_DOWN"

            # Close on opposite RF signal ALWAYS
            if state["open"] and sig:
                desired="long" if sig=="buy" else "short"
                if state["side"]!=desired and (reason is None):
                    close_market("opposite_signal")
                    qty=compute_size(bal, px or info["price"])
                    if qty>0:
                        open_market(sig, qty, px or info["price"])
                        last_signal_id=f"{info['time']}:{sig}"
                        snapshot(bal,info,ind,spread_bps,None)
                        time.sleep(SLEEP_S); continue

            # Open new position when flat
            if not state["open"] and (reason is None) and sig:
                qty=compute_size(bal, px or info["price"])
                if qty>0:
                    open_market(sig, qty, px or info["price"])
                    last_signal_id=f"{info['time']}:{sig}"
                else:
                    reason="qty<=0"

            state["ui_wait"] = (not state["open"]) or (reason is not None)

            snapshot(bal,info,ind,spread_bps,reason)

            if state["open"]:
                state["bars"] += 1
            if post_close_cooldown>0 and not state["open"]:
                post_close_cooldown -= 1

        except Exception as e:
            print(colored(f"❌ loop error: {e}\n{traceback.format_exc()}","red"))
        time.sleep(SLEEP_S)

# ------------ Keepalive + API ------------
def keepalive_loop():
    url = (SELF_URL or "").strip().rstrip("/")
    if not url:
        print(colored("⛔ keepalive: SELF_URL/RENDER_EXTERNAL_URL not set — skipping.", "yellow"))
        return
    sess = requests.Session()
    sess.headers.update({"User-Agent":"rf-pro-bot/keepalive"})
    print(colored(f"🛰️ keepalive: ping {url} every {KEEPALIVE_SECONDS}s","cyan"))
    while True:
        try:
            r = sess.get(url, timeout=8)
            if r.status_code==200:
                print(colored("🟢 keepalive ok (200)","green"))
            else:
                print(colored(f"🟠 keepalive status={r.status_code}","yellow"))
        except Exception as e:
            print(colored(f"🔴 keepalive error: {e}","red"))
        time.sleep(max(KEEPALIVE_SECONDS,15))

app = Flask(__name__)

@app.route("/")
def home():
    mode = 'LIVE' if MODE_LIVE else 'PAPER'
    return f"✅ مشروع البوت — {SYMBOL} {INTERVAL} — {mode} — {STRATEGY.upper()}"

@app.route("/metrics")
def metrics():
    return jsonify({
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "mode": "live" if MODE_LIVE else "paper",
        "leverage": LEVERAGE,
        "risk_alloc": RISK_ALLOC,
        "price": price_now(),
        "position": state,
        "compound_pnl": compound_pnl,
        "time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "strategy": STRATEGY
    })

@app.route("/ping")
def ping(): return "pong", 200

# Boot
threading.Thread(target=trade_loop, daemon=True).start()
threading.Thread(target=keepalive_loop, daemon=True).start()
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
