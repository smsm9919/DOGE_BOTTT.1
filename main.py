# -*- coding: utf-8 -*-
"""
RF Futures Bot — Smart Pro (BingX Perp, CCXT)
- Entries: TradingView Range Filter EXACT (BUY/SELL)
- Size: 60% balance × leverage (default 10x)
- Exit:
  • Opposite RF signal ALWAYS closes
  • Smart Profit: TP1 partial + move to breakeven + ATR trailing (trend-riding)
- Indicators (display-only): RSI/DI+/DI-/DX/ADX/ATR
- Robust keepalive (SELF_URL/RENDER_EXTERNAL_URL), retries, /metrics

Patched:
- BingX position mode support (oneway|hedge) with correct positionSide
- Safe state updates (don't flip local state on failed open)
- Logs: show WAIT reason + time left to candle close
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

# Strategy mode (default smart)
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

# ---- NEW: BingX position mode support ----
BINGX_POSITION_MODE = getenv("BINGX_POSITION_MODE", "oneway").lower()  # oneway | hedge

# ---- Optional: entry confirmation knobs (kept for future use, default off) ----
ENTRY_CONFIRM_SEC    = getenv("ENTRY_CONFIRM_SEC", 0, int)   # 0 = disabled
ENTRY_CONFIRM_STRICT = getenv("ENTRY_CONFIRM_STRICT", True, bool)

print(colored(f"MODE: {'LIVE' if MODE_LIVE else 'PAPER'} • SYMBOL={SYMBOL} • {INTERVAL}", "yellow"))
print(colored(f"STRATEGY: {STRATEGY.upper()} • SMART_EXIT={'ON' if USE_SMART_EXIT else 'OFF'}", "yellow"))
print(colored(f"KEEPALIVE: url={'SET' if SELF_URL else 'NOT SET'} • every {KEEPALIVE_SECONDS}s", "yellow"))
print(colored(f"BINGX_POSITION_MODE={BINGX_POSITION_MODE}", "yellow"))

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

# ---- NEW: time to candle close helpers ----
def _interval_seconds(iv: str) -> int:
    iv = (iv or "").lower().strip()
    if iv.endswith("m"): return int(float(iv[:-1]))*60
    if iv.endswith("h"): return int(float(iv[:-1]))*3600
    if iv.endswith("d"): return int(float(iv[:-1]))*86400
    return 15*60

def time_to_candle_close(df: pd.DataFrame, use_tv_bar: bool) -> int:
    tf = _interval_seconds(INTERVAL)
    if len(df) == 0: return tf
    # معظم المنصات تعطي time كبداية الشمعة بالـ ms
    cur_start_ms = int(df["time"].iloc[-1])
    now_ms = int(time.time()*1000)
    next_close_ms = cur_start_ms + tf*1000
    while next_close_ms <= now_ms:
        next_close_ms += tf*1000
    left = max(0, next_close_ms - now_ms)
    return int(left/1000)

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
state={"open":False,"side":None,"entry":None,"qty":0.0,"pnl":0.0,"bars":0,"trail":None,"tp1_done":False,"breakeven":None}
compound_pnl=0.0
last_signal_id=None
post_close_cooldown=0

def compute_size(balance, price):
    capital = (balance or 0.0)*RISK_ALLOC*LEVERAGE
    return max(0.0, capital/max(float(price or 0.0),1e-9))

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
            state.update({"open":True,"side":side,"entry":entry,"qty":qty,"pnl":0.0,"bars":0,"trail":None,"tp1_done":False,"breakeven":None})
            print(colored(f"✅ Synced position ⇒ {side.upper()} qty={fmt(qty,4)} @ {fmt(entry)}","green"))
            return
        print(colored("↔️  Sync: no open position on exchange.","yellow"))
    except Exception as e:
        print(colored(f"❌ sync error: {e}","red"))

# ------------ BingX params helpers (NEW) ------------
def _position_params_for_open(side: str):
    if BINGX_POSITION_MODE == "hedge":
        return {"positionSide": "LONG" if side == "buy" else "SHORT", "reduceOnly": False}
    return {"positionSide": "BOTH", "reduceOnly": False}

def _position_params_for_close():
    if BINGX_POSITION_MODE == "hedge":
        return {"positionSide": "LONG" if state.get("side")=="long" else "SHORT", "reduceOnly": True}
    return {"positionSide": "BOTH", "reduceOnly": True}

# ------------ Orders (PATCHED) ------------
def open_market(side, qty, price):
    global state
    if qty<=0:
        print(colored("❌ qty<=0 skip open","red")); return

    params = _position_params_for_open(side)

    if MODE_LIVE:
        try:
            lev_params = {"positionSide": params["positionSide"]} if BINGX_POSITION_MODE=="hedge" else {"positionSide":"BOTH"}
            try: ex.set_leverage(LEVERAGE, SYMBOL, params=lev_params)
            except Exception as e: print(colored(f"⚠️ set_leverage: {e}", "yellow"))
            ex.create_order(SYMBOL, "market", side, qty, params=params)
        except Exception as e:
            print(colored(f"❌ open: {e}", "red"))
            return  # لا نُحدّث الحالة إذا فشل التنفيذ الفعلي

    state={"open":True,"side":"long" if side=="buy" else "short","entry":price,"qty":qty,"pnl":0.0,"bars":0,"trail":None,"tp1_done":False,"breakeven":None}
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
        try: ex.create_order(SYMBOL,"market",side,qty_close,params=_position_params_for_close())
        except Exception as e: print(colored(f"❌ partial close: {e}","red")); return
    pnl=(px-state["entry"])*qty_close*(1 if state["side"]=="long" else -1)
    compound_pnl+=pnl
    state["qty"]-=qty_close
    print(colored(f"🔻 PARTIAL {reason} closed={fmt(qty_close,4)} pnl={fmt(pnl)} rem_qty={fmt(state['qty'],4)}","magenta"))
    if state["qty"]<=0:
        reset_after_full_close("fully_exited")

def reset_after_full_close(reason):
    global state, post_close_cooldown
    print(colored(f"🔚 CLOSE {reason} totalCompounded now={fmt(compound_pnl)}","magenta"))
    state={"open":False,"side":None,"entry":None,"qty":0.0,"pnl":0.0,"bars":0,"trail":None,"tp1_done":False,"breakeven":None}
    post_close_cooldown = COOLDOWN_AFTER_CLOSE_BARS

def close_market(reason):
    global state, compound_pnl
    if not state["open"]: return
    px=price_now() or state["entry"]; qty=state["qty"]
    side="sell" if state["side"]=="long" else "buy"
    if MODE_LIVE:
        try: ex.create_order(SYMBOL,"market",side,qty,params=_position_params_for_close())
        except Exception as e: print(colored(f"❌ close: {e}","red")); return
    pnl=(px-state["entry"])*qty*(1 if state["side"]=="long" else -1)
    compound_pnl+=pnl
    print(colored(f"🔚 CLOSE {state['side']} reason={reason} pnl={fmt(pnl)} total={fmt(compound_pnl)}","magenta"))
    reset_after_full_close(reason)

# ------------ Smart Profit (trend-aware) ------------
def smart_exit_check(info, ind):
    """Return True if full close happened."""
    if not (STRATEGY=="smart" and USE_SMART_EXIT and state["open"]):
        return None
    px = info["price"]; e=state["entry"]; side=state["side"]
    rr = (px - e)/e * 100.0 * (1 if side=="long" else -1)
    atr = ind.get("atr") or 0.0

    # انتظر كام شمعة بعد الدخول لتجنب الخروج السريع
    if state["bars"] < 2:
        return None

    # TP1 جزئي للموجات الصغيرة
    if (not state["tp1_done"]) and rr >= TP1_PCT:
        close_partial(TP1_CLOSE_FRAC, f"TP1@{TP1_PCT:.2f}%")
        state["tp1_done"]=True
        # حرك الوقف لبريك-إيفن بعد مكسب معقول
        if rr >= BREAKEVEN_AFTER:
            state["breakeven"]=e

    # Trailing ATR للاستفادة القصوى من الترند
    if rr >= TRAIL_ACTIVATE and atr and ATR_MULT_TRAIL>0:
        gap = atr * ATR_MULT_TRAIL
        if side=="long":
            new_trail = px - gap
            state["trail"] = max(state["trail"] or new_trail, new_trail)
            if state["breakeven"] is not None:
                state["trail"] = max(state["trail"], state["breakeven"])
            if px < state["trail"]:
                close_market(f"TRAIL_ATR({ATR_MULT_TRAIL}x)"); return True
        else:
            new_trail = px + gap
            state["trail"] = min(state["trail"] or new_trail, new_trail)
            if state["breakeven"] is not None:
                state["trail"] = min(state["trail"], state["breakeven"])
            if px > state["trail"]:
                close_market(f"TRAIL_ATR({ATR_MULT_TRAIL}x)"); return True
    return None

# ------------ HUD (PATCHED to show candle close countdown & wait reason) ------------
def snapshot(bal,info,ind,spread_bps,reason=None, df=None):
    left_s = time_to_candle_close(df if df is not None else fetch_ohlcv(), USE_TV_BAR)
    print(colored("─"*100,"cyan"))
    print(colored(f"📊 {SYMBOL} {INTERVAL} • {'LIVE' if MODE_LIVE else 'PAPER'} • {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC","cyan"))
    print(colored("─"*100,"cyan"))
    print("📈 INDICATORS")
    print(f"   💲 Price {fmt(info.get('price'))}  RF filt={fmt(info.get('filter'))}  hi={fmt(info.get('hi'))}  lo={fmt(info.get('lo'))}")
    print(f"   RSI({RSI_LEN})={fmt(ind['rsi'])}  +DI={fmt(ind['plus_di'])}  -DI={fmt(ind['minus_di'])}  DX={fmt(ind['dx'])}  ADX({ADX_LEN})={fmt(ind['adx'])}  ATR={fmt(ind['atr'])}")
    print(f"   ✅ BUY={info['long']}   ❌ SELL={info['short']}   🧮 spread_bps={fmt(spread_bps,2)}   Mode={STRATEGY.upper()}")
    print(f"   ⏱️ Candle closes in ~ {left_s}s")
    print()
    print("🧭 POSITION")
    print(f"   💰 Balance {fmt(bal,2)} USDT   Risk={int(RISK_ALLOC*100)}%×{LEVERAGE}x   PostCloseCooldown={post_close_cooldown}")
    if state["open"]:
        print(f"   📌 {'🟩 LONG' if state['side']=='long' else '🟥 SHORT'}  Entry={fmt(state['entry'])}  Qty={fmt(state['qty'],4)}  Bars={state['bars']}  PnL={fmt(state['pnl'])}  Trail={fmt(state['trail'])}  TP1_done={state['tp1_done']}")
    else:
        print("   ⚪ FLAT")
    print()
    print("📦 RESULTS")
    eff_eq = (bal or 0.0) + compound_pnl if MODE_LIVE else compound_pnl
    print(f"   🧮 CompoundPnL {fmt(compound_pnl)}   🚀 EffectiveEq {fmt(eff_eq)} USDT")
    if reason:
        print(colored(f"   ℹ️ WAIT — reason: {reason}","yellow"))
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

            # Smart profit (trend-aware)
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

            # Close on opposite RF signal ALWAYS
            if state["open"] and sig and (reason is None):
                desired="long" if sig=="buy" else "short"
                if state["side"]!=desired:
                    close_market("opposite_signal")
                    qty=compute_size(bal, px or info["price"])
                    if qty>0:
                        open_market(sig, qty, px or info["price"])
                        last_signal_id=f"{info['time']}:{sig}"
                        snapshot(bal,info,ind,spread_bps,None, df)
                        time.sleep(SLEEP_S); continue

            # Open new position when flat
            if not state["open"] and (reason is None) and sig:
                qty=compute_size(bal, px or info["price"])
                if qty>0:
                    open_market(sig, qty, px or info["price"])
                    last_signal_id=f"{info['time']}:{sig}"
                else:
                    reason="qty<=0"

            snapshot(bal,info,ind,spread_bps,reason, df)

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
    return f"✅ RF Bot — {SYMBOL} {INTERVAL} — {mode} — {STRATEGY.upper()}"

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
        "strategy": STRATEGY,
        "bingx_mode": BINGX_POSITION_MODE
    })

@app.route("/ping")
def ping(): return "pong", 200

# Boot
threading.Thread(target=trade_loop, daemon=True).start()
threading.Thread(target=keepalive_loop, daemon=True).start()
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
