# -*- coding: utf-8 -*-
"""
Smart DOGE Bot — LIVE (BingX Perp, ccxt)
- Entry = TradingView-like Range Filter on candle CLOSE (TV-sync)
- Post-entry Intelligence: TP1 → Breakeven → ATR Trailing + Hold-TP + Scale-In
- Logs: colored, icons, full preview (next qty @10x), reasons on no-trade
- Health: / and /metrics (Flask)

ENV (keep same names you already use):
  BINGX_API_KEY, BINGX_API_SECRET
  SYMBOL                e.g. "DOGE/USDT:USDT" (recommended) or "DOGEUSDT"
  INTERVAL              "15m"
  LEVERAGE              10
  RISK_PCT              60           # % of equity used notionally (with leverage)
  DECISION_EVERY_S      30
  KEEPALIVE_SECONDS     50
  PORT                  5000
  USE_TV_BAR            false        # wait for closed candle
  FORCE_TV_ENTRIES      true         # match TV on close
  # Indicators & exits
  ADX_LEN=14  ATR_LEN=14
  TP1_PCT=0.40  TP1_CLOSE_FRAC=0.50
  BREAKEVEN_AFTER_PCT=0.30
  TRAIL_ACTIVATE_PCT=0.60
  ATR_MULT_TRAIL=1.6
  # Trend/Range helpers
  RANGE_MIN_PCT=1.0
  MIN_TP_PERCENT=0.40
  MOVE_3BARS_PCT=0.8
  HOLD_TP_STRONG=true
  HOLD_TP_ADX=28
  HOLD_TP_SLOPE=0.50
  SCALE_IN_ENABLED=true
  SCALE_IN_MAX_ADDS=3
  SCALE_IN_ADX_MIN=25
  SCALE_IN_SLOPE_MIN=0.50
  SPIKE_FILTER_ATR_MULTIPLIER=3.0
  RENDER_EXTERNAL_URL=  (optional)
"""

import os, time, math, json, threading, traceback
from datetime import datetime
from typing import List, Dict, Any, Tuple

# ---------------- Colors & Icons ----------------
RESET="\x1b[0m"; DIM="\x1b[2m"
FG={"r":"\x1b[31m","g":"\x1b[32m","y":"\x1b[33m","b":"\x1b[34m","m":"\x1b[35m","c":"\x1b[36m","w":"\x1b[97m"}
ICON={"info":"🛈","ok":"✅","warn":"⚠️","err":"⛔","buy":"🟢","sell":"🔴","wait":"🟡","flat":"⚪","ind":"📈","pos":"📦","tp":"🎯","trail":"🪄","be":"🛡️","tv":"📺","candle":"🕯️"}

def log(sec, msg, col="w"):
    print(f"{FG.get(col,'w')}{sec:>10}{RESET} {msg}{RESET}", flush=True)

def getenv(k, d=None, typ=str):
    v=os.getenv(k, d)
    if typ is bool:
        return str(v).strip().lower() in ("1","true","yes","on")
    try:
        return typ(v)
    except Exception:
        return v if v is not None else d

# ---------------- ENV ----------------
ENV = {
  "BINGX_API_KEY": getenv("BINGX_API_KEY",""),
  "BINGX_API_SECRET": getenv("BINGX_API_SECRET",""),
  "SYMBOL": getenv("SYMBOL","DOGE/USDT:USDT"),
  "INTERVAL": getenv("INTERVAL","15m"),
  "LEVERAGE": getenv("LEVERAGE",10,int),
  "RISK_PCT": getenv("RISK_PCT",60,float),
  "DECISION_EVERY_S": getenv("DECISION_EVERY_S",30,int),
  "KEEPALIVE_SECONDS": getenv("KEEPALIVE_SECONDS",50,int),
  "PORT": getenv("PORT",5000,int),
  "USE_TV_BAR": getenv("USE_TV_BAR",False,bool),
  "FORCE_TV_ENTRIES": getenv("FORCE_TV_ENTRIES",True,bool),
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
  "HOLD_TP_STRONG": getenv("HOLD_TP_STRONG",True,bool),
  "HOLD_TP_ADX": getenv("HOLD_TP_ADX",28,int),
  "HOLD_TP_SLOPE": getenv("HOLD_TP_SLOPE",0.50,float),
  "SCALE_IN_ENABLED": getenv("SCALE_IN_ENABLED",True,bool),
  "SCALE_IN_MAX_ADDS": getenv("SCALE_IN_MAX_ADDS",3,int),
  "SCALE_IN_ADX_MIN": getenv("SCALE_IN_ADX_MIN",25,int),
  "SCALE_IN_SLOPE_MIN": getenv("SCALE_IN_SLOPE_MIN",0.50,float),
  "SPIKE_FILTER_ATR_MULTIPLIER": getenv("SPIKE_FILTER_ATR_MULTIPLIER",3.0,float),
  "RENDER_EXTERNAL_URL": getenv("RENDER_EXTERNAL_URL",""),
}
PAPER = not (ENV["BINGX_API_KEY"] and ENV["BINGX_API_SECRET"])

# -------------- Helpers --------------
def normalize_ccxt_symbol(sym_env:str)->str:
    s = sym_env.replace(" ","").upper()
    if ":" in s and "/" in s: return s
    if "/" in s and ":" not in s: return s+":USDT"
    if s.endswith("USDT"): return f"{s[:-4]}/USDT:USDT"
    return "DOGE/USDT:USDT"
SYM_CCXT = normalize_ccxt_symbol(ENV["SYMBOL"])

def ms_interval(tf:str)->int:
    tf=tf.strip().lower()
    if tf.endswith("m"): return int(tf[:-1])*60*1000
    if tf.endswith("h"): return int(tf[:-1])*60*60*1000
    if tf.endswith("d"): return int(tf[:-1])*24*60*60*1000
    raise ValueError("Bad INTERVAL")

BAR_MS = ms_interval(ENV["INTERVAL"])

def ema(arr:List[float], n:int)->List[float]:
    k=2/(n+1.0); out=[]; cur=None
    for v in arr:
        cur = v if cur is None else (v-cur)*k+cur
        out.append(cur)
    return out

def rsi(close:List[float], n:int=14)->List[float]:
    up=[0.0]; dn=[0.0]
    for i in range(1,len(close)):
        diff=close[i]-close[i-1]
        up.append(max(diff,0.0)); dn.append(max(-diff,0.0))
    au=ema(up,n); ad=ema(dn,n); rsis=[]
    for g,l in zip(au,ad):
        if l==0: rsis.append(100.0)
        else: rsis.append(100.0 - 100.0/(1.0+g/l))
    return rsis

def true_range(h,l,c):
    out=[]; prev=None
    for i in range(len(c)):
        tr = h[i]-l[i] if prev is None else max(h[i]-l[i], abs(h[i]-prev), abs(l[i]-prev))
        out.append(tr); prev=c[i]
    return out

def atr(h,l,c,n=14)->List[float]:
    return ema(true_range(h,l,c), n)

def adx(high,low,close,n=14)->Tuple[List[float],List[float],List[float],List[float]]:
    tr = true_range(high,low,close)
    dmp=[0.0]; dmm=[0.0]
    for i in range(1,len(high)):
        up=high[i]-high[i-1]; dn=low[i-1]-low[i]
        dmp.append(up if (up>dn and up>0) else 0.0)
        dmm.append(dn if (dn>up and dn>0) else 0.0)
    sm_tr=ema(tr,n); sm_p=ema(dmp,n); sm_m=ema(dmm,n)
    dip=[]; dim=[]
    for p,m,t in zip(sm_p,sm_m,sm_tr):
        dip.append(0.0 if t==0 else 100.0*(p/t))
        dim.append(0.0 if t==0 else 100.0*(m/t))
    dx=[]
    for p,m in zip(dip,dim):
        d = p+m
        dx.append(0.0 if d==0 else 100.0*abs(p-m)/d)
    adxv=ema(dx,n)
    return dip, dim, dx, adxv

def range_filter(closes:List[float], pct:float=1.0):
    base=ema(closes,20)
    up=[b*(1+pct/100.0) for b in base]
    lo=[b*(1-pct/100.0) for b in base]
    trend=[]
    for c,u,l in zip(closes,up,lo):
        if c>u: trend.append(+1)
        elif c<l: trend.append(-1)
        else: trend.append(0)
    return trend, up, lo

def candle_tag(o,h,l,c)->str:
    body=abs(c-o); rng=max(h,o,c,l)-min(h,o,c,l)
    up_w = h-max(o,c); lo_w = min(o,c)-l
    tags=[]
    if body < rng*0.15: tags.append("Doji")
    if lo_w > body*2 and c>o: tags.append("Hammer")
    if up_w > body*2 and o>c: tags.append("ShootingStar")
    return "|".join(tags) if tags else "NONE"

# -------------- Data: BingX public klines --------------
import urllib.request, urllib.parse
def fetch_klines(symbol_ccxt:str, interval:str, limit:int=300)->List[Dict[str,Any]]:
    sym = symbol_ccxt.replace(":USDT","").replace("/","-") # DOGE/USDT:USDT -> DOGE-USDT
    qs = urllib.parse.urlencode({"symbol":sym,"interval":interval,"limit":limit})
    urls = [
        f"https://open-api.bingx.com/openApi/swap/v3/quote/klines?{qs}",
        f"https://open-api.bingx.com/openApi/swap/v2/quote/klines?{qs}",
    ]
    for u in urls:
        try:
            with urllib.request.urlopen(u, timeout=10) as r:
                js=json.loads(r.read().decode())
                data=js.get("data") or []
                out=[]
                for k in data:
                    if isinstance(k,dict):
                        t=int(k.get("openTime")); o=float(k["open"]); h=float(k["high"]); l=float(k["low"]); c=float(k["close"])
                    else:
                        t=int(k[0]); o=float(k[1]); h=float(k[2]); l=float(k[3]); c=float(k[4])
                    out.append({"t":t,"o":o,"h":h,"l":l,"c":c})
                if out: return out
        except Exception as e:
            pass
    return []

# -------------- Broker (LIVE via ccxt) --------------
import ccxt
_ex=None
def get_ex():
    global _ex
    if _ex is not None: return _ex
    if PAPER: raise RuntimeError("BingX keys missing (PAPER mode)")
    _ex = ccxt.bingx({
        "apiKey": ENV["BINGX_API_KEY"],
        "secret": ENV["BINGX_API_SECRET"],
        "enableRateLimit": True,
        "timeout": 20000,
        "options": {"defaultType":"swap"}
    })
    try: _ex.load_markets()
    except Exception as e:
        log("BROKER", f"{ICON['err']} load_markets: {e}", "r"); raise
    return _ex

def ensure_lev(ex):
    try: ex.set_leverage(ENV["LEVERAGE"], SYM_CCXT, params={"side":"BOTH"})
    except Exception as e: log("BROKER", f"set_leverage warn: {e}", "y")

def place_order(side:str, qty:float, ref_price:float)->bool:
    if PAPER:
        log("BROKER", f"{ICON['ok']} PAPER {side} qty={qty:.2f} ~{ref_price:.6f}", "c"); return True
    try:
        ex=get_ex(); ensure_lev(ex)
        ccxt_side="buy" if side.upper()=="BUY" else "sell"
        o=ex.create_order(SYM_CCXT, "market", ccxt_side, float(qty), None, {"reduceOnly":False})
        oid=o.get("id") or o.get("orderId") or "N/A"
        avg=o.get("average") or o.get("price") or ref_price
        log("BROKER", f"{ICON['ok']} LIVE {ccxt_side.upper()} id={oid} qty={qty:.2f} avg={avg}", "g")
        return True
    except Exception as e:
        log("BROKER", f"{ICON['err']} order failed: {e}", "r")
        return False

def close_position_ccxt(side_pos:str, qty:float, ref_price:float)->bool:
    if PAPER:
        log("BROKER", f"{ICON['ok']} PAPER CLOSE {side_pos} qty={qty:.2f}", "c"); return True
    try:
        ex=get_ex(); ensure_lev(ex)
        ccxt_side = "sell" if side_pos=="LONG" else "buy"
        o=ex.create_order(SYM_CCXT,"market",ccxt_side,float(qty),None,{"reduceOnly":True})
        oid=o.get("id") or o.get("orderId") or "N/A"
        avg=o.get("average") or o.get("price") or ref_price
        log("BROKER", f"{ICON['ok']} LIVE CLOSE {side_pos}→{ccxt_side.upper()} id={oid} avg={avg}", "g")
        return True
    except Exception as e:
        log("BROKER", f"{ICON['err']} close failed: {e}", "r")
        return False

# -------------- State --------------
class Position:
    def __init__(self):
        self.side=None        # LONG/SHORT
        self.entry=0.0
        self.qty=0.0
        self.bars=0
        self.pnl=0.0
        self.trail=0.0
        self.tp1=False
        self.adds=0
POS=Position()
COMPOUND_PNL=0.0
LAST_BAR_TS=0
LAST_KEEPALIVE=0

# -------------- Sizing & Preview --------------
def next_qty(price:float, equity:float, risk_pct:float, lev:int)->float:
    notional = equity * (risk_pct/100.0)
    levered = notional * lev
    return max(0.0, levered/max(price,1e-9))

# -------------- Flask (health & metrics) --------------
from flask import Flask, jsonify
app=Flask(__name__)
@app.get("/")
def home(): return f"✅ Smart Bot • {SYM_CCXT} {ENV['INTERVAL']} • {'LIVE' if not PAPER else 'PAPER'}"
@app.get("/metrics")
def metrics():
    return jsonify({
        "symbol": SYM_CCXT, "interval": ENV["INTERVAL"],
        "mode": "live" if not PAPER else "paper",
        "leverage": ENV["LEVERAGE"], "risk_pct": ENV["RISK_PCT"],
        "position": {"open": POS.side is not None, "side": POS.side, "entry": POS.entry, "qty": POS.qty, "pnl": POS.pnl,
                     "trail": POS.trail, "tp1_done": POS.tp1, "adds": POS.adds},
        "compound_pnl": COMPOUND_PNL, "time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    })

def start_server():
    app.run(host="0.0.0.0", port=ENV["PORT"])

# -------------- Core Loop --------------
def core_loop():
    global LAST_BAR_TS, LAST_KEEPALIVE, COMPOUND_PNL
    equity = 100.0  # placeholder for preview; لو عايز تربطه برصيد فعلي، اجلب balance عبر ccxt
    while True:
        try:
            kl=fetch_klines(SYM_CCXT, ENV["INTERVAL"], limit=300)
            if len(kl)<120:
                log("INDICATORS", f"{ICON['err']} not enough data", "r"); time.sleep(ENV["DECISION_EVERY_S"]); continue

            t=[k["t"] for k in kl]; o=[k["o"] for k in kl]; h=[k["h"] for k in kl]; l=[k["l"] for k in kl]; c=[k["c"] for k in kl]
            rsi_arr=rsi(c,14); di_p,di_m,dx_arr,adx_arr=adx(h,l,c,ENV["ADX_LEN"]); atr_arr=atr(h,l,c,ENV["ATR_LEN"])
            trend, up, lo = range_filter(c, ENV["RANGE_MIN_PCT"])
            bar_closed = (t[-1]!=LAST_BAR_TS)
            secs_to_close = int((t[-1]+BAR_MS)/1000 - time.time())

            # Candle intel
            tag = candle_tag(o[-1],h[-1],l[-1],c[-1])
            engulf="NONE"
            if len(c)>=2:
                prev_body=abs(c[-2]-o[-2]); last_body=abs(c[-1]-o[-1])
                if c[-1]>o[-1] and last_body>prev_body and o[-1]<c[-2] and c[-1]>o[-2]: engulf="ENGULF_BULL"
                if c[-1]<o[-1] and last_body>prev_body and o[-1]>c[-2] and c[-1]<o[-2]: engulf="ENGULF_BEAR"

            # Logs — header & indicators
            log("TICK", f"{ICON['tv']} {SYM_CCXT} {ENV['INTERVAL']} • {datetime.utcfromtimestamp(t[-1]/1000).strftime('%Y-%m-%d %H:%M:%S')} UTC", "w")
            log("INDICATORS", f"{ICON['ind']} Price {c[-1]:.6f} | RF≈{(up[-1]+lo[-1])/2:.6f}  hi={h[-1]:.6f} lo={l[-1]:.6f} | "
                               f"RSI={rsi_arr[-1]:.2f}  +DI={di_p[-1]:.2f} -DI={di_m[-1]:.2f}  DX={dx_arr[-1]:.2f}  ADX={adx_arr[-1]:.2f}  ATR={atr_arr[-1]:.6f}", "c")
            log("CANDLES", f"{ICON['candle']} {engulf if engulf!='NONE' else ''} {tag} | candle closes in ~{max(0,secs_to_close)}s", "m")

            # Position snapshot
            if POS.side is None:
                log("POSITION", f"{ICON['flat']} FLAT • Eq≈{equity:.2f} USDT • Risk={ENV['RISK_PCT']:.0f}%×{ENV['LEVERAGE']}x", "y")
            else:
                log("POSITION", f"{('🟩 LONG' if POS.side=='LONG' else '🟥 SHORT')} Entry={POS.entry:.6f} Qty={POS.qty:.2f} Bars={POS.bars} "
                                 f"PnL={POS.pnl:+.6f} Trail={POS.trail:.6f} TP1={POS.tp1}", "y")

            # Preview next qty@10x
            q_preview = next_qty(c[-1], equity, ENV["RISK_PCT"], ENV["LEVERAGE"])
            log("PREVIEW", f"Next order @{ENV['LEVERAGE']}x ≈ {q_preview:.2f} DOGE (~{q_preview*c[-1]:.2f} USDT)", "b")

            # Signals (on CLOSED bar)
            buy_sig  = (trend[-2] <= 0 and trend[-1] > 0)
            sell_sig = (trend[-2] >= 0 and trend[-1] < 0)

            # Entry policy: wait bar close to match TV
            if ENV["FORCE_TV_ENTRIES"] and not bar_closed:
                log("RESULTS", f"{ICON['wait']} No trade — waiting bar close. close in ~{max(0,secs_to_close)}s", "w")
            else:
                # Spike filter: avoid giant single-bar moves
                bar_move = abs(c[-1]-o[-1])
                huge_spike = bar_move > ENV["SPIKE_FILTER_ATR_MULTIPLIER"]*atr_arr[-1]

                if POS.side is None:
                    reason=None; side=None
                    if huge_spike:
                        reason=f"spike {bar_move:.6f}>{ENV['SPIKE_FILTER_ATR_MULTIPLIER']}×ATR"
                    elif buy_sig: side="LONG"
                    elif sell_sig: side="SHORT"
                    else: reason="no signal"

                    if reason:
                        log("RESULTS", f"{ICON['wait']} No trade — reason: {reason}", "w")
                    else:
                        q = max(0.0, q_preview)
                        if q<=0:
                            log("RESULTS", f"{ICON['err']} sizing=0", "r")
                        else:
                            ok = place_order("BUY" if side=="LONG" else "SELL", q, c[-1])
                            if ok:
                                POS.side=side; POS.entry=c[-1]; POS.qty=q; POS.bars=0; POS.pnl=0.0; POS.trail=0.0; POS.tp1=False; POS.adds=0
                                log("POSITION", f"{ICON['buy'] if side=='LONG' else ICON['sell']} ACTIVE", "g" if side=="LONG" else "r")
                else:
                    # ------- Smart Management -------
                    POS.bars += 1
                    POS.pnl = (c[-1]-POS.entry)*POS.qty if POS.side=="LONG" else (POS.entry-c[-1])*POS.qty
                    rr = (c[-1]-POS.entry)/POS.entry if POS.side=="LONG" else (POS.entry-c[-1])/POS.entry

                    # TP1 partial
                    if not POS.tp1 and rr >= ENV["TP1_PCT"]/100.0:
                        close_qty = POS.qty * ENV["TP1_CLOSE_FRAC"]
                        if close_qty>0:
                            # reduce-only partial close
                            if close_position_ccxt(POS.side, close_qty, c[-1]):
                                POS.qty -= close_qty
                                POS.tp1 = True
                                COMPOUND_PNL += (c[-1]-POS.entry)*close_qty if POS.side=="LONG" else (POS.entry-c[-1])*close_qty
                                log("RESULTS", f"{ICON['tp']} TP1: closed {close_qty:.2f} DOGE • rr={rr*100:.2f}%", "g")

                    # Breakeven arm
                    if POS.tp1 and POS.trail==0.0 and rr>=ENV["BREAKEVEN_AFTER_PCT"]/100.0:
                        POS.trail = POS.entry
                        log("RESULTS", f"{ICON['be']} Breakeven armed @ {POS.trail:.6f}", "g")

                    # ATR Trailing
                    if rr>=ENV["TRAIL_ACTIVATE_PCT"]/100.0:
                        raw = (c[-1]-ENV["ATR_MULT_TRAIL"]*atr_arr[-1]) if POS.side=="LONG" else (c[-1]+ENV["ATR_MULT_TRAIL"]*atr_arr[-1])
                        if POS.trail==0.0:
                            POS.trail = raw; log("RESULTS", f"{ICON['trail']} Trail start @ {POS.trail:.6f}", "g")
                        else:
                            POS.trail = max(POS.trail, raw) if POS.side=="LONG" else min(POS.trail, raw)

                    # Hold-TP if trend strong
                    if ENV["HOLD_TP_STRONG"]:
                        slope = (c[-1]-c[-4])/max(1e-9,c[-4]) if len(c)>=4 else 0.0
                        if adx_arr[-1]>=ENV["HOLD_TP_ADX"] and slope>ENV["HOLD_TP_SLOPE"]/100.0:
                            log("RESULTS", f"Hold-TP: ADX {adx_arr[-1]:.1f} & slope {slope*100:.2f}%", "c")

                    # Scale-In (adds)
                    if ENV["SCALE_IN_ENABLED"] and POS.adds<ENV["SCALE_IN_MAX_ADDS"]:
                        slope = (c[-1]-c[-4])/max(1e-9,c[-4]) if len(c)>=4 else 0.0
                        if adx_arr[-1]>=ENV["SCALE_IN_ADX_MIN"] and slope>ENV["SCALE_IN_SLOPE_MIN"]/100.0:
                            add_qty = POS.qty*0.25
                            if add_qty>0:
                                ok = place_order("BUY" if POS.side=="LONG" else "SELL", add_qty, c[-1])
                                if ok:
                                    POS.qty += add_qty; POS.adds += 1
                                    log("RESULTS", f"Scale-In +{add_qty:.2f} DOGE (adds {POS.adds}/{ENV['SCALE_IN_MAX_ADDS']})", "b")

                    # Exit: trail or reverse
                    exit_reason=None
                    if POS.trail!=0.0:
                        if (POS.side=="LONG" and c[-1]<POS.trail) or (POS.side=="SHORT" and c[-1]>POS.trail):
                            exit_reason="trail hit"
                    if (POS.side=="LONG" and sell_sig) or (POS.side=="SHORT" and buy_sig):
                        exit_reason = exit_reason or "reverse signal"

                    if exit_reason:
                        realized = POS.pnl
                        ok = close_position_ccxt(POS.side, POS.qty, c[-1])
                        if ok:
                            COMPOUND_PNL += realized
                            log("RESULTS", f"{ICON['ok']} CLOSE {POS.side} • reason={exit_reason} • PnL={realized:+.4f} • Compound={COMPOUND_PNL:+.4f}", "g" if realized>=0 else "r")
                            POS.side=None; POS.entry=0.0; POS.qty=0.0; POS.bars=0; POS.pnl=0.0; POS.trail=0.0; POS.tp1=False; POS.adds=0

            # keepalive
            if int(time.time()) - LAST_KEEPALIVE >= ENV["KEEPALIVE_SECONDS"]:
                LAST_KEEPALIVE = int(time.time())
                log("KEEPALIVE", "ok (200)", "w")

            LAST_BAR_TS = t[-1]
            time.sleep(ENV["DECISION_EVERY_S"])

        except Exception as e:
            log("ERROR", f"{ICON['err']} {e}\n{traceback.format_exc()}", "r")
            time.sleep(ENV["DECISION_EVERY_S"])

# -------------- Boot --------------
if __name__=="__main__":
    log("BOOT", f"{ICON['ok']} Start • Mode={'LIVE' if not PAPER else 'PAPER'} • TV-sync={'ON' if ENV['FORCE_TV_ENTRIES'] else 'OFF'}", "g")
    log("CONF", f"Symbol={SYM_CCXT}  Interval={ENV['INTERVAL']}  Leverage={ENV['LEVERAGE']}x  Risk={ENV['RISK_PCT']}%", "w")
    threading.Thread(target=start_server, daemon=True).start()
    core_loop()
