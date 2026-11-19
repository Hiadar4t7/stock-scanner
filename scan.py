#!/usr/bin/env python3
"""
Daily data robot for Haidar's Expert Stock Playbook (Part C rules).

Runs on GitHub Actions every weekday morning and:
1. downloads daily prices for every stock in watchlist.csv (+ SPY, VIX, DAX, FX rates)
2. computes trend, momentum, volatility and volume indicators from COMPLETED daily bars
3. detects the playbook's strategy setups 1-4 with entry, stop, target and reward:risk
4. scores the best candidates with the automatic 10-row scorecard (max 20 points)
5. writes data/latest.json (read by the Claude daily scan), data/latest_all.csv,
   data/latest.md and a dated copy in data/archive/

Everything is rule-based and reproducible. It is not financial advice.
Run `python scan.py --selftest` to test the logic with made-up prices (no internet).
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data")

# ---------- Settings: keep in sync with the playbook (Part C) ----------
RISK_EUR = 20.0            # max loss per trade if the stop is hit (green weather)
POT_EUR = 500.0            # max money in one trade (the whole test pot)
FEES_EUR = 2.0             # 1 EUR to buy + 1 EUR to sell on Trade Republic
MIN_TURNOVER_EUR = 25e6    # 20-day average daily turnover: "large and heavily traded"
SHORTLIST_N = 25           # how many setups get the full scorecard
MIN_SCORE = 14             # minimum total score for a pick
FX = {"USD": "EURUSD=X", "GBP": "EURGBP=X", "CHF": "EURCHF=X", "DKK": "EURDKK=X"}
MARKET = ["SPY", "^VIX", "^GDAXI"]
STRATEGY_NAMES = {1: "Pullback in an uptrend", 2: "Breakout from a base",
                  3: "Post-earnings drift", 4: "Oversold bounce"}
PREFERENCE = {1: 0, 3: 1, 2: 2, 4: 3}   # beginner preference when R:R ties


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def num(x, n=2):
    """Round for JSON; NaN/inf/None become None."""
    try:
        if x is None:
            return None
        x = float(x)
        if math.isnan(x) or math.isinf(x):
            return None
        return round(x, n)
    except Exception:
        return None


# ---------------------------------------------------------------- data ---
def download(symbols, period="2y"):
    """Daily OHLCV per symbol from Yahoo Finance (via yfinance), with retries."""
    import yfinance as yf
    out, pending = {}, list(dict.fromkeys(symbols))
    for attempt in range(3):
        if not pending:
            break
        try:
            df = yf.download(pending, period=period, interval="1d", auto_adjust=False,
                             group_by="ticker", threads=True, progress=False)
        except Exception as e:  # noqa: BLE001
            log("download error:", e)
            df = None
        still = []
        for s in pending:
            sub = None
            try:
                if df is not None and isinstance(df.columns, pd.MultiIndex):
                    if s in df.columns.get_level_values(0):
                        sub = df[s]
                elif df is not None and len(pending) == 1:
                    sub = df
            except Exception:  # noqa: BLE001
                sub = None
            if sub is not None:
                sub = sub.rename(columns=lambda c: str(c).title())
                need = ["Open", "High", "Low", "Close", "Volume"]
                if all(c in sub.columns for c in need):
                    sub = sub[need].dropna(subset=["Close", "High", "Low"])
                    if len(sub):
                        out[s] = sub
                        continue
            still.append(s)
        pending = still
        if pending:
            log(f"retrying {len(pending)} symbols")
            time.sleep(5 * (attempt + 1))
    return out, pending


def completed_bars(df, today):
    """Drop today's unfinished bar (European markets are open when the robot runs)."""
    df = df.copy()
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = idx
    live = None
    if len(df) and df.index[-1].date() >= today:
        live = float(df["Close"].iloc[-1])
        df = df.iloc[:-1]
    return df, live


def rsi(close, n):
    d = close.diff()
    up, dn = d.clip(lower=0), (-d).clip(lower=0)
    au = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    ad = dn.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = au / ad.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(ad != 0, 100.0)


def atr(df, n=14):
    pc = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()],
                   axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def indicators(df):
    c, h, l, o, v = df["Close"], df["High"], df["Low"], df["Open"], df["Volume"]
    C = float(c.iloc[-1])
    sma = {n: c.rolling(n).mean() for n in (5, 10, 20, 50, 200)}
    a = atr(df)

    def f(s, i=-1):
        return float(s.iloc[i])

    def perf(n):
        return C / f(c, -1 - n) - 1 if len(c) > n else None

    ind = {
        "date": str(df.index[-1].date()),
        "open": f(o), "high": f(h), "low": f(l), "close": C,
        "prev_close": f(c, -2), "chg_1d": C / f(c, -2) - 1, "chg_3d": C / f(c, -4) - 1,
        "sma5": f(sma[5]), "sma10": f(sma[10]), "sma20": f(sma[20]),
        "sma50": f(sma[50]), "sma200": f(sma[200]), "sma50_10d_ago": f(sma[50], -11),
        "rsi14": f(rsi(c, 14)), "rsi2": f(rsi(c, 2)), "atr14": f(a), "atr_pct": f(a) / C,
        "vol": f(v), "avgvol10": float(v.iloc[-10:].mean()), "avgvol20": float(v.iloc[-20:].mean()),
        "avgvol50": float(v.iloc[-50:].mean()),
        "perf_1m": perf(21), "perf_3m": perf(63), "perf_6m": perf(126), "perf_12m": perf(252),
        "mom_12_1": (f(c, -22) / f(c, -253) - 1) if len(c) > 253 else None,
        "high_60": float(h.iloc[-60:].max()), "high_52w": float(h.iloc[-252:].max()),
        "low_52w": float(l.iloc[-252:].min()), "low_10": float(l.iloc[-10:].min()),
        "high_10": float(h.iloc[-10:].max()), "hh15_close": float(c.iloc[-15:].max()),
        "base_high_30": float(h.iloc[-30:].max()), "base_low_30": float(l.iloc[-30:].min()),
    }
    ind["dist_52w_high"] = C / ind["high_52w"] - 1
    ind["base_range_30"] = (ind["base_high_30"] - ind["base_low_30"]) / ind["base_high_30"]
    ind["range_10"] = (ind["high_10"] - ind["low_10"]) / ind["high_10"]
    ind["rel_vol"] = ind["vol"] / ind["avgvol20"] if ind["avgvol20"] else None

    streak = 0
    for i in range(len(c) - 1, 0, -1):
        if c.iloc[i] < c.iloc[i - 1]:
            streak += 1
        else:
            break
    ind["down_streak"] = streak

    ch = c.diff().iloc[-10:]
    v10 = v.iloc[-10:]
    up_v, dn_v = v10[ch > 0], v10[ch < 0]
    ind["upday_vol"] = float(up_v.mean()) if len(up_v) else None
    ind["downday_vol"] = float(dn_v.mean()) if len(dn_v) else None

    gaps = []
    for i in range(max(51, len(df) - 10), len(df)):
        pc, oo, cc = f(c, i - 1), f(o, i), f(c, i)
        hh, ll, vv = f(h, i), f(l, i), f(v, i)
        av = float(v.iloc[i - 50:i].mean())
        g = {"date": str(df.index[i].date()), "bars_ago": len(df) - 1 - i,
             "gap_pct": oo / pc - 1, "day_chg": cc / pc - 1, "vol_x": vv / av if av else None,
             "close_pos": (cc - ll) / (hh - ll) if hh > ll else 0.5,
             "low": ll, "high": hh, "prev_close": pc}
        if (abs(g["day_chg"]) >= 0.04 or abs(g["gap_pct"]) >= 0.04) and (g["vol_x"] or 0) >= 2:
            gaps.append(g)
    ind["event_gaps"] = gaps
    ind["bars_10"] = [[str(df.index[i].date()), num(f(o, i)), num(f(h, i)), num(f(l, i)),
                       num(f(c, i)), int(f(v, i))] for i in range(len(df) - 10, len(df))]
    return ind


# -------------------------------------------------------------- setups ---
def _rr(entry, stop, target):
    risk = entry - stop
    return (target - entry) / risk if risk > 0 else None


def setup_1(ind):
    C, A = ind["close"], ind["atr14"]
    depth = 1 - C / ind["hh15_close"]
    cond = {
        "trend_close_gt_sma50_gt_sma200": C > ind["sma50"] > ind["sma200"],
        "stronger_than_spy_3m": (ind.get("rs_3m") or -1) > 0,
        "pullback_3_to_8pct_from_15d_high": 0.03 <= depth <= 0.08,
        "near_sma20_or_sma50": (0.97 * ind["sma20"] <= C <= 1.02 * ind["sma20"])
        or (0.98 * ind["sma50"] <= C <= 1.04 * ind["sma50"]),
        "rsi14_35_to_55": 35 <= ind["rsi14"] <= 55,
        "lower_volume_on_down_days": ind["downday_vol"] is None or ind["upday_vol"] is None
        or ind["downday_vol"] < ind["upday_vol"],
    }
    entry = ind["high"] + 0.1 * A
    stop = min(ind["low_10"] - 0.1 * A, entry - 1.5 * A)
    target = ind["high_60"]
    rr = _rr(entry, stop, target)
    cond["target_at_least_1_5R_away"] = rr is not None and rr >= 1.5
    return {"strategy": 1, "entry_order": "stop-buy just above the last day's high",
            "entry": entry, "stop": stop, "target": target, "rr": rr,
            "target_note": "60-day high: sell half there, trail the rest (2R/3R levels below)",
            "hold_days": "3-15", "time_stop_days": 10, "conditions": cond,
            "valid": all(cond.values())}


def setup_2(ind):
    C, A = ind["close"], ind["atr14"]
    H, L = ind["base_high_30"], ind["base_low_30"]
    cond = {
        "trend_close_gt_sma50_gt_sma200": C > ind["sma50"] > ind["sma200"],
        "base_30d_range_max_15pct": ind["base_range_30"] <= 0.15,
        "range_tightening_last_10d": ind["range_10"] <= 0.6 * ind["base_range_30"],
        "close_within_3pct_of_base_top": 0.97 * H <= C <= H,
        "base_top_near_52w_high": H >= 0.90 * ind["high_52w"],
        "volume_drying_up": ind["avgvol10"] < ind["avgvol50"],
    }
    entry = H * 1.005
    stop = min(ind["low_10"] - 0.1 * A, entry - 1.0 * A)
    cond["stop_max_8pct_below_entry"] = stop >= entry * 0.92
    target = entry + (H - L)
    rr = _rr(entry, stop, target)
    cond["rr_at_least_1_5"] = rr is not None and rr >= 1.5
    return {"strategy": 2, "entry_order": "stop-buy 0.5% above the base top",
            "entry": entry, "stop": stop, "target": target, "rr": rr,
            "target_note": "measured move: base height added to the breakout level",
            "fill_rule": "keep the trade only if the fill day's volume is at least 1.5x normal; "
                         "exit if price closes back inside the base within 1-3 days",
            "hold_days": "5-30", "time_stop_days": 15, "conditions": cond,
            "valid": all(cond.values())}


def setup_3(ind):
    C, A = ind["close"], ind["atr14"]
    gaps = [g for g in ind["event_gaps"] if g["bars_ago"] <= 9
            and max(g["gap_pct"], g["day_chg"]) >= 0.05 and (g["vol_x"] or 0) >= 3
            and g["close_pos"] >= 0.5]
    if not gaps:
        return {"strategy": 3, "valid": False,
                "conditions": {"big_gap_up_on_3x_volume_last_10d": False}}
    g = gaps[-1]
    entry, stop = C, g["low"] - 0.1 * A
    risk = entry - stop
    cond = {
        "big_gap_up_on_3x_volume_last_10d": True,
        "still_above_gap_day_low": C > g["low"],
        "gap_not_filled": C > g["prev_close"],
        "above_sma200": C > ind["sma200"],
        "stop_max_10pct_below_entry": risk > 0 and risk / entry <= 0.10,
    }
    return {"strategy": 3, "entry_order": "limit near the last close",
            "entry": entry, "stop": stop, "target": entry + 2.5 * risk if risk > 0 else None,
            "rr": 2.5 if risk > 0 else None, "gap_day": g["date"],
            "target_note": "2.5R; always sell before the next earnings date",
            "must_verify": "gap came from earnings: sales AND EPS beat AND guidance raised",
            "hold_days": "5-40", "time_stop_days": 30, "conditions": cond,
            "valid": all(cond.values())}


def setup_4(ind):
    C, A = ind["close"], ind["atr14"]
    oversold = ind["rsi2"] < 10 or (ind["down_streak"] >= 3 and ind["chg_3d"] <= -0.05)
    news_gap = any(g["bars_ago"] <= 2 and min(g["gap_pct"], g["day_chg"]) <= -0.05
                   and (g["vol_x"] or 0) >= 2.5 for g in ind["event_gaps"])
    entry, stop, target = C, C - 2 * A, ind["sma20"]
    rr = _rr(entry, stop, target)
    cond = {"above_sma200": C > ind["sma200"], "oversold_rsi2_lt_10_or_3_down_days_5pct": oversold,
            "no_big_news_gap_down_last_3d": not news_gap, "rr_at_least_1": rr is not None and rr >= 1}
    return {"strategy": 4, "entry_order": "limit near the last close",
            "entry": entry, "stop": stop, "target": target, "rr": rr,
            "target_note": "20-day average; exit early when a close is above the 5-day average",
            "hold_days": "2-7", "time_stop_days": 7, "conditions": cond,
            "valid": all(cond.values())}


def all_setups(ind):
    return [setup_1(ind), setup_2(ind), setup_3(ind), setup_4(ind)]


def best_setup(setups):
    valid = [s for s in setups if s.get("valid")]
    if not valid:
        return None
    return sorted(valid, key=lambda s: (-(s["rr"] or 0), PREFERENCE[s["strategy"]]))[0]


# ------------------------------------------------------------- weather ---
def market_weather(spy, vix):
    if spy is None or vix is None:
        return "UNKNOWN"
    C = spy["close"]
    if C < spy["sma200"] or vix > 30:
        return "RED"
    if C < spy["sma50"] or vix >= 25:
        return "YELLOW"
    return "GREEN"


# -------------------------------------------------------- fundamentals ---
def fundamentals(sym, today, fx_eur, close, currency):
    """Business numbers from Yahoo for one stock. Missing values stay None."""
    import yfinance as yf
    t = yf.Ticker(sym)
    out = {"source": "Yahoo Finance via yfinance"}
    try:
        info = t.info or {}
    except Exception as e:  # noqa: BLE001
        info, out["error"] = {}, f"info: {e}"[:200]
    for k in ["longName", "sector", "industry", "financialCurrency", "profitMargins",
              "operatingMargins", "revenueGrowth", "earningsGrowth", "freeCashflow", "totalDebt",
              "totalCash", "ebitda", "forwardPE", "trailingPE", "trailingPegRatio", "pegRatio",
              "sharesOutstanding", "shortPercentOfFloat", "recommendationMean",
              "numberOfAnalystOpinions", "targetMeanPrice"]:
        out[k] = info.get(k)

    nxt = None
    try:
        cal = t.calendar
        dates = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if dates:
            future = [pd.Timestamp(d).date() for d in dates if pd.Timestamp(d).date() >= today]
            nxt = min(future) if future else None
    except Exception:  # noqa: BLE001
        pass
    if nxt is None:
        for k in ("earningsTimestampStart", "earningsTimestamp"):
            ts = info.get(k)
            if ts:
                d = dt.datetime.fromtimestamp(ts, dt.timezone.utc).date()
                if d >= today:
                    nxt = d
                    break
    out["next_earnings"] = str(nxt) if nxt else None
    out["trading_days_to_earnings"] = int(np.busday_count(today, nxt)) if nxt else None

    up = down = None
    try:
        er = t.eps_revisions
        if er is not None and len(er) and "0y" in er.index:
            row = er.loc["0y"]
            cols = {c.lower(): c for c in row.index}
            up = row[cols["uplast30days"]] if "uplast30days" in cols else None
            down = row[cols["downlast30days"]] if "downlast30days" in cols else None
    except Exception:  # noqa: BLE001
        pass
    out["eps_up_30d"], out["eps_down_30d"] = num(up, 0), num(down, 0)

    fin_cur = out.get("financialCurrency") or currency
    fx_fin = fx_eur.get(fin_cur)
    fx_px = fx_eur.get(currency)
    shares = out.get("sharesOutstanding")
    mcap_eur = shares * close * fx_px if shares and fx_px else None
    fcf = out.get("freeCashflow")
    out["market_cap_eur"] = mcap_eur
    out["fcf_yield"] = (fcf * fx_fin / mcap_eur) if (fcf is not None and fx_fin and mcap_eur) else None
    debt, cash, ebitda = out.get("totalDebt"), out.get("totalCash"), out.get("ebitda")
    out["net_debt_to_ebitda"] = ((debt - cash) / ebitda) if (debt is not None and cash is not None
                                                             and ebitda and ebitda > 0) else None
    out["net_cash"] = (cash > debt) if (debt is not None and cash is not None) else None
    return out


# ------------------------------------------------------------- scoring ---
def score(stock, weather):
    f, ind, best = stock.get("fund") or {}, stock["ind"], stock.get("best")
    rows, notes = {}, []
    pm, om = f.get("profitMargins"), f.get("operatingMargins")
    if pm is None:
        rows["1_business"] = 0
        notes.append("row 1: margins missing")
    else:
        rows["1_business"] = 2 if (pm > 0 and (om or 0) >= 0.15) else (1 if pm > 0 else 0)

    g = f.get("revenueGrowth")
    rows["2_revenue_growth"] = 0 if g is None else (2 if g >= 0.10 else (1 if g >= 0 else 0))
    if g is None:
        notes.append("row 2: revenue growth missing")

    fcf, fy = f.get("freeCashflow"), f.get("fcf_yield")
    if fcf is None:
        rows["3_free_cash_flow"] = 0
        notes.append("row 3: free cash flow missing")
    else:
        rows["3_free_cash_flow"] = 2 if (fcf > 0 and fy is not None and fy >= 0.025) else (1 if fcf > 0 else 0)

    sector, industry = (f.get("sector") or ""), (f.get("industry") or "")
    is_fin = sector == "Financial Services" and any(w in industry for w in
                                                     ("Bank", "Insurance", "Capital Markets", "Asset Management", "Credit"))
    nd = f.get("net_debt_to_ebitda")
    if is_fin:
        rows["4_debt"] = 1
        notes.append("row 4: bank/insurer - debt rule does not apply, set to 1")
    elif f.get("net_cash"):
        rows["4_debt"] = 2
    elif nd is None:
        rows["4_debt"] = 0
        notes.append("row 4: debt or EBITDA missing/negative")
    else:
        rows["4_debt"] = 2 if nd < 2 else (1 if nd <= 3 else 0)

    peg = f.get("trailingPegRatio") or f.get("pegRatio")
    fpe = f.get("forwardPE")
    if peg is not None and peg > 0:
        rows["5_valuation"] = 2 if peg <= 2 else (1 if peg <= 3 else 0)
    else:
        rows["5_valuation"] = 1 if (fpe is not None and 0 < fpe <= 30) else 0

    up, dn = f.get("eps_up_30d"), f.get("eps_down_30d")
    if up is None or dn is None:
        rows["6_estimates"] = 1
        notes.append("row 6: no revision data - Claude checks the news")
    else:
        rows["6_estimates"] = 2 if up > dn else (1 if up == dn else 0)

    td = f.get("trading_days_to_earnings")
    if td is None:
        rows["7_event_risk"] = 1
        notes.append("row 7: earnings date unknown - verify")
    else:
        rows["7_event_risk"] = 2 if td > 15 else (1 if td > 5 else 0)

    C = ind["close"]
    if C > ind["sma20"] and C > ind["sma50"] > ind["sma200"]:
        rows["8_trend"] = 2
    elif C > ind["sma50"] and C > ind["sma200"]:
        rows["8_trend"] = 1
    else:
        rows["8_trend"] = 0

    rows["9_market_weather"] = {"GREEN": 2, "YELLOW": 1}.get(weather, 0)

    if best is None:
        rows["10_setup_rr"] = 0
    elif best["strategy"] == 4:
        rows["10_setup_rr"] = 1 if (best["rr"] or 0) >= 1 else 0
    else:
        rr = best["rr"] or 0
        rows["10_setup_rr"] = 2 if rr >= 2 else (1 if rr >= 1.5 else 0)

    total = sum(rows.values())
    breakers_ok = all(rows[k] >= 1 for k in ("1_business", "4_debt", "7_event_risk", "10_setup_rr"))
    return rows, total, breakers_ok, notes


def sizing(setup, eur_per_unit, risk_eur=RISK_EUR, cap_eur=POT_EUR):
    e, s, t = (setup[k] * eur_per_unit for k in ("entry", "stop", "target"))
    risk_ps = e - s
    if risk_ps <= 0 or e <= 0:
        return {"tradeable": False, "reason": "invalid levels"}
    n = min(math.floor(risk_eur / risk_ps), math.floor(cap_eur / e))
    out = {"entry_eur": num(e), "stop_eur": num(s), "target_eur": num(t),
           "risk_per_share_eur": num(risk_ps), "stop_pct": num((s / e - 1) * 100, 1),
           "target_pct": num((t / e - 1) * 100, 1), "risk_budget_eur": risk_eur,
           "breakeven_win_rate_pct": num(100 / (1 + setup["rr"]), 0) if setup.get("rr") else None}
    if n < 1:
        out.update(tradeable=False, shares=0,
                   reason="1 share risks more than the risk budget or costs more than the pot")
        return out
    out.update(tradeable=True, shares=n, position_eur=num(n * e),
               loss_at_stop_eur=num(-(n * risk_ps) - FEES_EUR),
               gain_at_target_eur=num(n * (t - e) - FEES_EUR))
    return out


# ---------------------------------------------------------------- main ---
def run(download_fn=download, fundamentals_fn=fundamentals, today=None):
    today = today or dt.datetime.now(dt.timezone.utc).date()
    generated = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    wl = pd.read_csv(os.path.join(HERE, "watchlist.csv"))
    syms = list(wl["symbol"])
    prices, missing = download_fn(syms + MARKET + list(FX.values()))
    errors = [f"no price data: {s}" for s in missing]

    fx_eur = {"EUR": 1.0}
    for cur, tk in FX.items():
        if tk in prices:
            rate = float(prices[tk]["Close"].iloc[-1])   # units of `cur` per 1 EUR
            fx_eur[cur] = 1.0 / rate
    if "GBP" in fx_eur:
        fx_eur["GBp"] = fx_eur["GBP"] / 100.0

    mkt = {}
    for tk in MARKET:
        if tk in prices:
            df, _ = completed_bars(prices[tk], today)
            if len(df) >= 210:
                mkt[tk] = indicators(df)
    spy, vix = mkt.get("SPY"), mkt.get("^VIX")
    weather = market_weather(spy, vix["close"] if vix else None)

    stocks = []
    for row in wl.itertuples(index=False):
        s = row.symbol
        if s not in prices:
            continue
        try:
            df, live = completed_bars(prices[s], today)
            if len(df) < 210:
                errors.append(f"too little history: {s}")
                continue
            ind = indicators(df)
            fxr = fx_eur.get(row.currency)
            stale = (today - pd.Timestamp(ind["date"]).date()).days > 6
            ind["rs_3m"] = (ind["perf_3m"] - spy["perf_3m"]) if (spy and ind["perf_3m"] is not None) else None
            turnover = ind["avgvol20"] * ind["close"] * fxr if fxr else None
            setups = all_setups(ind)
            stocks.append({"symbol": s, "name": row.name, "region": row.region,
                           "currency": row.currency, "eur_per_unit": fxr, "live_price": live,
                           "ind": ind, "turnover_eur": turnover, "stale": stale,
                           "liquid": bool(turnover and turnover >= MIN_TURNOVER_EUR),
                           "setups": setups, "best": best_setup(setups)})
        except Exception as e:  # noqa: BLE001
            errors.append(f"{s}: {e}")
            log(traceback.format_exc())

    usable = [x for x in stocks if x["liquid"] and not x["stale"] and x["eur_per_unit"]]
    cands = [x for x in usable if x["best"] is not None]

    def pre_key(x):
        b = x["best"]
        rr = b["rr"] or 0
        trend = 2 if x["ind"]["close"] > x["ind"]["sma20"] > 0 else 1
        return (-(2 if rr >= 2 else 1), -trend, -(x["ind"]["rs_3m"] or -9))
    cands.sort(key=pre_key)
    shortlist = cands[:SHORTLIST_N]

    for x in shortlist:
        try:
            x["fund"] = fundamentals_fn(x["symbol"], today, fx_eur, x["ind"]["close"], x["currency"])
        except Exception as e:  # noqa: BLE001
            x["fund"] = {"error": str(e)[:200]}
        time.sleep(0.5)
        rows, total, ok, notes = score(x, weather)
        x.update(rows=rows, total=total, dealbreakers_ok=ok, notes=notes)
        x["sizing"] = sizing(x["best"], x["eur_per_unit"])
        x["eligible"] = bool(ok and total >= MIN_SCORE and weather in ("GREEN", "YELLOW")
                             and x["sizing"].get("tradeable"))

    shortlist.sort(key=lambda x: (not x["eligible"], -x["total"], -(x["best"]["rr"] or 0),
                                  -(x["ind"]["rs_3m"] or -9)))

    momentum = sorted([x for x in usable if x["ind"]["mom_12_1"] is not None
                       and x["ind"]["close"] > x["ind"]["sma200"]],
                      key=lambda x: -x["ind"]["mom_12_1"])[:5]

    def pack_ind(ind):
        keep = ["date", "close", "high", "low", "chg_1d", "sma20", "sma50", "sma200", "rsi14", "rsi2",
                "atr14", "atr_pct", "rel_vol", "perf_1m", "perf_3m", "perf_12m", "rs_3m", "mom_12_1",
                "high_60", "high_52w", "dist_52w_high", "down_streak", "base_range_30"]
        d = {k: num(ind.get(k), 4) for k in keep}
        d["date"] = ind["date"]
        d["event_gaps"] = [{k: (num(v, 4) if isinstance(v, float) else v) for k, v in g.items()}
                           for g in ind["event_gaps"]]
        d["bars_10"] = ind["bars_10"]
        return d

    def pack_setup(s):
        return {k: (num(v, 4) if isinstance(v, float) else v) for k, v in s.items()}

    def pack(x):
        fund = {k: (num(v, 4) if isinstance(v, float) else v) for k, v in (x.get("fund") or {}).items()}
        return {"symbol": x["symbol"], "name": x["name"], "region": x["region"],
                "currency": x["currency"], "eur_per_unit": num(x["eur_per_unit"], 6),
                "live_price_today": num(x["live_price"], 4), "eligible": x.get("eligible"),
                "total": x.get("total"), "rows": x.get("rows"), "dealbreakers_ok": x.get("dealbreakers_ok"),
                "notes": x.get("notes"), "best_setup": pack_setup(x["best"]) if x["best"] else None,
                "other_valid_setups": [pack_setup(s) for s in x["setups"]
                                       if s.get("valid") and s is not x["best"]],
                "sizing_green_20eur_risk": x.get("sizing"),
                "sizing_yellow_10eur_risk": sizing(x["best"], x["eur_per_unit"], risk_eur=10.0)
                if x["best"] else None,
                "indicators": pack_ind(x["ind"]), "fundamentals": fund,
                "turnover_eur_m": num((x["turnover_eur"] or 0) / 1e6, 1)}

    result = {
        "generated_at_utc": generated,
        "data_date_us": spy["date"] if spy else None,
        "settings": {"risk_eur": RISK_EUR, "pot_eur": POT_EUR, "fees_eur": FEES_EUR,
                     "min_turnover_eur": MIN_TURNOVER_EUR, "min_score": MIN_SCORE},
        "market": {
            "weather": weather,
            "spy": {k: num(spy.get(k), 4) for k in ("close", "sma50", "sma200", "perf_3m")} | {"date": spy["date"]} if spy else None,
            "vix": num(vix["close"]) if vix else None,
            "dax": {k: num(mkt["^GDAXI"].get(k), 2) for k in ("close", "sma50", "sma200")} if "^GDAXI" in mkt else None,
        },
        "fx_eur_per_unit": {k: num(v, 6) for k, v in fx_eur.items()},
        "counts": {"watchlist": len(syms), "with_data": len(stocks), "liquid_and_fresh": len(usable),
                   "with_valid_setup": len(cands), "scored": len(shortlist),
                   "eligible": sum(1 for x in shortlist if x.get("eligible"))},
        "shortlist": [pack(x) for x in shortlist],
        "momentum_top5_strategy6": [{"symbol": x["symbol"], "name": x["name"],
                                     "mom_12_1": num(x["ind"]["mom_12_1"], 4),
                                     "close": num(x["ind"]["close"], 2), "currency": x["currency"]}
                                    for x in momentum],
        "errors": errors[:50],
    }

    os.makedirs(os.path.join(OUT, "archive"), exist_ok=True)
    with open(os.path.join(OUT, "latest.json"), "w") as fh:
        json.dump(result, fh, indent=1, default=str)
    with open(os.path.join(OUT, "archive", f"{today}.json"), "w") as fh:
        json.dump(result, fh, default=str)

    table = []
    for x in stocks:
        i = x["ind"]
        table.append({"symbol": x["symbol"], "name": x["name"], "currency": x["currency"],
                      "date": i["date"], "close": num(i["close"], 4), "sma20": num(i["sma20"], 4),
                      "sma50": num(i["sma50"], 4), "sma200": num(i["sma200"], 4), "rsi14": num(i["rsi14"], 1),
                      "atr14": num(i["atr14"], 4), "perf_3m": num(i["perf_3m"], 4),
                      "rs_3m": num(i["rs_3m"], 4), "liquid": x["liquid"], "stale": x["stale"],
                      "best_strategy": x["best"]["strategy"] if x["best"] else None,
                      "eur_per_unit": num(x["eur_per_unit"], 6)})
    pd.DataFrame(table).to_csv(os.path.join(OUT, "latest_all.csv"), index=False)

    lines = [f"# Daily scan data - {result['data_date_us']}",
             f"Generated {generated} UTC. Market weather: **{weather}**.", "",
             "| # | Stock | Score | Strategy | Entry | Stop | Target | R:R | Eligible |",
             "|---|---|---|---|---|---|---|---|---|"]
    for n, x in enumerate(result["shortlist"], 1):
        b = x["best_setup"] or {}
        lines.append(f"| {n} | {x['symbol']} {x['name']} | {x['total']}/20 | {b.get('strategy')} | "
                     f"{b.get('entry')} | {b.get('stop')} | {b.get('target')} | {b.get('rr')} | {x['eligible']} |")
    lines += ["", f"Errors: {len(errors)}"] + [f"- {e}" for e in errors[:20]]
    with open(os.path.join(OUT, "latest.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    log(f"done: weather={weather} counts={result['counts']}")
    return result


# ------------------------------------------------------------ selftest ---
def _fake_download(symbols, period="2y"):
    rng = np.random.default_rng(7)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize() - pd.Timedelta(days=1), periods=500)
    out = {}
    for k, s in enumerate(symbols):
        base = {"^VIX": 16.0, "EURUSD=X": 1.08, "EURGBP=X": 0.85, "EURCHF=X": 0.94,
                "EURDKK=X": 7.46}.get(s, 50 + k)
        drift = 0.0 if s.startswith(("^", "EUR")) else rng.normal(0.0006, 0.0006)
        r = rng.normal(drift, 0.015 if not s.startswith("EUR") else 0.003, len(idx))
        close = base * np.exp(np.cumsum(r))
        if s.startswith("EUR") or s == "^VIX":
            close = base * (1 + r)
        if k % 7 == 0 and not s.startswith(("^", "EUR")):          # make some pullbacks
            close[-6:] = close[-7] * np.array([0.99, 0.98, 0.975, 0.97, 0.965, 0.968])
        if k % 11 == 0 and not s.startswith(("^", "EUR")):         # and a few earnings gaps
            close[-4:] = close[-5] * np.array([1.08, 1.09, 1.095, 1.10])
        high = close * (1 + abs(rng.normal(0, 0.008, len(idx))))
        low = close * (1 - abs(rng.normal(0, 0.008, len(idx))))
        opn = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.003, len(idx)))
        if k % 11 == 0 and not s.startswith(("^", "EUR")):
            opn[-4] = close[-5] * 1.07
            low[-4] = min(low[-4], opn[-4] * 0.99)
        vol = rng.integers(2_000_000, 6_000_000, len(idx)).astype(float)
        if k % 11 == 0:
            vol[-4] *= 4
        out[s] = pd.DataFrame({"Open": opn, "High": np.maximum.reduce([high, opn, close]),
                               "Low": np.minimum.reduce([low, opn, close]), "Close": close,
                               "Volume": vol}, index=idx)
    return out, []


def _fake_fundamentals(sym, today, fx_eur, close, currency):
    rng = np.random.default_rng(abs(hash(sym)) % 2**32)
    nxt = today + dt.timedelta(days=int(rng.integers(3, 60)))
    return {"longName": sym, "sector": "Technology", "industry": "Software",
            "profitMargins": float(rng.normal(0.15, 0.1)), "operatingMargins": float(rng.normal(0.2, 0.1)),
            "revenueGrowth": float(rng.normal(0.1, 0.1)), "freeCashflow": float(rng.normal(5e9, 3e9)),
            "fcf_yield": float(rng.normal(0.03, 0.02)), "net_debt_to_ebitda": float(rng.normal(1.5, 1)),
            "net_cash": bool(rng.random() < 0.3), "trailingPegRatio": float(rng.normal(2, 0.8)),
            "forwardPE": 25.0, "eps_up_30d": int(rng.integers(0, 6)), "eps_down_30d": int(rng.integers(0, 6)),
            "next_earnings": str(nxt), "trading_days_to_earnings": int(np.busday_count(today, nxt))}


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        OUT = os.path.join(HERE, "selftest_out")
        res = run(download_fn=_fake_download, fundamentals_fn=_fake_fundamentals)
        print(json.dumps(res["counts"]), res["market"]["weather"])
        for x in res["shortlist"][:5]:
            print(x["symbol"], x["total"], x["eligible"], (x["best_setup"] or {}).get("strategy"),
                  (x["best_setup"] or {}).get("rr"), x["sizing_green_20eur_risk"])
    else:
        run()
