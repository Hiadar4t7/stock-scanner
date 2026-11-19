#!/usr/bin/env python3
"""Print a compact, readable view of data/latest.json for the Claude daily scan.

Usage:
  python3 view.py latest.json            # market + all scored stocks (eligible first)
  python3 view.py latest.json TTWO       # full details for one scored stock
"""
import json
import sys


def pct(x, n=1):
    return "n/a" if x is None else f"{x * 100:+.{n}f}%"


def main():
    d = json.load(open(sys.argv[1]))
    want = sys.argv[2].upper() if len(sys.argv) > 2 else None
    if want:
        for x in d["shortlist"]:
            if x["symbol"].upper() == want:
                print(json.dumps(x, indent=1))
                return
        print(f"{want} is not in today's scored shortlist (see latest_all.csv for its prices).")
        return
    m = d["market"]
    print(f"DATA: generated {d['generated_at_utc']} UTC | last US close {d['data_date_us']}")
    print(f"MARKET: weather={m['weather']} | SPY {m['spy']} | VIX {m['vix']} | DAX {m['dax']}")
    print(f"FX (EUR per unit): {d['fx_eur_per_unit']}")
    print(f"COUNTS: {d['counts']}")
    if d.get("errors"):
        print(f"ERRORS ({len(d['errors'])}): {d['errors'][:10]}")
    print()
    for n, x in enumerate(d["shortlist"], 1):
        b, s, f, i = x["best_setup"] or {}, x["sizing_green_20eur_risk"] or {}, x["fundamentals"] or {}, x["indicators"]
        rows = " ".join(f"{k.split('_')[0]}={v}" for k, v in (x["rows"] or {}).items())
        print(f"#{n} {x['symbol']} ({x['name']}, {x['currency']}) SCORE {x['total']}/20 "
              f"eligible={x['eligible']} dealbreakers_ok={x['dealbreakers_ok']}")
        print(f"   rows: {rows}")
        if x.get("notes"):
            print(f"   notes: {'; '.join(x['notes'])}")
        print(f"   setup S{b.get('strategy')}: entry {b.get('entry')} ({b.get('entry_order')}) | stop {b.get('stop')} | "
              f"target {b.get('target')} | R:R {b.get('rr')} | hold {b.get('hold_days')}d, time stop {b.get('time_stop_days')}d")
        if b.get("must_verify"):
            print(f"   MUST VERIFY: {b['must_verify']} (gap day {b.get('gap_day')})")
        if b.get("fill_rule"):
            print(f"   fill rule: {b['fill_rule']}")
        print(f"   EUR: entry {s.get('entry_eur')} stop {s.get('stop_eur')} ({s.get('stop_pct')}%) target "
              f"{s.get('target_eur')} ({s.get('target_pct')}%) | shares {s.get('shares')} position {s.get('position_eur')} | "
              f"at stop {s.get('loss_at_stop_eur')} | at target {s.get('gain_at_target_eur')} | "
              f"break-even win rate {s.get('breakeven_win_rate_pct')}% | tradeable={s.get('tradeable')} {s.get('reason', '')}")
        print(f"   tech: close {i['close']} ({i['date']}) sma20 {i['sma20']} sma50 {i['sma50']} sma200 {i['sma200']} "
              f"rsi14 {i['rsi14']} atr {pct(i['atr_pct'])} 3m {pct(i['perf_3m'])} vs SPY {pct(i['rs_3m'])} "
              f"52w-high {pct(i['dist_52w_high'])}")
        print(f"   fund: earnings {f.get('next_earnings')} ({f.get('trading_days_to_earnings')} trading days) | "
              f"rev growth {pct(f.get('revenueGrowth'))} | op margin {pct(f.get('operatingMargins'))} | "
              f"FCF yield {pct(f.get('fcf_yield'))} | net debt/EBITDA {f.get('net_debt_to_ebitda')} | "
              f"PEG {f.get('trailingPegRatio') or f.get('pegRatio')} | fwd P/E {f.get('forwardPE')} | "
              f"EPS revisions 30d up {f.get('eps_up_30d')} / down {f.get('eps_down_30d')}")
        if i.get("event_gaps"):
            print(f"   event gaps: {i['event_gaps']}")
        print()
    print("STRATEGY 6 momentum top 5:", d.get("momentum_top5_strategy6"))


if __name__ == "__main__":
    main()
