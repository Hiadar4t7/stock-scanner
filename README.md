# stock-scanner

The free "data robot" behind Haidar's daily stock scan.

Every weekday at about 09:35 Berlin time (10:35 in summer), GitHub Actions runs `scan.py`. It:

1. downloads daily prices for every stock in `watchlist.csv`, plus SPY, VIX, DAX and FX rates (Yahoo Finance via `yfinance`)
2. works out moving averages, RSI, ATR, momentum, volume and price ranges from **completed** daily bars
3. checks each stock for the playbook's setups (1 pullback, 2 breakout, 3 post-earnings drift, 4 oversold bounce) with entry, stop, target and reward-to-risk
4. scores the best 25 setups with the automatic 10-row scorecard (max 20 points)
5. saves `data/latest.json`, `data/latest_all.csv`, `data/latest.md` and a dated copy in `data/archive/`

At about 11:15 Berlin time, a Claude scheduled task reads `data/latest.json`, checks the news and open trades, and emails the top 3–5 picks.

- Change the stocks: edit `watchlist.csv` (Yahoo symbols; currency USD, EUR, CHF or GBp).
- Change the rules: the settings at the top of `scan.py` (risk 20 €, pot 500 €, min score 14).
- Run it by hand: Actions tab → "Daily data robot" → Run workflow.
- Test without internet: `python scan.py --selftest`.

Rule-based and for learning. Not financial advice.
