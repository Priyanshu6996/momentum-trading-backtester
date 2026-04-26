# NSE Long-Only Momentum Strategy

A Python backtesting script that runs a 12-1 momentum strategy on a universe of NSE-listed stocks and benchmarks it against the Nifty 50.

---

## What it does

- Downloads historical price data for ~150 NSE stocks using `yfinance`
- Every month, ranks stocks by their 12-month momentum (skipping the most recent month)
- Goes long the top 20% of ranked stocks, weighted by inverse volatility
- Applies a transaction cost of 0.1% per leg on each rebalance
- Compares the strategy's performance against the Nifty 50 index

---

## Requirements

```
yfinance
pandas
numpy
matplotlib
```

Install with:

```bash
pip install yfinance pandas numpy matplotlib
```

---

## How to run

```bash
python momentum_strategy.py
```

That's it. The script will download data from 2018 to today, run the backtest, print the results, and save a chart.

---

## Output

**Console** — prints performance metrics:
- Final cumulative return
- Annualised return
- Sharpe ratio
- Max drawdown
- Nifty 50 annualised return
- Excess return (alpha)

**Chart** — saved as `strategy_performance.png` in the same directory:
- Equity curve vs Nifty 50
- Daily returns bar chart
- Drawdown chart

---

## Parameters

All key parameters are at the top of the file and easy to adjust:

| Parameter | Default | Description |
|---|---|---|
| `LOOKBACK` | 252 | Momentum lookback window (trading days) |
| `SKIP` | 21 | Days to skip at the end of lookback (12-1 momentum) |
| `HOLDING` | 21 | Rebalance frequency (trading days) |
| `TOP_PCT` | 0.2 | Fraction of top stocks to go long |
| `COST_PER_TRADE` | 0.001 | Cost per leg (0.1%) |
| `START_DATE` | 2018-01-01 | Backtest start date |

---

## Notes

- Data is fetched live each run — no local data files needed
- The script only goes long; there is no short leg
- Weights are normalised inverse-volatility, so lower-volatility stocks get higher allocation
- Transaction costs scale with actual turnover, not total portfolio size
