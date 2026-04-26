import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

today = datetime.date.today().strftime("%Y-%m-%d")

# --- Universe ---
symbolstop = [
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","ICICIBANK.NS","INFY.NS",
    "ITC.NS","HINDUNILVR.NS","SBIN.NS","BHARTIARTL.NS","KOTAKBANK.NS",
    "LT.NS","AXISBANK.NS","ASIANPAINT.NS","MARUTI.NS","SUNPHARMA.NS",
    "TITAN.NS","ULTRACEMCO.NS","NESTLEIND.NS","BAJFINANCE.NS","BAJAJFINSV.NS",
    "HCLTECH.NS","WIPRO.NS","ONGC.NS","NTPC.NS","POWERGRID.NS",
    "M&M.NS","TATASTEEL.NS","JSWSTEEL.NS","TECHM.NS","INDUSINDBK.NS",
    "ADANIENT.NS","ADANIPORTS.NS","GRASIM.NS","COALINDIA.NS","DRREDDY.NS",
    "CIPLA.NS","DIVISLAB.NS","EICHERMOT.NS","BRITANNIA.NS","HEROMOTOCO.NS",
    "BAJAJ-AUTO.NS","UPL.NS","SBILIFE.NS","HDFCLIFE.NS","ICICIPRULI.NS",
    "DABUR.NS","GODREJCP.NS","PIDILITIND.NS","BERGEPAINT.NS","HAVELLS.NS",
    "SIEMENS.NS","ABB.NS","BOSCHLTD.NS","CUMMINSIND.NS","THERMAX.NS",
    "ESCORTS.NS","ASHOKLEY.NS","TVSMOTOR.NS","MOTHERSON.NS","BALKRISIND.NS",
    "APOLLOHOSP.NS","MAXHEALTH.NS","FORTIS.NS","METROPOLIS.NS","LALPATHLAB.NS",
    "BIOCON.NS","TORNTPHARM.NS","AUROPHARMA.NS","ALKEM.NS","GLENMARK.NS",
    "INDIGO.NS","IRCTC.NS","ZOMATO.NS","NYKAA.NS","PAYTM.NS",
    "TRENT.NS","DMART.NS","VBL.NS","TATACONSUM.NS","UBL.NS",
    "COLPAL.NS","MARICO.NS","EMAMILTD.NS","PAGEIND.NS","RELAXO.NS",
    "CHOLAFIN.NS","SHRIRAMFIN.NS","BAJAJHLDNG.NS","MUTHOOTFIN.NS","LICHSGFIN.NS",
    "PNB.NS","BANKBARODA.NS","CANBK.NS","UNIONBANK.NS","IDFCFIRSTB.NS",
    "FEDERALBNK.NS","RBLBANK.NS","YESBANK.NS","IDBI.NS","BANDHANBNK.NS",
    "INDHOTEL.NS","OBEROIRLTY.NS","PHOENIXLTD.NS","DLF.NS","GODREJPROP.NS",
    "LODHA.NS","PRESTIGE.NS","SOBHA.NS","BRIGADE.NS","SUNTECK.NS",
    "TATAMOTORS.NS","MRF.NS","EXIDEIND.NS","AMARAJABAT.NS","TIINDIA.NS",
    "SUPRAJIT.NS","SKFINDIA.NS","ENDURANCE.NS","RAMCOCEM.NS","DALBHARAT.NS",
    "JKCEMENT.NS","ACC.NS","AMBUJACEM.NS","SHREECEM.NS",
    "PETRONET.NS","IGL.NS","MGL.NS","ATGL.NS","GSPL.NS",
    "GAIL.NS","OIL.NS","HINDPETRO.NS","BPCL.NS","IOC.NS",
    "CONCOR.NS","DELHIVERY.NS","BLUEDART.NS","TCI.NS","VRLLOG.NS",
    "POLYCAB.NS","KEI.NS","FINCABLES.NS","RRKABEL.NS","VGUARD.NS",
    "TATAELXSI.NS","LTTS.NS","MPHASIS.NS","PERSISTENT.NS","COFORGE.NS",
    "KPITTECH.NS","ZENSARTECH.NS","SONATSOFTW.NS","OFSS.NS","BIRLASOFT.NS",
    "DEEPAKNTR.NS","AARTIIND.NS","SRF.NS","NAVINFLUOR.NS","PIIND.NS",
    "COROMANDEL.NS","CHAMBLFERT.NS","GNFC.NS","RCF.NS","FACT.NS",
    "HAL.NS","BEL.NS","BDL.NS","MAZDOCK.NS","COCHINSHIP.NS",
    "IRFC.NS","RVNL.NS","IRCON.NS","NBCC.NS","HUDCO.NS",
]

# --- Params ---
lookback          = 252
skip              = 21      # skip last month (standard 12-1 momentum)
holding           = 21
top_pct           = 0.2
cost_per_trade    = 0.001   # 0.1% per leg (buy + sell = 0.2% round trip per rebalance)

# --- Data ---
prices_raw = yf.download(symbolstop, start="2018-01-01", end=today)["Close"]
market_raw = yf.download("^NSEI",    start="2018-01-01", end=today)["Close"].squeeze()

stock_ret_raw  = prices_raw.pct_change()
market_ret_raw = market_raw.pct_change()

# --- Align ---
data = stock_ret_raw.join(market_ret_raw.rename("MKT"), how="inner")
data = data[data["MKT"].notna()]

dates      = data.index
stock_ret  = data[symbolstop]
market_ret = data["MKT"]
prices     = prices_raw.reindex(dates)

T, N = stock_ret.shape

print(f"Data shape : {data.shape}")
print(f"Date range : {dates[0].date()} -> {dates[-1].date()}")
print(f"T={T}, lookback={lookback}, holding={holding}")
print(f"Expected rebalances: {len(range(lookback, T - holding, holding))}\n")

portfolio_returns = pd.Series(index=dates, dtype=float)
prev_weights      = pd.Series(0.0, index=stock_ret.columns)  # track previous weights for turnover cost

# --- Rolling Backtest ---
for t in range(lookback, T - holding, holding):
    print(f"Rebalancing at t={t} ({dates[t].date()})")

    # 12-1 momentum: return from [t-252] to [t-21]
    try:
        ret_12m = (prices.iloc[t - skip] / prices.iloc[t - lookback]) - 1
    except Exception:
        continue

    ret_12m = ret_12m.dropna()
    if len(ret_12m) < 10:
        print("  Skipping - too few stocks")
        continue

    # --- Rank: long top 20% by momentum ---
    ranking     = ret_12m.sort_values(ascending=False)
    k           = max(1, int(top_pct * len(ranking)))
    long_stocks = ranking.head(k).index

    print(f"  k={k} | Long: {list(long_stocks[:5])} ...")

    # --- Volatility-adjusted weights, normalised to sum = 1 ---
    vol = stock_ret.iloc[t - lookback:t].std().replace(0, np.nan)
    weights = pd.Series(0.0, index=stock_ret.columns)
    weights[long_stocks] = 1.0 / vol[long_stocks]
    weights = weights / weights.sum()

    # --- Transaction cost: proportional to turnover vs previous weights ---
    # Turnover = sum of absolute weight changes; cost applied on rebalance day
    turnover = (weights - prev_weights).abs().sum()
    cost_this_rebalance = cost_per_trade * turnover   # scales with how much you actually trade
    prev_weights = weights.copy()

    # --- Apply weights + deduct cost on first day of holding period ---
    for h in range(holding):
        idx = t + h
        if idx < T:
            day_ret = stock_ret.iloc[idx].fillna(0)
            port_ret = (weights * day_ret).sum()
            if h == 0:
                port_ret -= cost_this_rebalance  # deduct cost on rebalance day only
            portfolio_returns.iloc[idx] = port_ret

# --- Results ---
portfolio_returns = portfolio_returns.dropna()

if len(portfolio_returns) == 0:
    print("\nNo trades executed")
else:
    cum        = (1 + portfolio_returns).cumprod()
    annual_ret = portfolio_returns.mean() * 252
    sharpe     = (portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(252)
    peak       = cum.cummax()
    drawdown   = (cum - peak) / peak
    max_dd     = drawdown.min()

    nifty_ret_aligned = nifty_returns = market_ret.reindex(portfolio_returns.index).fillna(0)
    nifty_cum  = (1 + nifty_ret_aligned).cumprod()
    nifty_ann  = nifty_ret_aligned.mean() * 252

    print("\n========== STRATEGY PERFORMANCE ==========")
    print(f"Final Cumulative Return : {cum.iloc[-1] - 1:.2%}")
    print(f"Annualised Return       : {annual_ret:.2%}")
    print(f"Sharpe Ratio            : {sharpe:.2f}")
    print(f"Max Drawdown            : {max_dd:.2%}")
    print(f"\n========== NIFTY BENCHMARK ===============")
    print(f"Nifty Annualised Return : {nifty_ann:.2%}")
    print(f"Excess Return (Alpha)   : {annual_ret - nifty_ann:.2%}")

    # --- Plot ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Long-Only Momentum Strategy vs Nifty (NSE)", fontsize=14, fontweight="bold")

    axes[0].plot(cum.index, cum.values, color="steelblue", linewidth=1.5, label="Strategy")
    axes[0].plot(nifty_cum.index, nifty_cum.values, color="orange", linewidth=1.2,
                 linestyle="--", label="Nifty 50")
    axes[0].axhline(1, color="gray", linestyle=":", linewidth=0.8)
    axes[0].set_ylabel("Cumulative Return")
    axes[0].set_title("Equity Curve")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    colors = ["green" if r > 0 else "red" for r in portfolio_returns.values]
    axes[1].bar(portfolio_returns.index, portfolio_returns.values,
                color=colors, width=1, alpha=0.7)
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("Daily Return")
    axes[1].set_title("Daily Returns")
    axes[1].grid(True, alpha=0.3)

    axes[2].fill_between(drawdown.index, drawdown.values, 0, color="crimson", alpha=0.5)
    axes[2].set_ylabel("Drawdown")
    axes[2].set_title("Drawdown")
    axes[2].grid(True, alpha=0.3)
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[2].xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("strategy_performance.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nChart saved to strategy_performance.png")