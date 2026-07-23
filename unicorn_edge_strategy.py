"""
Unicorn Edge strategy
======================
Implementation of the value + short-term-reversal factor, gated by a
stock-specific "drift regime" filter, as described in:

    "Discovery of a 13-Sharpe OOS Factor: Drift Regimes Unlock Hidden
     Cross-Sectional Predictability" (Singha, 2025, arXiv:2511.12490)

This file gives you the actual mechanics (signal construction, regime gate,
market-neutral portfolio construction, vol/drawdown scaling, kill-switch,
and a backtester) so you can test the claim on real data yourself.

IMPORTANT — read before trusting the output
--------------------------------------------
The paper's headline numbers (OOS Sharpe > 13, ~159% annualized return,
Sharpe rising as you shrink the universe/AUM) are extreme relative to
anything documented in the academic or practitioner factor literature
(typical well-known equity factors run Sharpe ~0.3-0.8 unlevered). Numbers
like this on a widely-studied universe (S&P 500) with widely-known signals
(price-inverse "value", 10-day reversal) are the textbook signature of:
  - a parameter search over (drift window, threshold, weights) that wasn't
    truly frozen out-of-sample,
  - survivorship bias from using *current* constituents over 20 years
    (the paper admits this and estimates it inflates results 20-30%, which
    on its own does not explain a move from ~1 to ~13 Sharpe),
  - only 3 non-overlapping 1-year OOS test windows, each coinciding with
    unusually strong bull-market recoveries (2010-11, 2015-16, 2020-21) —
    that's a very small, cherry-timed OOS sample to hang a "13 Sharpe"
    headline on.
Use this code to check the mechanism on your own out-of-sample data and
form your own view — don't take the reported Sharpe at face value.

Data format expected
--------------------
A pandas DataFrame `prices` of daily adjusted close prices:
    index = trading dates (DatetimeIndex)
    columns = tickers
"""

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# 1. Signal construction
# ----------------------------------------------------------------------

def compute_value_signal(prices: pd.DataFrame) -> pd.DataFrame:
    """Price-based 'value': inverse price, converted to cross-sectional
    percentile rank in [0, 1] each day (Eq. in paper: no accounting data)."""
    inv_price = 1.0 / prices
    # cross-sectional percentile rank, per day
    value_rank = inv_price.rank(axis=1, pct=True)
    return value_rank


def compute_reversal_signal(prices: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
    """Short-term reversal: negative of trailing `lookback`-day return,
    standardized to a cross-sectional z-score each day."""
    trailing_ret = prices.pct_change(lookback)
    reversal_raw = -trailing_ret
    mu = reversal_raw.mean(axis=1)
    sd = reversal_raw.std(axis=1)
    reversal_z = reversal_raw.sub(mu, axis=0).div(sd, axis=0)
    return reversal_z


def compute_base_factor(prices: pd.DataFrame, value_weight: float = 0.7,
                         reversal_lookback: int = 10) -> pd.DataFrame:
    """BASE = value_weight * value + (1 - value_weight) * reversal"""
    value = compute_value_signal(prices)
    reversal = compute_reversal_signal(prices, reversal_lookback)
    base = value_weight * value + (1 - value_weight) * reversal
    return base


def compute_drift_regime(prices: pd.DataFrame, window: int = 63,
                          up_threshold: float = 0.60) -> pd.DataFrame:
    """REGIME[i,t] = 1 if fraction of positive daily returns for stock i
    over the trailing `window` days exceeds `up_threshold`, else 0."""
    daily_ret = prices.pct_change()
    is_up = (daily_ret > 0).astype(float)
    up_fraction = is_up.rolling(window, min_periods=window).mean()
    regime = (up_fraction > up_threshold).astype(float)
    return regime


def compute_edge_signal(prices: pd.DataFrame, drift_window: int = 63,
                         up_threshold: float = 0.60, value_weight: float = 0.7,
                         reversal_lookback: int = 10) -> pd.DataFrame:
    """EDGE = BASE * REGIME"""
    base = compute_base_factor(prices, value_weight, reversal_lookback)
    regime = compute_drift_regime(prices, drift_window, up_threshold)
    edge = base * regime
    # cells where regime == 0 should be NaN (not just zero) so they don't
    # enter the cross-sectional z-score / portfolio construction that day
    edge = edge.where(regime == 1)
    return edge


# ----------------------------------------------------------------------
# 2. Portfolio construction
# ----------------------------------------------------------------------

def edge_to_weights(edge: pd.DataFrame, gross_long: float = 0.5,
                     gross_short: float = 0.5) -> pd.DataFrame:
    """Convert daily EDGE scores into a market-neutral long/short weight
    matrix: z-score the active names each day, split into long/short
    buckets by sign, normalize each side to the target gross exposure."""
    weights = pd.DataFrame(0.0, index=edge.index, columns=edge.columns)

    for date, row in edge.iterrows():
        active = row.dropna()
        if active.shape[0] < 10:
            continue  # not enough names active that day
        z = (active - active.mean()) / active.std()

        longs = z[z > 0]
        shorts = z[z < 0]

        if len(longs) > 0 and longs.sum() != 0:
            w_long = (longs / longs.sum()) * gross_long
            weights.loc[date, w_long.index] = w_long.values
        if len(shorts) > 0 and shorts.sum() != 0:
            w_short = (shorts / shorts.sum()) * gross_short  # shorts are negative z
            weights.loc[date, w_short.index] = w_short.values

    return weights


# ----------------------------------------------------------------------
# 3. Backtest: returns, vol/DD scaling, kill-switch
# ----------------------------------------------------------------------

def backtest_returns(prices: pd.DataFrame, weights: pd.DataFrame,
                      cost_bp: float = 0.6) -> pd.Series:
    """Apply weights (decided using info available *through* day t-1) to
    day-t returns, net of simple turnover-based transaction costs."""
    fwd_ret = prices.pct_change().shift(-1)  # next-day return
    weights_lagged = weights  # weights already computed from info at t; realized next day
    gross_pnl = (weights_lagged * fwd_ret).sum(axis=1)

    turnover = weights_lagged.diff().abs().sum(axis=1).fillna(0)
    costs = turnover * (cost_bp / 10000.0)

    net_ret = gross_pnl - costs
    net_ret.index = prices.index
    return net_ret.shift(1).fillna(0)  # align so return[t] is realized ON day t


def scale_and_kill_switch(returns: pd.Series, train_returns: pd.Series,
                           vol_cap: float = 0.12, dd_cap: float = 0.15,
                           dd_kill: float = 0.30, roll_kill: float = -0.10,
                           roll_window: int = 63) -> pd.Series:
    """Apply a static scale factor fit on `train_returns`, then a dynamic
    kill-switch on the (scaled) `returns` series."""
    train_vol = train_returns.std() * np.sqrt(252)
    cum = (1 + train_returns).cumprod()
    train_dd = (cum / cum.cummax() - 1).min()

    scale = min(vol_cap / train_vol if train_vol > 0 else 1.0,
                dd_cap / abs(train_dd) if train_dd != 0 else 1.0)

    scaled = returns * scale

    cum_test = (1 + scaled).cumprod()
    dd_test = cum_test / cum_test.cummax() - 1
    roll_test = scaled.rolling(roll_window).sum()

    killed = (dd_test < -dd_kill) | (roll_test < roll_kill)
    # once killed, stays off for the rest of the test period (no re-entry)
    off = killed.cummax()
    scaled_after_kill = scaled.where(~off, 0.0)

    return scaled_after_kill, scale


def performance_summary(returns: pd.Series) -> dict:
    ann_ret = (1 + returns).prod() ** (252 / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    cum = (1 + returns).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()
    return {"ann_return": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe, "max_dd": max_dd}


# ----------------------------------------------------------------------
# 4. End-to-end convenience wrapper
# ----------------------------------------------------------------------

def run_backtest(prices: pd.DataFrame, drift_window: int = 63, up_threshold: float = 0.60,
                  value_weight: float = 0.7, reversal_lookback: int = 10,
                  cost_bp: float = 0.6, train_frac: float = 0.5) -> dict:
    """Run the full pipeline and return a performance dict. The first
    `train_frac` of the sample is used only to fit the vol/DD scale factor
    (mirroring the paper's train/test split), the rest is scored OOS."""
    edge = compute_edge_signal(prices, drift_window, up_threshold, value_weight, reversal_lookback)
    weights = edge_to_weights(edge)
    raw_returns = backtest_returns(prices, weights, cost_bp)

    split = int(len(raw_returns) * train_frac)
    train_returns = raw_returns.iloc[:split]
    test_returns = raw_returns.iloc[split:]

    scaled_test, scale = scale_and_kill_switch(test_returns, train_returns)
    perf = performance_summary(scaled_test)
    perf["scale_factor"] = scale
    perf["pct_active"] = edge.notna().mean(axis=1).mean()
    return perf


if __name__ == "__main__":
    # Minimal synthetic demo so the script runs end-to-end without network
    # access. Replace `prices` with real adjusted-close data (e.g. from
    # yfinance) to actually test the strategy.
    np.random.seed(0)
    n_days, n_stocks = 1500, 200
    dates = pd.bdate_range("2018-01-01", periods=n_days)
    rets = np.random.normal(0.0003, 0.018, size=(n_days, n_stocks))
    prices = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=dates,
                           columns=[f"STK{i:03d}" for i in range(n_stocks)])

    result = run_backtest(prices)
    print("Demo backtest on SYNTHETIC random-walk data (no real edge expected):")
    for k, v in result.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
