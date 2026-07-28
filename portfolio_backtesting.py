#!/usr/bin/env python
"""
Example: Run the strategy on multiple stock tickers (recommended)
"""
import pandas as pd
from strategy_main import run_backtest, fetch_stock_data

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "JPM", "V", "JNJ"]
    print(f"Backtesting strategy on {len(tickers)} stocks...")
    print(f"Tickers: {', '.join(tickers)}\n")
    
    prices = fetch_stock_data(tickers, "2020-01-01", "2024-07-23")
    print(f"Downloaded {len(prices)} days of data for {len(prices.columns)} stocks\n")
    
    result = run_backtest(prices)
    print(f"Backtest Results:")
    print(f"  Annualized Return: {result['ann_return']:.2%}")
    print(f"  Annualized Volatility: {result['ann_vol']:.2%}")
    print(f"  Sharpe Ratio: {result['sharpe']:.4f}")
    print(f"  Max Drawdown: {result['max_dd']:.2%}")
    print(f"  Scale Factor: {result['scale_factor']:.4f}")
    print(f"  % Active: {result['pct_active']:.2%}")
