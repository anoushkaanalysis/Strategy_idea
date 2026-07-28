#!/usr/bin/env python
"""
Example: Run the strategy on a single stock ticker
"""
import pandas as pd
import yfinance as yf
from strategy_main import run_backtest, fetch_stock_data

if __name__ == "__main__":
    ticker = "AAPL"
    print(f"Backtesting strategy on {ticker}...")
    
    prices = fetch_stock_data(ticker, "2020-01-01", "2024-07-23")
    print(f"Downloaded {len(prices)} days of data")
    
    result = run_backtest(prices)
    print(f"\nBacktest Results for {ticker}:")
    print(f"  Annualized Return: {result['ann_return']:.2%}")
    print(f"  Annualized Volatility: {result['ann_vol']:.2%}")
    print(f"  Sharpe Ratio: {result['sharpe']:.4f}")
    print(f"  Max Drawdown: {result['max_dd']:.2%}")
    print(f"  Scale Factor: {result['scale_factor']:.4f}")
    print(f"  % Active: {result['pct_active']:.2%}")
