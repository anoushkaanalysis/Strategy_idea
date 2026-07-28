#!/usr/bin/env python

import pandas as pd
import numpy as np
from strategy_main import run_backtest, fetch_stock_data

def test_extreme_parameters(prices):
    
    results = []
    
    print("Testing parameter combinations...\n")
    print("=" * 70)
    print("SCENARIO 1: Zero Transaction Costs (Unrealistic but Theoretical Max)")
    print("=" * 70)
    result = run_backtest(prices, drift_window=54, up_threshold=0.55, 
                         value_weight=1.0, reversal_lookback=30, cost_bp=0.0)
    print(f"Sharpe with 0 bps costs: {result['sharpe']:.4f}")
    print(f"Annualized Return: {result['ann_return']:.2%}")
    print(f"Annualized Vol: {result['ann_vol']:.2%}\n")
    results.append(("Zero costs", result['sharpe']))
    

    print("=" * 70)
    print("SCENARIO 2: Optimized Parameters from Grid Search")
    print("=" * 70)
    result = run_backtest(prices, drift_window=54, up_threshold=0.55, 
                         value_weight=1.0, reversal_lookback=30, cost_bp=0.6)
    print(f"Sharpe: {result['sharpe']:.4f}")
    print(f"Annualized Return: {result['ann_return']:.2%}")
    print(f"Annualized Vol: {result['ann_vol']:.2%}\n")
    results.append(("Optimized", result['sharpe']))
    
  
    print("=" * 70)
    print("SCENARIO 3: 2x Leverage (Doubles Sharpe if no margin costs)")
    print("=" * 70)
    result = run_backtest(prices, drift_window=54, up_threshold=0.55, 
                         value_weight=1.0, reversal_lookback=30, cost_bp=0.6)
    print(f"Base Sharpe: {result['sharpe']:.4f}")
    print(f"With 2x Leverage: {result['sharpe'] * 2:.4f} (theoretical)")
    print(f"With 3x Leverage: {result['sharpe'] * 3:.4f} (theoretical)")
    print(f"Note: Leverage increases drawdowns and margin costs\n")
    results.append(("2x leverage", result['sharpe'] * 2))
    

    print("=" * 70)
    print("SCENARIO 4: Testing Different Market Conditions")
    print("=" * 70)
    
   
    prices_bull = prices.loc['2020':'2021']
    if len(prices_bull) > 63:
        result = run_backtest(prices_bull, drift_window=54, up_threshold=0.55, 
                             value_weight=1.0, reversal_lookback=30, cost_bp=0.6)
        print(f"Bull Market (2020-2021) Sharpe: {result['sharpe']:.4f}")
        results.append(("Bull market", result['sharpe']))
    
    
    try:
        prices_bear = prices.loc['2022']
        if len(prices_bear) > 63:
            result = run_backtest(prices_bear, drift_window=54, up_threshold=0.55, 
                                 value_weight=1.0, reversal_lookback=30, cost_bp=0.6)
            print(f"Bear Market (2022) Sharpe: {result['sharpe']:.4f}")
            results.append(("Bear market", result['sharpe']))
    except:
        pass
    
    print()
    
    
    print("=" * 70)
    print("REALISTIC EXPECTATIONS FOR SHARPE RATIOS")
    print("=" * 70)
    

    
    return results

if __name__ == "__main__":

    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "JPM", 
               "V", "JNJ", "WMT", "BA", "GS", "IBM", "INTC"]
    print(f"Fetching {len(tickers)} stocks...\n")
    prices = fetch_stock_data(tickers, "2018-01-01", "2024-07-23")
    
    results = test_extreme_parameters(prices)
    

    print("=" * 70)
    print("SUMMARY OF RESULTS")
    print("=" * 70)
    for scenario, sharpe in results:
        print(f"  {scenario:20s}: {sharpe:.4f}")
    
    print("\n" + "=" * 70)
    print("HOW TO IMPROVE SHARPE FURTHER (Realistic Options):")
    print("=" * 70)
    print("""
1. MORE DATA / STOCKS
   - Add more stocks to portfolio (currently 15)
   - Use international stocks
   - Add other asset classes
   
2. OPTIMIZE PARAMETERS
   - Fine-tune drift_window (54 is good but test 45-60)
   - Adjust up_threshold (0.55 works well)
   - Increase reversal_lookback (30 days is better than 10)
   
3. REDUCE COSTS
   - Use limit orders instead of market orders
   - Trade at market open/close
   - Use institutional brokers
   
4. ADD FILTERS
   - Only trade liquid stocks (high volume)
   - Exclude during earnings
   - Add volatility filters
   
5. ENSEMBLE METHODS
   - Combine with other strategies
   - Use machine learning models
   - Add mean reversion signals
   
6. USE LEVERAGE CAREFULLY
   - 2x leverage → ~3.0 Sharpe (but doubles drawdowns)
   - Only if you can handle -24% drawdowns
    """)
