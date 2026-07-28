#!/usr/bin/env python

import pandas as pd
from strategy_main import run_backtest, fetch_stock_data

if __name__ == "__main__":
   
    tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "JPM", "V", "JNJ",
        "WMT", "BA", "GS", "IBM", "INTC", "AMD", "PG", "KO", "MCD", "NFLX",
        "DIS", "PYPL", "ADBE", "CSCO", "QCOM"
    ]
    
    print(f"Fetching {len(tickers)} stocks...\n")
    prices = fetch_stock_data(tickers, "2018-01-01", "2024-07-23")
    
    print("=" * 70)
    print("LEVERAGE OPTIMIZATION ANALYSIS")
    print("=" * 70)
    print("\nTesting different leverage levels...\n")
    
    results = []
    
    for leverage in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        result = run_backtest(
            prices,
            drift_window=54,
            up_threshold=0.55,
            value_weight=1.0,
            reversal_lookback=30,
            cost_bp=0.6,
            leverage=leverage
        )
        
        results.append({
            'leverage': leverage,
            'sharpe': result['sharpe'],
            'ann_return': result['ann_return'],
            'ann_vol': result['ann_vol'],
            'max_dd': result['max_dd']
        })
        
        print(f"Leverage {leverage:.1f}x:")
        print(f"  Return:    {result['ann_return']:7.2%}")
        print(f"  Volatility: {result['ann_vol']:7.2%}")
        print(f"  Sharpe:    {result['sharpe']:7.4f}")
        print(f"  Max DD:    {result['max_dd']:7.2%}")
        print()
    
    
    best = max(results, key=lambda x: x['sharpe'])
    
    print("=" * 70)
    print(f"OPTIMAL LEVERAGE: {best['leverage']:.1f}x")
    print("=" * 70)
    print(f"  Sharpe:    {best['sharpe']:.4f}")
    print(f"  Return:    {best['ann_return']:.2%}")
    print(f"  Volatility: {best['ann_vol']:.2%}")
    print(f"  Max DD:    {best['max_dd']:.2%}")
    
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    print(f"""
Best Strategy:
  - Use 1.0x leverage (no leverage) for stable Sharpe {results[1]['sharpe']:.4f}
  - Use 1.5x leverage for better returns ({results[2]['ann_return']:.2%}) with Sharpe {results[2]['sharpe']:.4f}
  - Avoid 3.0x (too much drawdown risk: {results[5]['max_dd']:.2%})

Practical considerations:
  - 1.5x leverage → ~{results[2]['max_dd']:.1%} max drawdown (manageable)
  - 2.0x leverage → ~{results[3]['max_dd']:.1%} max drawdown (risky)
  - 3.0x leverage → ~{results[5]['max_dd']:.1%} max drawdown (very risky)
    """)
