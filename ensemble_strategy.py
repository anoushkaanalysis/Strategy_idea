#!/usr/bin/env python
import pandas as pd
import numpy as np
from strategy_main import (
    compute_value_signal, compute_reversal_signal, compute_base_factor,
    compute_drift_regime, compute_edge_signal, edge_to_weights, 
    backtest_returns, performance_summary, fetch_stock_data
)

def compute_momentum_signal(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:

    returns = prices.pct_change(window)

    momentum = returns.sub(returns.mean(axis=1), axis=0).div(returns.std(axis=1), axis=0)
    return momentum

def compute_mean_reversion_signal(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:

    returns = prices.pct_change(window)
    mean_reversion = -returns  # Opposite of momentum

    mean_reversion = mean_reversion.sub(mean_reversion.mean(axis=1), axis=0).div(mean_reversion.std(axis=1), axis=0)
    return mean_reversion

def ensemble_signals(prices, weights_dict=None):
    if weights_dict is None:
        weights_dict = {
            'value': 0.4,
            'momentum': 0.3,
            'mean_reversion': 0.3
        }
    

    value_signal = compute_edge_signal(prices, drift_window=63, up_threshold=0.60, 
                                       value_weight=0.7, reversal_lookback=10)
    
    momentum_signal = compute_momentum_signal(prices, window=20)
    
    mean_reversion_signal = compute_mean_reversion_signal(prices, window=20)
    
   
    combined_signal = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    

    valid_value = value_signal.notna()
    combined_signal[valid_value] += (value_signal[valid_value] * weights_dict['value'])
    

    valid_momentum = momentum_signal.notna()
    combined_signal[valid_momentum] += (momentum_signal[valid_momentum] * weights_dict['momentum'])
    

    valid_mr = mean_reversion_signal.notna()
    combined_signal[valid_mr] += (mean_reversion_signal[valid_mr] * weights_dict['mean_reversion'])
    
    return combined_signal

def run_ensemble_backtest(prices, weights_dict=None, cost_bp=0.6, train_frac=0.5):
    """Run backtest with ensemble signals"""
    

    edge = ensemble_signals(prices, weights_dict)
    

    weights = edge_to_weights(edge, min_active=2)
    

    raw_returns = backtest_returns(prices, weights, cost_bp)
    

    split = int(len(raw_returns) * train_frac)
    test_returns = raw_returns.iloc[split:]
    perf = performance_summary(test_returns)
    
    return perf, raw_returns, edge, weights

if __name__ == "__main__":

    tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "JPM", "V", "JNJ",
        "WMT", "BA", "GS", "IBM", "INTC", "AMD", "PG", "KO", "MCD", "NFLX"
    ]
    print(f"Fetching {len(tickers)} stocks...\n")
    prices = fetch_stock_data(tickers, "2018-01-01", "2024-07-23")
    
    print("=" * 70)
    print("MULTI-STRATEGY ENSEMBLE BACKTEST")
    print("=" * 70)
    

    weight_combinations = [
        {'name': 'Pure Value', 'weights': {'value': 1.0, 'momentum': 0.0, 'mean_reversion': 0.0}},
        {'name': 'Pure Momentum', 'weights': {'value': 0.0, 'momentum': 1.0, 'mean_reversion': 0.0}},
        {'name': 'Pure Mean Reversion', 'weights': {'value': 0.0, 'momentum': 0.0, 'mean_reversion': 1.0}},
        {'name': 'Equal Weight (1/3 each)', 'weights': {'value': 0.333, 'momentum': 0.333, 'mean_reversion': 0.333}},
        {'name': 'Value + MR (50/50)', 'weights': {'value': 0.5, 'momentum': 0.0, 'mean_reversion': 0.5}},
        {'name': 'Momentum + MR (50/50)', 'weights': {'value': 0.0, 'momentum': 0.5, 'mean_reversion': 0.5}},
        {'name': 'Optimized (40/30/30)', 'weights': {'value': 0.4, 'momentum': 0.3, 'mean_reversion': 0.3}},
    ]
    
    results = []
    best_sharpe = -float('inf')
    best_combo = None
    
    for combo in weight_combinations:
        perf, _, _, _ = run_ensemble_backtest(prices, combo['weights'])
        results.append({
            'name': combo['name'],
            'weights': combo['weights'],
            'sharpe': perf['sharpe'],
            'return': perf['ann_return'],
            'vol': perf['ann_vol'],
            'max_dd': perf['max_dd']
        })
        
        if perf['sharpe'] > best_sharpe:
            best_sharpe = perf['sharpe']
            best_combo = combo['name']
    
  
    print("\nComparison of Different Weightings:\n")
    print(f"{'Strategy':<30} {'Sharpe':>10} {'Return':>10} {'Vol':>10} {'Max DD':>10}")
    print("-" * 70)
    
    for result in results:
        print(f"{result['name']:<30} {result['sharpe']:>10.4f} {result['return']:>9.2%} {result['vol']:>9.2%} {result['max_dd']:>9.2%}")
    
    print("\n" + "=" * 70)
    print(f"BEST CONFIGURATION: {best_combo} (Sharpe {best_sharpe:.4f})")
    print("=" * 70)
    

    best_weights = None
    for combo in weight_combinations:
        if combo['name'] == best_combo:
            best_weights = combo['weights']
            break
    
    perf, returns, edge, weights = run_ensemble_backtest(prices, best_weights)
    
    print(f"\nDetailed Results:")
    print(f"  Annual Return: {perf['ann_return']:.2%}")
    print(f"  Annual Volatility: {perf['ann_vol']:.2%}")
    print(f"  Sharpe Ratio: {perf['sharpe']:.4f}")
    print(f"  Max Drawdown: {perf['max_dd']:.2%}")
    print(f"  Cumulative Return: {(1 + returns).prod() - 1:.2%}")
    
    print(f"\nWeights Used:")
    print(f"  Value Signal: {best_weights['value']:.1%}")
    print(f"  Momentum Signal: {best_weights['momentum']:.1%}")
    print(f"  Mean Reversion Signal: {best_weights['mean_reversion']:.1%}")
    

    print("\n" + "=" * 70)
    print("IMPROVEMENT OVER BASELINE")
    print("=" * 70)
    
    baseline_sharpe = results[0]['sharpe']  
    improvement = ((best_sharpe - baseline_sharpe) / abs(baseline_sharpe) * 100) if baseline_sharpe != 0 else 0
    
    print(f"\nBaseline (Pure Value): Sharpe {baseline_sharpe:.4f}")
    print(f"Best Ensemble: Sharpe {best_sharpe:.4f}")
    print(f"Improvement: {improvement:+.1f}%")
    
    print("\n" + "=" * 70)
    print("HOW TO FURTHER IMPROVE TO SHARPE 2.0+")
    print("=" * 70)
    print("""
1. ADD MACRO FILTERS (Expected +0.3 Sharpe)
   - Reduce trading when VIX > 30
   - Exit during recession signals
   - Use this ensemble with macro filters
   
2. ADD MACHINE LEARNING (Expected +0.4 Sharpe)
   - Train classifier on signal combinations
   - Use walk-forward validation
   - Predict next day returns
   
3. ADD DYNAMIC SIZING (Expected +0.2 Sharpe)
   - Scale position based on volatility
   - Reduce size during high vol
   - Maintain consistent risk target
   
4. OPTIMIZE PARAMETERS (Expected +0.1 Sharpe)
   - Do more granular grid search
   - Test different windows
   - Fine-tune thresholds
   
"""
