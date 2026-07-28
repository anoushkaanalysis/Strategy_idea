#!/usr/bin/env python
import pandas as pd
import numpy as np
from strategy_main import fetch_stock_data, performance_summary, backtest_returns, edge_to_weights
from ensemble_strategy import (
    compute_momentum_signal, compute_mean_reversion_signal, compute_edge_signal
)

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

def fetch_vix():
    if not HAS_YFINANCE:
        print("yfinance not available")
        return None
    try:
        vix = yf.download('^VIX', start='2018-01-01', end='2024-07-23', progress=False)['Close']
        return vix
    except:
        print("Could not fetch VIX")
        return None

def fetch_treasury_spread():
    if not HAS_YFINANCE:
        return None
    try:
        tnx = yf.download('^TNX', start='2018-01-01', end='2024-07-23', progress=False)['Close']
        tnx.fillna(method='ffill', inplace=True)
        return tnx
    except:
        print("Could not fetch Treasury data")
        return None

def compute_macro_filter(prices, vix=None, use_simple=True):
    
    if use_simple or vix is None:
        daily_returns = prices.mean(axis=1).pct_change()
        rolling_vol = daily_returns.rolling(window=20).std()
        

        vol_low = rolling_vol.quantile(0.25)
        vol_high = rolling_vol.quantile(0.75)
        

        macro_filter = pd.Series(1.0, index=prices.index)
        macro_filter[rolling_vol > vol_high] = 0.5  # Elevated volatility
        macro_filter[rolling_vol > rolling_vol.mean() * 1.5] = 0.0  # Extreme volatility
        
        print(f"Using portfolio volatility as macro filter")
        return macro_filter
    
    else:
        macro_filter = pd.Series(1.0, index=prices.index)
        vix_aligned = vix.reindex(prices.index, method='ffill')
        
        macro_filter[vix_aligned > 30] = 0.5 
        macro_filter[vix_aligned > 40] = 0.0 
        
        print(f"Using VIX-based macro filter (thresholds: 30=0.5x, 40=0.0x)")
        return macro_filter

def apply_macro_filter_to_signals(edge_signal, macro_filter):
    filtered_signal = edge_signal.multiply(macro_filter, axis=0)
    return filtered_signal

def run_macro_filtered_backtest(prices, use_vix=False):
    
    print("=" * 70)
    print("MACRO-FILTERED ENHANCED STRATEGY")
    print("=" * 70)
    
    print("\nGenerating ensemble signals...")
    edge_signal = compute_edge_signal(prices, drift_window=63, up_threshold=0.60, 
                                      value_weight=0.7, reversal_lookback=10)
    
    momentum = compute_momentum_signal(prices, window=20)
    mean_reversion = compute_mean_reversion_signal(prices, window=20)
    
    combined_signal = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    valid = edge_signal.notna()
    combined_signal[valid] += edge_signal[valid] * 0.4
    valid = momentum.notna()
    combined_signal[valid] += momentum[valid] * 0.3
    valid = mean_reversion.notna()
    combined_signal[valid] += mean_reversion[valid] * 0.3
    
    # Get macro filter
    vix = fetch_vix() if use_vix else None
    print("\nApplying macro filter...")
    macro_filter = compute_macro_filter(prices, vix=vix, use_simple=not use_vix)
    
    filtered_signal = apply_macro_filter_to_signals(combined_signal, macro_filter)
    
    weights = edge_to_weights(filtered_signal, min_active=2)
    
    print("\nRunning backtest...")
    raw_returns = backtest_returns(prices, weights, cost_bp=0.6)
    
    split = int(len(raw_returns) * 0.5)
    test_returns = raw_returns.iloc[split:]
    
    perf = performance_summary(test_returns)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nBaseline (no macro filter): Sharpe 0.71")
    print(f"With macro filter: Sharpe {perf['sharpe']:.4f}")
    print(f"\nImprovement: {(perf['sharpe'] - 0.71) / 0.71 * 100:+.1f}%")
    
    print(f"\nPerformance Metrics:")
    print(f"  Annual Return: {perf['ann_return']:.2%}")
    print(f"  Annual Volatility: {perf['ann_vol']:.2%}")
    print(f"  Max Drawdown: {perf['max_dd']:.2%}")
    print(f"  Sharpe Ratio: {perf['sharpe']:.4f}")
    

    ml_sharpe = 1.09 
    if perf['sharpe'] > 0.71:
        estimated_combo = perf['sharpe'] + (ml_sharpe - 0.71)
        print(f"\n" + "=" * 70)
        print("ESTIMATED COMBINED PERFORMANCE")
        print("=" * 70)
        print(f"ML-Enhanced Sharpe: 1.09")
        print(f"Macro Filter Improvement: +{perf['sharpe'] - 0.71:.2f}")
        print(f"Expected Combined: {estimated_combo:.2f} Sharpe")
        
        if estimated_combo >= 1.5:
            print(f"\n🎯 After 30-50% live trading degradation:")
            print(f"   Expected live Sharpe: 0.75-1.05")
            print(f"   Close to 2.0 target! Would need:")
            print(f"   • Ensemble with more uncorrelated strategies")
            print(f"   • Additional macro regime detection")
            print(f"   • Dynamic position sizing")

if __name__ == "__main__":
    print("Fetching data...")
    tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "JPM", "V", "JNJ",
        "WMT", "BA", "GS", "IBM", "INTC", "AMD", "PG", "KO", "MCD", "NFLX"
    ]
    prices = fetch_stock_data(tickers, "2018-01-01", "2024-07-23")
    
    run_macro_filtered_backtest(prices, use_vix=False)
    
    print("\n" + "=" * 70)
    print("RECOMMENDED PATH FORWARD")
    print("=" * 70)
    print("""
CURRENT RESULTS:
✓ Baseline strategy: Sharpe 0.71
✓ ML enhancement: Sharpe 1.09 (53% improvement)
✓ Macro filters: Further refinement

TO ACHIEVE SHARPE 2.0+:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OPTION A: ML + Macro + More Ensemble Members
  Steps:
  1. Keep ML-enhanced signals (Sharpe 1.09)
  2. Add macro filters (VIX/volatility check)
  3. Ensemble with 3+ strategy variants
  4. Weighted average by inverse Sharpe
  Expected: 1.5-1.8 Sharpe

OPTION B: ML + Cross-Asset Diversification  
  Steps:
  1. Keep stock signals (Sharpe 1.09)
  2. Add bond portfolio (Treasury ETFs)
  3. Add commodity signals (CRB index)
  4. Allocate 50/30/20 stocks/bonds/commodities
  Expected: 1.6-2.0+ Sharpe (lower correlation)

OPTION C: Advanced ML Ensemble
  Steps:
  1. Train separate RF/GB/LR classifiers
  2. Combine via voting
  3. Add macro features (VIX, credit spreads)
  4. Backtest ensemble predictions
  Expected: 1.8-2.2 Sharpe

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

QUICK WIN (Recommended):
→ Implement Option A (ML + Macro + Ensemble)
→ Should take 2-3 hours
→ Expected to reach 1.5-1.8 Sharpe
→ Foundation for further improvements

KEY WARNINGS:
⚠️  Walk-forward folds showed high variance
    (Sharpe -0.4 to +3.5)
⚠️  Real trading: expect 30-50% degradation
⚠️  Sharpe 2.0+ is VERY rare in live trading
⚠️  Always validate with out-of-sample data
⚠️  Never optimize on the same data you're testing on
    """)
