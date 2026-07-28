#!/usr/bin/env python

import pandas as pd
import numpy as np
import yfinance as yf
from strategy_main import (
    fetch_stock_data, edge_to_weights, backtest_returns, performance_summary
)
from ensemble_strategy import (
    compute_momentum_signal, compute_mean_reversion_signal, compute_edge_signal
)

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    HAS_ML = True
except ImportError:
    HAS_ML = False
    print("scikit-learn not found. Install with: pip install scikit-learn")

def create_ml_features(prices, lookback=5):
    features_dict = {}
    
    # Price-based features (aggregate across all stocks)
    returns = prices.pct_change()
    for lag in range(1, lookback+1):
        features_dict[f'return_{lag}d_ago'] = returns.shift(lag).mean(axis=1)
    
    # Value signal (aggregate)
    value_signal = compute_edge_signal(prices, drift_window=63, up_threshold=0.60, 
                                       value_weight=0.7, reversal_lookback=10)
    features_dict['value_signal'] = value_signal.mean(axis=1)
    
    # Momentum signal (aggregate)
    momentum_signal = compute_momentum_signal(prices, window=20)
    features_dict['momentum_signal'] = momentum_signal.mean(axis=1)
    
    # Mean reversion signal (aggregate)
    mean_reversion_signal = compute_mean_reversion_signal(prices, window=20)
    features_dict['mean_reversion_signal'] = mean_reversion_signal.mean(axis=1)
    
    # Volatility (aggregate)
    rolling_vol = returns.rolling(window=20).std().mean(axis=1)
    features_dict['volatility_20d'] = rolling_vol
    
    # Cumulative returns (aggregate)
    for lookback_ret in [5, 10, 20]:
        cum_ret = ((1 + returns).rolling(window=lookback_ret).apply(np.prod, raw=True) - 1).mean(axis=1)
        features_dict[f'cumret_{lookback_ret}d'] = cum_ret
    
    features = pd.DataFrame(features_dict, index=prices.index)
    return features

def create_ml_target(prices, forward_days=1):
    returns = prices.pct_change().shift(-forward_days)
    target = (returns > 0).astype(int)
    return target, returns

def prepare_ml_data(prices, lookback=5, forward_days=1):
    """Prepare data for ML model"""
    features = create_ml_features(prices, lookback=lookback)
    target_class, target_ret = create_ml_target(prices, forward_days=forward_days)
    
    data = pd.concat([features, target_class.rename('target_class'), 
                      target_ret.rename('target_return')], axis=1)
    data = data.dropna()
    
    return data, features

def train_ml_model(data_train, feature_cols):
    """Train ML classifier on training data using scikit-learn"""
    X = data_train[feature_cols].fillna(0)
    y = data_train['target_class']
    
    X = X.loc[:, X.std() > 0.001]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
        verbose=0
    )
    
    model.fit(X_scaled, y)
    return model, X.columns.tolist(), scaler

def generate_ml_signals(prices, model, feature_cols, scaler, lookback=5):
    """Generate trading signals from ML model predictions"""
    features = create_ml_features(prices, lookback=lookback)

    X = features[feature_cols].fillna(0)
    X_scaled = scaler.transform(X)
    
    predictions = model.predict_proba(X_scaled)[:, 1]
    ml_signal = 2 * predictions - 1  # Map [0,1] to [-1, 1]
    
    return pd.DataFrame(ml_signal, index=prices.index, columns=prices.columns)

def run_walk_forward_backtest(prices, train_period=252*2, test_period=252, step=63):
    """
    Walk-forward validation:
    - Train on period of train_period days
    - Test on period of test_period days
    - Step forward by step days
    - Repeat until end of data
    """
    
    if not HAS_ML:
        print("ERROR: scikit-learn not installed. Install with: pip install scikit-learn")
        return None
    
    all_predictions = []
    all_returns = []
    all_sharpes = []
    
    total_length = len(prices)
    start_idx = 0
    
    fold = 0
    while start_idx + train_period + test_period <= total_length:
        fold += 1
        train_end = start_idx + train_period
        test_end = start_idx + train_period + test_period
        

        prices_train = prices.iloc[:train_end]
        prices_test = prices.iloc[train_end:test_end]
        
        print(f"\nFold {fold}: Training on {train_end} days, Testing on {test_period} days")
        
        data_train, features = prepare_ml_data(prices_train, lookback=5)
        feature_cols = [col for col in data_train.columns if col.startswith(('return', 'signal', 'volatility', 'cumret'))]
        
        if len(data_train) < 100:
            print(f"  Skipping: Not enough training data ({len(data_train)} rows)")
            start_idx += step
            continue
        
        try:
            model, used_features, scaler = train_ml_model(data_train, feature_cols)
            print(f"  Model trained on {len(data_train)} samples using {len(used_features)} features")
        except Exception as e:
            print(f"  Training failed: {e}")
            start_idx += step
            continue
        
        try:
            ml_signals = generate_ml_signals(prices_test, model, used_features, scaler, lookback=5)
            
            weights = edge_to_weights(ml_signals, min_active=2)
            
            raw_returns = backtest_returns(prices_test, weights, cost_bp=0.6)
            
            if len(raw_returns) > 0 and raw_returns.std() > 0:
                perf = performance_summary(raw_returns)
                all_sharpes.append(perf['sharpe'])
                print(f"  Test Sharpe: {perf['sharpe']:.4f}, Return: {perf['ann_return']:.2%}, Vol: {perf['ann_vol']:.2%}")
            else:
                print(f"  No valid returns generated")
                all_sharpes.append(np.nan)
        
        except Exception as e:
            print(f"  Backtesting failed: {e}")
        
        start_idx += step
    
    return all_sharpes

def run_ml_enhanced_strategy(prices, use_walk_forward=True):
    print("=" * 70)
    print("MACHINE LEARNING ENHANCED STRATEGY")
    print("=" * 70)
    
    if not HAS_ML:
        print("\n❌ ERROR: scikit-learn not installed")
        print("Install with: pip install scikit-learn")
        return None
    
    if use_walk_forward:
        print("\n📊 Running Walk-Forward Validation...")
        print("(Training on 2 years, testing on 1 year, stepping every 63 days)")
        
        sharpes = run_walk_forward_backtest(prices, train_period=252*2, 
                                            test_period=252, step=63)
        
        if sharpes:
            valid_sharpes = [s for s in sharpes if not np.isnan(s)]
            if valid_sharpes:
                print("\n" + "=" * 70)
                print("WALK-FORWARD RESULTS")
                print("=" * 70)
                print(f"Folds Completed: {len(valid_sharpes)}")
                print(f"Average Sharpe: {np.mean(valid_sharpes):.4f}")
                print(f"Std Dev Sharpe: {np.std(valid_sharpes):.4f}")
                print(f"Min Sharpe: {np.min(valid_sharpes):.4f}")
                print(f"Max Sharpe: {np.max(valid_sharpes):.4f}")
                
                
                median_sharpe = np.median(valid_sharpes)
                print(f"\n🎯 Expected Live Trading Sharpe: {median_sharpe:.4f}")
                print(f"   (With 30-50% degradation: {median_sharpe * 0.5:.4f} - {median_sharpe * 0.7:.4f})")
                
                if median_sharpe > 0.8:
                    print(f"\n✅ ML Enhancement: SUCCESSFUL - Improved from 0.71 baseline")
                else:
                    print(f"\n⚠️  ML Enhancement: Limited improvement over baseline 0.71")
            else:
                print("No valid folds completed")
        
    else:
        print("\n📊 Running Single-Pass ML Model...")
        data, features = prepare_ml_data(prices, lookback=5)
        feature_cols = [col for col in data.columns if col != 'target_class' and col != 'target_return']
        
        split = int(len(data) * 0.7)
        data_train = data.iloc[:split]
        data_test = data.iloc[split:]
        
        model, used_features, scaler = train_ml_model(data_train, feature_cols)
        
        X_test = data_test[used_features].fillna(0)
        X_test_scaled = scaler.transform(X_test)
        y_test = data_test['target_class']
        accuracy = model.score(X_test_scaled, y_test)
        
        print(f"\nModel Accuracy on Test Set: {accuracy:.2%}")
        
        ml_signals = generate_ml_signals(prices.iloc[split:], model, used_features, scaler)
        weights = edge_to_weights(ml_signals, min_active=2)
        raw_returns = backtest_returns(prices.iloc[split:], weights, cost_bp=0.6)
        
        if len(raw_returns) > 0 and raw_returns.std() > 0:
            perf = performance_summary(raw_returns)
            print(f"\nBacktest Results (Test Set):")
            print(f"  Sharpe: {perf['sharpe']:.4f}")
            print(f"  Return: {perf['ann_return']:.2%}")
            print(f"  Vol: {perf['ann_vol']:.2%}")
            print(f"  Max DD: {perf['max_dd']:.2%}")

if __name__ == "__main__":
    tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "JPM", "V", "JNJ",
        "WMT", "BA", "GS", "IBM", "INTC", "AMD", "PG", "KO", "MCD", "NFLX"
    ]
    print(f"Fetching {len(tickers)} stocks...\n")
    prices = fetch_stock_data(tickers, "2018-01-01", "2024-07-23")
    
    run_ml_enhanced_strategy(prices, use_walk_forward=True)
    
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    
