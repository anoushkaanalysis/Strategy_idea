"""
MACHINE LEARNING ENHANCED STRATEGY - VERSION 2
Combines ensemble signals with ML classification for Sharpe 2.0+"""
import pandas as pd
import numpy as np
from strategy_main import (
    fetch_stock_data, edge_to_weights, backtest_returns, performance_summary
)
from ensemble_strategy import (
    compute_momentum_signal, compute_mean_reversion_signal, compute_edge_signal
)

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    HAS_ML = True
except ImportError:
    HAS_ML = False
    print("scikit-learn not found. Install with: pip install scikit-learn")

def extract_portfolio_features(prices, lookback=5):
    """Extract time-series features aggregated across portfolio"""
    
    returns = prices.pct_change()
    
    features_dict = {}
    
    # 1. Recent returns (aggregated)
    for lag in range(1, min(lookback + 1, 6)):
        features_dict[f'return_{lag}d_ago'] = returns.shift(lag).mean(axis=1)
    
    # 2. Signals (aggregated)
    value_sig = compute_edge_signal(prices, drift_window=63, up_threshold=0.60, 
                                     value_weight=0.7, reversal_lookback=10)
    features_dict['value_signal'] = value_sig.mean(axis=1)
    
    mom_sig = compute_momentum_signal(prices, window=20)
    features_dict['momentum_signal'] = mom_sig.mean(axis=1)
    
    mr_sig = compute_mean_reversion_signal(prices, window=20)
    features_dict['mean_reversion_signal'] = mr_sig.mean(axis=1)
    
    # 3. Volatility (aggregated)
    vol_20d = returns.rolling(window=20).std().mean(axis=1)
    features_dict['volatility_20d'] = vol_20d
    
    # 4. Cumulative returns (aggregated)
    for window in [5, 10, 20]:
        cum_ret = ((1 + returns).rolling(window=window).apply(np.prod, raw=True) - 1).mean(axis=1)
        features_dict[f'cumret_{window}d'] = cum_ret
    
    features_df = pd.DataFrame(features_dict, index=prices.index)
    return features_df

def create_target(prices, forward_days=1):

    portfolio_ret = prices.mean(axis=1).pct_change(forward_days).shift(-forward_days)
    target = (portfolio_ret > 0).astype(int)
    return target

def prepare_ml_dataset(prices, lookback=5, forward_days=1):

    features = extract_portfolio_features(prices, lookback=lookback)
    target = create_target(prices, forward_days=forward_days)
    
    data = pd.concat([features, target.rename('target')], axis=1)
    data = data.dropna()
    
    return data

def train_model(X_train, y_train):

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    

    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_scaled, y_train)
    
    return model, scaler

def generate_signals_from_model(prices, model, scaler, feature_cols):

    features = extract_portfolio_features(prices, lookback=5)
    

    X = features[feature_cols].fillna(0)
    X_scaled = scaler.transform(X)
    

    probs = model.predict_proba(X_scaled)[:, 1]
    

    signal = 2 * probs - 1
    

    signal_df = pd.DataFrame(
        np.tile(signal.reshape(-1, 1), (1, prices.shape[1])),
        index=prices.index,
        columns=prices.columns
    )
    
    return signal_df

def run_single_fold(prices_train, prices_test, fold_num):

    print(f"\n{'='*60}")
    print(f"Fold {fold_num}")
    print(f"{'='*60}")
    print(f"Train: {len(prices_train)} days | Test: {len(prices_test)} days")
    

    data_train = prepare_ml_dataset(prices_train, lookback=5, forward_days=1)
    
    if len(data_train) < 50:
        print(f"❌ Insufficient training data ({len(data_train)} rows)")
        return None

    feature_cols = [col for col in data_train.columns if col != 'target']
    X_train = data_train[feature_cols]
    y_train = data_train['target']
    

    print(f"Training on {len(X_train)} samples with {len(feature_cols)} features...")
    model, scaler = train_model(X_train, y_train)
    

    X_test_full = data_train[feature_cols]
    X_test_scaled = scaler.transform(X_test_full)
    train_accuracy = model.score(X_test_scaled, y_train)
    print(f"Train accuracy: {train_accuracy:.2%}")

    print(f"Generating signals for test period...")
    ml_signals = generate_signals_from_model(prices_test, model, scaler, feature_cols)
    

    weights = edge_to_weights(ml_signals, min_active=2)
    

    raw_returns = backtest_returns(prices_test, weights, cost_bp=0.6)
    
    if len(raw_returns) == 0 or raw_returns.std() == 0:
        print(f"❌ No valid returns")
        return None
    

    perf = performance_summary(raw_returns)
    
    print(f"✓ Sharpe Ratio: {perf['sharpe']:.4f}")
    print(f"✓ Annual Return: {perf['ann_return']:.2%}")
    print(f"✓ Annual Volatility: {perf['ann_vol']:.2%}")
    print(f"✓ Max Drawdown: {perf['max_dd']:.2%}")
    
    return perf['sharpe']

def run_walk_forward_validation(prices, train_window=504, test_window=252, step=63):
    """Run walk-forward validation"""
    all_sharpes = []
    fold = 0
    idx = 0
    
    print(f"\n{'='*60}")
    print(f"WALK-FORWARD VALIDATION")
    print(f"{'='*60}")
    print(f"Train window: {train_window} days (~2 years)")
    print(f"Test window: {test_window} days (~1 year)")
    print(f"Step size: {step} days (~3 months)")
    
    while idx + train_window + test_window <= len(prices):
        fold += 1
        

        train_end = idx + train_window
        test_end = idx + train_window + test_window
        
        prices_train = prices.iloc[:train_end]
        prices_test = prices.iloc[train_end:test_end]
        

        sharpe = run_single_fold(prices_train, prices_test, fold)
        
        if sharpe is not None:
            all_sharpes.append(sharpe)
        
        idx += step
    
    return all_sharpes

def main():

    tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "JPM", "V", "JNJ",
        "WMT", "BA", "GS", "IBM", "INTC", "AMD", "PG", "KO", "MCD", "NFLX"
    ]
    
    print("=" * 70)
    print("ML-ENHANCED STRATEGY FOR SHARPE 2.0+")
    print("=" * 70)
    print(f"\nFetching {len(tickers)} stocks (2018-2024)...\n")
    
    prices = fetch_stock_data(tickers, "2018-01-01", "2024-07-23")
    
    if not HAS_ML:
        print("ERROR: scikit-learn not installed. Run: pip install scikit-learn")
        return
    

    sharpes = run_walk_forward_validation(prices, train_window=504, test_window=252, step=63)
    

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    if sharpes:
        avg_sharpe = np.mean(sharpes)
        median_sharpe = np.median(sharpes)
        std_sharpe = np.std(sharpes)
        min_sharpe = np.min(sharpes)
        max_sharpe = np.max(sharpes)
        
        print(f"\nFolds Completed: {len(sharpes)}")
        print(f"Average Sharpe: {avg_sharpe:.4f}")
        print(f"Median Sharpe: {median_sharpe:.4f}")
        print(f"Std Dev: {std_sharpe:.4f}")
        print(f"Range: {min_sharpe:.4f} to {max_sharpe:.4f}")
        
        print(f"\n{'='*70}")
        print("PERFORMANCE ESTIMATE")
        print(f"{'='*70}")
        print(f"\nUsing MEDIAN (most conservative): {median_sharpe:.4f}")
        print(f"Baseline (non-ML): 0.71")
        print(f"Improvement: {(median_sharpe - 0.71) / 0.71 * 100:.1f}%")
        
        if median_sharpe > 0.8:
            print(f"\n✅ ML ENHANCEMENT SUCCESSFUL")
            print(f"Expected live trading Sharpe (with 30-50% degradation):")
            print(f"  Conservative: {median_sharpe * 0.5:.4f}")
            print(f"  Optimistic: {median_sharpe * 0.7:.4f}")
        else:
            print(f"\n⚠️  Limited improvement - consider additional enhancements:")
            print(f"  • Add macro filters (VIX/yield curve)")
            print(f"  • Add more diverse asset classes")
            print(f"  • Increase training data")
    else:
        print("No valid folds completed")

if __name__ == "__main__":
    main()
