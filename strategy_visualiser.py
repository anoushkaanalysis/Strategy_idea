import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)
from matplotlib import cm
try:
    from strategy_main import fetch_stock_data
    from ml_strategy_v2 import (
        extract_portfolio_features, create_target, run_walk_forward_validation,
    )
    HAS_STRATEGY = True
except ImportError as e:
    HAS_STRATEGY = False
    print(f"[info] Could not import strategy modules ({e}).")
    print("[info] Falling back to synthetic demo data so the plots still work.\n")


TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "JPM", "V", "JNJ",
    "WMT", "BA", "GS", "IBM", "INTC", "AMD", "PG", "KO", "MCD", "NFLX",
]


def make_synthetic_prices(n_days=1500, n_assets=20, seed=7):
    """Random-walk price panel, used only when the real data/modules aren't available."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n_days)
    drift = rng.normal(0.0003, 0.0002, n_assets)
    vol = rng.uniform(0.01, 0.025, n_assets)
    rets = rng.normal(drift, vol, size=(n_days, n_assets))
    prices = 100 * np.cumprod(1 + rets, axis=0)
    cols = TICKERS[:n_assets] if n_assets <= len(TICKERS) else [f"A{i}" for i in range(n_assets)]
    return pd.DataFrame(prices, index=dates, columns=cols)


def get_prices():
    if HAS_STRATEGY:
        try:
            print("Fetching real price data (2018-01-01 to 2024-07-23)...")
            return fetch_stock_data(TICKERS, "2018-01-01", "2024-07-23")
        except Exception as e:
            print(f"[warn] fetch_stock_data failed ({e}); using synthetic data instead.")
    return make_synthetic_prices()


def get_features_and_target(prices):
    if HAS_STRATEGY:
        feats = extract_portfolio_features(prices, lookback=5)
        target = create_target(prices, forward_days=1)
        return feats, target

    returns = prices.pct_change()
    feats = pd.DataFrame(index=prices.index)
    feats["momentum_signal"] = returns.rolling(20).mean().mean(axis=1)
    feats["mean_reversion_signal"] = -returns.rolling(20).mean().mean(axis=1)
    feats["value_signal"] = (prices / prices.rolling(63).mean() - 1).mean(axis=1)
    target = (prices.mean(axis=1).pct_change().shift(-1) > 0).astype(int)
    return feats, target


def get_fold_sharpes(prices):
    if HAS_STRATEGY:
        try:
            sharpes = run_walk_forward_validation(prices, train_window=504, test_window=252, step=63)
            if sharpes:
                return sharpes
        except Exception as e:
            print(f"[warn] walk-forward validation failed ({e}); using synthetic Sharpe values.")
    rng = np.random.default_rng(3)
    n_folds = max(3, (len(prices) - 504 - 252) // 63 + 1)
    return list(np.clip(rng.normal(0.75, 0.35, n_folds), -0.5, 2.2))


def get_equity_curve(prices):
    """Simple equal-weight long-only equity curve, used for the ribbon plot."""
    returns = prices.pct_change().fillna(0)
    port_ret = returns.mean(axis=1)
    equity = (1 + port_ret).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    return equity, drawdown


def plot_feature_space(feats, target):
    data = pd.concat([feats, target.rename("target")], axis=1).dropna()
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        data["momentum_signal"], data["mean_reversion_signal"], data["value_signal"],
        c=data["target"], cmap="coolwarm", alpha=0.6, s=14, edgecolors="none",
    )
    ax.set_xlabel("Momentum signal")
    ax.set_ylabel("Mean-reversion signal")
    ax.set_zlabel("Value signal")
    ax.set_title("Feature space - colored by next-day direction (red = up)")
    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("Target (1 = next-day up)")
    fig.tight_layout()
    return fig

def plot_walkforward_bars(sharpes):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    n = len(sharpes)
    xs = np.arange(n)
    ys = np.zeros(n)
    zs = np.zeros(n)
    dx = np.ones(n) * 0.6
    dy = np.ones(n) * 0.6
    dz = np.array(sharpes, dtype=float)
    span = dz.max() - dz.min()
    colors = cm.viridis((dz - dz.min()) / (span if span > 1e-9 else 1))
    ax.bar3d(xs, ys, zs, dx, dy, dz, color=colors, shade=True)
    ax.set_xlabel("Fold #")
    ax.set_yticks([])
    ax.set_zlabel("Sharpe ratio")
    ax.set_title("Walk-forward validation - Sharpe ratio per fold")
    fig.tight_layout()
    return fig

def plot_equity_ribbon(equity, drawdown):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    t = np.arange(len(equity))
    ax.plot(t, equity.values, np.zeros(len(equity)), color="steelblue", lw=1.5, label="Equity curve")
    ax.plot(t, np.ones(len(equity)) * equity.max(), drawdown.values * 100, color="firebrick", lw=1.2, label="Drawdown (%)")
    ax.set_xlabel("Trading day")
    ax.set_ylabel("Equity (growth of $1)")
    ax.set_zlabel("Drawdown (%) / 0")
    ax.set_title("Equity curve and drawdown depth over time")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def main():
    prices = get_prices()
    feats, target = get_features_and_target(prices)
    sharpes = get_fold_sharpes(prices)
    equity, drawdown = get_equity_curve(prices)

    print(f"\nPrices: {prices.shape[0]} days x {prices.shape[1]} assets")
    print(f"Feature rows: {len(feats.dropna())}")
    print(f"Folds: {len(sharpes)}  |  Median Sharpe: {np.median(sharpes):.3f}")

    plot_feature_space(feats, target)
    plot_walkforward_bars(sharpes)
    plot_equity_ribbon(equity, drawdown)

    print("\nThree windows will open. Click-and-drag inside each one to rotate,")
    print("scroll to zoom, and right-click-drag to pan.")
    plt.show()


if __name__ == "__main__":
    main()
