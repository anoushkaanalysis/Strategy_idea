
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d proj)

from unicorn_edge_strategy import run_backtest


def sweep_sharpe_surface(prices: pd.DataFrame, windows, thresholds) -> np.ndarray:
    sharpe_grid = np.zeros((len(windows), len(thresholds)))
    for i, w in enumerate(windows):
        for j, th in enumerate(thresholds):
            try:
                perf = run_backtest(prices, drift_window=w, up_threshold=th)
                sharpe_grid[i, j] = perf["sharpe"] if np.isfinite(perf["sharpe"]) else 0.0
            except Exception:
                sharpe_grid[i, j] = np.nan
    return sharpe_grid


def plot_surface(windows, thresholds, sharpe_grid, out_path, title):
    W, T = np.meshgrid(thresholds, windows)  # note order matches sharpe_grid[i,j] -> (windows,thresholds)

    fig = plt.figure(figsize=(10, 7.5))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(W, T, sharpe_grid, cmap="RdYlGn", edgecolor="k",
                            linewidth=0.3, antialiased=True, vmin=np.nanmin(sharpe_grid),
                            vmax=max(np.nanmax(sharpe_grid), 1e-6))

    ax.set_xlabel("Up-fraction threshold (θ)")
    ax.set_ylabel("Drift window (days)")
    ax.set_zlabel("OOS Sharpe ratio")
    ax.set_title(title)
    fig.colorbar(surf, shrink=0.6, aspect=12, label="Sharpe ratio")
    ax.view_init(elev=25, azim=-135)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    # --- Replace this block with real price data to test the real claim ---
    np.random.seed(1)
    n_days, n_stocks = 1500, 200
    dates = pd.bdate_range("2018-01-01", periods=n_days)
    rets = np.random.normal(0.0003, 0.018, size=(n_days, n_stocks))
    prices = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=dates,
                           columns=[f"STK{i:03d}" for i in range(n_stocks)])
    
    windows = [44, 54, 63, 72, 82]
    thresholds = [0.42, 0.51, 0.60, 0.69, 0.78]

    grid = sweep_sharpe_surface(prices, windows, thresholds)
    print("Sharpe grid (rows=drift_window, cols=threshold):")
    print(pd.DataFrame(grid, index=windows, columns=thresholds).round(2))

    plot_surface(windows, thresholds, grid,
                 "/mnt/user-data/outputs/sharpe_sensitivity_3d.png",
                 "OOS Sharpe sensitivity — drift window × up-threshold\n(demo run on synthetic random-walk data)")
    print("\nSaved plot to /mnt/user-data/outputs/sharpe_sensitivity_3d.png")
