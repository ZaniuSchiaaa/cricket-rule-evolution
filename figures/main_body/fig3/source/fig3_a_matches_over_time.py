"""
fig3_a_matches_over_time.py

Plots the cumulative number of cricket matches played over time on a log-scaled
y-axis, fits an ordinary least squares regression in log space, and overlays the
fit on the original-scale data.

Usage
-----
    # From the repo root:
    python figures/main_body/fig3/source/fig3_a_matches_over_time.py

Outputs
-------
    figures/main_body/fig3/components/fig3_a_matches_over_time.svg
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress
import shutil

# ==========================
# Configuration
# ==========================
DATA_PATH = "./figures/main_body/fig3/source/fig3_a_matches_over_time.csv"
SAVE_PATH = "./figures/main_body/fig3/components/fig3_a_matches_over_time.svg"
COLUMNS = ["Cumulative Matches Played"]

DATA_COLOR = "#7f7f7f"
FIT_COLOR = "#9c0412ff"

USE_TEX = shutil.which("latex") is not None

plt.rcParams["text.usetex"] = USE_TEX
plt.rcParams["font.family"] = "serif"
if USE_TEX:
    plt.rcParams["font.serif"] = ["Computer Modern Roman"]
else:
    plt.rcParams['mathtext.fontset'] = 'cm'
    print("Note: LaTeX not found on PATH; falling back to matplotlib's mathtext.")


def plot_column(df, col):
    """Plot cumulative counts for a single column with a log-space OLS fit.

    Parameters
    ----------
    df : pandas.DataFrame
        Data indexed by year.
    col : str
        Name of the column to plot.

    Returns
    -------
    matplotlib.figure.Figure
    """
    log_col = f"Log of {col} (Cumulative)"
    df[log_col] = np.log10(df[col].replace(0, np.nan))

    x = df.index
    y = df[log_col]

    # Linear regression in log space.
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    y_pred = intercept + slope * x

    fig = plt.figure(figsize=(10, 6))

    # Plot data and regression line back in the original scale.
    plt.scatter(x, 10 ** y, color=DATA_COLOR)
    plt.plot(x, 10 ** y_pred, color=FIT_COLOR, label=f"OLS Fit: slope = {slope:.3f}")
    plt.yscale("log")

    plt.xlabel("Year", fontsize=30)
    plt.ylabel("Cumulative matches played", fontsize=30)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(False)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.gca().set_box_aspect(1)  # Make the axes box square.

    return fig


def main():
    """Load the data, generate the figure, and save it to disk."""
    df = pd.read_csv(DATA_PATH, index_col="Year")

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    for col in COLUMNS:
        fig = plot_column(df, col)
        fig.savefig(SAVE_PATH)
        plt.close(fig)


if __name__ == "__main__":
    main()