"""
figS10_b_num_words_over_time.py 

Computes and plots the number of words for each cricket ruleset edition year,
based on the plain-text ruleset files, against time.

Usage
-----
    # From the repo root:
    python figures/supplemental/figS10/source/figS10_b_num_words_over_time.py 

Outputs
-------
    figures/supplemental/figS10/components/figS10_b_num_words_over_time.svg
"""

import os
import re
import shutil
from scipy.stats import linregress

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Matplotlib / LaTeX config (falls back to mathtext if LaTeX is unavailable)
# ---------------------------------------------------------------------------
USE_TEX = shutil.which("latex") is not None

plt.rcParams["text.usetex"] = USE_TEX
plt.rcParams["font.family"] = "serif"
if USE_TEX:
    plt.rcParams["font.serif"] = ["Computer Modern Roman"]
else:
    plt.rcParams["mathtext.fontset"] = "cm"
    print("Note: LaTeX not found on PATH; falling back to matplotlib's mathtext.")

# ---------------------------------------------------------------------------
# Paths (relative to repo root)
# ---------------------------------------------------------------------------
TXT_DIR = "./data/datasets/rule_texts/processed"
FIGURES_DIR = "./figures/supplemental/figS10/components/"
# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
TARGET_YEARS = [
    1752, 1755, 1774, 1785, 1786, 1788, 1803, 1806, 1809,
    1816, 1820, 1823, 1828, 1830, 1835, 1857, 1884, 1890,
    1892, 1896, 1900, 1902, 1906, 1908, 1910, 1911, 1913,
    1914, 1918, 1920, 1923, 1932, 1939, 1947, 1952, 1962,
    1968, 1980, 1992, 2000, 2008, 2010, 2017, 2019,
]

WORD_PATTERN = re.compile(r"\b\w+\b")

# ---------------------------------------------------------------------------
# Text measures
# ---------------------------------------------------------------------------

def count_words(text: str) -> int:
    """Count word tokens in `text` (\\b\\w+\\b matches)."""
    return len(WORD_PATTERN.findall(text))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_file_for_year(folder: str, year: int, extension: str) -> str | None:
    """Return the path of the first file in `folder` matching {year}*{extension}."""
    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"Directory not found: {folder!r}. "
            "Check the path constants at the top of this script."
        )
    for filename in sorted(os.listdir(folder)):
        if filename.startswith(str(year)) and filename.endswith(extension):
            return os.path.join(folder, filename)
    return None


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------

def build_dataframe() -> pd.DataFrame:
    """
    Build a DataFrame indexed by Year with word counts (from .txt files).
    """
    records = []
    for year in TARGET_YEARS:
        record = {"Year": year, "Number of Words": np.nan}

        txt_path = find_file_for_year(TXT_DIR, year, ".txt")
        if txt_path is not None:
            with open(txt_path, "r", encoding="utf-8") as fh:
                record["Number of Words"] = count_words(fh.read())
        else:
            print(f"Warning: no .txt file found for {year} in {TXT_DIR}")

        records.append(record)

    return pd.DataFrame(records).set_index("Year").sort_index()


# ---------------------------------------------------------------------------
# Figure helper (matches the square-axes construction used elsewhere)
# ---------------------------------------------------------------------------

def create_square_ax(fig_size: float = 8, axes_fraction: float = 0.75):
    """Create a square figure with square axes of consistent physical size."""
    fig = plt.figure(figsize=(fig_size, fig_size))
    ax = fig.add_axes([0.15, 0.15, axes_fraction, axes_fraction])
    ax.set_box_aspect(1)
    return fig, ax


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot(df: pd.DataFrame) -> None:
    # x is now Year (the DataFrame's index) rather than cumulative matches,
    # so no join against an external matches CSV is needed.
    x = df.index.to_series()
    y = df["Number of Words"]

    # Mask NaNs and non-positive values (log10 is undefined for <= 0).
    mask = x.notna() & y.notna() & (x > 0) & (y > 0)
    x_clean = x[mask]
    y_clean = y[mask]

    # Regression in log space.
    log_y_clean = np.log10(y_clean)

    # Year is a linear time axis, not naturally log-scaled, so the predictor
    # is left un-logged by default. Flip this to True if you want a
    # log(Year)-vs-log(Number of Words) fit instead.
    log_predictor_var = False
    if log_predictor_var:
        log_x_clean = np.log10(x_clean)
        slope, intercept, r_value, p_value, std_err = linregress(log_x_clean, log_y_clean)
        log_y_pred = intercept + slope * log_x_clean
    else:
        slope, intercept, r_value, p_value, std_err = linregress(x_clean, log_y_clean)
        log_y_pred = intercept + slope * x_clean

    y_pred = 10 ** log_y_pred  # convert back from log space

    # Square figure + square axes, matching the first script.
    fig, ax = create_square_ax(fig_size=8, axes_fraction=0.75)

    # Sort by x so the fitted line draws cleanly.
    order = np.argsort(x_clean.values)
    x_sorted = x_clean.values[order]
    y_pred_sorted = y_pred.values[order]

    ax.scatter(x_clean, y_clean, color="#7f7f7f")
    ax.plot(x_sorted, y_pred_sorted, color="#9c0412ff",
            label=f"OLS Fit: slope = {slope:.3f}")
    r_squared = r_value ** 2
    print(f"R^2 = {r_squared:.4f}")

    # Log scale + formatting.
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda y, _: f"$10^{{{int(np.log10(y))}}}$")
    )
    if log_predictor_var:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda y, _: f"$10^{{{int(np.log10(y))}}}$")
        )

    ax.set_xlabel("Year", fontsize=30, labelpad=10)
    ax.set_ylabel(r"\# of words", fontsize=30, labelpad=10)
    ax.grid(False)
    ax.tick_params(axis="x", labelsize=18)
    ax.tick_params(axis="y", labelsize=18)
    ax.legend(fontsize=20)

    # Match the first script: reposition then re-square the axes box.
    ax.set_position([0.15, 0.2, 0.7, 0.65])
    ax.set_box_aspect(1)

    fig.savefig(os.path.join(FIGURES_DIR, "figS10_b_num_words_over_time.svg"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    df = build_dataframe()
    print(df.round(4).to_string())
    plot(df)


if __name__ == "__main__":
    main()