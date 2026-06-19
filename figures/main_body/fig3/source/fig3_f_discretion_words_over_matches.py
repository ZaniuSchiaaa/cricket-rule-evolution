"""
fig3_f_discretion_words_over_matches.py

Computes and plots the share of discretion-granting words (as a percentage of
all discretion words) for each cricket ruleset edition year, based on the
plain-text ruleset files, against cumulative matches played.

"Discretion-granting" words permit action (may); "discretion-constraining"
words compel or forbid it (must, shall, should, is to be). The plotted quantity
is granting / (granting + constraining), expressed as a percentage.

Usage
-----
    # From the repo root:
    python figures/main_body/fig3/source/fig3_f_discretion_words_over_matches.py

Outputs
-------
    figures/main_body/fig3/components/fig3_f_discretion_words_over_matches.svg
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
FIGURES_DIR = "./figures/main_body/fig3/components"
MATCHES_CSV = "./figures/main_body/fig3/source/fig3_a_matches_over_time.csv"

# Name of the cumulative-matches column in fig3_a_matches_over_time.csv
MATCHES_COL = "Cumulative Matches Played"  # <-- adjust to match the actual CSV header

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

# Discretion lexicon.
CONSTRAINING_PATTERN = re.compile(r"\b(?:must|shall|should|is to be)\b", re.IGNORECASE)
GRANTING_PATTERN = re.compile(r"\bmay\b", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Text measures
# ---------------------------------------------------------------------------

def discretion_granting_pct(text: str) -> float:
    """
    Return discretion-granting words as a percentage of all discretion words:
    granting / (granting + constraining) * 100.

    Returns NaN if the text contains no discretion words (ratio undefined).
    """
    granting = len(GRANTING_PATTERN.findall(text))
    constraining = len(CONSTRAINING_PATTERN.findall(text))
    total = granting + constraining
    if total == 0:
        return np.nan
    return 100.0 * granting / total

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
    Build a DataFrame indexed by Year with the discretion-granting percentage
    (from .txt files).
    """
    records = []
    for year in TARGET_YEARS:
        record = {"Year": year, "Discretion Granting %": np.nan}

        txt_path = find_file_for_year(TXT_DIR, year, ".txt")
        if txt_path is not None:
            with open(txt_path, "r", encoding="utf-8") as fh:
                record["Discretion Granting %"] = discretion_granting_pct(fh.read())
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
    # Load matches CSV and merge on Year so x/y align regardless of row order.
    matches_df = pd.read_csv(MATCHES_CSV).set_index("Year")
    merged = df.join(matches_df[[MATCHES_COL]], how="inner")

    x = merged[MATCHES_COL]
    y = merged["Discretion Granting %"]

    # Mask NaNs; x must be > 0 for the log axis, y is a bounded percentage.
    mask = x.notna() & y.notna() & (x > 0)
    x_clean = x[mask]
    y_clean = y[mask]

    fig, ax = create_square_ax(fig_size=8, axes_fraction=0.75)

    ax.scatter(x_clean, y_clean, color="#7f7f7f")

    # Log x-axis (matches), linear y-axis (percentage).
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"$10^{{{int(np.log10(v))}}}$")
    )

    ax.set_xlabel("Cumulative matches played", fontsize=30, labelpad=10)
    ax.set_ylabel(r"\% of discretion granting words", fontsize=30, labelpad=10)
    ax.grid(False)
    ax.tick_params(axis="x", labelsize=18)
    ax.tick_params(axis="y", labelsize=18)
    ax.legend(fontsize=20)

    # Match the other fig3 scripts: reposition then re-square the axes box.
    ax.set_position([0.15, 0.2, 0.7, 0.65])
    ax.set_box_aspect(1)

    fig.savefig(os.path.join(FIGURES_DIR, "fig3_f_discretion_words_over_matches.svg"))
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