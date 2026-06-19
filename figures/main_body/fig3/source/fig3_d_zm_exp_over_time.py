"""
fig3_d_zm_exp_over_time.py

Computes and plots the Zipf-Mandelbrot exponent (alpha) for each cricket
ruleset edition year, based on the plain-text ruleset files, against time.

Usage
-----
    # From the repo root:
    python figures/main_body/fig3/source/fig3_d_zm_exp_over_time.py

Outputs
-------
    figures/main_body/fig3/components/fig3_d_zm_exp_over_time.svg
"""

import os
import re
import shutil
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

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

LANGUAGE = "english"

# ---------------------------------------------------------------------------
# Text measures / Zipf-Mandelbrot fitting
# ---------------------------------------------------------------------------

def clean_and_tokenize(text: str) -> list[str]:
    """Lowercase, tokenize, keep alphabetic non-stopword tokens."""
    tokens = word_tokenize(text.lower())
    stop = set(stopwords.words(LANGUAGE))
    return [t for t in tokens if t.isalpha() and t not in stop]


def zipf_mandelbrot(r, C, b, a):
    """Zipf-Mandelbrot law: C / (r + b)^a."""
    return C / (r + b) ** a


def fit_zm_exponent(text: str) -> float:
    """
    Fit the Zipf-Mandelbrot law to the rank-frequency distribution of `text`
    and return the exponent alpha (np.nan if the fit fails).
    """
    tokens = clean_and_tokenize(text)
    freq = Counter(tokens)
    total = sum(freq.values())
    if total == 0:
        return np.nan

    freqs = np.array(sorted((c / total for c in freq.values()), reverse=True))
    ranks = np.arange(1, len(freqs) + 1)

    mask = freqs > 0
    ranks = ranks[mask]
    freqs = freqs[mask]

    try:
        popt, _ = curve_fit(
            zipf_mandelbrot,
            ranks,
            freqs,
            p0=[1.0, 1.0, 1.0],
            bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
            maxfev=10000,
        )
        _, _, a = popt
        return a
    except RuntimeError:
        return np.nan


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
    Build a DataFrame indexed by Year with the Zipf-Mandelbrot exponent alpha.
    """
    records = []
    for year in TARGET_YEARS:
        record = {"Year": year, "Alpha": np.nan}

        txt_path = find_file_for_year(TXT_DIR, year, ".txt")
        if txt_path is not None:
            with open(txt_path, "r", encoding="utf-8") as fh:
                record["Alpha"] = fit_zm_exponent(fh.read())
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
    y = df["Alpha"]
    mask = y.notna()
    x_clean = df.index[mask]
    y_clean = y[mask]

    fig, ax = create_square_ax(fig_size=8, axes_fraction=0.75)

    ax.scatter(x_clean, y_clean, marker="o", linestyle="-", color="#7f7f7f")

    ax.set_xlabel("Year", fontsize=30, labelpad=10)
    ax.set_ylabel(r"Zipf-Mandelbrot exponent ($\alpha$)", fontsize=30, labelpad=10)
    ax.grid(False)
    ax.tick_params(axis="x", labelsize=18)
    ax.tick_params(axis="y", labelsize=18)

    # Match the other fig3 scripts: reposition then re-square the axes box.
    # ax.set_position([0.15, 0.2, 0.7, 0.65])
    ax.set_box_aspect(1)

    fig.savefig(os.path.join(FIGURES_DIR, "fig3_d_zm_exp_over_time.svg"))
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