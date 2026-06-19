"""
fig3_e_rank_freq.py

Computes and plots the normalized rank-frequency distribution of words for each
cricket ruleset edition year, based on the plain-text ruleset files. Each year's
distribution is drawn as a log-log scatter, colored by year.

Usage
-----
    # From the repo root:
    python figures/main_body/fig3/source/fig3_e_rank_freq.py

Outputs
-------
    figures/main_body/fig3/components/fig3_e_rank_freq.png
"""

import os
import re
import shutil
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np

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
TEXT_DIR = "./data/datasets/rule_texts/processed"
FIGURES_DIR = "./figures/main_body/fig3/components"

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
YEAR_PATTERN = re.compile(r"(\d{4})")
LANGUAGE = "english"
POINT_SIZE = 8
POINT_ALPHA = 0.3

# ---------------------------------------------------------------------------
# Text measures
# ---------------------------------------------------------------------------

def clean_and_tokenize(text: str) -> list[str]:
    """Lowercase, tokenize, and keep alphabetic non-stopword tokens."""
    stop = set(stopwords.words(LANGUAGE))
    tokens = word_tokenize(text.lower())
    return [t for t in tokens if t.isalpha() and t not in stop]


def get_word_frequency(text: str) -> Counter:
    """Return a Counter of word-token frequencies for `text`."""
    return Counter(clean_and_tokenize(text))


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------

def build_frequencies(folder: str) -> dict[int, Counter]:
    """
    Map each ruleset edition year to its word-frequency Counter, parsed from
    the .txt files in `folder` (year taken from the first 4-digit run in the
    filename).
    """
    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"Directory not found: {folder!r}. "
            "Check the path constants at the top of this script."
        )

    freq_by_year: dict[int, Counter] = {}
    for filename in sorted(os.listdir(folder)):
        if not filename.endswith(".txt"):
            continue
        match = YEAR_PATTERN.search(filename)
        if match is None:
            print(f"Warning: no year found in filename {filename!r}; skipping.")
            continue
        year = int(match.group(1))
        with open(os.path.join(folder, filename), "r", encoding="utf-8") as fh:
            freq_by_year[year] = get_word_frequency(fh.read())

    return freq_by_year


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

def plot(freq_by_year: dict[int, Counter]) -> None:
    years = sorted(freq_by_year)
    norm = mcolors.Normalize(vmin=min(years), vmax=max(years))
    cmap = plt.get_cmap("coolwarm")

    fig, ax = create_square_ax(fig_size=8, axes_fraction=0.75)

    for year in years:
        total = sum(freq_by_year[year].values())
        sorted_freqs = sorted(
            (count / total for count in freq_by_year[year].values()),
            reverse=True,
        )
        ranks = range(1, len(sorted_freqs) + 1)
        ax.scatter(
            ranks, sorted_freqs,
            color=cmap(norm(year)),
            s=POINT_SIZE,
            alpha=POINT_ALPHA,
            edgecolors="none",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Rank", fontsize=30, labelpad=10)
    ax.set_ylabel("Relative word frequency", fontsize=30, labelpad=10)
    ax.tick_params(axis="x", labelsize=18)
    ax.tick_params(axis="y", labelsize=18)

    # Colorbar doubles as a year legend.
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Year", fontsize=30, labelpad=10)
    cbar.ax.tick_params(labelsize=20)

    fig.savefig(os.path.join(FIGURES_DIR, "fig3_e_rank_freq.png"), dpi=600)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    freq_by_year = build_frequencies(TEXT_DIR)
    plot(freq_by_year)


if __name__ == "__main__":
    main()