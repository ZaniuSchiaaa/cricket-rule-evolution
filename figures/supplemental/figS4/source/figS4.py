"""
figS4.py

Generates the first Zipf-Mandelbrot fit grid (6x4, first 24 edition years) for the
cricket rule-set texts. Each subplot shows the rank-frequency data (log-log) for a
single edition year overlaid with its fitted Zipf-Mandelbrot curve, annotated with
the fitted alpha, beta, and R^2.

Usage
-----
    # From the repo root:
    python figures/supplemental/figS4/source/figS4.py

Outputs
-------
    ./figures/supplemental/figS4/figS4_final.svg
"""

# ──────────────────────────────────────────────────────────────────────────────
# [0] IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import os
import re
import shutil
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# ──────────────────────────────────────────────────────────────────────────────
# [1] MATPLOTLIB / LaTeX CONFIG
# ──────────────────────────────────────────────────────────────────────────────
if shutil.which("latex"):
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Computer Modern Roman"]
else:
    plt.rcParams["text.usetex"] = False
    plt.rcParams["mathtext.fontset"] = "cm"

# ──────────────────────────────────────────────────────────────────────────────
# [2] CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
TEXT_FOLDER = "./data/datasets/rule_texts/processed"
OUTPUT_DIR = "./figures/supplemental/figS4"

FILE_PATTERN = re.compile(r"(\d{4})")  # Extract year
LANGUAGE = "english"

N_ROWS = 6
N_COLS = 4

os.makedirs(OUTPUT_DIR, exist_ok=True)

nltk.download("punkt")
nltk.download("stopwords")

# ──────────────────────────────────────────────────────────────────────────────
# [3] HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def clean_and_tokenize(text):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha() and t not in stopwords.words(LANGUAGE)]
    return tokens


def get_rank_frequency(text):
    tokens = clean_and_tokenize(text)
    freq = Counter(tokens)
    total = sum(freq.values())
    return freq, total


# --- Define Zipf-Mandelbrot function ---
# Thanks to Dawoon Jeong!
def zipf_mandelbrot(r, C, b, a):
    return C / (r + b) ** a


def r_squared(y_true, y_pred):
    residual = np.sum((y_true - y_pred) ** 2)
    total = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (residual / total)


# ──────────────────────────────────────────────────────────────────────────────
# [4] LOAD FILES AND COMPUTE FREQUENCIES
# ──────────────────────────────────────────────────────────────────────────────
freq_by_year = {}
totals_by_year = {}

for filename in os.listdir(TEXT_FOLDER):
    if filename.endswith(".txt"):
        match = FILE_PATTERN.search(filename)
        if match:
            year = int(match.group(1))
            with open(os.path.join(TEXT_FOLDER, filename), "r", encoding="utf-8") as f:
                text = f.read()
            freq, total = get_rank_frequency(text)
            freq_by_year[year] = freq
            totals_by_year[year] = total

years = sorted(freq_by_year.keys())

# ──────────────────────────────────────────────────────────────────────────────
# [5] FIT ZIPF-MANDELBROT PER YEAR
# ──────────────────────────────────────────────────────────────────────────────
fit_data_by_year = {}

for year in years:
    total = totals_by_year[year]
    freqs = sorted([count / total for count in freq_by_year[year].values()], reverse=True)
    ranks = np.arange(1, len(freqs) + 1)

    ranks = np.array(ranks)
    freqs = np.array(freqs)
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
        C, b, a = popt
        y_hat = zipf_mandelbrot(ranks, C, b, a)
        r2 = r_squared(freqs, y_hat)
        fit_data_by_year[year] = (ranks, freqs, C, b, a, r2)
    except RuntimeError:
        print(f"Fit failed for year {year}")

# ──────────────────────────────────────────────────────────────────────────────
# [6] GRID PLOT (6x4, first 24)
# ──────────────────────────────────────────────────────────────────────────────
sorted_fit_years = sorted(fit_data_by_year.keys())
batch = sorted_fit_years[:24]

fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(N_COLS * 4, N_ROWS * 3))

for idx, (ax, year) in enumerate(zip(axes.flat, batch)):
    ranks, freqs, C, b, a, r2 = fit_data_by_year[year]
    y_hat = zipf_mandelbrot(ranks, C, b, a)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.scatter(ranks, freqs, color="black", s=8, alpha=0.4, label="Data")
    ax.plot(ranks, y_hat, color="orange", linewidth=2, label="Fit")
    ax.set_title(str(year), fontsize=12)
    ax.set_xlabel("Rank", fontsize=16, labelpad=5) if (idx // N_COLS == N_ROWS - 1) else ax.set_xlabel("")
    ax.set_ylabel("Relative frequency", fontsize=16, labelpad=5) if (idx % N_COLS == 0) else ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=10)
    ax.text(
        0.05, 0.95,
        rf"$\alpha$ = {a:.2f}" + "\n" + rf"$\beta$ = {b:.2f}" + "\n" + rf"$R^2$ = {r2:.3f}",
        transform=ax.transAxes,
        fontsize=7,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )

# Hide unused cells
for ax in axes.flat[len(batch):]:
    ax.set_visible(False)

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "figS4_final.svg")
plt.savefig(out_path, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved grid → {out_path}")

print("\nDone.")