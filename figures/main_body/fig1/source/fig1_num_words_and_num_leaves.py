"""
fig1_num_words_and_num_leaves.py

Computes, for each cricket ruleset edition year:

    1. Number of words   -- from the plain-text ruleset files
    2. Number of leaves  -- from the YAML ruleset trees

then plots both series directly on a dual-axis figure (log-scale y-axes).
The data flow is entirely in memory: no intermediate CSV is written.

Usage
-----
    # From the repo root:
    python figures/main_body/fig1/source/fig1_num_words_and_num_leaves.py

Outputs
-------
    figures/main_body/fig1/components/fig1_num_words_and_num_leaves.svg
"""

import os
import re
import shutil

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import yaml

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
YAML_DIR = "./data/datasets/rule_set_structure/yaml_files/flattened"
FIGURES_DIR = "./figures/main_body/fig1/components"

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
# Tree measures
# ---------------------------------------------------------------------------

def load_yaml_ignore_comments(filepath: str):
    """Load a YAML file, silently dropping full-line comment rows."""
    with open(filepath, "r", encoding="utf-8") as fh:
        lines = [line for line in fh if not line.strip().startswith("#")]
    return yaml.safe_load("".join(lines))


def count_leaf_nodes(node) -> int:
    """Recursively count leaf nodes (anything that is not a dict or list)."""
    if isinstance(node, dict):
        return sum(count_leaf_nodes(v) for v in node.values())
    if isinstance(node, list):
        return sum(count_leaf_nodes(item) for item in node)
    return 1


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
    Build a DataFrame indexed by Year with word counts (from .txt files)
    and leaf counts (from .yaml trees).
    """
    records = []
    for year in TARGET_YEARS:
        record = {"Year": year, "Number of Words": np.nan, "Number of Leaves": np.nan}

        txt_path = find_file_for_year(TXT_DIR, year, ".txt")
        if txt_path is not None:
            with open(txt_path, "r", encoding="utf-8") as fh:
                record["Number of Words"] = count_words(fh.read())
        else:
            print(f"Warning: no .txt file found for {year} in {TXT_DIR}")

        yaml_path = find_file_for_year(YAML_DIR, year, ".yaml")
        if yaml_path is not None:
            tree = load_yaml_ignore_comments(yaml_path)
            record["Number of Leaves"] = count_leaf_nodes(tree)
        else:
            print(f"Warning: no .yaml file found for {year} in {YAML_DIR}")

        records.append(record)

    return pd.DataFrame(records).set_index("Year").sort_index()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_words_and_leaves(df: pd.DataFrame) -> None:
    """Plot Number of Words (left axis) and Number of Leaves (right axis),
    both on log-scale y-axes."""
    color_words  = "#9c0412"
    color_leaves = "#4C78A8"

    x        = df.index
    y_words  = df["Number of Words"]
    y_leaves = df["Number of Leaves"]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # --- Number of Words (left axis) ---
    mask1 = y_words.notna().to_numpy()
    ax1.plot(x[mask1], y_words[mask1], color=color_words, linewidth=2.5,
             label="Number of Words")
    ax1.scatter(x[mask1], y_words[mask1], color=color_words, s=40,
                edgecolor="white", linewidth=0.7)
    ax1.set_ylabel("Number of Words", fontsize=20, color=color_words, labelpad=10)
    ax1.tick_params(axis="y", labelcolor=color_words, colors=color_words, labelsize=18)
    ax1.set_yscale("log")

    # --- Number of Leaves (right axis) ---
    ax2 = ax1.twinx()
    mask2 = y_leaves.notna().to_numpy()
    ax2.plot(x[mask2], y_leaves[mask2], color=color_leaves, linewidth=2.5,
             label="Number of Leaves")
    ax2.scatter(x[mask2], y_leaves[mask2], color=color_leaves, s=40,
                edgecolor="white", linewidth=0.7)
    ax2.set_ylabel("Number of Leaves", fontsize=20, color=color_leaves, labelpad=10)
    ax2.tick_params(axis="y", labelcolor=color_leaves, colors=color_leaves, labelsize=18)
    ax2.set_yscale("log")

    # --- Formatting ---
    ax1.set_title("Number of Words and Number of Leaves (log scale) over Year",
                  fontsize=24, pad=12)
    ax1.set_xlabel("Year", fontsize=20, labelpad=10)
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=10))
    ax1.tick_params(axis="x", labelsize=18)
    ax1.grid(False)

    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=16, frameon=False)

    fig.tight_layout()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = os.path.join(FIGURES_DIR, "fig1_num_words_and_num_leaves.svg")
    plt.savefig(out_path)
    print(f"Saved: {out_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    df = build_dataframe()
    print(df.round(4).to_string())
    plot_words_and_leaves(df)


if __name__ == "__main__":
    main()