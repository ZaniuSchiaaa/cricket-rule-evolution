"""
fig6_e_ave_depth_over_matches.py

Plot the average depth of each cricket ruleset tree against the cumulative
number of matches played.

Usage
-----
    # From the repo root:
    python figures/main_body/fig6/source/fig6_e_ave_depth_over_matches.py

Outputs
-------
    ./figures/main_body/fig6/components/fig6_e_ave_depth_over_matches.svg
"""


# ===========
# [0] IMPORTS
# ===========

import os
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ===============
# [1] MATPLOTLIB / LaTeX CONFIG
# ===============

USE_TEX = shutil.which("latex") is not None

plt.rcParams["text.usetex"] = USE_TEX
plt.rcParams["font.family"] = "serif"
if USE_TEX:
    plt.rcParams["font.serif"] = ["Computer Modern Roman"]
else:
    plt.rcParams["mathtext.fontset"] = "cm"
    print("Note: LaTeX not found on PATH; falling back to matplotlib's mathtext.")


# ==============
# [2] CONSTANTS
# ==============

YEARS = [
    1752, 1755, 1774, 1785, 1786, 1788, 1803, 1806, 1809,
    1816, 1820, 1823, 1828, 1830, 1835, 1857, 1884, 1890,
    1892, 1896, 1900, 1902, 1906, 1908, 1910, 1911, 1913,
    1914, 1918, 1920, 1923, 1932, 1939, 1947, 1952, 1962,
    1968, 1980, 1992, 2000, 2008, 2010, 2017, 2019,
]

YAML_DIR = "./data/datasets/rule_set_structure/yaml_files/flattened"
MATCHES_CSV = "./data/datasets/num_matches/cumulative_matches_played.csv"

VIS_FILEPATH = "./figures/main_body/fig6/components/fig6_e_ave_depth_over_matches.svg"

INDENT_UNIT = 2  # spaces per nesting level in the flattened YAML files

COLUMN = "Average Depth"
COLOR = "#7F7F7F"


# ===================
# [3] DEPTH UTILITIES
# ===================

def line_depth(line):
    """Return the nesting depth of a single YAML line from its leading indentation."""
    stripped = line.lstrip(" ")
    leading_spaces = len(line) - len(stripped)
    return leading_spaces // INDENT_UNIT


def compute_average_depth(yaml_text):
    """
    Compute the average node depth of a YAML tree from its raw text.

    Each non-blank, non-comment line is treated as one node; its depth is
    inferred from leading indentation. Returns the mean depth across all nodes.
    """
    depth_counts = {}  # depth -> number of nodes at that depth

    for line in yaml_text.splitlines():
        stripped = line.strip()
        if stripped == "" or stripped == "---" or stripped.startswith("#"):
            continue
        depth = line_depth(line)
        depth_counts[depth] = depth_counts.get(depth, 0) + 1

    num_nodes = sum(depth_counts.values())
    if num_nodes == 0:
        return 0.0

    sum_product = sum(depth * count for depth, count in depth_counts.items())
    return sum_product / num_nodes


def load_yaml_text_ignore_comments(filepath):
    """Read a YAML file as raw text, dropping full-line comments."""
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line for line in f if not line.strip().startswith("#")]
    return "".join(lines)


# =================
# [4] DATA LOADING
# =================

def build_depth_dataframe(yaml_dir):
    """Build a DataFrame of average depth per year from flattened YAML trees."""
    df = pd.DataFrame(index=YEARS)
    df.index.name = "Year"

    for file in os.listdir(yaml_dir):
        if not file.endswith(".yaml"):
            continue
        year = int(file[0:4])
        yaml_text = load_yaml_text_ignore_comments(os.path.join(yaml_dir, file))
        df.loc[year, COLUMN] = compute_average_depth(yaml_text)

    return df


def merge_with_matches(df, matches_csv):
    """Join the average-depth data with cumulative-matches data, indexed by year."""
    match_df = pd.read_csv(matches_csv, index_col="Year")
    merged_df = df.join(match_df, how="left")

    for col_name in match_df.columns:
        merged_df[col_name] = merged_df[col_name].astype("Int64")

    return merged_df.sort_values("Year")


# ==============
# [5] PLOTTING
# ==============

def create_square_ax(fig_size=6, axes_fraction=0.7):
    """Create a figure with a single square axis."""
    fig = plt.figure(figsize=(fig_size, fig_size))
    ax = fig.add_axes([0.15, 0.15, axes_fraction, axes_fraction])
    ax.set_box_aspect(1)
    return fig, ax


def plot_depth(merged_df, vis_filepath):
    """Scatter the average depth against cumulative matches played (log x-axis)."""
    fig, ax = create_square_ax(fig_size=8, axes_fraction=0.75)

    x = merged_df["Cumulative Matches Played"]
    y = merged_df[COLUMN]

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"$10^{{{int(np.log10(v))}}}$")
    )

    ax.scatter(x, y, color=COLOR)

    ax.set_xlabel("Cumulative matches played", fontsize=30, labelpad=10)
    ax.set_ylabel("Average depth", fontsize=30, labelpad=10)
    ax.grid(False)
    ax.tick_params(axis="both", labelsize=18)
    ax.set_position([0.15, 0.2, 0.7, 0.65])

    os.makedirs(os.path.dirname(vis_filepath), exist_ok=True)
    fig.savefig(vis_filepath, dpi=300)
    plt.close(fig)
    print(f"Saved figure to {vis_filepath}")


# ==========
# [6] MAIN
# ==========

def main():
    df = build_depth_dataframe(YAML_DIR)
    merged_df = merge_with_matches(df, MATCHES_CSV)
    plot_depth(merged_df, VIS_FILEPATH)


if __name__ == "__main__":
    main()