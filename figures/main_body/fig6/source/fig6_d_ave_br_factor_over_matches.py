"""
fig6_d_ave_br_factor_over_matches.py

Plot the average branching factor of each cricket ruleset tree against the
cumulative number of matches played.

Usage
-----
    # From the repo root:
    python figures/main_body/fig6/source/fig6_d_ave_br_factor_over_matches.py

Outputs
-------
    ./figures/main_body/fig6/components/fig6_d_ave_br_factor_over_matches.svg
"""


# ===========
# [0] IMPORTS
# ===========

import os
import shutil

import numpy as np
import matplotlib.ticker as mticker
import pandas as pd
import yaml
import matplotlib.pyplot as plt


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

VIS_FILEPATH = "./figures/main_body/fig6/components/fig6_d_ave_br_factor_over_matches.svg"

BR_FACTOR_COLUMN = "Average Branching Factor"
BR_FACTOR_COLOR = "#7f7f7f"


# ===================
# [3] YAML UTILITIES
# ===================

def load_yaml_ignore_comments(filepath):
    """Load a YAML file while ignoring full-line comments."""
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line for line in f if not line.strip().startswith("#")]
    return yaml.safe_load("".join(lines))


def analyze_tree(node):
    """
    Recursively analyze the tree.

    Returns a (total_nodes, leaf_nodes, branching_factors) tuple, where
    branching_factors collects the number of children at each list (branch) node.
    """
    if isinstance(node, dict):
        total_nodes = 1
        leaf_nodes = 0
        branching_factors = []

        for value in node.values():
            child_total, child_leaves, child_bf = analyze_tree(value)
            total_nodes += child_total
            leaf_nodes += child_leaves
            branching_factors.extend(child_bf)

        return (total_nodes, leaf_nodes, branching_factors)

    elif isinstance(node, list):
        total_nodes = 0
        leaf_nodes = 0
        branching_factors = [len(node)]  # how many children this node branches into

        for item in node:
            child_total, child_leaves, child_bf = analyze_tree(item)
            total_nodes += child_total
            leaf_nodes += child_leaves
            branching_factors.extend(child_bf)

        return (total_nodes, leaf_nodes, branching_factors)

    else:  # leaf
        return (1, 1, [])


def compute_ave_branching_factor(yaml_as_nest):
    """Return the mean branching factor for a parsed YAML tree (0 if none)."""
    _, _, branching_factors = analyze_tree(yaml_as_nest)
    return np.mean(branching_factors) if branching_factors else 0.0


# =================
# [4] DATA LOADING
# =================

def build_branching_dataframe(yaml_dir):
    """Build a DataFrame of average branching factor per year from flattened YAML trees."""
    df = pd.DataFrame(index=YEARS)
    df.index.name = "Year"

    for file in os.listdir(yaml_dir):
        if not file.endswith(".yaml"):
            continue
        year = int(file[0:4])
        yaml_as_nest = load_yaml_ignore_comments(os.path.join(yaml_dir, file))
        df.loc[year, BR_FACTOR_COLUMN] = compute_ave_branching_factor(yaml_as_nest)

    return df


def merge_with_matches(df, matches_csv):
    """Join the branching-factor data with cumulative-matches data, indexed by year."""
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


def plot_branching_factor(merged_df, vis_filepath):
    """Scatter the average branching factor against cumulative matches played."""
    fig, ax = create_square_ax(fig_size=8, axes_fraction=0.75)

    x = merged_df["Cumulative Matches Played"]
    y = merged_df[BR_FACTOR_COLUMN]

    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"$10^{{{int(np.log10(y))}}}$"))
    
    ax.scatter(x, y, color=BR_FACTOR_COLOR)

    ax.set_xlabel("Cumulative matches played", fontsize=30, labelpad=10)
    ax.set_ylabel("Average branching factor", fontsize=30, labelpad=10)
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
    df = build_branching_dataframe(YAML_DIR)
    merged_df = merge_with_matches(df, MATCHES_CSV)
    plot_branching_factor(merged_df, VIS_FILEPATH)


if __name__ == "__main__":
    main()