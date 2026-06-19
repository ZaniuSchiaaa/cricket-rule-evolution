"""
fig6_b_nodes_and_leaves_over_matches.py

Plot the number of nodes and number of leaf nodes in each cricket ruleset tree
against the cumulative number of matches played, on log-log axes, with OLS fits.

Usage
-----
    # From the repo root:
    python figures/main_body/fig6/source/fig6_b_nodes_and_leaves_over_matches.py

Outputs
-------
    ./figures/main_body/fig6/components/fig6_b_nodes_and_leaves_over_matches.svg
"""


# ===========
# [0] IMPORTS
# ===========

import os
import shutil

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import linregress


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

VIS_FILEPATH = "./figures/main_body/fig6/components/fig6_b_nodes_and_leaves_over_matches.svg"

COLUMNS = ["Number of Nodes", "Number of Leaves"]
COLORS = {
    "Number of Nodes": "#4C78A8",
    "Number of Leaves": "#9C0412",
}


# ===================
# [3] YAML UTILITIES
# ===================

def load_yaml_ignore_comments(filepath):
    """Load a YAML file while ignoring full-line comments."""
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line for line in f if not line.strip().startswith("#")]
    return yaml.safe_load("".join(lines))


def count_nodes(data):
    """Recursively count all nodes (internal + leaf) in a nested YAML structure."""
    if isinstance(data, dict):
        return len(data) + sum(count_nodes(v) for v in data.values())
    elif isinstance(data, list):
        return sum(count_nodes(item) for item in data)
    else:
        return 1  # Leaf node


def count_leaf_nodes(data):
    """Recursively count leaf nodes (non-dict, non-list values)."""
    if isinstance(data, dict):
        return sum(count_leaf_nodes(v) for v in data.values())
    elif isinstance(data, list):
        return sum(count_leaf_nodes(item) for item in data)
    else:
        return 1  # Leaf node


# =================
# [4] DATA LOADING
# =================

def build_node_dataframe(yaml_dir):
    """Build a DataFrame of node and leaf counts per year from flattened YAML trees."""
    df = pd.DataFrame(index=YEARS)
    df.index.name = "Year"

    for file in os.listdir(yaml_dir):
        if not file.endswith(".yaml"):
            continue
        year = int(file[0:4])
        yaml_as_nest = load_yaml_ignore_comments(os.path.join(yaml_dir, file))
        df.loc[year, "Number of Nodes"] = count_nodes(yaml_as_nest)
        df.loc[year, "Number of Leaves"] = count_leaf_nodes(yaml_as_nest)

    return df


def merge_with_matches(df, matches_csv):
    """Join node-count data with cumulative-matches data, indexed by year."""
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


def plot_nodes_and_leaves(merged_df, vis_filepath):
    """Produce a log-log scatter of node/leaf counts vs cumulative matches with OLS fits."""
    fig, ax = create_square_ax(fig_size=8, axes_fraction=0.75)

    x = merged_df["Cumulative Matches Played"]

    for col in COLUMNS:
        y = merged_df[col]

        mask = x.notna() & y.notna() & (x > 0) & (y > 0)
        x_clean = x[mask]
        y_clean = y[mask]

        log_x = np.log10(x_clean)
        log_y = np.log10(y_clean)

        slope, intercept, r_value, p_value, std_err = linregress(log_x, log_y)
        y_pred = 10 ** (intercept + slope * log_x)

        print(f"{col}: raw={len(x)}, after mask={len(x_clean)}, slope={slope:.3f}")

        ax.scatter(x_clean, y_clean, color=COLORS[col], alpha=0.7)

        label = col.replace("Number of ", "")
        ax.plot(
            x_clean, y_pred,
            color=COLORS[col],
            linewidth=2.5,
            label=f"{label} OLS: slope={slope:.3f}",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"$10^{{{int(np.log10(v))}}}$")
    )
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"$10^{{{int(np.log10(v))}}}$")
    )

    ax.set_xlabel("Cumulative matches played", fontsize=30, labelpad=10)
    ax.set_ylabel("Count", fontsize=30, labelpad=10)
    ax.grid(False)
    ax.tick_params(axis="both", labelsize=18)
    ax.legend(fontsize=20, loc="upper left")
    ax.set_position([0.15, 0.2, 0.7, 0.65])

    os.makedirs(os.path.dirname(vis_filepath), exist_ok=True)
    fig.savefig(vis_filepath, dpi=300)
    plt.close(fig)
    print(f"Saved figure to {vis_filepath}")


# ==========
# [6] MAIN
# ==========

def main():
    df = build_node_dataframe(YAML_DIR)
    df["Log Number of Nodes"] = np.log10(df["Number of Nodes"].replace(0, np.nan))
    df["Log Number of Leaves"] = np.log10(df["Number of Leaves"].replace(0, np.nan))

    merged_df = merge_with_matches(df, MATCHES_CSV)
    plot_nodes_and_leaves(merged_df, VIS_FILEPATH)


if __name__ == "__main__":
    main()