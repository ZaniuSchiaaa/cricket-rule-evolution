"""
fig1_num_laws_and_ave_depth.py

Loads YAML cricket ruleset trees, computes the number of laws and average
depth for each year, and plots both series on a dual-axis figure.

Usage
-----
    # From the repo root:
    python figures/main_body/fig1/source/fig1_num_laws_and_ave_depth.py

Outputs
-------
    <FIGURES_DIR>/fig1_num_laws_and_ave_depth.svg
"""

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import yaml

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Computer Modern Roman"]

# ---------------------------------------------------------------------------
# Paths – edit these to match your local directory layout
# ---------------------------------------------------------------------------
YAML_DIR = (
    "/Users/Chia Jia Nuo Daniel Personal Folder/After Touchdown/"
    "Santa Fe Institute/Rules Project Materials/Code and Data/Cricket/"
    "ANALYSIS/DATASET FOR ANALYSIS/yaml-rule-trees"
)
FIGURES_DIR = (
    "/Users/Chia Jia Nuo Daniel Personal Folder/After Touchdown/"
    "Santa Fe Institute/Rules Project Materials/Github Repo/paper/figures/"
    "main_body/fig1/components"
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
TREE_TYPES    = ["flattened"]
SUBTREE_TYPES = ["full-ruleset"]

# Number of laws entered manually from inspecting the trees
NUM_LAWS_DICT = {
    1752: 33, 1755: 35, 1774: 34, 1785: 33, 1786: 34, 1788: 37,
    1803: 40, 1806: 45, 1809: 41, 1816: 44, 1820: 44, 1823: 43,
    1828: 43, 1830: 47, 1835: 47, 1857: 47, 1884: 54, 1890: 54,
    1892: 54, 1896: 54, 1900: 54, 1902: 54, 1906: 54, 1908: 54,
    1910: 54, 1911: 54, 1913: 54, 1914: 55, 1918: 55, 1920: 55,
    1923: 55, 1932: 55, 1939: 55, 1947: 47, 1952: 47, 1962: 47,
    1968: 47, 1980: 42, 1992: 42, 2000: 42, 2008: 42, 2010: 42,
    2017: 42, 2019: 42,
}

# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

def load_yaml_ignore_comments(filepath: str):
    """Load a YAML file, silently dropping full-line comment rows."""
    with open(filepath, "r", encoding="utf-8") as fh:
        lines = [line for line in fh if not line.strip().startswith("#")]
    return yaml.safe_load("".join(lines))


# ---------------------------------------------------------------------------
# Average-depth computation (operates on the raw YAML string)
# ---------------------------------------------------------------------------

def _indent_depth(line: str) -> int:
    """Return the indentation depth of a YAML line (2 spaces = 1 level)."""
    return (len(line) - len(line.lstrip())) // 2


def compute_average_depth(yaml_str: str) -> float:
    """
    Compute the average depth of all nodes, weighted by the number of nodes
    at each depth level.
    """
    depth_counts: dict[int, int] = {}
    for line in yaml_str.splitlines():
        stripped = line.strip()
        if not stripped or stripped == "---" or stripped.startswith("#"):
            continue
        depth = _indent_depth(line)
        depth_counts[depth] = depth_counts.get(depth, 0) + 1

    total_nodes  = sum(depth_counts.values())
    weighted_sum = sum(depth * count for depth, count in depth_counts.items())
    return weighted_sum / total_nodes if total_nodes else 0.0


# ---------------------------------------------------------------------------
# Per-subtree data pipeline
# ---------------------------------------------------------------------------

def build_dataframe(tree_type: str, subtree_type: str) -> pd.DataFrame:
    """
    Walk the YAML files for one (tree_type, subtree_type) combination and
    return a DataFrame with columns ['Number of Laws', 'Average Depth'],
    indexed by Year.
    """
    subtree_path = os.path.join(YAML_DIR, tree_type, subtree_type)

    records = []
    for filename in sorted(os.listdir(subtree_path)):
        if not filename.endswith(".yaml"):
            continue

        year = int(filename[:4])
        filepath = os.path.join(subtree_path, filename)

        with open(filepath, "r", encoding="utf-8") as fh:
            yaml_str = fh.read()

        records.append({
            "Year":           year,
            "Number of Laws": NUM_LAWS_DICT.get(year, np.nan),
            "Average Depth":  compute_average_depth(yaml_str),
        })

    df = pd.DataFrame(records).set_index("Year").sort_index()
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_laws_and_depth(df: pd.DataFrame) -> None:
    """Plot Number of Laws (left axis) and Average Depth (right axis)."""
    color_laws  = "#9c0412"
    color_depth = "#4C78A8"

    x       = df.index
    y_laws  = df["Number of Laws"]
    y_depth = df["Average Depth"]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # --- Number of Laws (left axis) ---
    mask1 = y_laws.notna()
    ax1.plot(x[mask1], y_laws[mask1], color=color_laws, linewidth=2.5, label="Number of Laws")
    ax1.scatter(x[mask1], y_laws[mask1], color=color_laws, s=40, edgecolor="white", linewidth=0.7)
    ax1.set_ylabel("Number of Laws", fontsize=20, color=color_laws, labelpad=10)
    ax1.tick_params(axis="y", labelcolor=color_laws, labelsize=18)

    # --- Average Depth (right axis) ---
    ax2 = ax1.twinx()
    mask2 = y_depth.notna()
    ax2.plot(x[mask2], y_depth[mask2], color=color_depth, linewidth=2.5, label="Average Depth")
    ax2.scatter(x[mask2], y_depth[mask2], color=color_depth, s=40, edgecolor="white", linewidth=0.7)
    ax2.set_ylabel("Average Depth", fontsize=20, color=color_depth, labelpad=10)
    ax2.tick_params(axis="y", labelcolor=color_depth, labelsize=18)

    # --- Formatting ---
    ax1.set_title("Number of Laws and Average Depth over Year", fontsize=24, pad=12)
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
    out_path = os.path.join(FIGURES_DIR, "fig1_num_laws_and_ave_depth.svg")
    plt.savefig(out_path)
    print(f"Saved: {out_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    for tree_type in TREE_TYPES:
        for subtree_type in SUBTREE_TYPES:
            df = build_dataframe(tree_type, subtree_type)
            plot_laws_and_depth(df)


if __name__ == "__main__":
    main()
