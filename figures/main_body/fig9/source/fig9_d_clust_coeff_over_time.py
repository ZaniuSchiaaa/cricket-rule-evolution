"""
fig9_d_clust_coeff_over_time.py

Plots three clustering-coefficient measures of the multi-count interdependency
networks over time: global clustering (transitivity, undirected/unweighted),
average local clustering (directed, unweighted), and average local clustering
(directed, weighted).

Usage
-----
    # From the repo root:
    python figures/main_body/fig9/source/fig9_d_clust_coeff_over_time.py

Outputs
-------
    ./figures/main_body/fig9/components/fig9_d_clust_coeff_over_time.svg
"""

# =============================================================================
# [0] IMPORTS
# =============================================================================
import os
import re
import glob
import shutil

import numpy as np
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt

# =============================================================================
# [1] MATPLOTLIB / LaTeX CONFIG
# =============================================================================
if shutil.which("latex"):
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Computer Modern Roman"]
else:
    plt.rcParams["text.usetex"] = False
    matplotlib.rcParams["mathtext.fontset"] = "cm"

# =============================================================================
# [2] CONSTANTS
# =============================================================================
GRAPH_DIR = "./data/datasets/interdependency_networks/graph_files/gexf/multi_count"

OUTPUT_DIR = "./figures/main_body/fig9/components"
OUTPUT_NAME = "fig9_d_clust_coeff_over_time.svg"

WEIGHT_ATTR = "weight"

# Aesthetic constants
FIG_SIZE = 8
AXES_FRACTION = 0.75
GLOBAL_CC_COLOR = "#228B22"
LOCAL_UNWEIGHTED_COLOR = "#4C78A8"
LOCAL_WEIGHTED_COLOR = "#9c0412"
LABEL_FONTSIZE = 30
TICK_LABELSIZE = 20
LEGEND_FONTSIZE = 16
LABEL_PAD = 10
DPI = 300


# =============================================================================
# CLUSTERING ANALYSIS
# =============================================================================
def compute_clustering(G):
    """
    Compute the three clustering measures for a single graph.

    Returns a dict:
        - avg_local_cc_unweighted_directed
        - avg_local_cc_weighted_directed
        - global_cc_transitivity
    """
    # (1) Local clustering coefficients — directed, unweighted vs. weighted
    local_cc_unweighted = nx.clustering(G, weight=None)
    local_cc_weighted = nx.clustering(G, weight=WEIGHT_ATTR)

    # (2) Average local clustering coefficients
    avg_local_unweighted = (
        float(np.mean(list(local_cc_unweighted.values()))) if local_cc_unweighted else np.nan
    )
    avg_local_weighted = (
        float(np.mean(list(local_cc_weighted.values()))) if local_cc_weighted else np.nan
    )

    # (3) Global clustering (transitivity). nx.transitivity treats the graph as
    #     undirected & unweighted internally; this is the standard global baseline.
    global_cc = nx.transitivity(G)

    return {
        "avg_local_cc_unweighted_directed": avg_local_unweighted,
        "avg_local_cc_weighted_directed": avg_local_weighted,
        "global_cc_transitivity": global_cc,
    }


# =============================================================================
# GEXF LOADING
# =============================================================================
def extract_year(filename):
    """Pull the first 4-digit year from a filename; None if absent."""
    match = re.search(r"(\d{4})", os.path.basename(filename))
    return int(match.group(1)) if match else None


def load_graphs_by_year(graph_dir):
    """
    Read all GEXF files in graph_dir, returning (year, DiGraph) sorted by year.
    Coerces edge weights to float; skips files with no extractable year.
    """
    paths = sorted(glob.glob(os.path.join(graph_dir, "*.gexf")))
    if not paths:
        raise FileNotFoundError(f"No .gexf files found in {graph_dir}")

    graphs = []
    for path in paths:
        year = extract_year(path)
        if year is None:
            print(f"Skipping (no year found): {path}")
            continue

        G = nx.read_gexf(path)
        if not G.is_directed():
            G = G.to_directed()

        for u, v, data in G.edges(data=True):
            data[WEIGHT_ATTR] = float(data.get(WEIGHT_ATTR, 1))

        graphs.append((year, G))

    graphs.sort(key=lambda t: t[0])
    return graphs


# =============================================================================
# PLOTTING
# =============================================================================
def create_square_ax(fig_size=6, axes_fraction=0.7):
    fig = plt.figure(figsize=(fig_size, fig_size))
    ax = fig.add_axes([0.15, 0.15, axes_fraction, axes_fraction])
    ax.set_box_aspect(1)
    return fig, ax


def plot_time_series(years, global_cc, local_unweighted, local_weighted,
                     output_dir, output_name):
    fig, ax = create_square_ax(fig_size=FIG_SIZE, axes_fraction=AXES_FRACTION)

    years = np.asarray(years, dtype=int)

    ax.plot(years, global_cc,
            linestyle="--", marker="s",
            label="Global CC (transitivity, undirected)",
            color=GLOBAL_CC_COLOR)
    ax.plot(years, local_unweighted,
            marker="o",
            label="Average local CC (directed, unweighted)",
            color=LOCAL_UNWEIGHTED_COLOR)
    ax.plot(years, local_weighted,
            marker="o",
            label="Average local CC (directed, weighted)",
            color=LOCAL_WEIGHTED_COLOR)

    ax.set_xlabel("Year", fontsize=LABEL_FONTSIZE, labelpad=LABEL_PAD)
    ax.set_ylabel("Clustering coefficient", fontsize=LABEL_FONTSIZE, labelpad=LABEL_PAD)
    ax.legend(fontsize=LEGEND_FONTSIZE)
    ax.grid(False)
    ax.tick_params(axis="x", labelsize=TICK_LABELSIZE)
    ax.tick_params(axis="y", labelsize=TICK_LABELSIZE)

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, output_name), dpi=DPI)
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================
def main():
    graphs = load_graphs_by_year(GRAPH_DIR)

    years, global_cc, local_unweighted, local_weighted = [], [], [], []
    for year, G in graphs:
        stats = compute_clustering(G)
        years.append(year)
        global_cc.append(stats["global_cc_transitivity"])
        local_unweighted.append(stats["avg_local_cc_unweighted_directed"])
        local_weighted.append(stats["avg_local_cc_weighted_directed"])
        print(f"{year}: "
              f"avg_local_unweighted={stats['avg_local_cc_unweighted_directed']:.4f}, "
              f"avg_local_weighted={stats['avg_local_cc_weighted_directed']:.4f}, "
              f"global_cc={stats['global_cc_transitivity']:.4f}")

    plot_time_series(years, global_cc, local_unweighted, local_weighted,
                     OUTPUT_DIR, OUTPUT_NAME)
    print(f"\nSaved figure to {os.path.join(OUTPUT_DIR, OUTPUT_NAME)}")


if __name__ == "__main__":
    main()