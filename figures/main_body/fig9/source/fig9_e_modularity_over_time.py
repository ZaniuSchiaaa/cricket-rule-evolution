"""
fig9_e_modularity_over_time.py

Compute and plot network modularity (Q) over time, based on the greedy-modularity community partition.

Usage
-----
    # From the repo root:
    python figures/main_body/fig9/source/fig9_e_modularity_over_time.py

Outputs
-------
    ./figures/main_body/fig9/components/fig9_e_modularity_over_time.svg
"""

# [0] IMPORTS

import os
import re
import glob
import shutil

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt


# [1] MATPLOTLIB / LaTeX CONFIG

USE_TEX = shutil.which("latex") is not None
plt.rcParams["text.usetex"] = USE_TEX
plt.rcParams["font.family"] = "serif"
if USE_TEX:
    plt.rcParams["font.serif"] = ["Computer Modern Roman"]
else:
    plt.rcParams["mathtext.fontset"] = "cm"


# [2] CONSTANTS

GEXF_DIR = (
    "/Users/Chia Jia Nuo Daniel Personal Folder/After Touchdown/"
    "Santa Fe Institute/Rules Project Materials/repo/data/datasets/"
    "interdependency_networks/graph_files/gexf/multi_count"
)

OUTPUT_DIR = "./figures/main_body/fig9/components"
MODULARITY_SVG = os.path.join(OUTPUT_DIR, "fig9_e_modularity_over_time.svg")

WEIGHT_KEY = "weight"
REMOVE_SELF_EDGES = True

SCATTER_COLOR = "black"
LABEL_FONTSIZE = 30
TICK_LABELSIZE = 20


# [3] FUNCTIONS

def create_square_ax(fig_size=8, axes_fraction=0.75):
    fig = plt.figure(figsize=(fig_size, fig_size))
    ax = fig.add_axes([0.15, 0.15, axes_fraction, axes_fraction])
    ax.set_box_aspect(1)
    return fig, ax


def load_graphs(gexf_dir, remove_self_edges):
    """Load all GEXF files in gexf_dir, keyed by the year parsed from each filename."""
    graphs = {}
    for path in sorted(glob.glob(os.path.join(gexf_dir, "*.gexf"))):
        match = re.search(r"(\d{4})", os.path.basename(path))
        if match is None:
            print(f"Skipping (no year in filename): {path}")
            continue
        year = int(match.group(1))

        G = nx.read_gexf(path)
        if remove_self_edges:
            G.remove_edges_from(nx.selfloop_edges(G))
        graphs[year] = G
    return graphs


def network_modularity(G, weight):
    G_undirected = G.to_undirected()
    communities = nx.community.greedy_modularity_communities(G_undirected, weight=weight)
    Q = nx.community.modularity(G_undirected, communities, weight=weight)
    return Q


def compute_modularity_series(graphs, weight):
    """Return a modularity dict keyed by year."""
    modularity_series = {}
    for year in sorted(graphs):
        Q = network_modularity(graphs[year], weight)
        modularity_series[year] = Q
        print(f"{year}: Q = {Q:.4f}")
    return modularity_series


def plot_series(series, ylabel, out_path):
    df = pd.Series(series).sort_index()

    fig, ax = create_square_ax(fig_size=8, axes_fraction=0.75)
    ax.scatter(df.index, df.values, marker="o", color=SCATTER_COLOR)
    ax.set_xlabel("Year", fontsize=LABEL_FONTSIZE, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE, labelpad=10)
    ax.tick_params(axis="x", labelsize=TICK_LABELSIZE)
    ax.tick_params(axis="y", labelsize=TICK_LABELSIZE)

    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# [4] MAIN

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    graphs = load_graphs(GEXF_DIR, REMOVE_SELF_EDGES)
    if not graphs:
        raise FileNotFoundError(f"No GEXF files found in: {GEXF_DIR}")

    modularity_series = compute_modularity_series(graphs, WEIGHT_KEY)
    plot_series(modularity_series, "Modularity (Q)", MODULARITY_SVG)


if __name__ == "__main__":
    main()