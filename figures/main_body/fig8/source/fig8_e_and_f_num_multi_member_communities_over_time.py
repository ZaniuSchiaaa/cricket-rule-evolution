"""
fig8_e_and_f_num_multi_member_communities_over_time.py

Robustness check for fig8_f: compute and plot over time:

1. The number of communities based on the greedy-modularity community
   partition, excluding singleton communities.
2. The number of Laws belonging to multi-member communities
   (i.e. communities of size >= 2).

Usage
-----

    # From the repo root:

    python figures/main_body/fig8/source/fig8_e_and_f_num_multi_member_communities_over_time.py

Outputs
-------

    ./figures/main_body/fig8/components/fig8_e_num_multi_member_communities_over_time.svg
    ./figures/main_body/fig8/components/fig8_f_num_laws_in_multi_member_communities_over_time.svg

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
    "./data/datasets/"
    "interdependency_networks/graph_files/gexf/multi_count"
)

OUTPUT_DIR = "./figures/main_body/fig8/components"

NUM_COMM_SVG = os.path.join(
    OUTPUT_DIR, "fig8_e_num_multi_member_communities_over_time.svg"
)

NUM_LAWS_IN_COMM_SVG = os.path.join(
    OUTPUT_DIR, "fig8_f_num_laws_in_multi_member_communities_over_time.svg"
)

WEIGHT_KEY = "weight"
REMOVE_SELF_EDGES = True

MIN_COMMUNITY_SIZE = 2
# Communities smaller than this (i.e. singletons) are not counted.

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


def get_communities(G, weight):
    """
    Convert the directed graph to an undirected weighted graph and
    calculate greedy-modularity communities.

    If both A->B and B->A exist, their weights are summed into the
    undirected A--B edge.
    """

    G_undirected = nx.Graph()

    G_undirected.add_nodes_from(G.nodes(data=True))

    for u, v, d in G.edges(data=True):

        if G_undirected.has_edge(u, v):
            G_undirected[u][v][weight] += d[weight]

        else:
            G_undirected.add_edge(
                u,
                v,
                **{weight: d[weight]}
            )

    communities = nx.community.greedy_modularity_communities(
        G_undirected,
        weight=weight
    )

    return communities


def community_statistics(G, weight, min_size=MIN_COMMUNITY_SIZE):
    """
    Return:

    1. Number of multi-member communities.
    2. Number of Laws belonging to multi-member communities.

    Singleton communities are excluded from both quantities.
    """

    communities = get_communities(G, weight)

    multi_member_communities = [
        c for c in communities
        if len(c) >= min_size
    ]

    num_communities = len(multi_member_communities)

    num_laws_in_communities = sum(
        len(c) for c in multi_member_communities
    )

    return num_communities, num_laws_in_communities


def compute_community_series(graphs, weight):
    """
    Return two dictionaries keyed by year:

    - number of multi-member communities
    - number of Laws in multi-member communities
    """

    num_communities_series = {}
    num_laws_in_communities_series = {}

    for year in sorted(graphs):

        num_communities, num_laws_in_communities = (
            community_statistics(graphs[year], weight)
        )

        num_communities_series[year] = num_communities
        num_laws_in_communities_series[year] = num_laws_in_communities

        print(
            f"{year}: "
            f"{num_communities} communities, "
            f"{num_laws_in_communities} Laws in multi-member communities"
        )

    return (
        num_communities_series,
        num_laws_in_communities_series
    )


def plot_series(series, ylabel, out_path):
    """Plot a time series as a scatter plot."""

    df = pd.Series(series).sort_index()

    fig, ax = create_square_ax(
        fig_size=8,
        axes_fraction=0.75
    )

    ax.scatter(
        df.index,
        df.values,
        marker="o",
        color=SCATTER_COLOR
    )

    ax.set_xlabel(
        "Year",
        fontsize=LABEL_FONTSIZE,
        labelpad=10
    )

    ax.set_ylabel(
        ylabel,
        fontsize=LABEL_FONTSIZE,
        labelpad=10
    )

    ax.tick_params(
        axis="x",
        labelsize=TICK_LABELSIZE
    )

    ax.tick_params(
        axis="y",
        labelsize=TICK_LABELSIZE
    )

    fig.savefig(
        out_path,
        dpi=300
    )

    plt.close(fig)


# [4] MAIN

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    graphs = load_graphs(
        GEXF_DIR,
        REMOVE_SELF_EDGES
    )

    if not graphs:
        raise FileNotFoundError(
            f"No GEXF files found in: {GEXF_DIR}"
        )

    if USE_TEX:
        num_comm_label = r"\# of multi-member communities"
        num_laws_label = r"\# of Laws in multi-member communities"
    else:
        num_comm_label = "# of multi-member communities"
        num_laws_label = "# of Laws in multi-member communities"

    (
        num_communities_series,
        num_laws_in_communities_series
    ) = compute_community_series(
        graphs,
        WEIGHT_KEY
    )

    # Plot 1: number of multi-member communities
    plot_series(
        num_communities_series,
        num_comm_label,
        NUM_COMM_SVG
    )

    # Plot 2: number of Laws belonging to multi-member communities
    plot_series(
        num_laws_in_communities_series,
        num_laws_label,
        NUM_LAWS_IN_COMM_SVG
    )


if __name__ == "__main__":
    main()