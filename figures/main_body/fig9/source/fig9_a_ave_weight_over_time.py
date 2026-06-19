"""
fig9_a_ave_weight_over_time.py

Plot the average edge weight of dyads (over all possible dyads) over time.

Usage
-----
    # From the repo root:
    python figures/main_body/fig9/source/fig9_a_ave_weight_over_time.py

Outputs
-------
    ./figures/main_body/fig9/components/fig9_a_ave_weight_over_time.svg
"""

import os
import shutil

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
YEARS = [
    1835, 1857, 1884, 1890,
    1892, 1896, 1900, 1902, 1906, 1908, 1910, 1911, 1913,
    1914, 1918, 1920, 1923, 1932, 1939, 1947, 1952, 1962,
    1968, 1980, 1992, 2000, 2008, 2010, 2017, 2019,
]

GEXF_FOLDER = "./data/datasets/interdependency_networks/graph_files/gexf/multi_count"
OUTPUT_PATH = "./figures/main_body/fig9/components/fig9_a_ave_weight_over_time.svg"

USE_TEX = shutil.which("latex") is not None
plt.rcParams["text.usetex"] = USE_TEX
plt.rcParams["font.family"] = "serif"
if USE_TEX:
    plt.rcParams["font.serif"] = ["Computer Modern Roman"]
else:
    plt.rcParams["mathtext.fontset"] = "cm"
    print("Note: LaTeX not found on PATH; falling back to matplotlib's mathtext.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def create_graph(year):
    """Load the multi-count graph for `year`, keeping only numeric nodes."""
    G = nx.read_gexf(os.path.join(GEXF_FOLDER, f"{year}_multi_count.gexf"))
    numeric_nodes = [n for n in G.nodes if str(n).isdigit()]
    return G.subgraph(numeric_nodes).copy()


def compute_graph_stats(G):
    """Return (num_edges, total_weight, avg_weight_existing, avg_weight_all_dyads)."""
    num_edges = G.number_of_edges()
    total_weight = sum(data["weight"] for _, _, data in G.edges(data=True))

    avg_weight_existing = total_weight / num_edges if num_edges > 0 else 0.0

    num_nodes = G.number_of_nodes()
    possible_edges = num_nodes * (num_nodes - 1)  # directed, no self-loops
    avg_weight_all_dyads = total_weight / possible_edges if possible_edges > 0 else 0.0

    return num_edges, total_weight, avg_weight_existing, avg_weight_all_dyads


def create_square_ax(fig_size=6, axes_fraction=0.7):
    fig = plt.figure(figsize=(fig_size, fig_size))
    ax = fig.add_axes([0.15, 0.15, axes_fraction, axes_fraction])
    ax.set_box_aspect(1)
    return fig, ax


def main():
    results = []
    for year in YEARS:
        print(f"Processing {year}...")
        G = create_graph(year)
        num_edges, total_weight, avg_existing, avg_all_dyads = compute_graph_stats(G)
        results.append(
            {
                "Year": year,
                "num_edges": num_edges,
                "total_weight": total_weight,
                "avg_weight_existing": avg_existing,
                "avg_weight_all_dyads": avg_all_dyads,
            }
        )

    df_results = pd.DataFrame(results).set_index("Year")

    fig, ax = create_square_ax(fig_size=8, axes_fraction=0.75)
    ax.scatter(
        df_results.index,
        df_results["avg_weight_all_dyads"],
        marker="o",
        color="black",
    )
    ax.set_xlabel("Year", fontsize=30, labelpad=10)
    ax.set_ylabel("Average weight (all dyads)", fontsize=30, labelpad=10)
    ax.tick_params(axis="x", labelsize=18)
    ax.tick_params(axis="y", labelsize=18)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fig.savefig(OUTPUT_PATH)
    plt.close(fig)


if __name__ == "__main__":
    main()