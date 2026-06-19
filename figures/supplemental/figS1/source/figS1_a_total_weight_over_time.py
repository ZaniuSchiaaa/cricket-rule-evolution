"""
figS1_a_total_weight_over_time.py

Plots total edge weight of the interdependency network over time (one point per
edition year). Edge weights are read directly from the multi-count GEXF graph
files.

Statistic
---------
    Total weight = sum of all edge weights in the graph.

Usage
-----
    # From the repo root:
    python figures/supplemental/figS1/source/figS1_a_total_weight_over_time.py

Outputs
-------
    ./figures/supplemental/figS1/components/figS1_a_total_weight_over_time.svg
"""

# ──────────────────────────────────────────────────────────────────────────────
# [0] IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import os
import re
import glob
import shutil

import networkx as nx
import matplotlib.pyplot as plt

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
GEXF_DIR = "./data/datasets/interdependency_networks/graph_files/gexf/multi_count"
OUTPUT_DIR = "./figures/supplemental/figS1/components"

REMOVE_SELF_EDGES = True
FILE_PATTERN = re.compile(r"(\d{4})")  # Extract year from filename

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# [3] HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def create_square_ax(fig_size=6, axes_fraction=0.7, label_pad=10):
    fig = plt.figure(figsize=(fig_size, fig_size))
    ax = fig.add_axes([0.15, 0.15, axes_fraction, axes_fraction])
    ax.set_box_aspect(1)
    return fig, ax


def load_graph(path):
    """Read a GEXF file, assert it is directed, and optionally drop self-loops."""
    G = nx.read_gexf(path)
    if not G.is_directed():
        raise ValueError(
            f"{os.path.basename(path)} read as undirected; the all-dyads denominator "
            "N(N-1) assumes a directed graph. Verify the GEXF export."
        )
    if REMOVE_SELF_EDGES:
        G.remove_edges_from(nx.selfloop_edges(G))
    return G


def total_weight(G):
    """(1) Sum of all edge weights."""
    return sum(data.get("weight", 1) for _, _, data in G.edges(data=True))


# ──────────────────────────────────────────────────────────────────────────────
# [4] LOAD GRAPHS AND COMPUTE STATISTIC
# ──────────────────────────────────────────────────────────────────────────────
gexf_files = sorted(glob.glob(os.path.join(GEXF_DIR, "*.gexf")))

if not gexf_files:
    print(f"No GEXF files found in {GEXF_DIR}")
else:
    years = []
    values = []
    for fpath in gexf_files:
        match = FILE_PATTERN.search(os.path.basename(fpath))
        if not match:
            print(f"  Skipping (no year in name): {os.path.basename(fpath)}")
            continue
        year = int(match.group(1))
        G = load_graph(fpath)
        values.append(total_weight(G))
        years.append(year)

    # Sort by year
    order = sorted(range(len(years)), key=lambda i: years[i])
    years = [years[i] for i in order]
    values = [values[i] for i in order]

    # ──────────────────────────────────────────────────────────────────────────
    # [5] PLOT
    # ──────────────────────────────────────────────────────────────────────────
    fig, ax = create_square_ax(fig_size=8, axes_fraction=0.75)
    ax.scatter(years, values, marker="o", color="black")
    ax.set_xlabel("Year", fontsize=30, labelpad=10)
    ax.set_ylabel("Total weight", fontsize=30, labelpad=10)
    ax.tick_params(axis="x", labelsize=18)
    ax.tick_params(axis="y", labelsize=18)

    out_path = os.path.join(OUTPUT_DIR, "figS1_a_total_weight_over_time.svg")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")

print("\nDone.")