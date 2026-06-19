"""
figS7.py

Generates a 5x4 grid of radial network visualizations of the cricket ruleset
YAML trees (last 20 edition years, 1910-2019), colored by node depth, with a
shared depth colorbar.

Usage
-----
    # From the repo root:
    python figures/supplemental/figS7/source/figS7.py

Outputs
-------
    ./figures/supplemental/figS7/figS7_final.png
"""

# ──────────────────────────────────────────────────────────────────────────────
# [0] USER INPUT: FILL IN!
# ──────────────────────────────────────────────────────────────────────────────

# Repo-root-relative paths (run from repo root).
YAML_DIR = "./data/datasets/rule_set_structure/yaml_files/flattened"
OUTPUT_DIR = "./figures/supplemental/figS7"
OUTPUT_NAME = "figS7_final.png"

TARGET_YEARS = [
    1910, 1911, 1913,
    1914, 1918, 1920, 1923, 1932, 1939, 1947, 1952, 1962,
    1968, 1980, 1992, 2000, 2008, 2010, 2017, 2019,
]

N_ROWS = 5
N_COLS = 4

# ──────────────────────────────────────────────────────────────────────────────
# [1] IMPORTS
# ──────────────────────────────────────────────────────────────────────────────

import os
import shutil
import colorsys

import yaml
import networkx as nx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.colors import BoundaryNorm
from networkx.drawing.nx_agraph import graphviz_layout

# ──────────────────────────────────────────────────────────────────────────────
# [2] MATPLOTLIB / LaTeX CONFIG
# ──────────────────────────────────────────────────────────────────────────────

USE_TEX = shutil.which("latex") is not None

plt.rcParams["text.usetex"] = USE_TEX
plt.rcParams["font.family"] = "serif"
if USE_TEX:
    plt.rcParams["font.serif"] = ["Computer Modern Roman"]
else:
    plt.rcParams["mathtext.fontset"] = "cm"
    print("Note: LaTeX not found on PATH; falling back to matplotlib's mathtext.")

# ──────────────────────────────────────────────────────────────────────────────
# [3] CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

N_DEPTH_LEVELS = 7
COLORMAP_NAME = "viridis"
LIGHTEN_AMOUNT = 0.6

# ──────────────────────────────────────────────────────────────────────────────
# [4] HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def lighten_color(color, amount=0.5):
    """Lighten an RGBA color by blending toward white and reducing saturation."""
    r, g, b, a = color
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = 1 - amount * (1 - l)
    r, g, b = colorsys.hls_to_rgb(h, l, s * 0.6)
    return (r, g, b, a)


def extract_edges_and_labels(node, path=None, edges=None, labels=None):
    """Recursively traverse a nested YAML structure into edges and labels."""
    if path is None:
        path = []
    if edges is None:
        edges = []
    if labels is None:
        labels = {}

    current_id = "/".join(path)
    labels[current_id] = str(path[-1])

    if isinstance(node, dict):
        for k, v in node.items():
            child_path = path + [str(k)]
            child_id = "/".join(child_path)
            edges.append((current_id, child_id))
            extract_edges_and_labels(v, child_path, edges, labels)
    elif isinstance(node, list):
        for child in node:
            if isinstance(child, (dict, list)):
                extract_edges_and_labels(child, path, edges, labels)
            else:
                child_path = path + [str(child)]
                child_id = "/".join(child_path)
                edges.append((current_id, child_id))
                labels[child_id] = str(child)
    else:
        child_path = path + [str(node)]
        child_id = "/".join(child_path)
        edges.append((current_id, child_id))
        labels[child_id] = str(node)

    return edges, labels


def compute_depths(labels):
    """Return {node_id: depth} where depth = number of '/' separators in the ID."""
    return {node: node.count("/") for node in labels}


def build_colormap(n_levels=N_DEPTH_LEVELS):
    """Build the shared discrete colormap and BoundaryNorm for depth coloring."""
    bounds = list(range(n_levels + 1))  # [0, 1, ..., n_levels]
    base_cmap = plt.get_cmap(COLORMAP_NAME)
    raw_colors = [base_cmap(i / (n_levels - 1)) for i in range(n_levels)]
    colors = [lighten_color(c, LIGHTEN_AMOUNT) for c in raw_colors]
    cmap = mcolors.ListedColormap(colors)
    norm = BoundaryNorm(bounds, ncolors=n_levels, clip=True)
    return cmap, norm, bounds, n_levels


def draw_year(ax, year, yaml_dir, cmap, norm):
    """Load YAML for a given year and draw the network onto ax."""
    yaml_path = os.path.join(yaml_dir, f"{year}_flattened.yaml")
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    if isinstance(data, dict) and len(data) == 1:
        root_key = next(iter(data))
        edges, labels = extract_edges_and_labels(data[root_key], path=[str(root_key)])
    else:
        edges, labels = extract_edges_and_labels(data)

    G = nx.DiGraph()
    G.add_edges_from(edges)

    pos = graphviz_layout(G, prog="twopi", args="-Gnodesep=0.4 -Granksep=1")

    depths = compute_depths(labels)
    node_depths = [depths[node] for node in G.nodes()]
    node_colors = [cmap(norm(d)) for d in node_depths]
    node_sizes = [max(500 * (0.4 ** d), 10) for d in node_depths]

    # Square axis limits with padding
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    max_range = max(max(xs) - min(xs), max(ys) - min(ys))
    center_x = (min(xs) + max(xs)) / 2
    center_y = (min(ys) + max(ys)) / 2
    padding = 0.05 * max_range
    half = (max_range / 2) + padding

    ax.set_xlim(center_x - half, center_x + half)
    ax.set_ylim(center_y - half, center_y + half)
    ax.set_aspect("equal")

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes, node_color=node_colors)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="gray", alpha=0.3, arrows=False)

    ax.text(
        0.01, 0.99, f"Year: {year}",
        transform=ax.transAxes,
        fontsize=8, fontweight="bold",
        ha="left", va="top",
    )
    ax.axis("off")

# ──────────────────────────────────────────────────────────────────────────────
# [5] MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cmap, norm, bounds, n_levels = build_colormap()

    fig = plt.figure(figsize=(N_COLS * 4, N_ROWS * 4))
    gs = gridspec.GridSpec(
        N_ROWS, N_COLS + 1,
        figure=fig,
        width_ratios=[1] * N_COLS + [0.08],
        hspace=0.05,
        wspace=0.05,
    )

    for idx, year in enumerate(TARGET_YEARS):
        row = idx // N_COLS
        col = idx % N_COLS
        ax = fig.add_subplot(gs[row, col])
        draw_year(ax, year, YAML_DIR, cmap, norm)

    # Hide unused cells if the year count doesn't fill the grid.
    for idx in range(len(TARGET_YEARS), N_ROWS * N_COLS):
        row = idx // N_COLS
        col = idx % N_COLS
        fig.add_subplot(gs[row, col]).set_visible(False)

    # Shared colorbar in the rightmost column
    cbar_ax = fig.add_subplot(gs[:, N_COLS])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    tick_positions = [(bounds[i] + bounds[i + 1]) / 2 for i in range(n_levels)]
    tick_labels = [str(i) for i in range(n_levels)]
    cbar = fig.colorbar(sm, cax=cbar_ax, ticks=tick_positions)
    cbar.set_label("Node Depth", fontsize=18)
    cbar.ax.yaxis.labelpad = 12
    cbar.ax.set_yticklabels(tick_labels)
    cbar.ax.tick_params(labelsize=13)

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved to {output_path}")

# ──────────────────────────────────────────────────────────────────────────────
# [6] ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()