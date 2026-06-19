"""
Cricket Ruleset Network Visualizer
===================================
Generates radial network visualizations of cricket ruleset YAML trees,
colored by node depth, for each edition year.

Usage
-----
    # From the repo root:
    python data/visualizations/rule_set_structure/network_layout/source/rule_set_network_viz.py
"""

import colorsys
import os
from multiprocessing import Pool, freeze_support

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import yaml
from matplotlib.colors import BoundaryNorm
from networkx.drawing.nx_agraph import graphviz_layout

# ── Matplotlib / LaTeX config ──────────────────────────────────────────────────
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman']

# ── Global settings ────────────────────────────────────────────────────────────
IS_FLATTENED = False
RULE_TREE_TYPE = 'flattened' if IS_FLATTENED else 'original'

N_DEPTH_LEVELS = 8
COLORMAP_NAME = 'viridis'
LIGHTEN_AMOUNT = 0.6

TARGET_YEARS = [
    1752, 1755, 1774, 1785, 1786, 1788, 1803, 1806, 1809,
    1816, 1820, 1823, 1828, 1830, 1835, 1857, 1884, 1890,
    1892, 1896, 1900, 1902, 1906, 1908, 1910, 1911, 1913,
    1914, 1918, 1920, 1923, 1932, 1939, 1947, 1952, 1962,
    1968, 1980, 1992, 2000, 2008, 2010, 2017, 2019,
]

YAML_PATH_TEMPLATE = (
    "./data/datasets/rule_set_structure/yaml_files/{rule_tree_type}"
    "/{year}_{rule_tree_type}.yaml"
)
PNG_PATH_TEMPLATE = (
    "./data/visualizations/rule_set_structure/network_layout"
    "/network_viz/{rule_tree_type}/pngs/{year}_{rule_tree_type}_network_viz.png"
)
SVG_PATH_TEMPLATE = (
    "./data/visualizations/rule_set_structure/network_layout"
    "/network_viz/{rule_tree_type}/svgs/{year}_{rule_tree_type}_network_viz.svg"
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def lighten_color(color: tuple, amount: float = 0.5) -> tuple:
    """Lighten an RGBA color by blending toward white and reducing saturation."""
    r, g, b, a = color
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = 1 - amount * (1 - l)
    r, g, b = colorsys.hls_to_rgb(h, l, s * 0.6)
    return (r, g, b, a)


def extract_edges_and_labels(
    node, path: list = None, edges: list = None, labels: dict = None
) -> tuple[list, dict]:
    """
    Recursively traverse a nested YAML structure and extract graph edges
    and node labels.

    Parameters
    ----------
    node : dict | list | scalar
        Current YAML node being traversed.
    path : list of str
        Path from the root to the current node (used as node IDs).
    edges : list of tuple
        Accumulated (parent_id, child_id) edge pairs.
    labels : dict
        Accumulated {node_id: label} mapping.

    Returns
    -------
    edges : list of tuple
    labels : dict
    """
    if path is None:
        path = []
    if edges is None:
        edges = []
    if labels is None:
        labels = {}

    current_id = '/'.join(path)
    labels[current_id] = str(path[-1])

    if isinstance(node, dict):
        for k, v in node.items():
            child_path = path + [str(k)]
            child_id = '/'.join(child_path)
            edges.append((current_id, child_id))
            extract_edges_and_labels(v, child_path, edges, labels)

    elif isinstance(node, list):
        for child in node:
            if isinstance(child, (dict, list)):
                extract_edges_and_labels(child, path, edges, labels)
            else:
                child_path = path + [str(child)]
                child_id = '/'.join(child_path)
                edges.append((current_id, child_id))
                labels[child_id] = str(child)

    else:
        child_path = path + [str(node)]
        child_id = '/'.join(child_path)
        edges.append((current_id, child_id))
        labels[child_id] = str(node)

    return edges, labels


def truncate_label(label: str, maxlen: int = 10) -> str:
    """Truncate a label to `maxlen` characters, appending '...' if needed."""
    label = str(label)
    return label if len(label) <= maxlen else label[:maxlen - 3] + "..."


def compute_depths(labels: dict) -> dict:
    """Return {node_id: depth} where depth = number of '/' separators in the ID."""
    return {node: node.count('/') for node in labels}


def build_depth_colormap(n_levels: int = N_DEPTH_LEVELS) -> tuple:
    """
    Build a discrete colormap and BoundaryNorm for depth-based node coloring.

    Returns
    -------
    cmap : ListedColormap
    norm : BoundaryNorm
    bounds : list of int
    """
    base_cmap = plt.get_cmap(COLORMAP_NAME)
    raw_colors = [base_cmap(i / (n_levels - 1)) for i in range(n_levels)]
    colors = [lighten_color(c, LIGHTEN_AMOUNT) for c in raw_colors]
    cmap = mcolors.ListedColormap(colors)
    bounds = list(range(n_levels + 1))
    norm = BoundaryNorm(bounds, ncolors=n_levels, clip=True)
    return cmap, norm, bounds


# ── Per-year processing ────────────────────────────────────────────────────────

def process_year(year: int) -> None:
    """Load the YAML for `year`, build a network graph, and save PNG + SVG."""

    # Load YAML
    yaml_path = YAML_PATH_TEMPLATE.format(rule_tree_type=RULE_TREE_TYPE, year=year)
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    # Build graph
    if isinstance(data, dict) and len(data) == 1:
        root_key = next(iter(data))
        edges, labels = extract_edges_and_labels(data[root_key], path=[str(root_key)])
    else:
        edges, labels = extract_edges_and_labels(data)

    G = nx.DiGraph()
    G.add_edges_from(edges)

    # Layout
    pos = graphviz_layout(G, prog='twopi', args='-Gnodesep=0.4 -Granksep=1')

    # Depth-based visual properties
    depths = compute_depths(labels)
    cmap, norm, bounds = build_depth_colormap()

    node_depths = [depths[node] for node in G.nodes()]
    node_colors = [cmap(norm(d)) for d in node_depths]
    node_sizes = [max(500 * (0.4 ** d), 10) for d in node_depths]

    # Square, padded axis limits
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    max_range = max(max(xs) - min(xs), max(ys) - min(ys))
    center_x = (min(xs) + max(xs)) / 2
    center_y = (min(ys) + max(ys)) / 2
    padding = 0.05 * max_range
    half = max_range / 2 + padding

    # Draw
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xlim(center_x - half, center_x + half)
    ax.set_ylim(center_y - half, center_y + half)
    ax.set_aspect('equal')

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes, node_color=node_colors)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="gray", alpha=0.3, arrows=False)

    # Colorbar
    n_levels = len(bounds) - 1
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    tick_positions = [(bounds[i] + bounds[i + 1]) / 2 for i in range(n_levels)]
    tick_labels = [str(i) for i in range(n_levels)]

    cbar = plt.colorbar(sm, ax=ax, ticks=tick_positions)
    cbar.set_label('Node Depth', fontsize=46)
    cbar.ax.yaxis.labelpad = 16
    cbar.ax.set_yticklabels(tick_labels)
    cbar.ax.tick_params(labelsize=32)

    # Year annotation
    ax.text(
        0.01, 0.99, f"Year: {year}",
        transform=ax.transAxes,
        fontsize=30, fontweight='bold',
        ha='left', va='top',
    )

    plt.tight_layout()
    plt.savefig(PNG_PATH_TEMPLATE.format(rule_tree_type=RULE_TREE_TYPE, year=year), dpi=300)
    plt.savefig(SVG_PATH_TEMPLATE.format(rule_tree_type=RULE_TREE_TYPE, year=year))
    plt.close(fig)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    freeze_support()
    with Pool(processes=4) as pool:
        pool.map(process_year, TARGET_YEARS)