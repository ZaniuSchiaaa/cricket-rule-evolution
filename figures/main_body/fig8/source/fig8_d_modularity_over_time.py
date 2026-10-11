"""
fig8_d_modularity_over_time.py

Robustness check for fig8_e: normalise the greedy-modularity Q of each year's
interdependency network against a batched, strength-preserving
configuration-model baseline, and plot the normalised modularity Q_NM vs. year
with 95% CI error bars.

Method (per year)
-----------------
1. Q_obs: collapse the directed multi_count graph into an undirected weighted
   graph (summed weights, self-edges removed), run greedy modularity
   maximisation, and record the optimised Q.
2. Baseline: preserve each node's weighted degree (strength) of the collapsed
   undirected graph. Each unit of weight is one stub; stubs are paired
   uniformly at random, self-loops are removed with strength-preserving pair
   swaps, and parallel edges are collapsed back into summed weights.
   Zero-strength nodes get no stubs and stay isolated. Record the optimised Q
   of each null graph with the SAME procedure (optimised vs. optimised).
3. Normalise each null draw i:  Q_NM,i = (Q_obs - Q_rand,i) / (1 - Q_rand,i).
4. Pool Q_NM,i across batches; report the median, stdev and 95% percentile CI.

Requirements
------------
Collapsed edge weights must be integers (stub matching treats one unit of
weight as one stub). A ValueError is raised otherwise.

Usage
-----
    # From the repo root:
    python figures/main_body/fig8/source/fig8_d_modularity_over_time.py

Outputs
-------
    ./figures/main_body/fig8/components/fig8_d_modularity_over_time.svg
"""

# =============================================================================
# [0] IMPORTS
# =============================================================================
import os
import re
import glob
import shutil

import numpy as np
import pandas as pd
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

OUTPUT_DIR = "./figures/main_body/fig8/components"
OUTPUT_NAME = "fig8_d_modularity_over_time.svg"
# SUMMARY_NAME = "fig8_d_modularity_over_time.csv"

WEIGHT_ATTR = "weight"
REMOVE_SELF_EDGES = True

# Each null draw runs a full greedy-modularity optimisation, so these are
# much smaller than in the reciprocity script. Scale up once runtime is known.
NUM_RANDOM_PER_BATCH = 20
NUM_BATCHES = 5
SEED = 42
MAX_REPAIR_ITER = 10_000

# Aesthetic constants
FIG_SIZE = 8
AXES_FRACTION = 0.75
MARKER_COLOR = "black"
ERROR_COLOR = "#9C0313"
ELINEWIDTH = 1.2
CAPSIZE = 4
LABEL_FONTSIZE = 30
TICK_LABELSIZE = 20
LABEL_PAD = 10
DPI = 300


# =============================================================================
# MODULARITY ANALYSIS
# =============================================================================
def collapse_to_undirected(G, weight=WEIGHT_ATTR):
    """Collapse a directed graph into an undirected graph with summed weights."""
    G_undirected = nx.Graph()
    G_undirected.add_nodes_from(G.nodes(data=True))
    for u, v, d in G.edges(data=True):
        if G_undirected.has_edge(u, v):
            G_undirected[u][v][weight] += d[weight]
        else:
            G_undirected.add_edge(u, v, **{weight: d[weight]})
    return G_undirected


def optimised_modularity(G_undirected, weight=WEIGHT_ATTR):
    """Greedy-modularity partition and its modularity (default resolution)."""
    communities = nx.community.greedy_modularity_communities(
        G_undirected, weight=weight
    )
    return nx.community.modularity(G_undirected, communities, weight=weight)


def integer_strengths(G_undirected, weight=WEIGHT_ATTR):
    """Weighted degree per node (node order = list(G.nodes)); asserts integer weights."""
    weights = np.array(
        [d[weight] for _, _, d in G_undirected.edges(data=True)], dtype=float
    )
    if weights.size and not np.allclose(weights, np.round(weights)):
        raise ValueError(
            "Non-integer edge weights found; the stub-matching null requires "
            "integer weights."
        )
    nodes = list(G_undirected.nodes)
    strengths = np.array(
        [int(round(G_undirected.degree(n, weight=weight))) for n in nodes],
        dtype=np.int64,
    )
    return nodes, strengths


def repair_self_loops(a, b, rng, max_iter=MAX_REPAIR_ITER):
    """
    Remove self-loops from stub pairs (a[i], b[i]) in place, preserving every
    node's stub count. Each bad pair (u, u) is swapped with a random other pair
    so that neither resulting pair is a self-loop. Returns True on success.
    """
    n_pairs = len(a)
    for _ in range(max_iter):
        bad = np.flatnonzero(a == b)
        if bad.size == 0:
            return True
        for i in bad:
            if a[i] != b[i]:
                continue
            j = rng.integers(n_pairs)
            if j == i:
                continue
            if rng.random() < 0.5:
                # (a_i, b_i), (a_j, b_j) -> (a_i, b_j), (a_j, b_i)
                if a[i] != b[j] and a[j] != b[i]:
                    b[i], b[j] = b[j], b[i]
            else:
                # (a_i, b_i), (a_j, b_j) -> (a_i, a_j), (b_i, b_j)
                if a[i] != a[j] and b[i] != b[j]:
                    b[i], a[j] = a[j], b[i]
    return bool(np.all(a != b))


def configuration_null(nodes, strengths, rng, weight=WEIGHT_ATTR):
    """
    One strength-preserving null graph: random stub matching, self-loops
    repaired by swaps, parallel edges summed into weights. Zero-strength nodes
    are kept as isolated nodes.
    """
    n = len(nodes)
    stubs = np.repeat(np.arange(n), strengths)
    rng.shuffle(stubs)
    a, b = stubs[0::2].copy(), stubs[1::2].copy()

    if not repair_self_loops(a, b, rng):
        raise RuntimeError(
            "Could not remove all self-loops while preserving strengths."
        )

    lo, hi = np.minimum(a, b), np.maximum(a, b)
    keys, counts = np.unique(lo * n + hi, return_counts=True)

    G_null = nx.Graph()
    G_null.add_nodes_from(nodes)
    G_null.add_weighted_edges_from(
        ((nodes[k // n], nodes[k % n], int(c)) for k, c in zip(keys, counts)),
        weight=weight,
    )
    return G_null


def analyze_modularity(nodes, strengths, q_obs, weight=WEIGHT_ATTR,
                       num_random=NUM_RANDOM_PER_BATCH, rng=None):
    """
    Compute Q_NM draws against a strength-preserving null.

    Returns a dict with Q_obs, the per-null-graph Q_rand list, and the
    per-null-graph Q_NM list.
    """
    if rng is None:
        rng = np.random.default_rng()

    q_rand_list = np.empty(num_random, dtype=float)
    for i in range(num_random):
        G_null = configuration_null(nodes, strengths, rng, weight)
        q_rand_list[i] = optimised_modularity(G_null, weight)

    q_nm_list = (q_obs - q_rand_list) / (1 - q_rand_list)

    return {"Q_obs": q_obs, "Q_rand_list": q_rand_list, "Q_NM_list": q_nm_list}


def analyze_modularity_with_batches(G, year, weight=WEIGHT_ATTR,
                                    num_random_per_batch=NUM_RANDOM_PER_BATCH,
                                    num_batches=NUM_BATCHES, seed=SEED):
    """Pool Q_NM samples across batches; report median, stdev, and 95% CI."""
    G_undirected = collapse_to_undirected(G, weight)
    q_obs = optimised_modularity(G_undirected, weight)

    nodes, strengths = integer_strengths(G_undirected, weight)
    if strengths.sum() % 2 != 0:
        raise ValueError(f"{year}: total strength is odd; cannot pair stubs.")

    all_q_rand, all_q_nm = [], []
    for batch_idx in range(num_batches):
        # Independent, reproducible stream per (seed, year, batch)
        rng = np.random.default_rng([seed, year, batch_idx])
        result = analyze_modularity(
            nodes, strengths, q_obs, weight=weight,
            num_random=num_random_per_batch, rng=rng,
        )
        all_q_rand.extend(result["Q_rand_list"])
        all_q_nm.extend(result["Q_NM_list"])

    all_q_rand = np.array(all_q_rand, dtype=float)
    all_q_nm = np.array(all_q_nm, dtype=float)
    return {
        "Q_obs": q_obs,
        "Q_rand_mean": float(np.nanmean(all_q_rand)),
        "Q_rand_stdev": float(np.nanstd(all_q_rand)),
        "Q_NM_median": float(np.nanmedian(all_q_nm)),
        "Q_NM_stdev": float(np.nanstd(all_q_nm)),
        "Q_NM_ci": tuple(np.nanpercentile(all_q_nm, [2.5, 97.5])),
    }


# =============================================================================
# GEXF LOADING
# =============================================================================
def extract_year(filename):
    """Pull the first 4-digit year from a filename; None if absent."""
    match = re.search(r"(\d{4})", os.path.basename(filename))
    return int(match.group(1)) if match else None


def to_simple_digraph(G, weight=WEIGHT_ATTR):
    """Return a DiGraph with float weights (parallel edges, if any, are summed)."""
    D = nx.DiGraph()
    D.add_nodes_from(G.nodes(data=True))
    for u, v, data in G.edges(data=True):
        w = float(data.get(weight, 1))
        if D.has_edge(u, v):
            D[u][v][weight] += w
        else:
            D.add_edge(u, v, **{weight: w})
    return D


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
        G = to_simple_digraph(G, WEIGHT_ATTR)

        if REMOVE_SELF_EDGES:
            G.remove_edges_from(nx.selfloop_edges(G))

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


def plot_q_nm(years, medians, ci_lower, ci_upper, output_dir, output_name):
    fig, ax = create_square_ax(fig_size=FIG_SIZE, axes_fraction=AXES_FRACTION)

    years = np.asarray(years, dtype=float)
    medians = np.asarray(medians, dtype=float)
    yerr = [medians - np.asarray(ci_lower), np.asarray(ci_upper) - medians]

    ax.errorbar(
        years, medians,
        yerr=yerr,
        color=MARKER_COLOR,
        fmt="o",
        ecolor=ERROR_COLOR,
        elinewidth=ELINEWIDTH,
        capsize=CAPSIZE,
    )
    ax.axhline(0, linestyle="--", linewidth=1, color=ERROR_COLOR)

    ax.set_xlabel("Year", fontsize=LABEL_FONTSIZE, labelpad=LABEL_PAD)
    ax.set_ylabel(r"$Q_{NM}$", fontsize=LABEL_FONTSIZE, labelpad=LABEL_PAD)
    ax.tick_params(axis="x", labelsize=TICK_LABELSIZE)
    ax.tick_params(axis="y", labelsize=TICK_LABELSIZE)
    ax.grid(False)
    ax.set_position([0.15, 0.2, 0.7, 0.65])

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, output_name), dpi=DPI)
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================
def main():
    graphs = load_graphs_by_year(GRAPH_DIR)

    rows = []
    for year, G in graphs:
        stats = analyze_modularity_with_batches(
            G,
            year,
            weight=WEIGHT_ATTR,
            num_random_per_batch=NUM_RANDOM_PER_BATCH,
            num_batches=NUM_BATCHES,
            seed=SEED,
        )
        lo, hi = stats["Q_NM_ci"]
        rows.append({
            "year": year,
            "n_nodes": G.number_of_nodes(),
            "n_edges": G.number_of_edges(),
            "Q_obs": stats["Q_obs"],
            "Q_rand_mean": stats["Q_rand_mean"],
            "Q_rand_stdev": stats["Q_rand_stdev"],
            "Q_NM_median": stats["Q_NM_median"],
            "Q_NM_stdev": stats["Q_NM_stdev"],
            "Q_NM_ci_lower": lo,
            "Q_NM_ci_upper": hi,
        })
        print(f"{year}: Q_obs={stats['Q_obs']:.4f} "
              f"Q_rand={stats['Q_rand_mean']:.4f} "
              f"Q_NM={stats['Q_NM_median']:.4f} "
              f"CI=({lo:.4f}, {hi:.4f}) stdev={stats['Q_NM_stdev']:.4f}")

    summary = pd.DataFrame(rows).set_index("year")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # summary.to_csv(os.path.join(OUTPUT_DIR, SUMMARY_NAME))

    plot_q_nm(
        summary.index, summary["Q_NM_median"],
        summary["Q_NM_ci_lower"], summary["Q_NM_ci_upper"],
        OUTPUT_DIR, OUTPUT_NAME,
    )
    print(f"\nSaved figure to {os.path.join(OUTPUT_DIR, OUTPUT_NAME)}")


if __name__ == "__main__":
    main()
