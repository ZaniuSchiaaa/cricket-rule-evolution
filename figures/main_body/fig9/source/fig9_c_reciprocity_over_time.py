"""
fig9_c_reciprocity_over_time.py

Reads multi-count interdependency-network GEXF files sequentially, computes the
weighted-reciprocity statistic (rho_NM) for each year with a batched random-graph
baseline, and plots rho_NM vs. year with 95% CI error bars.

Usage
-----
    # From the repo root:
    python figures/main_body/fig9/source/fig9_c_reciprocity_over_time.py

Outputs
-------
    ./figures/main_body/fig9/components/fig9_c_reciprocity_over_time.svg
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

OUTPUT_DIR = './figures/main_body/fig9/components'
OUTPUT_NAME = "fig9_c_reciprocity_over_time.svg"

WEIGHT_ATTR = "weight"
NUM_RANDOM_PER_BATCH = 200
NUM_BATCHES = 10
SEED = 42
RAND_GRAPH_MODE = "permute"  # 'permute' or 'draw from distribution'

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
# RECIPROCITY ANALYSIS
# =============================================================================
def get_recip_weights(G, weight=WEIGHT_ATTR):
    """Collate min reciprocated weight from each reciprocated dyad (once per pair)."""
    recip_weights = []
    seen = set()
    for u, v in G.edges():
        if G.has_edge(v, u) and (v, u) not in seen:
            w1 = G[u][v].get(weight, 1)
            w2 = G[v][u].get(weight, 1)
            recip_weights.append(min(w1, w2))
            seen.add((u, v))
            seen.add((v, u))
    return recip_weights


def get_weighted_reciprocity(G, recip_weights, weight=WEIGHT_ATTR):
    """r = 2 * sum(reciprocated weight) / total weight."""
    total_weight = G.size(weight=weight)
    if total_weight == 0:
        return np.nan
    total_recip_weight = sum(recip_weights)
    return (2 * total_recip_weight) / total_weight


def analyze_reciprocity(G, weight=WEIGHT_ATTR, num_random=NUM_RANDOM_PER_BATCH,
                        seed=None, generate_rand_graph_mode=RAND_GRAPH_MODE):
    """
    Compute rho_NM for graph G against a random-graph baseline.

    Returns a dict with rho_NM, its stdev, 95% CI, and the per-random-graph list.
    """
    recip_weights = get_recip_weights(G, weight=weight)
    weighted_reciprocity_observed = get_weighted_reciprocity(G, recip_weights, weight=weight)

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    weights_orig = [G[u][v].get(weight, 1) for u, v in G.edges()]

    all_weighted_reciprocities = []
    for i in range(num_random):
        graph_seed = (seed + i) if seed is not None else None
        R = nx.gnm_random_graph(num_nodes, num_edges, directed=True, seed=graph_seed)

        if generate_rand_graph_mode == "draw from distribution":
            for (u, v) in R.edges():
                R[u][v][weight] = np.random.choice(weights_orig)
        elif generate_rand_graph_mode == "permute":
            weights_shuffled = np.random.permutation(weights_orig)
            for (u, v), w in zip(R.edges(), weights_shuffled):
                R[u][v][weight] = w
        else:
            raise ValueError(
                "mode of random graph generation must be "
                "'draw from distribution' or 'permute'"
            )

        recip_weights_rand = get_recip_weights(R, weight=weight)
        all_weighted_reciprocities.append(
            get_weighted_reciprocity(R, recip_weights_rand, weight=weight)
        )

    all_weighted_reciprocities = np.array(all_weighted_reciprocities, dtype=float)
    aggr_rand = np.nanmean(all_weighted_reciprocities)

    rho_NM = (weighted_reciprocity_observed - aggr_rand) / (1 - aggr_rand)
    rho_rn_list = [
        (weighted_reciprocity_observed - r_i) / (1 - r_i)
        for r_i in all_weighted_reciprocities
    ]

    return {
        "rho_NM": rho_NM,
        "rho_NM_stdev": float(np.nanstd(rho_rn_list)),
        "rho_NM_ci": tuple(np.nanpercentile(rho_rn_list, [2.5, 97.5])),
        "rho_NM_list": rho_rn_list,
    }


def analyze_reciprocity_with_batches(G, weight=WEIGHT_ATTR,
                                     num_random_per_batch=NUM_RANDOM_PER_BATCH,
                                     num_batches=NUM_BATCHES, seed=SEED,
                                     generate_rand_graph_mode=RAND_GRAPH_MODE):
    """Pool rho_NM samples across batches; report median, stdev, and 95% CI."""
    all_rho_rn = []
    for batch_idx in range(num_batches):
        # Offset by a full batch width so per-iteration seeds never overlap
        batch_seed = (seed + batch_idx * num_random_per_batch) if seed is not None else None
        result = analyze_reciprocity(
            G, weight=weight, num_random=num_random_per_batch,
            seed=batch_seed, generate_rand_graph_mode=generate_rand_graph_mode,
        )
        all_rho_rn.extend(result["rho_NM_list"])

    all_rho_rn = np.array(all_rho_rn, dtype=float)
    return {
        "rho_NM_mean": float(np.nanmedian(all_rho_rn)),
        "rho_NM_stdev": float(np.nanstd(all_rho_rn)),
        "rho_NM_ci": tuple(np.nanpercentile(all_rho_rn, [2.5, 97.5])),
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


def plot_rho_rn(years, means, ci_lower, ci_upper, output_dir, output_name):
    fig, ax = create_square_ax(fig_size=FIG_SIZE, axes_fraction=AXES_FRACTION)

    years = np.asarray(years, dtype=float)
    means = np.asarray(means, dtype=float)
    yerr = [means - np.asarray(ci_lower), np.asarray(ci_upper) - means]

    ax.errorbar(
        years, means,
        yerr=yerr,
        color=MARKER_COLOR,
        fmt="o",
        ecolor=ERROR_COLOR,
        elinewidth=ELINEWIDTH,
        capsize=CAPSIZE,
    )
    ax.axhline(0, linestyle="--", linewidth=1, color=ERROR_COLOR)

    ax.set_xlabel("Year", fontsize=LABEL_FONTSIZE, labelpad=LABEL_PAD)
    ax.set_ylabel(r"$\rho_{NM}$", fontsize=LABEL_FONTSIZE, labelpad=LABEL_PAD)
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

    years, means, ci_lower, ci_upper = [], [], [], []
    for year, G in graphs:
        stats = analyze_reciprocity_with_batches(
            G,
            weight=WEIGHT_ATTR,
            num_random_per_batch=NUM_RANDOM_PER_BATCH,
            num_batches=NUM_BATCHES,
            seed=SEED,
            generate_rand_graph_mode=RAND_GRAPH_MODE,
        )
        lo, hi = stats["rho_NM_ci"]
        years.append(year)
        means.append(stats["rho_NM_mean"])
        ci_lower.append(lo)
        ci_upper.append(hi)
        print(f"{year}: rho_NM={stats['rho_NM_mean']:.4f} "
              f"CI=({lo:.4f}, {hi:.4f}) stdev={stats['rho_NM_stdev']:.4f}")

    plot_rho_rn(years, means, ci_lower, ci_upper, OUTPUT_DIR, OUTPUT_NAME)
    print(f"\nSaved figure to {os.path.join(OUTPUT_DIR, OUTPUT_NAME)}")


if __name__ == "__main__":
    main()