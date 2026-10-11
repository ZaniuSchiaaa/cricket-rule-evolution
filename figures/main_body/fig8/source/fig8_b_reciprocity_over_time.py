"""
fig8_b_reciprocity_over_time.py

Reads multi-count interdependency-network GEXF files sequentially, computes the
weighted-reciprocity statistic (rho_NM) for each year against a batched
Weighted Configuration Model (WCM) baseline, and plots rho_NM vs. year with 95%
CI error bars.

This is fig8_c_reciprocity_over_time.py with the G(n, m) + weight-permutation
null replaced by a directed WCM. The reciprocity statistic, the rho_NM
normalisation, the batching, the CI construction and the plotting are unchanged.

Method (per year)
-----------------
1. Load the directed multi_count graph (parallel edges summed, self-edges
   removed) and compute the observed weighted reciprocity
       r_obs = 2 * sum(min reciprocated weight per dyad) / total weight.
2. Baseline (WCM): every node keeps its observed in-strength and out-strength.
   Each unit of weight is one stub. Out-stubs are paired with a random
   permutation of in-stubs, self-loops are removed with strength-preserving
   pair swaps, and parallel edges are collapsed back into summed weights.
   Zero-strength nodes get no stubs and stay isolated.
3. Compute r_rand,i on each null graph with the SAME reciprocity function.
4. Normalise each null draw i:  rho_NM,i = (r_obs - r_rand,i) / (1 - r_rand,i).
5. Pool rho_NM,i across batches; report the median, stdev and 95% percentile CI.

Requirements
------------
Edge weights must be integers (stub matching treats one unit of weight as one
stub). A ValueError is raised otherwise.

Usage
-----
    # From the repo root:
    python figures/main_body/fig8/source/fig8_b_reciprocity_over_time.py

Outputs
-------
    ./figures/main_body/fig8/components/fig8_b_reciprocity_over_time.svg
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
OUTPUT_NAME = "fig8_b_reciprocity_over_time.svg"
# SUMMARY_NAME = "fig8_c_reciprocity_over_time.csv"

WEIGHT_ATTR = "weight"
REMOVE_SELF_EDGES = True
NUM_RANDOM_PER_BATCH = 200
NUM_BATCHES = 10
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
# RECIPROCITY STATISTIC (unchanged)
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


# =============================================================================
# WEIGHTED CONFIGURATION MODEL (directed)
# =============================================================================
def integer_in_out_strengths(G, weight=WEIGHT_ATTR):
    """
    In- and out-strength per node (node order = list(G.nodes)); asserts integer
    weights and that a self-loop-free stub pairing exists.
    """
    weights = np.array([d[weight] for _, _, d in G.edges(data=True)], dtype=float)
    if weights.size and not np.allclose(weights, np.round(weights)):
        raise ValueError(
            "Non-integer edge weights found; the stub-matching null requires "
            "integer weights."
        )
    nodes = list(G.nodes)
    out_strength = np.array(
        [int(round(G.out_degree(n, weight=weight))) for n in nodes], dtype=np.int64
    )
    in_strength = np.array(
        [int(round(G.in_degree(n, weight=weight))) for n in nodes], dtype=np.int64
    )

    # A node's out-stubs can only pair with other nodes' in-stubs, so a
    # self-loop-free pairing needs out_u + in_u <= total weight for every u.
    total = out_strength.sum()
    if np.any(out_strength + in_strength > total):
        raise ValueError(
            "A node holds more than half of all stubs; no self-loop-free "
            "stub pairing exists."
        )
    return nodes, out_strength, in_strength


def repair_self_loops(out_stubs, in_stubs, rng, max_iter=MAX_REPAIR_ITER):
    """
    Remove self-loops from stub pairs (out_stubs[i], in_stubs[i]) in place,
    preserving every node's in- and out-stub counts. Each bad pair (u, u) swaps
    its in-stub with that of a random other pair, accepted only if neither
    resulting pair is a self-loop. Returns True on success.
    """
    n_pairs = len(out_stubs)
    for _ in range(max_iter):
        bad = np.flatnonzero(out_stubs == in_stubs)
        if bad.size == 0:
            return True
        for i in bad:
            if out_stubs[i] != in_stubs[i]:
                continue
            j = rng.integers(n_pairs)
            if j == i:
                continue
            # (o_i, in_i), (o_j, in_j) -> (o_i, in_j), (o_j, in_i)
            if out_stubs[i] != in_stubs[j] and out_stubs[j] != in_stubs[i]:
                in_stubs[i], in_stubs[j] = in_stubs[j], in_stubs[i]
    return bool(np.all(out_stubs != in_stubs))


def weighted_configuration_null(nodes, out_strength, in_strength, rng,
                                weight=WEIGHT_ATTR):
    """
    One WCM null graph: out-stubs paired with shuffled in-stubs, self-loops
    repaired by swaps, parallel edges summed into weights. In- and out-strength
    of every node match the observed graph exactly. Zero-strength nodes are
    kept as isolated nodes.
    """
    n = len(nodes)
    out_stubs = np.repeat(np.arange(n), out_strength)
    in_stubs = np.repeat(np.arange(n), in_strength)
    rng.shuffle(in_stubs)

    if not repair_self_loops(out_stubs, in_stubs, rng):
        raise RuntimeError(
            "Could not remove all self-loops while preserving strengths."
        )

    keys, counts = np.unique(out_stubs * n + in_stubs, return_counts=True)

    R = nx.DiGraph()
    R.add_nodes_from(nodes)
    R.add_weighted_edges_from(
        ((nodes[k // n], nodes[k % n], int(c)) for k, c in zip(keys, counts)),
        weight=weight,
    )
    return R


# =============================================================================
# RECIPROCITY ANALYSIS
# =============================================================================
def analyze_reciprocity(nodes, out_strength, in_strength, r_obs,
                        weight=WEIGHT_ATTR, num_random=NUM_RANDOM_PER_BATCH,
                        rng=None):
    """
    Compute rho_NM for the observed graph against a WCM baseline.

    Returns a dict with rho_NM (ratio using the mean null), its stdev, 95% CI,
    and the per-null-graph rho_NM and r_rand lists.
    """
    if rng is None:
        rng = np.random.default_rng()

    all_weighted_reciprocities = []
    for _ in range(num_random):
        R = weighted_configuration_null(nodes, out_strength, in_strength, rng, weight)
        recip_weights_rand = get_recip_weights(R, weight=weight)
        all_weighted_reciprocities.append(
            get_weighted_reciprocity(R, recip_weights_rand, weight=weight)
        )

    all_weighted_reciprocities = np.array(all_weighted_reciprocities, dtype=float)
    aggr_rand = np.nanmean(all_weighted_reciprocities)

    rho_NM = (r_obs - aggr_rand) / (1 - aggr_rand)
    rho_rn_list = [
        (r_obs - r_i) / (1 - r_i) for r_i in all_weighted_reciprocities
    ]

    return {
        "rho_NM": rho_NM,
        "rho_NM_stdev": float(np.nanstd(rho_rn_list)),
        "rho_NM_ci": tuple(np.nanpercentile(rho_rn_list, [2.5, 97.5])),
        "rho_NM_list": rho_rn_list,
        "r_rand_list": all_weighted_reciprocities,
    }


def analyze_reciprocity_with_batches(G, year, weight=WEIGHT_ATTR,
                                     num_random_per_batch=NUM_RANDOM_PER_BATCH,
                                     num_batches=NUM_BATCHES, seed=SEED):
    """Pool rho_NM samples across batches; report median, stdev, and 95% CI."""
    recip_weights = get_recip_weights(G, weight=weight)
    r_obs = get_weighted_reciprocity(G, recip_weights, weight=weight)

    nodes, out_strength, in_strength = integer_in_out_strengths(G, weight)

    all_rho_rn, all_r_rand = [], []
    for batch_idx in range(num_batches):
        # Independent, reproducible stream per (seed, year, batch)
        rng = np.random.default_rng([seed, year, batch_idx])
        result = analyze_reciprocity(
            nodes, out_strength, in_strength, r_obs, weight=weight,
            num_random=num_random_per_batch, rng=rng,
        )
        all_rho_rn.extend(result["rho_NM_list"])
        all_r_rand.extend(result["r_rand_list"])

    all_rho_rn = np.array(all_rho_rn, dtype=float)
    all_r_rand = np.array(all_r_rand, dtype=float)
    return {
        "r_obs": r_obs,
        "r_rand_mean": float(np.nanmean(all_r_rand)),
        "r_rand_stdev": float(np.nanstd(all_r_rand)),
        "rho_NM_median": float(np.nanmedian(all_rho_rn)),
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
    Coerces edge weights to float; removes self-loops (the WCM null has none);
    skips files with no extractable year.
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


def plot_rho_rn(years, medians, ci_lower, ci_upper, output_dir, output_name):
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

    rows = []
    for year, G in graphs:
        stats = analyze_reciprocity_with_batches(
            G,
            year,
            weight=WEIGHT_ATTR,
            num_random_per_batch=NUM_RANDOM_PER_BATCH,
            num_batches=NUM_BATCHES,
            seed=SEED,
        )
        lo, hi = stats["rho_NM_ci"]
        rows.append({
            "year": year,
            "n_nodes": G.number_of_nodes(),
            "n_edges": G.number_of_edges(),
            "total_weight": G.size(weight=WEIGHT_ATTR),
            "r_obs": stats["r_obs"],
            "r_rand_mean": stats["r_rand_mean"],
            "r_rand_stdev": stats["r_rand_stdev"],
            "rho_NM_median": stats["rho_NM_median"],
            "rho_NM_stdev": stats["rho_NM_stdev"],
            "rho_NM_ci_lower": lo,
            "rho_NM_ci_upper": hi,
        })
        print(f"{year}: r_obs={stats['r_obs']:.4f} "
              f"r_rand={stats['r_rand_mean']:.4f} "
              f"rho_NM={stats['rho_NM_median']:.4f} "
              f"CI=({lo:.4f}, {hi:.4f}) stdev={stats['rho_NM_stdev']:.4f}")

    summary = pd.DataFrame(rows).set_index("year")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # summary.to_csv(os.path.join(OUTPUT_DIR, SUMMARY_NAME))

    plot_rho_rn(
        summary.index, summary["rho_NM_median"],
        summary["rho_NM_ci_lower"], summary["rho_NM_ci_upper"],
        OUTPUT_DIR, OUTPUT_NAME,
    )
    print(f"\nSaved figure to {os.path.join(OUTPUT_DIR, OUTPUT_NAME)}")


if __name__ == "__main__":
    main()
