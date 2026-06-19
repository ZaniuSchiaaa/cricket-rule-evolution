"""
fig8.py

Combined pipeline for the cricket-rule interdependency-network analysis.

Step 1 (fit): build the interdependency network for every year with a network
(1835 onwards) and fit three candidate models (power-law, Poisson, and
power-law with exponential cutoff) to each of its in-degree, out-degree,
in-weight, and out-weight distributions. The best model per distribution is
selected by AIC.

Step 2 (write): save all fit parameters to a single CSV.

Step 3 (plot): using the same in-memory results (no CSV round-trip), draw a
colour-band timeline of the best-fit weight distribution per edition.

Usage
-----
    # From the repo root:
    python figures/main_body/fig8/source/fig8.py

Outputs
-------
    ./figures/main_body/fig8/fit_parameters.csv
    ./figures/main_body/fig8/fig8_final.svg
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import poisson
from scipy.optimize import minimize, minimize_scalar


# ===============
# [1] MATPLOTLIB / LaTeX CONFIG
# ===============

USE_TEX = shutil.which("latex") is not None

plt.rcParams["text.usetex"] = USE_TEX
plt.rcParams["font.family"] = "serif"
if USE_TEX:
    plt.rcParams["font.serif"] = ["Computer Modern Roman"]
else:
    plt.rcParams["mathtext.fontset"] = "cm"
    print("Note: LaTeX not found on PATH; falling back to matplotlib's mathtext.")


# ==============
# [2] CONSTANTS
# ==============

FIRST_NETWORK_YEAR = 1823  # earliest year with an interdependency network
REMOVE_SELF_EDGES = True

CITATION_TABLE_TEMPLATE = "./data/datasets/interdependency_networks/citation_tables/{year}_citation_table.csv"
OUTPUT_CSV = "./figures/main_body/fig8/fit_parameters.csv"
VIS_FILEPATH = "./figures/main_body/fig8/fig8_final.svg"

NUM_LAWS_BY_YEAR = {
    1752: 33, 1755: 35, 1774: 34, 1785: 33, 1786: 34, 1788: 37,
    1803: 40, 1806: 45, 1809: 41, 1816: 44, 1820: 44, 1823: 43,
    1828: 43, 1830: 47, 1835: 47, 1857: 47, 1884: 54, 1890: 54,
    1892: 54, 1896: 54, 1900: 54, 1902: 54, 1906: 54, 1908: 54,
    1910: 54, 1911: 54, 1913: 54, 1914: 55, 1918: 55, 1920: 55,
    1923: 55, 1932: 55, 1939: 55, 1947: 47, 1952: 47, 1962: 47,
    1968: 47, 1980: 42, 1992: 42, 2000: 42, 2008: 42, 2010: 42,
    2017: 42, 2019: 42,
}

DIST_KEYS = ["in_degree", "out_degree", "in_weight", "out_weight"]

CSV_COLUMNS = [
    "year", "distribution", "best_fit",
    "pl_alpha", "pl_ll",
    "po_lambda", "po_ll",
    "plc_alpha", "plc_lam", "plc_ll",
]

# ── Plotting (fig8) ────────────────────────────────────────────────────────
COLORS: dict[str | None, str] = {
    "powerlaw": "#ECC7D8",
    "poisson": "#C5E5D4",
    "powerlaw_cutoff": "#FFE9C8",
    None: "#B0B0B0",
}

LABELS: dict[str | None, str] = {
    "powerlaw": "Power-law",
    "poisson": "Poisson",
    "powerlaw_cutoff": "PL + exp cutoff",
    None: "Unknown / missing",
}

# Timeline rows to plot, in order. Each entry is (distribution name, y-axis label).
DISTRIBUTIONS: list[tuple[str, str]] = [
    ("in_weight", "In-weight"),
    ("out_weight", "Out-weight"),
]


# ===================
# [3] GRAPH CONSTRUCTION
# ===================

def clean_laws(entry):
    """From a comma-separated list of cited rules, extract the high-level law numbers."""
    if pd.isna(entry):
        return []
    result = []
    for law in entry.split(","):
        law = law.strip()
        result.append(law.split(".")[0] if "." in law else law)
    return result


def extract_first_num(value):
    """Extract the high-level law number from a leftmost rule identifier."""
    return str(value).split(".")[0]


def create_graph(year):
    """Build the directed, weighted interdependency graph for a given year."""
    df = pd.read_csv(CITATION_TABLE_TEMPLATE.format(year=year))
    df["rule_first_char"] = df["rule"].apply(extract_first_num)
    df["cited_laws_clean"] = df["cited_laws"].apply(clean_laws)

    G = nx.DiGraph()
    node_list = [str(i + 1) for i in range(int(NUM_LAWS_BY_YEAR[int(year)]))]
    G.add_nodes_from(node_list)

    for _, row in df.iterrows():
        source = row["rule_first_char"]
        for target in row["cited_laws_clean"]:
            if G.has_edge(source, target):
                G[source][target]["weight"] += 1
            else:
                G.add_edge(source, target, weight=1)

    if REMOVE_SELF_EDGES:
        G.remove_edges_from(nx.selfloop_edges(G))

    degree_dict = dict(G.out_degree(weight="weight"))
    for n in G.nodes():
        G.nodes[n]["label"] = str(n)
        G.nodes[n]["degree"] = degree_dict[n]

    # Restrict to the main ruleset (numeric law nodes only).
    numeric_nodes = [n for n in G.nodes if str(n).isdigit()]
    return G.subgraph(numeric_nodes).copy()


# ===================
# [4] DISTRIBUTION MODELS
# ===================

def power_law_normalized(x, alpha):
    """Normalized power-law PMF over discrete support x: P(x_i) ∝ x_i^{-alpha}."""
    w = np.power(x.astype(float), -alpha)
    return w / w.sum()


def poisson_pmf(x, lam):
    """Poisson PMF at integer values x."""
    return poisson.pmf(x.astype(int), lam)


def log_likelihood(counts, probs):
    """Log-likelihood of observed counts given model probabilities."""
    probs = np.clip(np.asarray(probs, dtype=float), 1e-300, None)
    return float(np.sum(counts * np.log(probs)))


# ===================
# [5] FITTERS
# ===================

def fit_powerlaw(x, counts):
    """Fit a power-law by maximizing log-likelihood over alpha. Returns (alpha, ll)."""
    try:
        mask = x > 0
        if mask.sum() < 3:
            return None, -np.inf

        x_fit = x[mask].astype(float)
        c_fit = counts[mask].astype(float)

        def neg_ll(alpha):
            return -log_likelihood(c_fit, power_law_normalized(x_fit, alpha))

        result = minimize_scalar(neg_ll, bounds=(0.5, 10.0), method="bounded")
        return float(result.x), -float(result.fun)
    except Exception:
        return None, -np.inf


def fit_poisson(x, counts):
    """Fit a Poisson (MLE lambda) renormalized over the observed x > 0 support."""
    try:
        mask = x > 0
        if mask.sum() < 3:
            return None, -np.inf

        x_fit = x[mask].astype(int)
        c_fit = counts[mask].astype(float)

        total = c_fit.sum()
        if total == 0:
            return None, -np.inf

        lam = float(np.sum(x_fit * c_fit) / total)
        if lam <= 0:
            return None, -np.inf

        raw_probs = poisson_pmf(x_fit, lam)
        prob_sum = raw_probs.sum()
        if prob_sum == 0:
            return None, -np.inf
        probs = raw_probs / prob_sum

        return lam, log_likelihood(c_fit, probs)
    except Exception:
        return None, -np.inf


def fit_powerlaw_cutoff(x, counts):
    """Fit a power-law with exponential cutoff. Returns (alpha, lam, ll)."""
    try:
        mask = x > 0
        if mask.sum() < 3:
            return None, None, -np.inf

        x_fit = x[mask].astype(float)
        c_fit = counts[mask].astype(float)

        def neg_ll(params):
            alpha, lam = params
            if lam <= 0:
                return np.inf
            w = np.power(x_fit, -alpha) * np.exp(-lam * x_fit)
            if w.sum() == 0:
                return np.inf
            return -log_likelihood(c_fit, w / w.sum())

        best_ll = -np.inf
        best_params = (None, None)
        for a0 in [0.5, 1.5, 2.5]:
            for l0 in [0.01, 0.1, 0.5]:
                res = minimize(
                    neg_ll, x0=[a0, l0], method="Nelder-Mead",
                    options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 5000},
                )
                ll_val = -float(res.fun)
                if ll_val > best_ll:
                    best_ll = ll_val
                    best_params = (float(res.x[0]), float(res.x[1]))

        alpha, lam = best_params
        if alpha is None or alpha <= 0 or lam is None or lam <= 0:
            return None, None, -np.inf
        return alpha, lam, best_ll
    except Exception:
        return None, None, -np.inf


# ===================
# [6] DISTRIBUTIONS
# ===================

def empirical_distribution(values):
    """Return (x, counts, probs) for a list of non-negative numeric values."""
    values = np.array(values, dtype=float)
    if len(values) == 0:
        return np.array([]), np.array([]), np.array([])
    vmin, vmax = values.min(), values.max()
    if vmin == vmax:
        return np.array([vmin]), np.array([float(len(values))]), np.array([1.0])
    bins = np.arange(vmin, vmax + 2)
    counts, edges = np.histogram(values, bins=bins)
    x = edges[:-1]
    total = counts.sum()
    probs = counts / total if total > 0 else counts.astype(float)
    return x, counts.astype(float), probs


def get_distributions(G):
    """Extract in/out degree and in/out weight distributions from G."""
    if G.is_directed():
        in_deg = [d for _, d in G.in_degree()]
        out_deg = [d for _, d in G.out_degree()]
    else:
        deg = [d for _, d in G.degree()]
        in_deg = out_deg = deg

    in_weights = {n: 0.0 for n in G.nodes()}
    out_weights = {n: 0.0 for n in G.nodes()}
    for u, v, data in G.edges(data=True):
        w = float(data.get("weight", 1.0))
        out_weights[u] += w
        in_weights[v] += w

    return {
        "in_degree": empirical_distribution(in_deg),
        "out_degree": empirical_distribution(out_deg),
        "in_weight": empirical_distribution(list(in_weights.values())),
        "out_weight": empirical_distribution(list(out_weights.values())),
    }


def fit_distributions(distributions):
    """Fit all three models per distribution and select the best by AIC."""
    records = []

    for key in DIST_KEYS:
        x, counts, probs = distributions[key]

        if len(x) < 3:
            records.append({
                "distribution": key, "best_fit": None,
                "pl_alpha": None, "pl_ll": None,
                "po_lambda": None, "po_ll": None,
                "plc_alpha": None, "plc_lam": None, "plc_ll": None,
            })
            continue

        alpha, pl_ll = fit_powerlaw(x, counts)
        lam, po_ll = fit_poisson(x, counts)
        plc_alpha, plc_lam, plc_ll = fit_powerlaw_cutoff(x, counts)

        # AIC = 2k - 2*LL; lower is better. k = number of free parameters.
        aics = {
            "powerlaw": 2 * 1 - 2 * pl_ll,
            "poisson": 2 * 1 - 2 * po_ll,
            "powerlaw_cutoff": 2 * 2 - 2 * plc_ll,
        }
        best = min(aics, key=aics.get)

        records.append({
            "distribution": key, "best_fit": best,
            "pl_alpha": alpha, "pl_ll": pl_ll,
            "po_lambda": lam, "po_ll": po_ll,
            "plc_alpha": plc_alpha, "plc_lam": plc_lam, "plc_ll": plc_ll,
        })

    return records


# =================
# [7] DATA LOADING
# =================

def networked_years():
    """Return sorted years (1835+) that have an interdependency network."""
    return sorted(y for y in NUM_LAWS_BY_YEAR if y >= FIRST_NETWORK_YEAR)


def build_records():
    """Fit every networked year and return a flat list of per-distribution records."""
    all_records = []

    for year in networked_years():
        csv_path = CITATION_TABLE_TEMPLATE.format(year=year)
        if not os.path.exists(csv_path):
            print(f"  Skipping {year}: no citation table at {csv_path}")
            continue

        G = create_graph(year)
        print(f"  Processed {year}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        distributions = get_distributions(G)
        for record in fit_distributions(distributions):
            record["year"] = year
            all_records.append(record)

    return all_records


# ===================
# [8] PLOTTING (TIMELINE)
# ===================

def _draw_band_row(ax, df, years, dist_col, y_label):
    """Fill one timeline row with colour bands and thin dividers."""
    for i, year in enumerate(years):
        # Span runs to the next edition year (or an estimated gap for the last).
        gap = year - years[i - 1] if i > 0 else years[1] - years[0]
        next_year = years[i + 1] if i + 1 < len(years) else year + gap

        subset = df[(df["year"] == year) & (df["distribution"] == dist_col)]
        best = (
            None
            if subset.empty or pd.isna(subset["best_fit"].values[0])
            else subset["best_fit"].values[0]
        )

        ax.axvspan(
            year, next_year,
            facecolor=COLORS.get(best, COLORS[None]),
            alpha=0.85, linewidth=0,
        )

    for year in years:
        ax.axvline(year, color="white", linewidth=0.8, alpha=0.6)

    ax.set_yticks([])
    ax.set_ylabel(y_label, rotation=0, labelpad=40, va="center", fontsize=14)
    ax.set_xlim(min(years), years[-1] + (years[-1] - years[-2]))
    ax.tick_params(axis="both", labelsize=12)


def _format_x_axis(ax, years):
    """Set sensible ticks (every 20 years) on the shared x-axis."""
    start = int(np.floor(min(years) / 20) * 20)
    end = int(np.ceil(max(years) / 20) * 20)
    ax.set_xticks(np.arange(start, end + 1, 20))
    ax.set_xlabel("Year", fontsize=14)


def _add_legend(fig):
    """Add a centred four-column legend below the subplots."""
    handles = [
        mpatches.Patch(facecolor=color, label=LABELS[key])
        for key, color in COLORS.items()
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.14),
        frameon=True,
        ncol=4,
    )


def plot_best_fit_timeline(df, output_path):
    """Draw a colour-band timeline of best-fit distributions from a results DataFrame."""
    df = df.copy()
    df["year"] = df["year"].astype(int)
    years = sorted(df["year"].unique())

    fig, axes = plt.subplots(
        len(DISTRIBUTIONS), 1,
        figsize=(12, len(DISTRIBUTIONS) * 1.7),
        sharex=True,
    )

    for ax, (dist_col, y_label) in zip(axes, DISTRIBUTIONS):
        _draw_band_row(ax, df, years, dist_col, y_label)

    _format_x_axis(axes[-1], years)

    fig.suptitle("Best-fit distribution over time", y=0.90, fontsize=20)
    _add_legend(fig)
    plt.tight_layout(rect=[0, 0.10, 1, 0.95])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {output_path}")

    return fig


# ==========
# [9] MAIN
# ==========

def main():
    # Step 1 + 2: fit all networked years and build the results DataFrame.
    records = build_records()
    df = pd.DataFrame(records)[CSV_COLUMNS]

    # Step 2: write the CSV.
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, float_format="%.6g")
    print(f"\nSaved fit parameters for {df['year'].nunique()} years to {OUTPUT_CSV}")

    # Step 3: plot the timeline from the same in-memory results.
    plot_best_fit_timeline(df, VIS_FILEPATH)


if __name__ == "__main__":
    main()