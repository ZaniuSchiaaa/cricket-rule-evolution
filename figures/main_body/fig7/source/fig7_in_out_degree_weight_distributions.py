"""
fig7_in_out_degree_weight_distributions.py

Build a cricket-rule interdependency network and plot its in-degree,
out-degree, in-weight, and out-weight distributions in a single 2x2 figure.

For each of the four distributions the script fits three candidate models
(power-law, Poisson, and power-law with exponential cutoff), selects the best
by AIC, and overlays all three with the selected model highlighted.

Usage
-----
    # From the repo root:
    python figures/main_body/fig7/source/fig7_in_out_degree_weight_distributions.py

Outputs
-------
    ./figures/main_body/fig7/components/fig7_{year}_in_out_degree_weight_distributions.svg
"""


# ===========
# [0] IMPORTS
# ===========

import os
import shutil

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    # amssymb provides \bigstar, which is not in matplotlib's default LaTeX preamble.
    plt.rcParams["text.latex.preamble"] = r"\usepackage{amssymb}"
else:
    plt.rcParams["mathtext.fontset"] = "cm"
    print("Note: LaTeX not found on PATH; falling back to matplotlib's mathtext.")

# Prefix marking the AIC-selected (best) model in legend labels. Both mathtext
# and (with amssymb loaded) LaTeX understand \bigstar.
BEST_MARKER = r"$\bigstar$ "


# ==============
# [2] CONSTANTS
# ==============

YEAR = "1918" # Substitute in with "1962" or "2019",
              # or any other year with an interdependency network (i.e. 1835 onwards)
REMOVE_SELF_EDGES = True

CITATION_TABLE_TEMPLATE = "./data/datasets/interdependency_networks/citation_tables/{year}_citation_table.csv"
VIS_FILEPATH = f"./figures/main_body/fig7/components/fig7_{YEAR}_in_out_degree_weight_distributions.svg"

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
TITLES = {
    "in_degree": "In-Degree",
    "out_degree": "Out-Degree",
    "in_weight": "In-Weight",
    "out_weight": "Out-Weight",
}

COLOR_POWERLAW = "#9D0614"
COLOR_POISSON = "#4C78A8"
COLOR_CUTOFF = "#228B22"


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
    fit_results = {}

    for key, (x, counts, probs) in distributions.items():
        if len(x) < 3:
            fit_results[key] = dict(
                best=None,
                pl_alpha=None, pl_ll=-np.inf,
                po_lam=None, po_ll=-np.inf,
                plc_alpha=None, plc_lam=None, plc_ll=-np.inf,
            )
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

        fit_results[key] = dict(
            best=best,
            pl_alpha=alpha, pl_ll=pl_ll,
            po_lam=lam, po_ll=po_ll,
            plc_alpha=plc_alpha, plc_lam=plc_lam, plc_ll=plc_ll,
        )

    return fit_results


# ===================
# [7] PLOTTING
# ===================

def _draw_panel(ax, key, distributions, fit_results):
    """Draw one distribution panel with its three candidate-model overlays."""
    x, counts, probs = distributions[key]
    fr = fit_results[key]
    title = TITLES[key]

    if len(x) == 0:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    mask_pos = x > 0
    x_display = x[mask_pos]
    probs_display = probs[mask_pos]
    if len(x_display) == 0:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No positive values\n(all mass at 0)",
                ha="center", va="center", transform=ax.transAxes)
        return

    probs_display = probs_display / probs_display.sum()
    best = fr.get("best")

    ax.bar(
        x_display, probs_display,
        width=max((x_display.max() - x_display.min()) / len(x_display) * 0.9, 0.8),
        fill=False, edgecolor="#555555", linewidth=1, zorder=2,
    )

    if fr["pl_alpha"] is not None:
        x_plot = x[x > 0]
        is_best = best == "powerlaw"
        ax.plot(
            x_plot, power_law_normalized(x_plot, fr["pl_alpha"]),
            color=COLOR_POWERLAW, lw=5 if is_best else 2,
            alpha=1.0 if is_best else 0.8,
            label=f"{BEST_MARKER if is_best else ''}Power-law\n  $\\alpha$={fr['pl_alpha']:.3f}",
        )

    if fr["po_lam"] is not None:
        x_int = np.arange(max(1, int(x.min())), int(x.max()) + 1)
        y_fit = poisson_pmf(x_int, fr["po_lam"])
        prob_sum = y_fit.sum()
        if prob_sum > 0:
            y_fit = y_fit / prob_sum
        is_best = best == "poisson"
        ax.plot(
            x_int, y_fit, color=COLOR_POISSON, lw=5 if is_best else 2,
            alpha=1.0 if is_best else 0.8, marker="o", ms=4 if is_best else 2,
            label=f"{BEST_MARKER if is_best else ''}Poisson\n  $\\lambda$={fr['po_lam']:.3f}",
        )

    if fr["plc_alpha"] is not None and fr["plc_lam"] is not None:
        x_plot = x[x > 0].astype(float)
        w = np.power(x_plot, -fr["plc_alpha"]) * np.exp(-fr["plc_lam"] * x_plot)
        is_best = best == "powerlaw_cutoff"
        ax.plot(
            x_plot, w / w.sum(), color=COLOR_CUTOFF, lw=5 if is_best else 2,
            alpha=1.0 if is_best else 0.8,
            label=(f"{BEST_MARKER if is_best else ''}PL + exp cutoff\n"
                   f"  $\\alpha$={fr['plc_alpha']:.3f}  $\\lambda$={fr['plc_lam']:.3f}"),
        )

    ax.set_xlabel(title[0] + title[1:].lower(), fontsize=24, labelpad=5)
    ax.set_ylabel("Probability", fontsize=24, labelpad=5)
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(fontsize=10, loc="upper right")
    ax.set_xlim(left=max(x.min() - 0.5, 0))


def make_plot(name, distributions, fit_results, out_path, fig_size=6):
    """Render the 2x2 distribution figure and save it to out_path."""
    fig, axes = plt.subplots(2, 2, figsize=(fig_size * 2, fig_size * 2))
    fig.suptitle(f"Degree / Weight Distributions -- {name}", fontsize=32, fontweight="bold")

    for ax, key in zip(axes.flat, DIST_KEYS):
        _draw_panel(ax, key, distributions, fit_results)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {out_path}")


# ==========
# [8] MAIN
# ==========

def main():
    G = create_graph(YEAR)
    print(f"Processing {YEAR}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    distributions = get_distributions(G)
    fit_results = fit_distributions(distributions)
    make_plot(YEAR, distributions, fit_results, VIS_FILEPATH)


if __name__ == "__main__":
    main()