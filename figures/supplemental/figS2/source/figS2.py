"""
figS2.py

Analyzes degree and weight distributions for the cricket interdependency
network of a single edition year (default: 2019). The network is loaded from a
pre-built GEXF file. For the in-/out-degree and in-/out-weight distributions it:
  1. Computes the empirical distribution and normalizes to probabilities.
  2. Fits a power-law, a Poisson, and a power-law with exponential cutoff.
  3. Scores each fit by log-likelihood and selects the best model by AIC.
  4. Saves a 2x2 SVG plot.

Usage
-----
    # From the repo root:
    python figures/supplemental/figS2/source/figS2.py


Outputs
-------
    ./figures/supplemental/figS2/figS2_final.svg

"""

# ──────────────────────────────────────────────────────────────────────────────
# [0] USER INPUT: FILL IN!
# ──────────────────────────────────────────────────────────────────────────────

reference_list = [
    1835, 1857, 1884, 1890,
    1892, 1896, 1900, 1902, 1906, 1908, 1910, 1911, 1913,
    1914, 1918, 1920, 1923, 1932, 1939, 1947, 1952, 1962,
    1968, 1980, 1992, 2000, 2008, 2010, 2017, 2019,
]

YEAR = 2019 # figS2 features the 2x2 plot for the year 2019. However, feel free
            # to change the YEAR variable to any one of the years from the
            # reference_list above. 

# Repo-root-relative paths (run from repo root).
GEXF_DIR = "./data/datasets/interdependency_networks/graph_files/gexf/multi_count"
OUTPUT_DIR = "./figures/supplemental/figS2"
OUTPUT_NAME = "figS2_final.svg"

# ──────────────────────────────────────────────────────────────────────────────
# [1] IMPORTS
# ──────────────────────────────────────────────────────────────────────────────

import os
import glob
import shutil
import warnings

import numpy as np
import networkx as nx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# [2] MATPLOTLIB / LaTeX CONFIG
# ──────────────────────────────────────────────────────────────────────────────

if shutil.which("latex"):
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = r"\usepackage{amssymb}"
else:
    plt.rcParams["text.usetex"] = False
    plt.rcParams["mathtext.fontset"] = "cm"

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Computer Modern Roman"]

# ──────────────────────────────────────────────────────────────────────────────
# [3] CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

TITLES = {
    "in_degree": "In-Degree",
    "out_degree": "Out-Degree",
    "in_weight": "In-Weight",
    "out_weight": "Out-Weight",
}

# ──────────────────────────────────────────────────────────────────────────────
# [4] GRAPH LOADING HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def create_square_ax(fig_size=6, axes_fraction=0.7, label_pad=10):
    fig = plt.figure(figsize=(fig_size, fig_size))
    ax = fig.add_axes([0.15, 0.15, axes_fraction, axes_fraction])
    ax.set_box_aspect(1)
    return fig, ax


def load_graphs_from_gexf(gexf_dir):
    """
    Load every .gexf file in ``gexf_dir`` into a NetworkX graph.

    The edition year for each file is the first four characters of its
    filename. Returns a dict mapping {year (int): graph}.
    """
    graphs = {}
    for path in sorted(glob.glob(os.path.join(gexf_dir, "*.gexf"))):
        year = int(os.path.basename(path)[:4])
        graphs[year] = nx.read_gexf(path)
    return graphs

# ──────────────────────────────────────────────────────────────────────────────
# [5] DISTRIBUTION MODELS
# ──────────────────────────────────────────────────────────────────────────────

def power_law_normalized(x, alpha):
    """Normalized power-law PMF over discrete support x."""
    w = np.power(x.astype(float), -alpha)
    return w / w.sum()


def poisson_pmf(x, lam):
    """Poisson PMF at integer values x."""
    return poisson.pmf(x.astype(int), lam)


def log_likelihood(counts, probs):
    """LL = sum_i [ count_i * log P(x_i) ]."""
    probs = np.clip(np.asarray(probs, dtype=float), 1e-300, None)
    return float(np.sum(counts * np.log(probs)))

# ──────────────────────────────────────────────────────────────────────────────
# [6] FITTERS
# ──────────────────────────────────────────────────────────────────────────────

def fit_powerlaw(x, counts):
    try:
        mask = x > 0
        if mask.sum() < 3:
            return None, -np.inf

        x_fit = x[mask].astype(float)
        c_fit = counts[mask].astype(float)

        def neg_ll(alpha):
            probs = power_law_normalized(x_fit, alpha)
            return -log_likelihood(c_fit, probs)

        result = minimize_scalar(neg_ll, bounds=(0.5, 10.0), method="bounded")
        return float(result.x), -float(result.fun)
    except Exception:
        return None, -np.inf


def fit_poisson(x, counts):
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

        # Renormalize PMF over x > 0 to match the truncated support.
        raw_probs = poisson_pmf(x_fit, lam)
        prob_sum = raw_probs.sum()
        if prob_sum == 0:
            return None, -np.inf
        probs = raw_probs / prob_sum

        return lam, log_likelihood(c_fit, probs)
    except Exception:
        return None, -np.inf


def fit_powerlaw_cutoff(x, counts):
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
            probs = w / w.sum()
            return -log_likelihood(c_fit, probs)

        best_ll = -np.inf
        best_params = (None, None)

        for a0 in [0.5, 1.5, 2.5]:
            for l0 in [0.01, 0.1, 0.5]:
                res = minimize(
                    neg_ll, x0=[a0, l0], method="Nelder-Mead",
                    options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 5000},
                )
                if res.success or res.fun < -best_ll:
                    ll_val = -float(res.fun)
                    if ll_val > best_ll:
                        best_ll = ll_val
                        best_params = (float(res.x[0]), float(res.x[1]))

        alpha, lam = best_params
        if alpha is None or alpha <= 0:
            return None, None, -np.inf
        if lam is None or lam <= 0:
            return None, None, -np.inf

        return alpha, lam, best_ll
    except Exception:
        return None, None, -np.inf

# ──────────────────────────────────────────────────────────────────────────────
# [7] BUILD DISTRIBUTIONS
# ──────────────────────────────────────────────────────────────────────────────

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
    """Extract the four distributions from a NetworkX graph G."""
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

# ──────────────────────────────────────────────────────────────────────────────
# [8] FIT ALL FOUR DISTRIBUTIONS
# ──────────────────────────────────────────────────────────────────────────────

def fit_distributions(distributions):
    fit_results = {}
    records = []

    for key, (x, counts, probs) in distributions.items():
        if len(x) < 3:
            fit_results[key] = dict(
                best=None,
                pl_alpha=None, pl_ll=-np.inf,
                po_lam=None, po_ll=-np.inf,
                plc_alpha=None, plc_lam=None, plc_ll=-np.inf,
            )
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
        records.append({
            "distribution": key, "best_fit": best,
            "pl_alpha": alpha, "pl_ll": pl_ll,
            "po_lambda": lam, "po_ll": po_ll,
            "plc_alpha": plc_alpha, "plc_lam": plc_lam, "plc_ll": plc_ll,
        })

    return fit_results, records

# ──────────────────────────────────────────────────────────────────────────────
# [9] PLOTTING
# ──────────────────────────────────────────────────────────────────────────────

def make_plot(name, distributions, fit_results, out_path, fig_size=6):
    keys = ["in_degree", "out_degree", "in_weight", "out_weight"]

    out_fig, out_axes = plt.subplots(2, 2, figsize=(fig_size * 2, fig_size * 2))
    out_fig.suptitle(
        f"Degree / Weight Distributions -- {name}", fontsize=32, fontweight="bold"
    )

    for ax, key in zip(out_axes.flat, keys):
        x, counts, probs = distributions[key]
        fr = fit_results[key]
        title = TITLES[key]

        if len(x) == 0:
            ax.set_title(title)
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        mask_pos = x > 0
        x_display = x[mask_pos]
        probs_display = probs[mask_pos]

        if len(x_display) == 0:
            ax.set_title(title)
            ax.text(0.5, 0.5, "No positive values\n(all mass at 0)",
                    ha="center", va="center", transform=ax.transAxes)
            continue

        probs_display = probs_display / probs_display.sum()

        ax.bar(
            x_display, probs_display,
            width=max((x_display.max() - x_display.min()) / len(x_display) * 0.9, 0.8),
            fill=False, edgecolor="#555555", linewidth=1, zorder=2,
        )

        best = fr.get("best")

        if fr["pl_alpha"] is not None:
            x_plot = x[x > 0]
            y_fit = power_law_normalized(x_plot, fr["pl_alpha"])
            is_best = best == "powerlaw"
            ax.plot(
                x_plot, y_fit, color="#9D0614",
                lw=5 if is_best else 2, linestyle="-",
                alpha=1.0 if is_best else 0.8,
                label=(f"{'$\\bigstar$ ' if is_best else ''}Power-law\n"
                       f"  $\\alpha$={fr['pl_alpha']:.3f}"),
            )

        if fr["po_lam"] is not None:
            x_int = np.arange(max(1, int(x.min())), int(x.max()) + 1)
            y_fit = poisson_pmf(x_int, fr["po_lam"])
            prob_sum = y_fit.sum()
            if prob_sum > 0:
                y_fit = y_fit / prob_sum
            is_best = best == "poisson"
            ax.plot(
                x_int, y_fit, color="#4C78A8",
                lw=5 if is_best else 2, linestyle="-",
                alpha=1.0 if is_best else 0.8,
                marker="o", ms=4 if is_best else 2,
                label=(f"{'$\\bigstar$ ' if is_best else ''}Poisson\n"
                       f"  $\\lambda$={fr['po_lam']:.3f}"),
            )

        if fr["plc_alpha"] is not None and fr["plc_lam"] is not None:
            x_plot = x[x > 0]
            w = np.power(x_plot.astype(float), -fr["plc_alpha"]) * np.exp(
                -fr["plc_lam"] * x_plot.astype(float)
            )
            y_fit = w / w.sum()
            is_best = best == "powerlaw_cutoff"
            ax.plot(
                x_plot, y_fit, color="#228B22",
                lw=5 if is_best else 2, linestyle="-",
                alpha=1.0 if is_best else 0.8,
                label=(f"{'$\\bigstar$ ' if is_best else ''}PL + exp cutoff\n"
                       f"  $\\alpha$={fr['plc_alpha']:.3f}  $\\lambda$={fr['plc_lam']:.3f}"),
            )

        ax.set_xlabel(title[0] + title[1:].lower(), fontsize=24, labelpad=5)
        ax.set_ylabel("Probability", fontsize=24, labelpad=5)
        ax.tick_params(axis="x", labelsize=14)
        ax.tick_params(axis="y", labelsize=14)
        ax.legend(fontsize=10, loc="upper right")
        ax.set_xlim(left=max(x.min() - 0.5, 0))

    out_fig.tight_layout()
    out_fig.savefig(out_path, bbox_inches="tight")
    plt.close(out_fig)
    print(f"    Plot saved → {out_path}")

# ──────────────────────────────────────────────────────────────────────────────
# [10] MAIN ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def analyze_graphs(graphs: dict, output_dir: str, output_name: str):
    os.makedirs(output_dir, exist_ok=True)

    for name, G in graphs.items():
        print(f"  Processing: {name}  ({G.number_of_nodes()} nodes, "
              f"{G.number_of_edges()} edges)")
        distributions = get_distributions(G)
        fit_results, _ = fit_distributions(distributions)

        plot_path = os.path.join(output_dir, output_name)
        make_plot(name, distributions, fit_results, plot_path)

# ──────────────────────────────────────────────────────────────────────────────
# [11] ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    all_graphs = load_graphs_from_gexf(GEXF_DIR)

    graphs = {YEAR: all_graphs[YEAR]}

    analyze_graphs(graphs, output_dir=OUTPUT_DIR, output_name=OUTPUT_NAME)