"""
figS3_plot_year.py 

Plots the distribution of weighted local clustering coefficients for a given year's
interdependency network, overlaid with a maximum-likelihood Uniform fit and a
power-law fit (via the `powerlaw` package). The better-fitting model (by
log-likelihood ratio) is marked with a star in the legend.

Usage
-----
    # From the repo root:
    python figures/supplemental/figS3/source/figS3_plot_year.py

Outputs
-------
    figures/supplemental/figS3/components/figS3_{year}.svg
"""

# ──────────────────────────────────────────────────────────────────────────────
# [0] IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import os
import re
import glob
import shutil

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import powerlaw

# ──────────────────────────────────────────────────────────────────────────────
# [1] MATPLOTLIB / LaTeX CONFIG
# ──────────────────────────────────────────────────────────────────────────────
if shutil.which("latex"):
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Computer Modern Roman"]
    plt.rcParams["text.latex.preamble"] = r"\usepackage{amssymb}"  # for \bigstar
else:
    plt.rcParams["text.usetex"] = False
    plt.rcParams["mathtext.fontset"] = "cm"

# ──────────────────────────────────────────────────────────────────────────────
# [2] CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

reference_year_list = [
    "1835", "1857", "1884", "1890",
    "1892", "1896", "1900", "1902", "1906", "1908", "1910", "1911", "1913",
    "1914", "1918", "1920", "1923", "1932", "1939", "1947", "1952", "1962",
    "1968", "1980", "1992", "2000", "2008", "2010", "2017", "2019",
]

YEAR = "2019" # <-- can insert any year from reference year list above. 
            # figS3 has the plots from 1918, 1968 and 2019. 

GEXF_DIR = "./data/datasets/interdependency_networks/graph_files/gexf/multi_count"
OUTPUT_DIR = "./figures/supplemental/figS3/components"

REMOVE_SELF_EDGES = True
FILE_PATTERN = re.compile(r"(\d{4})")  # Extract year from filename

COL_UNIFORM = "#F58518"   # orange
COL_POWERLAW = "#9D0614"  # dark red

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# [3] HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def create_square_ax(fig_size=6, axes_fraction=0.7, label_pad=10):
    fig = plt.figure(figsize=(fig_size, fig_size))
    ax = fig.add_axes([0.15, 0.15, axes_fraction, axes_fraction])
    ax.set_box_aspect(1)
    return fig, ax


def find_gexf_for_year(year):
    """Locate the GEXF file whose name contains the given 4-digit year."""
    for fpath in sorted(glob.glob(os.path.join(GEXF_DIR, "*.gexf"))):
        match = FILE_PATTERN.search(os.path.basename(fpath))
        if match and match.group(1) == year:
            return fpath
    raise FileNotFoundError(f"No GEXF file for year {year} in {GEXF_DIR}")


def load_graph(path):
    """Read a GEXF file, assert it is directed, and optionally drop self-loops."""
    G = nx.read_gexf(path)
    if not G.is_directed():
        raise ValueError(
            f"{os.path.basename(path)} read as undirected; directed clustering "
            "is expected for these networks. Verify the GEXF export."
        )
    if REMOVE_SELF_EDGES:
        G.remove_edges_from(nx.selfloop_edges(G))
    return G


def weighted_local_cc_values(G):
    """Weighted directed local clustering coefficients as a 1-D array."""
    cc = nx.clustering(G, weight="weight")
    return np.array(list(cc.values()), dtype=float)


def prepare_distribution(values):
    """Drop NaNs and non-positive values (powerlaw/​log fits need positive support)."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    values = values[values > 0]
    return values


def uniform_loglikelihood(data, a, b):
    """Log-likelihood for Uniform(a, b): -n log(b - a) if all data in [a, b], else -inf."""
    if b <= a or np.any(data < a) or np.any(data > b):
        return -np.inf
    return -len(data) * np.log(b - a)


# ──────────────────────────────────────────────────────────────────────────────
# [4] COMPUTE WEIGHTED LOCAL CLUSTERING DISTRIBUTION
# ──────────────────────────────────────────────────────────────────────────────
gexf_path = find_gexf_for_year(YEAR)
G = load_graph(gexf_path)
vals = prepare_distribution(weighted_local_cc_values(G))

if len(vals) < 5:
    raise ValueError(
        f"Only {len(vals)} positive weighted CC values for {YEAR}; too few to fit."
    )

# ──────────────────────────────────────────────────────────────────────────────
# [5] FIT UNIFORM AND POWER LAW, RESOLVE WINNER
# ──────────────────────────────────────────────────────────────────────────────
# Uniform MLE
a_mle = vals.min()
b_mle = vals.max()
uniform_ll = uniform_loglikelihood(vals, a_mle, b_mle)

# Power-law fit
alpha = xmin = None
x_pl = y_pl = None
power_law_ll = np.nan
try:
    fit = powerlaw.Fit(vals, verbose=False)
    alpha = fit.power_law.alpha
    xmin = fit.power_law.xmin
    x_pl = np.linspace(xmin, max(vals), 200)
    C = (alpha - 1) / xmin
    y_pl = C * (x_pl / xmin) ** (-alpha)
    power_law_ll = fit.power_law.loglikelihoods(vals).sum()
except Exception as e:
    print(f"{YEAR}: power law fit failed -> {e}")

if not np.isnan(power_law_ll):
    LLR = power_law_ll - uniform_ll
    winner = "power_law" if LLR > 0 else "uniform"
else:
    winner = "uniform"  # fall back when PL fit failed

# ──────────────────────────────────────────────────────────────────────────────
# [6] PLOT
# ──────────────────────────────────────────────────────────────────────────────
fig, ax = create_square_ax(fig_size=8, axes_fraction=0.75)

# Histogram: hollow bars with dark outline
ax.hist(vals, bins=20, density=True, fill=False,
        edgecolor="#555555", linewidth=1, zorder=2)

# Uniform line
is_best_uniform = (winner == "uniform")
ax.plot([a_mle, b_mle], [1.0 / (b_mle - a_mle)] * 2,
        color=COL_UNIFORM,
        lw=5 if is_best_uniform else 2,
        linestyle="-",
        alpha=1.0 if is_best_uniform else 0.8,
        label=f"{'$\\bigstar$ ' if is_best_uniform else ''}Uniform")

# Power-law line
if x_pl is not None:
    is_best_pl = (winner == "power_law")
    ax.plot(x_pl, y_pl,
            color=COL_POWERLAW,
            lw=5 if is_best_pl else 2,
            linestyle="-",
            alpha=1.0 if is_best_pl else 0.8,
            label=(f"{'$\\bigstar$ ' if is_best_pl else ''}Power law"
                   f"\n  $\\alpha$={alpha:.3f}"))

ax.set_xlabel("Local clustering coefficient", fontsize=24, labelpad=5)
ax.set_ylabel("Probability", fontsize=24, labelpad=5)
ax.tick_params(axis="x", labelsize=14)
ax.tick_params(axis="y", labelsize=14)
ax.legend(fontsize=10, loc="upper right")
ax.set_title(YEAR, fontsize=32, pad=8)

out_path = os.path.join(OUTPUT_DIR, f"figS3_{YEAR}.svg")
plt.savefig(out_path, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out_path}")

print("\nDone.")