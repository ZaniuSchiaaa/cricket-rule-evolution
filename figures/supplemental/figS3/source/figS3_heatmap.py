"""
figS3_heatmap.py

Computes, the best-fitting distribution (Uniform vs. power-law) for the
distribution of weighted local clustering coefficients in each year's
interdependency network, then renders a best-fit timeline: a horizontal band per
year coloured by which distribution wins (by log-likelihood ratio).

Usage
-----
    # From the repo root:
    python figures/supplemental/figS3/source/figS3_heatmap.py

Outputs
-------
    figures/supplemental/figS3/components/figS3_heatmap.svg
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
import matplotlib.patches as mpatches
import powerlaw

# ──────────────────────────────────────────────────────────────────────────────
# [1] MATPLOTLIB / LaTeX CONFIG
# ──────────────────────────────────────────────────────────────────────────────
if shutil.which("latex"):
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Computer Modern Roman"]
    plt.rcParams["text.latex.preamble"] = r"\usepackage{amssymb}"
else:
    plt.rcParams["text.usetex"] = False
    plt.rcParams["mathtext.fontset"] = "cm"

# ──────────────────────────────────────────────────────────────────────────────
# [2] CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
GEXF_DIR = "./data/datasets/interdependency_networks/graph_files/gexf/multi_count"
OUTPUT_DIR = "./figures/supplemental/figS3/components"

REMOVE_SELF_EDGES = True
FILE_PATTERN = re.compile(r"(\d{4})")  # Extract year from filename

COLOR_MAP = {
    "power_law": "#ECC7D8",
    "uniform":   "#FFE9C8",
}
LABEL_MAP = {
    "power_law": "Power-law",
    "uniform":   "Uniform",
}
COLOR_MISSING = "#B0B0B0"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# [3] HELPERS — GRAPH + FIT
# ──────────────────────────────────────────────────────────────────────────────
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
    """Drop NaNs and non-positive values."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    values = values[values > 0]
    return values


def uniform_loglikelihood(data, a, b):
    """Log-likelihood for Uniform(a, b): -n log(b - a) if all data in [a, b], else -inf."""
    if b <= a or np.any(data < a) or np.any(data > b):
        return -np.inf
    return -len(data) * np.log(b - a)


def best_fit_for_year(year, path):
    """Return the winning distribution name for one year, or None if undetermined."""
    G = load_graph(path)
    vals = prepare_distribution(weighted_local_cc_values(G))

    if len(vals) < 5:
        print(f"{year}: only {len(vals)} positive values; winner undetermined.")
        return None

    # Uniform MLE
    a_mle = vals.min()
    b_mle = vals.max()
    uniform_ll = uniform_loglikelihood(vals, a_mle, b_mle)

    # Power-law fit
    power_law_ll = np.nan
    try:
        fit = powerlaw.Fit(vals, verbose=False)
        power_law_ll = fit.power_law.loglikelihoods(vals).sum()
    except Exception as e:
        print(f"{year}: power law fit failed -> {e}")

    if not np.isnan(power_law_ll):
        LLR = power_law_ll - uniform_ll
        return "power_law" if LLR > 0 else "uniform"
    return "uniform"  # fall back when PL fit failed


# ──────────────────────────────────────────────────────────────────────────────
# [4] DISCOVER YEARS AND COMPUTE WINNERS
# ──────────────────────────────────────────────────────────────────────────────
gexf_files = sorted(glob.glob(os.path.join(GEXF_DIR, "*.gexf")))

# Map year -> path (first match wins if duplicates), then sort by year
year_to_path = {}
for fpath in gexf_files:
    match = FILE_PATTERN.search(os.path.basename(fpath))
    if match:
        year = int(match.group(1))
        year_to_path.setdefault(year, fpath)

if not year_to_path:
    raise FileNotFoundError(f"No year-tagged GEXF files found in {GEXF_DIR}")

# years = sorted(year_to_path.keys())
# winner_by_year = {}
# for year in years:
#     print(f"Processing {year}...")
#     winner_by_year[year] = best_fit_for_year(year, year_to_path[year])
years = sorted(year_to_path.keys())
winner_by_year = {}
for year in years:
    print(f"Processing {year}...")
    w = best_fit_for_year(year, year_to_path[year])
    if w is not None:                     # skip sub-threshold years, as script 2/3 does
        winner_by_year[year] = w

# Only keep years that produced a winner — this is what makes the greys vanish
years = sorted(winner_by_year.keys())

# ──────────────────────────────────────────────────────────────────────────────
# [5] PLOT BEST-FIT TIMELINE
# ──────────────────────────────────────────────────────────────────────────────
# Guard: need >= 2 years to define band widths from spacing.
if len(years) < 2:
    raise ValueError("Need at least 2 years to build the timeline.")

fig, ax = plt.subplots(figsize=(14, 3))

# Continuous coloured spans: each year extends to the next observed year.
for i, year in enumerate(years):
    if i + 1 < len(years):
        next_year = years[i + 1]
    else:
        # extrapolate final band width from the last gap
        next_year = year + (year - years[i - 1])

    best = winner_by_year.get(year)
    color = COLOR_MAP.get(best, COLOR_MISSING)
    ax.axvspan(year, next_year, facecolor=color, alpha=0.85, linewidth=0)

# Thin white separators at each observed year.
for year in years:
    ax.axvline(year, color="white", linewidth=1.6, alpha=0.95)

ax.set_ylim(0, 1)
ax.set_yticks([])
ax.set_xlim(min(years), years[-1] + (years[-1] - years[-2]))

ax.set_xlabel("Year", fontsize=14, labelpad=5)
ax.tick_params(axis="x", labelsize=12)

start = int(np.floor(min(years) / 20) * 20)
end = int(np.ceil(max(years) / 20) * 20)
ax.set_xticks(np.arange(start, end + 1, 20))

legend_handles = [
    mpatches.Patch(facecolor=COLOR_MAP[k], label=LABEL_MAP[k])
    for k in COLOR_MAP.keys()
]
fig.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.02),
    frameon=True,
    ncol=2,
    fontsize=10,
)

ax.set_title(
    "Best-fit distribution for local clustering coefficient over time",
    fontsize=20,
    pad=8,
)

plt.tight_layout(rect=[0, 0.10, 1, 1])

out_path = os.path.join(OUTPUT_DIR, "figS3_heatmap.svg")
plt.savefig(out_path, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved → {out_path}")

print("\nDone.")