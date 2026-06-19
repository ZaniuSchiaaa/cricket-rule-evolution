"""
figS9.py

Generates the first depth-distribution grid (5x4, last 20 edition years) for the
cricket rule-set YAML tree structures. Each subplot is a bar chart of node counts
by tree depth for a single edition year.

Usage
-----
    # From the repo root:
    python figures/supplemental/figS9/source/figS9.py

Outputs
-------
    ./figures/supplemental/figS9/fig9_final.svg

Toggle LEAVES_ONLY to switch between counting all nodes and leaf nodes only.
"""

# ──────────────────────────────────────────────────────────────────────────────
# [0] IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import os
import glob
import shutil
from collections import defaultdict

import yaml
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ──────────────────────────────────────────────────────────────────────────────
# [1] MATPLOTLIB / LaTeX CONFIG
# ──────────────────────────────────────────────────────────────────────────────
if shutil.which("latex"):
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Computer Modern Roman"]
else:
    plt.rcParams["text.usetex"] = False
    plt.rcParams["mathtext.fontset"] = "cm"

# ──────────────────────────────────────────────────────────────────────────────
# [2] CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
INPUT_DIR = "./data/datasets/rule_set_structure/yaml_files/flattened"
OUTPUT_DIR = "./figures/supplemental/figS9"

LEAVES_ONLY = False   # ← Toggle: True = leaf nodes only, False = all nodes

N_ROWS = 5
N_COLS = 4

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# [3] HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def get_depths(node, current_depth=0, leaves_only=False):
    if isinstance(node, dict):
        for key, value in node.items():
            yield current_depth          # ← yield at the key level, not the container
            if not leaves_only or isinstance(value, (dict, list)):
                yield from get_depths(value, current_depth + 1, leaves_only)
    elif isinstance(node, list):
        for item in node:
            if isinstance(item, (dict, list)):
                yield from get_depths(item, current_depth, leaves_only)
            else:
                yield current_depth      # ← scalar list items are leaf nodes


def depth_distribution(tree, leaves_only=True):
    dist = defaultdict(int)
    for d in get_depths(tree, leaves_only=leaves_only):
        dist[d] += 1
    return dict(sorted(dist.items()))


def draw_depth_chart(ax, dist, title, idx, n_rows, n_cols, leaves_only):
    """Draw a depth distribution bar chart onto an existing axes."""
    depths = list(dist.keys())
    counts = list(dist.values())
    row = idx // n_cols
    col = idx % n_cols
    is_left_col   = (col == 0)
    is_bottom_row = (row == n_rows - 1)

    bars = ax.bar(depths, counts, color="#4C72B0", edgecolor="white", linewidth=0.6)

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            str(count),
            ha="center", va="bottom", fontsize=14, color="#333333"
        )

    ax.set_title(title, fontsize=20, fontweight="bold")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_xlim(min(depths) - 0.5, max(depths) + 0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis='x', labelsize=16, labelbottom=True)
    ax.tick_params(axis='y', labelsize=16, labelleft=False)

    if is_left_col:
        ax.set_ylabel(r"\# of nodes", fontsize=24)
    if is_bottom_row:
        ax.set_xlabel("Depth", fontsize=24)


# ──────────────────────────────────────────────────────────────────────────────
# [4] LOAD AND SORT ALL FILES
# ──────────────────────────────────────────────────────────────────────────────
yaml_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*_flattened.yaml")))

if not yaml_files:
    print(f"No YAML files found in {INPUT_DIR}")
else:
    print(f"Found {len(yaml_files)} YAML file(s). Mode: {'leaves only' if LEAVES_ONLY else 'all nodes'}\n")

    all_data = []  # list of (stem, dist)
    for fpath in yaml_files:
        fname = os.path.basename(fpath)
        stem  = fname.replace("_flattened.yaml", "")
        with open(fpath, "r") as f:
            tree = yaml.safe_load(f)
        dist = depth_distribution(tree, leaves_only=LEAVES_ONLY)
        print(f"  {fname}: {dist}")
        all_data.append((stem, dist))

    # ──────────────────────────────────────────────────────────────────────────
    # [5] GRID PLOT (5x4, last 20)
    # ──────────────────────────────────────────────────────────────────────────
    mode_suffix = "leaves" if LEAVES_ONLY else "all-nodes"
    batch = all_data[24:]
    filename = f"figS9_final.svg"

    fig, axes = plt.subplots(
        N_ROWS, N_COLS,
        figsize=(N_COLS * 4, N_ROWS * 4),
        sharex=False, sharey=False,  # we handle labels manually
    )

    for idx, (ax, (stem, dist)) in enumerate(zip(axes.flat, batch)):
        draw_depth_chart(ax, dist, stem, idx, N_ROWS, N_COLS, LEAVES_ONLY)

    # Hide unused cells
    for ax in axes.flat[len(batch):]:
        ax.set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved grid → {out_path}")

print("\nDone.")