"""
fig6_c_laws_over_time.py

Plot the number of laws in each cricket ruleset edition over time.

Usage
-----
    # From the repo root:
    python figures/main_body/fig6/source/fig6_c_laws_over_time.py

Outputs
-------
    ./figures/main_body/fig6/components/fig6_c_laws_over_time.svg
"""


# ===========
# [0] IMPORTS
# ===========

import os
import shutil

import matplotlib.pyplot as plt


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

VIS_FILEPATH = "./figures/main_body/fig6/components/fig6_c_laws_over_time.svg"


# ==============
# [3] PLOTTING
# ==============

def create_square_ax(fig_size=6, axes_fraction=0.7):
    """Create a figure with a single square axis."""
    fig = plt.figure(figsize=(fig_size, fig_size))
    ax = fig.add_axes([0.15, 0.15, axes_fraction, axes_fraction])
    ax.set_box_aspect(1)
    return fig, ax


def plot_laws_over_time(num_laws_by_year, vis_filepath):
    """Scatter the number of laws against edition year."""
    fig, ax = create_square_ax(fig_size=8, axes_fraction=0.75)

    years = sorted(num_laws_by_year)
    counts = [num_laws_by_year[year] for year in years]

    ax.scatter(years, counts, marker="o", color="#7F7F7F")

    ylabel = r"\# of laws" if plt.rcParams["text.usetex"] else "# of laws"
    ax.set_ylabel(ylabel, fontsize=30, labelpad=10)
    ax.set_xlabel("Year", fontsize=30, labelpad=10)
    ax.grid(False)
    ax.tick_params(axis="both", labelsize=18)
    ax.set_position([0.15, 0.2, 0.7, 0.65])

    os.makedirs(os.path.dirname(vis_filepath), exist_ok=True)
    fig.savefig(vis_filepath, dpi=300)
    plt.close(fig)
    print(f"Saved figure to {vis_filepath}")


# ==========
# [4] MAIN
# ==========

def main():
    plot_laws_over_time(NUM_LAWS_BY_YEAR, VIS_FILEPATH)


if __name__ == "__main__":
    main()