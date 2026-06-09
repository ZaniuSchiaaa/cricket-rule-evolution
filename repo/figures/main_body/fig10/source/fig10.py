"""
fig10.py
────────
Plot the categories of the top-5 laws across cricket ruleset revisions.

Each row represents a rank position (#1–#5), and each colored segment
indicates the category of the law occupying that rank for a given ruleset
period.

Usage
-----
    # From the repo root:
    python figures/main_body/fig10/source/fig10.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Computer Modern Roman"]


COLORS = {
    1: "#C5E5D4",
    2: "#ECC7D8",
    3: "#FFE9C8",
    0: "#B0B0B0",
}

LABELS = {
    1: "Batter dismissals",
    2: "Umpires, unfair play and conduct",
    3: "Fielder",
    0: "Miscellaneous",
}

YEAR_LIST = [
    1835, 1857, 1884, 1890, 1892, 1896, 1900, 1902, 1906, 1908,
    1910, 1911, 1913, 1914, 1918, 1920, 1923, 1932, 1939, 1947,
    1952, 1962, 1968, 1980, 1992, 2000, 2008, 2010, 2017, 2019,
]

DATA = [
    [1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,2,0,2,0,2,2,2,2,2,3,3,3],
    [1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,1,1,0,1,2,2,0,0,0,2,2],
    [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,3,2,3,3,0,2,0,1,2],
    [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,1,2,0,2,0,0,2,3,2,2,1],
    [0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,1,0,0,1,1,1,0,1,1,1,3,0,2,1,2],
] # these are numerical labels for the categories of the top five laws
  # over the duration of the rule set


def main():
    df = pd.DataFrame(DATA, columns=YEAR_LIST)

    n_ranks = df.shape[0]
    end_year = YEAR_LIST[-1] + (YEAR_LIST[-1] - YEAR_LIST[-2])

    fig, axes = plt.subplots(
        n_ranks,
        1,
        figsize=(12, n_ranks * 1.2),
        sharex=True,
    )

    for rank_idx, ax in enumerate(axes):
        for i, year in enumerate(YEAR_LIST):
            next_year = (
                YEAR_LIST[i + 1]
                if i + 1 < len(YEAR_LIST)
                else end_year
            )

            category = df.loc[rank_idx, year]

            ax.axvspan(
                year,
                next_year,
                facecolor=COLORS[category],
                alpha=0.85,
                linewidth=0,
            )

        ax.set_yticks([])
        ax.set_ylabel(
            f"\#{rank_idx + 1}",
            rotation=0,
            labelpad=18,
            va="center",
            fontsize=16,
        )

        ax.set_xlim(YEAR_LIST[0], end_year)

        for year in YEAR_LIST:
            ax.axvline(
                year,
                color="white",
                linewidth=0.8,
                alpha=0.6,
            )

        ax.tick_params(axis="both", labelsize=12)

    axes[-1].set_xlabel("Year", fontsize=16)

    fig.suptitle(
        "Top-5 laws by category over time",
        y=0.93,
        fontsize=24,
    )

    legend_handles = [
        mpatches.Patch(
            facecolor=COLORS[category],
            label=LABELS[category],
        )
        for category in COLORS
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(1.00, 1.0),
        frameon=True,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    plt.savefig(
        "./figures/main_body/fig10/fig10_final.svg",
        bbox_inches="tight",
    )

    plt.show()


if __name__ == "__main__":
    main()