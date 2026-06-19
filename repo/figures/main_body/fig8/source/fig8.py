"""
fig8.py
─────────────────────────
Visualise the best-fit degree-weight distribution for each cricket-ruleset
edition as a horizontal colour-band timeline.

Usage (CLI)
-----------
    python fig8.py \\
        --csv  path/to/fig8_fit_parameters.csv \\
        --out  path/to/fig8_final.svg

CSV schema
----------
Required columns:
    name          : edition year (integer)
    distribution  : one of {"in_weight", "out_weight"}
    best_fit      : one of {"powerlaw", "poisson", "powerlaw_cutoff"} or NaN
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Matplotlib / LaTeX defaults ───────────────────────────────────────────────
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Computer Modern Roman"]

# ── Visual constants ──────────────────────────────────────────────────────────
COLORS: dict[str | None, str] = {
    "powerlaw":        "#ECC7D8",
    "poisson":         "#C5E5D4",
    "powerlaw_cutoff": "#FFE9C8",
    None:              "#B0B0B0",
}

LABELS: dict[str | None, str] = {
    "powerlaw":        "Power-law",
    "poisson":         "Poisson",
    "powerlaw_cutoff": "PL + exp cutoff",
    None:              "Unknown / missing",
}

# Rows to plot, in order.  Each entry is (column_name, y-axis label).
DISTRIBUTIONS: list[tuple[str, str]] = [
    ("in_weight",  "In-weight"),
    ("out_weight", "Out-weight"),
]


# ── Core function ─────────────────────────────────────────────────────────────
def plot_best_fit_timeline(
    csv_path: str | Path,
    output_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot a colour-band timeline of best-fit distributions.

    Parameters
    ----------
    csv_path : str or Path
        Path to the CSV file (see module docstring for schema).
    output_path : str or Path, optional
        If provided, save the figure to this path.
    show : bool
        Whether to call ``plt.show()``.  Set to False when running
        non-interactively (e.g. in a pipeline).

    Returns
    -------
    matplotlib.figure.Figure
    """
    df = pd.read_csv(csv_path)
    df["name"] = df["name"].astype(int)
    years = sorted(df["name"].unique())

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

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved → {output_path}")

    if show:
        plt.show()

    return fig


# ── Private helpers ───────────────────────────────────────────────────────────
def _draw_band_row(
    ax: plt.Axes,
    df: pd.DataFrame,
    years: list[int],
    dist_col: str,
    y_label: str,
) -> None:
    """Fill one timeline row with colour bands and thin dividers."""
    for i, year in enumerate(years):
        # Span runs to the next edition year (or an estimated gap for the last).
        gap = year - years[i - 1] if i > 0 else years[1] - years[0]
        next_year = years[i + 1] if i + 1 < len(years) else year + gap

        subset = df[(df["name"] == year) & (df["distribution"] == dist_col)]
        best = (
            None
            if subset.empty or pd.isna(subset["best_fit"].values[0])
            else subset["best_fit"].values[0]
        )

        ax.axvspan(year, next_year, facecolor=COLORS.get(best, COLORS[None]),
                   alpha=0.85, linewidth=0)

    for year in years:
        ax.axvline(year, color="white", linewidth=0.8, alpha=0.6)

    ax.set_yticks([])
    ax.set_ylabel(y_label, rotation=0, labelpad=40, va="center", fontsize=14)
    ax.set_xlim(
        min(years),
        years[-1] + (years[-1] - years[-2]),
    )
    ax.tick_params(axis="both", labelsize=12)


def _format_x_axis(ax: plt.Axes, years: list[int]) -> None:
    """Set sensible decade ticks on the shared x-axis."""
    start = int(np.floor(min(years) / 20) * 20)
    end   = int(np.ceil(max(years)  / 20) * 20)
    ax.set_xticks(np.arange(start, end + 1, 20))
    ax.set_xlabel("Year", fontsize=14)


def _add_legend(fig: plt.Figure) -> None:
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


# ── CLI entry point ───────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot best-fit distribution timeline for cricket rulesets."
    )
    parser.add_argument("--csv", required=True, help="Path to input CSV file.")
    parser.add_argument("--out", default=None, help="Path for the output figure.")
    parser.add_argument(
        "--no-show", action="store_true",
        help="Suppress the interactive plot window.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    plot_best_fit_timeline(
        csv_path=args.csv,
        output_path=args.out,
        show=not args.no_show,
    )