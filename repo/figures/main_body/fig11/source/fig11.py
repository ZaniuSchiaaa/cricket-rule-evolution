"""
fig11.py
────────
Plots the eigenvector-centrality rankings of selected laws over time.

Usage
-----
    # From the repo root:
    python figures/main_body/fig11/source/fig11.py
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

plt.rcParams["text.usetex"]   = True
plt.rcParams["font.family"]   = "serif"
plt.rcParams["font.serif"]    = ["Computer Modern Roman"]

# ── Data ──────────────────────────────────────────────────────────────────────

YEARS = [
    1835, 1857, 1884, 1890, 1892, 1896, 1900, 1902, 1906, 1908,
    1910, 1911, 1913, 1914, 1918, 1920, 1923, 1932, 1939, 1947,
    1952, 1962, 1968, 1980, 1992, 2000, 2008, 2010, 2017, 2019,
]

# Rankings by law (1 = highest eigenvector centrality).
# "Fair/Unfair Play" was only codified from 1939 onward.
LAWS: dict[str, list[int]] = {
    "Caught": [
        13, 11, 24, 22, 20, 20, 18, 14, 14, 5,
         5,  5,  5,  6,  6,  5,  6,  6, 22, 15,
        18, 17, 19, 29, 25, 32, 34, 25, 33, 37,
    ],
    "Fair/Unfair Play": [2, 3, 1, 10, 1, 1, 1, 1, 1, 4, 2, 2],
    "The Fieldsman": [
         8,  5,  7,  9,  7,  7,  6, 10, 10, 14,
        14, 14, 14, 15, 15, 16, 16, 18, 18, 18,
         8,  2,  8,  3,  3,  5,  2,  1,  1,  1,
    ],
}

# "Fair/Unfair Play" data begins at 1939 — align to the tail of YEARS.
YEARS_BY_LAW: dict[str, list[int]] = {
    law: YEARS[-len(ranks):] for law, ranks in LAWS.items()
}

COLORS     = ["#79A899", "#A8748D", "#CDA173"]
LINESTYLES = ["-", "--", "-."]

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots()

for (label, ranks), color, ls in zip(LAWS.items(), COLORS, LINESTYLES):
    ax.plot(YEARS_BY_LAW[label], ranks, label=label, color=color, linestyle=ls)

ax.invert_yaxis()
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

ax.set_xlabel("Year",          fontsize=16)
ax.set_ylabel("Rank (1 = best)", fontsize=16)
ax.tick_params(axis="x",       labelsize=12)
ax.tick_params(axis="y",       labelsize=12)
ax.set_title("Law rankings over time", fontsize=20)
ax.legend()

plt.tight_layout()
plt.savefig("./figures/main_body/fig11/fig11_final_copy.svg", bbox_inches="tight")
plt.show()