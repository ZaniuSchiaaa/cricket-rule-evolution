"""
Cricket Ruleset Interdependency Network Extractor
==================================================
Builds weighted directed citation networks from cricket ruleset data and
exports them as .gexf files, one per (year, count_type) combination.

Output structure
----------------
    # From the repo root:
    data/datasets/interdependency_networks/graph_files/graph_viz
        single_count/
            {year}_single_count.gexf
        multi_count/
            {year}_multi_count.gexf

Usage
-----
    # From the repo root:
    python data/datasets/interdependency_networks/graph_files/source/extract_interdependency_network.py
"""

import os
import networkx as nx
import pandas as pd

# ── Global settings ────────────────────────────────────────────────────────────
REMOVE_SELF_EDGES = True
COUNT_TYPES = ["single_count", "multi_count"]

FULL_YEAR_LIST = [
    "1835", "1857", "1884", "1890", "1892",
    "1896", "1900", "1902", "1906", "1908", "1910", "1911", "1913",
    "1914", "1918", "1920", "1923", "1932", "1939", "1947", "1952", "1962",
    "1968", "1980", "1992", "2000", "2008", "2010", "2017", "2019",
]

# ── Path templates ─────────────────────────────────────────────────────────────

NUM_LAWS_CSV = "./data/datasets/rule_set_structure/tree_measures/original/full-ruleset.csv"

CITATION_CSV_TEMPLATE = (
    "./data/datasets/interdependency_networks/citation_tables"
    "/{year}_citation_table.csv"
)

GEXF_OUTPUT_TEMPLATE = (
    "./data/datasets/interdependency_networks/graph_files/graph_viz"
    "/{count_type}/{year}_{count_type}.gexf"
)

NUM_LAWS_DICT = {
    "1752": 33, "1755": 35, "1774": 34, "1785": 33, "1786": 34, "1788": 37,
    "1803": 40, "1806": 45, "1809": 41, "1816": 44, "1820": 44, "1823": 43,
    "1828": 43, "1830": 47, "1835": 47, "1857": 47, "1884": 54, "1890": 54,
    "1892": 54, "1896": 54, "1900": 54, "1902": 54, "1906": 54, "1908": 54,
    "1910": 54, "1911": 54, "1913": 54, "1914": 55, "1918": 55, "1920": 55,
    "1923": 55, "1932": 55, "1939": 55, "1947": 47, "1952": 47, "1962": 47,
    "1968": 47, "1980": 42, "1992": 42, "2000": 42, "2008": 42, "2010": 42,
    "2017": 42, "2019": 42,
}

# ── Helpers ────────────────────────────────────────────────────────────────────


def clean_laws(entry) -> list:
    """
    Parse a comma-separated citation string into a list of top-level law numbers.

    Entries like '17.3, 22' become ['17', '22'].

    Parameters
    ----------
    entry : str or float (NaN)
        Raw value from the 'cited_laws' column.

    Returns
    -------
    list of str
    """
    if pd.isna(entry):
        return []
    laws = [law.strip() for law in entry.split(",")]
    return [law.split(".")[0] if "." in law else law for law in laws]


def extract_first_num(str_input) -> str:
    """Return the top-level law number (before the first '.') from a rule string."""
    return str(str_input).split(".")[0]


# ── Graph construction ─────────────────────────────────────────────────────────

def create_graph(year: str, year_to_num_laws: dict, count_type: str) -> nx.DiGraph:
    """
    Build a weighted directed citation graph for a given year and count type,
    restricted to numerically-labelled (main-ruleset) nodes.

    Parameters
    ----------
    year : str
        Edition year (e.g. '2019').
    year_to_num_laws : dict
        Mapping of year string to number-of-laws string.
    count_type : str
        'single_count' — each law pair counted at most once per rule;
        'multi_count'  — every sub-rule citation counted separately.

    Returns
    -------
    nx.DiGraph
        Nodes are law numbers (strings); edges carry a 'weight' attribute.
    """
    csv_path = CITATION_CSV_TEMPLATE.format(year=year)
    df = pd.read_csv(csv_path)

    df["rule_first_char"] = df["rule"].apply(extract_first_num)
    df["cited_laws_clean"] = df["cited_laws"].apply(clean_laws)

    # Initialise graph with all known law nodes
    G = nx.DiGraph()
    node_list = [str(i + 1) for i in range(int(year_to_num_laws[year]))]
    G.add_nodes_from(node_list)

    # Add weighted edges
    for _, row in df.iterrows():
        source = row["rule_first_char"]

        if count_type == "single_count":
            targets = set(target.split('.')[0] for target in row["cited_laws_clean"])
        elif count_type == "multi_count":
            targets = row["cited_laws_clean"]
        else:
            raise ValueError(f"Unknown count_type: '{count_type}'")

        for target in targets:
            if G.has_edge(source, target):
                G[source][target]["weight"] += 1
            else:
                G.add_edge(source, target, weight=1)

    if REMOVE_SELF_EDGES:
        G.remove_edges_from(nx.selfloop_edges(G))

    # Restrict to numeric (main-ruleset) nodes
    numeric_nodes = [n for n in G.nodes() if str(n).isdigit()]
    G = G.subgraph(numeric_nodes).copy()

    # Annotate nodes for Gephi compatibility
    degree_dict = dict(G.out_degree(weight="weight"))
    for n in G.nodes():
        G.nodes[n]["label"] = str(n)
        G.nodes[n]["degree"] = degree_dict[n]

    return G


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    for count_type in COUNT_TYPES:
        for year in FULL_YEAR_LIST:
            G = create_graph(year, NUM_LAWS_DICT, count_type=count_type)

            out_path = GEXF_OUTPUT_TEMPLATE.format(count_type=count_type, year=year)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            nx.write_gexf(G, out_path)

            print(f"Saved: {out_path}  ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")