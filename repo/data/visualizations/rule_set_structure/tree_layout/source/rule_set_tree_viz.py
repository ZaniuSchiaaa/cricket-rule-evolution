"""
rule_set_tree_viz.py
────────────────────
Builds interactive HTML tree visualisations from ruleset YAML files.

For each target year and each tree type (original / flattened), the script:
  1. Loads the YAML rule tree.
  2. Builds an anytree structure from it.
  3. Renders the tree to an SVG entirely in memory.
  4. Embeds the SVG in an HTML template with pan/zoom and tooltip support,
     and writes the result to the output directory.

Usage
-----
    # From the repo root:
    python data/visualizations/rule_set_structure/tree_layout/source/rule_set_tree_viz.py

Inputs
------
    data/datasets/rule_set_structure/yaml_files/{original|flattened}/<year>_{type}.yaml

Outputs
-------
    data/visualizations/rule_set_structure/tree_layout/tree_viz/{original|flattened}/<year>_tree_viz_{type}.html

Dependencies
------------
    pip install pyyaml anytree graphviz python-slugify pandas
    (also requires the Graphviz system package: https://graphviz.org/download/)
"""

from __future__ import annotations

import os
import re

import graphviz
import pandas as pd
import yaml
from anytree import NodeMixin, RenderTree
from anytree.exporter import UniqueDotExporter
from slugify import slugify

# ── User inputs ───────────────────────────────────────────────────────────────

TARGET_YEARS: list[str] = [
    "1752", "1755", "1774", "1785", "1786", "1788", "1803", "1806", "1809",
    "1816", "1820", "1823", "1828", "1830", "1835", "1857", "1884", "1890",
    "1892", "1896", "1900", "1902", "1906", "1908", "1910", "1911", "1913",
    "1914", "1918", "1920", "1923", "1932", "1939", "1947", "1952", "1962",
    "1968", "1980", "1992", "2000", "2008", "2010", "2017", "2019",
]

RULE_TREE_TYPES: list[str] = ["original", "flattened"]

# Whether to display node labels in the tree diagram.
LABELLED: bool = True

# ── File paths ────────────────────────────────────────────────────────────────

YAML_DIR  = "./data/datasets/rule_set_structure/yaml_files"
HTML_DIR  = "./data/visualizations/rule_set_structure/tree_layout/tree_viz"

DOT_OPTIONS = [
    "rankdir=TB",
    "splines=polyline",
    "nodesep=0.6",
    "ranksep=2.0",
]


# ── Tree node class ───────────────────────────────────────────────────────────

class Rule(NodeMixin):
    """A single node in a ruleset tree."""

    def __init__(
        self,
        title: str,
        year_label: str,
        description: str | None = None,
        number_label: str | None = None,
        parent: "Rule | None" = None,
        children: list | None = None,
        full_path: str | None = None,
    ) -> None:
        super().__init__()
        self.title      = title
        self.year_label = year_label
        self.full_path  = full_path

        title_slug = slugify(title) if not pd.isna(title) else title

        if pd.isna(number_label):
            self.name = title_slug
        elif pd.isna(title):
            self.name = str(number_label)
        else:
            self.name = f"{number_label} {title_slug}"

        self.description  = description or title_slug
        self.number_label = number_label
        self.parent       = parent
        if children:
            self.children = children

    def __str__(self)  -> str: return f"{self.title}, {self.description}"
    def __repr__(self) -> str: return f"{self.title}, {self.description}"


# ── Tree construction ─────────────────────────────────────────────────────────

def build_tree(
    name,
    content,
    year_label: str,
    parent: Rule | None = None,
    path: str = "",
) -> Rule:
    """Recursively build a Rule tree from a parsed YAML structure."""
    label     = str(name)
    full_path = f"{path}/{label}" if path else label
    node      = Rule(title=label, year_label=year_label, parent=parent, full_path=full_path)

    if isinstance(content, dict):
        for k, v in content.items():
            build_tree(k, v, year_label, node, full_path)
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                for k, v in item.items():
                    build_tree(k, v, year_label, node, full_path)
            else:
                build_tree(item, [], year_label, node, full_path)

    return node


# ── Rendering ─────────────────────────────────────────────────────────────────

def _nodenamefunc(node: Rule) -> str:
    """Unique Graphviz node ID (internal use only)."""
    return f'"{node.full_path}"'


def _nodeattrfunc_labelled(node: Rule) -> str:
    return f'label="{node.title}", shape=box'


def _nodeattrfunc_unlabelled(node: Rule) -> str:
    return 'label="", shape=box'


def render_tree_to_svg(root: Rule, labelled: bool) -> str:
    """
    Render an anytree Rule tree to an SVG string entirely in memory.
    No .dot or .svg files are written to disk.
    """
    nodeattrfunc = _nodeattrfunc_labelled if labelled else _nodeattrfunc_unlabelled

    exporter  = UniqueDotExporter(
        root,
        options=DOT_OPTIONS,
        nodenamefunc=_nodenamefunc,
        nodeattrfunc=nodeattrfunc,
    )
    dot_source = "\n".join(list(exporter))
    svg_bytes  = graphviz.Source(dot_source).pipe(format="svg")
    return svg_bytes.decode("utf-8")


# ── HTML output ───────────────────────────────────────────────────────────────

def _embed_tooltips(svg: str) -> str:
    """Replace each SVG <g> title element with a data-tooltip attribute."""
    return re.sub(
        r'(<g[^>]*?)>\s*<title>(.*?)</title>',
        r'\1 data-tooltip="\2" style="pointer-events:all">',
        svg,
        flags=re.DOTALL,
    )


def build_html(svg_content: str, year_label: str, tree_type: str) -> str:
    """Wrap an SVG string in an interactive HTML page."""
    svg_with_tooltips = _embed_tooltips(svg_content)

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Interactive Rule Tree — {year_label} ({tree_type})</title>
    <style>
        .tooltip {{
            position: absolute;
            background: #333;
            color: #fff;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 8px;
            pointer-events: none;
            white-space: nowrap;
            display: none;
            z-index: 10;
        }}
        svg {{ width: 100%; height: auto; }}
        .info-box {{
            position: fixed;
            bottom: 2vh;
            right: 2vw;
            width: 25vw;
            background: rgba(0, 0, 0, 0.7);
            color: white;
            padding: 1.5vh 1vw;
            border-radius: 1vh;
            font-size: 2.0vh;
            line-height: 1.4;
            z-index: 20;
            box-sizing: border-box;
            text-align: left;
            box-shadow: 0 0.5vh 2vh rgba(0, 0, 0, 0.3);
        }}
    </style>
</head>
<body>
    <div class="tooltip" id="tooltip"></div>
    <div id="svg-wrapper">
        {svg_with_tooltips}
    </div>
    <div class="info-box">
        <strong>Year:</strong> {year_label}<br>
        <strong>Type:</strong> {tree_type}
    </div>
    <script>
        let scale = 1.0;
        const wrapper = document.getElementById("svg-wrapper");

        document.addEventListener("keydown", (e) => {{
            if (e.metaKey || e.ctrlKey) {{
                if (e.key === "=" || e.key === "+") {{
                    scale = Math.min(scale + 1, 50);
                    e.preventDefault();
                }} else if (e.key === "-") {{
                    scale = Math.max(scale - 1, 0.1);
                    e.preventDefault();
                }}
                wrapper.style.transform = `scale(${{scale}})`;
                wrapper.style.transformOrigin = "0 0";
            }}
        }});

        const tooltip  = document.getElementById("tooltip");
        const nodes    = document.querySelectorAll("g[data-tooltip]");

        nodes.forEach(node => {{
            const text = node.getAttribute("data-tooltip");
            if (text) {{
                node.addEventListener("mouseenter", () => {{
                    tooltip.innerHTML   = text.replace(/\\n/g, "<br>");
                    tooltip.style.display = "block";
                }});
                node.addEventListener("mousemove", (e) => {{
                    tooltip.style.left = (e.pageX + 10) + "px";
                    tooltip.style.top  = (e.pageY + 10) + "px";
                }});
                node.addEventListener("mouseleave", () => {{
                    tooltip.style.display = "none";
                }});
            }}
        }});
    </script>
</body>
</html>"""


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(
    target_years: list[str] = TARGET_YEARS,
    tree_types: list[str]   = RULE_TREE_TYPES,
    labelled: bool          = LABELLED,
) -> None:
    for year in target_years:
        for tree_type in tree_types:
            yaml_path = f"{YAML_DIR}/{tree_type}/{year}_{tree_type}.yaml"
            html_path = f"{HTML_DIR}/{tree_type}/{year}_tree_viz_{tree_type}.html"

            if not os.path.exists(yaml_path):
                print(f"Skipping {year} ({tree_type}): file not found.")
                continue

            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f)

            if not data:
                print(f"Skipping {year} ({tree_type}): empty YAML.")
                continue

            # Build tree (last top-level key wins, consistent with original behaviour)
            for key, val in data.items():
                root = build_tree(key, val, year)

            svg     = render_tree_to_svg(root, labelled)
            html    = build_html(svg, year, tree_type)

            os.makedirs(os.path.dirname(html_path), exist_ok=True)
            with open(html_path, "w") as f:
                f.write(html)

            print(f"Written → {html_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run()