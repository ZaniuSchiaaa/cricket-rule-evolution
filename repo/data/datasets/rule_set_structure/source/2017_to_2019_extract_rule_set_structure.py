"""
2017_to_2019_extract_rule_set_structure.py
──────────────────────────────────────────
Detects the tree structure of a cricket ruleset, writes it to a YAML file,
and optionally flattens it by merging each enumeration parent with its
leaf children.

How it works
------------
The script loops over every line of the processed ruleset text and performs
three main tasks in sequence:

  1. Detect a worded title (e.g. "THE PREAMBLE", "APPENDIX A") or a
     dotted rule decimal (e.g. 3.1, 5.5.6, A.3.2).

  2. On a new title or rule:
       a. Flush paragraphs from the *previous* rule into the YAML output.
       b. Update the candidate patterns based on the newly found rule
          (e.g. after finding 3.1, search for 3.1.1, 3.2, or 4.1).
       c. Reset the paragraph counter.
       d. Flag the previous rule if it was empty (likely an OCR error).
       e. Write the new title or rule into the YAML output.

  3. Increment the paragraph counter using bullet- and period-based logic.

After the loop, write the YAML tree to disk (and a flattened version if
IS_FLATTENED is True), then print any flagged empty rules.

Flattening
----------
"Flattening" collapses enumeration nodes whose children are all leaves by
concatenating the parent name and each leaf name into a single node at the
parent's depth. For example:

    - "3.1":          →    - "3.1 P1"
      - P1                 - "3.1 P2"
      - P2

This removes one layer of nesting wherever the parent acts purely as a
label for its leaf children.

Usage
-----
    # From the repo root:
    python data/datasets/rule_set_structure/source/2017_to_2019_extract_rule_set_structure.py

Inputs
------
    data/datasets/rule_texts/processed/<year>_processed.txt

Outputs
-------
    data/datasets/rule_set_structure/yaml_files/original/<year>_original_raw.yaml
    data/datasets/rule_set_structure/yaml_files/flattened/<year>_flattened.yaml  (if IS_FLATTENED)
"""

import re
import os

# ── User inputs ───────────────────────────────────────────────────────────────

DESIRED_YEAR  = "2017"
IS_FLATTENED  = False

# Worded titles present in this edition's document.
WORDED_TITLES = ["THE PREAMBLE", "APPENDIX A", "APPENDIX B"]

# ── File paths ────────────────────────────────────────────────────────────────

INPUT_PATH      = f"./data/datasets/rule_texts/processed/{DESIRED_YEAR}_processed.txt"
OUTPUT_ORIGINAL = f"./data/datasets/rule_set_structure/yaml_files/original/{DESIRED_YEAR}_original_raw.yaml"
OUTPUT_FLAT     = f"./data/datasets/rule_set_structure/yaml_files/flattened/{DESIRED_YEAR}_flattened.yaml"


# ═══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════════

def build_pattern(candidates: list[str]) -> re.Pattern:
    return re.compile("|".join(map(re.escape, candidates)))


# ═══════════════════════════════════════════════════════════════════════════════
# Part 1 — Rule-structure extraction
# ═══════════════════════════════════════════════════════════════════════════════

# ── Rule parsing ──────────────────────────────────────────────────────────────

def parse_rule(rule: str) -> list:
    """Split a dotted rule identifier into its components.

    Examples
    --------
    >>> parse_rule("A.3.2")
    ['A', 3, 2]
    >>> parse_rule("17.1")
    ['17', 1]
    """
    parts = rule.strip().split(".")
    return [parts[0]] + list(map(int, parts[1:]))


def rule_depth(rule: str) -> int:
    """Return the depth of a rule (number of dotted components)."""
    return len(parse_rule(rule))


def generate_next_candidates(rule: str) -> list[str]:
    """
    Given the current rule, return all plausible next rule strings to search for.

    Candidates are:
      - the first child          (e.g. 3.1   → 3.1.1)
      - each sibling/ancestor+1  (e.g. 3.1   → 3.2, 4.1)
      - next numeric law         (e.g. 17.1  → 18.1)
      - first appendix           (e.g. 42.x  → A.1)   any numeric rule
      - next appendix            (e.g. A.x   → B.1)
      - each candidate with a trailing space or newline, so the regex
        matches the rule itself rather than a longer child identifier
    """
    parts = parse_rule(rule)
    candidates = []

    # Child: append .1
    candidates.append(".".join(map(str, parts + [1])))

    # Siblings and higher-level next rules (skip the very top level)
    for i in range(len(parts) - 1, 0, -1):
        bumped = parts[:i] + [parts[i] + 1]
        candidates.append(".".join(map(str, bumped)))

    if parts[0].isdigit():
        # Next numeric law (e.g. 17.x → 18.1)
        next_law = f"{int(parts[0]) + 1}.1"
        if next_law not in candidates:
            candidates.append(next_law)
        # First appendix — always a candidate from any numeric rule,
        # since we don't know which numeric law is the last one.
        if "A.1" not in candidates:
            candidates.append("A.1")

    elif len(parts[0]) == 1 and parts[0].isalpha():
        # Next letter-prefixed appendix (e.g. A.x → B.1, B.x → C.1)
        next_letter = chr(ord(parts[0].upper()) + 1)
        next_appendix = f"{next_letter}.1"
        if next_appendix not in candidates:
            candidates.append(next_appendix)

    # Duplicate with trailing space and newline so the regex does not
    # accidentally match a child rule (e.g. "3.1" matching "3.1.2").
    return [c + " " for c in candidates] + [c + "\n" for c in candidates]


# ── Paragraph detection ───────────────────────────────────────────────────────

def starts_with_bullet(line: str) -> bool:
    return line.startswith("■") or line.startswith("□")


def ends_with_period(line: str) -> bool:
    stripped = line.strip()
    return stripped.endswith(".") or stripped.endswith('."') or stripped.endswith('."')


# ── YAML string builders ──────────────────────────────────────────────────────

def write_in_new_paras(
    curr_depth: int,
    num_paras: int,
    existing_str: str,
    is_child: bool,
) -> str:
    """
    Append paragraph entries for the just-completed rule to the YAML string.

    If the rule had no detected paragraphs and the incoming rule is *not* its
    child, a placeholder paragraph P1 is added anyway — a childless, empty
    rule is always an anomaly.
    """
    indent   = "  " * (curr_depth + 1)
    addition = "".join(f"{indent}- P{i + 1}\n" for i in range(num_paras))

    if num_paras == 0 and not is_child:
        addition += f"{indent}- P1\n"

    return existing_str + addition


def new_big_law(existing_str: str, rule: str) -> str:
    """
    Write a top-level law header pair to the YAML string.

    For a rule like "17.1" this produces:
        - LAW 17:
          - "17.1":
    """
    law_num = parse_rule(rule)[0]
    return (
        existing_str
        + f"  - LAW {law_num}:\n"
        + f'    - "{rule.rstrip()}":\n'
    )


def new_little_law(existing_str: str, rule: str, curr_depth: int) -> str:
    """Write a non-top-level rule entry to the YAML string."""
    indent = "  " * curr_depth
    return existing_str + f'{indent}- "{rule.rstrip()}":\n'


def new_title(existing_str: str, title: str) -> str:
    """Write a worded title entry (e.g. 'THE PREAMBLE') to the YAML string."""
    return existing_str + f"  - {title}:\n"


# ── Main extraction ───────────────────────────────────────────────────────────

def extract_rule_structure(
    input_path: str,
    worded_titles: list[str],
) -> tuple[str, list[str]]:
    """
    Parse a processed ruleset text file and return its YAML tree as a string.

    Parameters
    ----------
    input_path    : Path to the processed ruleset .txt file.
    worded_titles : Worded section titles to detect (e.g. ["THE PREAMBLE"]).

    Returns
    -------
    yaml_contents : The full YAML tree as a string.
    flagged       : List of empty rule identifiers (likely OCR errors).
    """
    yaml_contents = "---\n(full ruleset):\n"

    # Initial candidate: every edition starts at rule 1.1
    candidates            = ["1.1 ", "1.1\n"]
    pattern               = build_pattern(candidates)
    worded_titles_pattern = build_pattern(worded_titles)

    # State variables
    depth        = 0
    para_counter = 0
    curr_rule    = None
    in_bullet    = False
    is_empty     = True
    is_child     = False
    flagged: list[str] = []

    with open(input_path, "r") as f:
        for line in f:
            line = line.replace("\f", "")   # strip form-feed characters

            # ── (a) Worded title match ────────────────────────────────────────
            wt_match = worded_titles_pattern.search(line)
            if wt_match and line.startswith(wt_match.group()):
                yaml_contents = write_in_new_paras(depth, para_counter, yaml_contents, is_child=True)
                para_counter  = 0
                yaml_contents = new_title(yaml_contents, wt_match.group())
                depth         = 1
                in_bullet     = False
                continue

            # ── (b) Rule decimal match (e.g. 3.1, A.3.2) ─────────────────────
            match = pattern.search(line)
            if match and line.startswith(match.group()):
                new_rule = match.group()

                # Compute is_child BEFORE flushing, so the flush for the
                # current rule uses the correct value.
                # e.g. when A.10.1 is seen, "is A.10.1 a child of A.10?" = True
                # suppresses the spurious P1 under A.10.
                if curr_rule is not None:
                    curr_parts = parse_rule(curr_rule)
                    new_parts  = parse_rule(new_rule)
                    is_child   = (
                        len(new_parts) == len(curr_parts) + 1
                        and new_parts[:-1] == curr_parts
                    )
                else:
                    is_child = False

                # Flush paragraphs from the current rule with correct is_child
                yaml_contents = write_in_new_paras(depth, para_counter, yaml_contents, is_child)
                para_counter  = 0

                # Flag current rule if empty
                if is_empty and curr_rule is not None:
                    flagged.append(curr_rule.strip())

                # Update state
                curr_rule  = new_rule
                depth      = rule_depth(curr_rule)
                in_bullet  = False

                candidates = generate_next_candidates(curr_rule)
                pattern    = build_pattern(candidates)

                # Check whether this line contains text beyond the rule decimal
                is_empty = (line.removeprefix(match.group())).strip() == ""

                # Write rule into YAML
                parts = parse_rule(curr_rule)
                if rule_depth(curr_rule) == 2 and parts[-1] == 1 and parts[0].isdigit():
                    yaml_contents = new_big_law(yaml_contents, match.group())
                else:
                    yaml_contents = new_little_law(yaml_contents, match.group(), depth)

            # ── (c) No rule detected — update is_empty only ───────────────────
            else:
                if line.strip():
                    is_empty = False

            # ── (d) Paragraph counter ─────────────────────────────────────────
            if starts_with_bullet(line):
                para_counter += 1
                in_bullet     = True

            if ends_with_period(line):
                if not in_bullet:
                    para_counter += 1
                else:
                    in_bullet = False   # bullet paragraph ends; do not double-count

        # ── (e) Flush final rule's paragraphs ─────────────────────────────────
        yaml_contents = write_in_new_paras(depth, para_counter, yaml_contents, is_child=True)

    return yaml_contents, flagged


# ═══════════════════════════════════════════════════════════════════════════════
# Part 2 — Flattening
# ═══════════════════════════════════════════════════════════════════════════════

# ── Enumeration pattern (all node types across all editions) ──────────────────
#
# Covers:
#   2017–2019      : 1.1.1, A.1.1, B.1.1
#   2000–2010      : LAW 1, 1., (a), (i)
#   1980–1992      : LAW 1, 1., (a), (i), N(a), N(i)
#   1947–1968      : 1, N1, N(i), (a), (i)
#   1932 and before: 1, 48a

_ENUM_PATTERNS = [
    r'\d+(?:\.\d+)+',          # law decimal:              3.1, 5.5.6
    r'[A-Za-z](?:\.\d+)+',     # appendix decimal:         A.3.2
    r'LAW\s+\d+',              # law header:               LAW 17
    r'\d+\.',                  # number with period:       1.
    r'\([a-zA-Z]\)',           # letter in parentheses:    (a)
    r'\([ivxlcdm]+\)',         # roman in parentheses:     (iv)
    r'N\d+',                   # note with number:         N3
    r'N\([a-zA-Z]\)',          # note with letter:         N(a)
    r'N\([ivxlcdm]+\)',        # note with roman:          N(iv)
    r'\d+',                    # bare number:              1
    r'\d+[a-zA-Z]',            # number with letter:       48a
]

_ENUM_PATTERN = re.compile("|".join(f"(?:{p})" for p in _ENUM_PATTERNS))


# ── Flattening helpers ────────────────────────────────────────────────────────

def _get_depth(line: str) -> int:
    """Return the indentation depth of a YAML line (2 spaces = 1 level)."""
    return (len(line) - len(line.lstrip())) // 2


def _get_cleaned_node(line: str) -> str:
    """Strip YAML punctuation from a line to get the bare node label."""
    return line.strip().strip("-").strip(":").strip().strip('"').strip()


def _strip_inline_comments(line: str) -> str:
    """Remove inline YAML comments (everything from # onward)."""
    return line.split("#")[0]


def _is_enumeration_node(line: str) -> bool:
    """Return True if the node label matches any known enumeration pattern."""
    label = _get_cleaned_node(line)
    m     = _ENUM_PATTERN.match(label)
    return bool(m) and m.group() == label   # full match only


def _remove_last_line(s: str) -> str:
    """Remove the last non-empty line from a string."""
    lines = s.rsplit("\n", 2)
    return lines[0] + "\n" if len(lines) > 1 else s


# ── Main flattening function ──────────────────────────────────────────────────

def flatten_yaml(yaml_contents: str) -> str:
    """
    Flatten a ruleset YAML string.

    For every enumeration parent whose children are *all* leaves, replace the
    parent + its children with concatenated "parent leaf" nodes at the
    parent's depth. Non-enumeration parents (e.g. section headers) are left
    unchanged.

    Parameters
    ----------
    yaml_contents : Raw YAML string (as produced by extract_rule_structure).

    Returns
    -------
    Flattened YAML string.
    """
    # Strip comments and blank lines
    lines = []
    for raw in yaml_contents.splitlines():
        if raw == "---":
            lines.append(raw)
            continue
        cleaned = _strip_inline_comments(raw)
        if cleaned.strip() and not cleaned.strip().startswith("#"):
            lines.append(cleaned)

    mdf_yaml     = ""
    curr_parent  = None
    prev_depth   = 0
    all_leaves   = True
    og_addition  = ""
    mdf_addition = ""

    def _flush(all_leaves, curr_parent, mdf_yaml, og_addition, mdf_addition):
        """Commit the buffered additions to mdf_yaml."""
        if all_leaves and curr_parent:
            return _remove_last_line(mdf_yaml) + mdf_addition
        else:
            return mdf_yaml + og_addition

    for line in lines:

        # Pass through the YAML document separator
        if line == "---":
            mdf_yaml += line + "\n"
            continue

        is_leaf    = not line.rstrip().endswith(":")
        curr_depth = _get_depth(line)

        # Depth change → flush the buffered group
        if curr_depth != prev_depth:
            mdf_yaml = _flush(all_leaves, curr_parent, mdf_yaml, og_addition, mdf_addition)

            if curr_depth < prev_depth:
                curr_parent = None

            prev_depth   = curr_depth
            all_leaves   = True
            og_addition  = ""
            mdf_addition = ""

        if is_leaf:
            og_addition += line + "\n"

            if curr_parent:
                cleaned_leaf = _get_cleaned_node(line)
                indent       = "  " * (curr_depth - 1)
                mdf_addition += f'{indent}- "{curr_parent} {cleaned_leaf}"\n'

        else:
            # Parent node
            if _is_enumeration_node(line):
                curr_parent = _get_cleaned_node(line)
            else:
                curr_parent = None  # non-enumeration parent: do not flatten
            all_leaves   = False
            og_addition += line + "\n"

    # Final flush after the last line
    mdf_yaml = _flush(all_leaves, curr_parent, mdf_yaml, og_addition, mdf_addition)

    return mdf_yaml


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def _write(path: str, contents: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(contents)
    print(f"Written → {path}")


if __name__ == "__main__":
    yaml_contents, flagged = extract_rule_structure(INPUT_PATH, WORDED_TITLES)

    _write(OUTPUT_ORIGINAL, yaml_contents)

    if IS_FLATTENED:
        flat_contents = flatten_yaml(yaml_contents)
        _write(OUTPUT_FLAT, flat_contents)

    if flagged:
        print(f"\nFlagged empty rules ({len(flagged)}) — likely OCR errors:")
        print(", ".join(flagged))