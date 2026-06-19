"""
2017_to_2019_extract_rule_set_structure.py
──────────────────────────────────────────
Detects the tree structure of a cricket ruleset and writes it to a YAML file.
A secondary output flags empty rules (likely OCR errors) for manual review.

Although tailored to the 2000–2019 ruleset format, the general algorithm can
be adapted to other years with some adjustments.

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

After the loop, write the YAML tree to disk and (if any were found) print
the flagged empty rules.

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
"""

import re

# ── User inputs ───────────────────────────────────────────────────────────────

DESIRED_YEAR = "2017"

# Worded titles present in this edition's document.
WORDED_TITLES = ["THE PREAMBLE", "APPENDIX A", "APPENDIX B"]

# ── File paths ────────────────────────────────────────────────────────────────

INPUT_PATH  = f"./data/datasets/rule_texts/processed/{DESIRED_YEAR}_processed.txt"
OUTPUT_YAML = f"./data/datasets/rule_set_structure/yaml_files/original/{DESIRED_YEAR}_original_raw.yaml"


# ═══════════════════════════════════════════════════════════════════════════════
# Helper functions
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
      - the first child          (e.g. 3.1 → 3.1.1)
      - each sibling/ancestor+1  (e.g. 3.1 → 3.2, 4.1)
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

    # Next top-level law (only when the leading component is an integer)
    if parts[0].isdigit():
        next_law = f"{int(parts[0]) + 1}.1"
        if next_law not in candidates:
            candidates.append(next_law)

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
    indent = "  " * (curr_depth + 1)
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


# ── Regex helpers ─────────────────────────────────────────────────────────────

def build_pattern(candidates: list[str]) -> re.Pattern:
    return re.compile("|".join(map(re.escape, candidates)))


# ═══════════════════════════════════════════════════════════════════════════════
# Main extraction
# ═══════════════════════════════════════════════════════════════════════════════

def extract_rule_structure(
    input_path: str,
    output_yaml_path: str,
    worded_titles: list[str],
) -> None:
    """
    Parse a processed ruleset text file and write its tree structure to YAML.

    Parameters
    ----------
    input_path       : Path to the processed ruleset .txt file.
    output_yaml_path : Destination .yaml file.
    worded_titles    : Worded section titles to detect (e.g. ["THE PREAMBLE"]).
    """
    yaml_contents = "---\n(full ruleset):\n"

    # Initial candidate: every edition starts at rule 1.1
    candidates            = ["1.1 ", "1.1\n"]
    pattern               = build_pattern(candidates)
    worded_titles_pattern = build_pattern(worded_titles)

    # State variables
    depth       = 0
    para_counter = 0
    curr_rule   = None
    in_bullet   = False
    is_empty    = True
    is_child    = False
    flagged     : list[str] = []

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
                # Flush paragraphs from the previous rule
                yaml_contents = write_in_new_paras(depth, para_counter, yaml_contents, is_child)
                para_counter  = 0

                # Flag previous rule if empty
                if is_empty and curr_rule is not None:
                    flagged.append(curr_rule.strip())

                # Update state
                curr_rule = match.group()
                depth     = rule_depth(curr_rule)
                in_bullet = False

                candidates = generate_next_candidates(curr_rule)
                pattern    = build_pattern(candidates)

                # Determine whether the *next* rule will be a child of this one
                is_child = False    # reset; will be re-evaluated next iteration

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

    # ── Write YAML output ─────────────────────────────────────────────────────
    with open(output_yaml_path, "w") as f:
        f.write(yaml_contents)
    print(f"YAML written → {output_yaml_path}")

    # ── Report flagged (empty) rules ──────────────────────────────────────────
    if flagged:
        print(f"\nFlagged empty rules ({len(flagged)}) — likely OCR errors:")
        print(", ".join(flagged))


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    extract_rule_structure(
        input_path       = INPUT_PATH,
        output_yaml_path = OUTPUT_YAML,
        worded_titles    = WORDED_TITLES,
    )