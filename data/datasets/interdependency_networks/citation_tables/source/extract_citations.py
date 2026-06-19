"""
extract_citations.py

Parses the plain-text enumeration of each cricket ruleset and extracts every
cross-reference (cited law number, appendix, or worded title) found in each
rule's text.  Results are written back to CSV as a new ``cited_laws`` column.

# From the repo root:
Input  (per year): data/datasets/interdependency_networks/source/enumerated_text/<year>_enumerated_text.csv
Output (per year): data/datasets/interdependency_networks/citation_tables/<year>_citation_table.csv

Usage
-----
    # From the repo root:
    python data/datasets/interdependency_networks/citation_tables/source/extract_citations.py
"""

import re

import pandas as pd
import spacy
from tqdm import tqdm

tqdm.pandas()

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
YEARS = [
    "1823", "1828", "1830", "1835", "1857", "1884", "1890", "1892",
    "1896", "1900", "1902", "1906", "1908", "1910", "1911", "1913",
    "1914", "1918", "1920", "1923", "1932", "1939", "1947", "1952",
    "1962", "1968", "1980", "1992", "2000", "2008", "2010", "2017", "2019",
]

# Laws referenced by title rather than number
WORDED_TITLES = [
    "One Day Matches",
    "Single Wicket",
    "Notes for Scorers and Umpires",
    "Regulations for Drying the Wickets and Ground in First-Class Matches in England",
]

# If True, every occurrence of a citation is kept; if False, deduplicate per rule.
COUNT_EVERY_REFERENCE = True

# ---------------------------------------------------------------------------
# Pre-compiled constants
# ---------------------------------------------------------------------------
nlp = spacy.load("en_core_web_sm")

_UNITS = {
    "in", "ins", "cm", "m", "ounce", "mm", "ft", "g", "ounces",
    "minutes", "runs", "yards", "wickets", "balls", "run", "-run", "Penalty",
}
_UNITS_PATTERN = r"|".join(re.escape(u) for u in _UNITS)

_WORDED_TITLES_PATTERN = re.compile(
    "|".join(fr'See [""]{re.escape(s)}[""]' for s in WORDED_TITLES),
    re.IGNORECASE,
)

_MONTH_NAMES = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
}

_APPENDIX_RE        = re.compile(r'\bAppendix\s+([A-Z](?:\.\d+)*\.?)\b', re.IGNORECASE)
_LETTER_DECIMAL_RE  = re.compile(r'\b([A-E]\.\d+(?:\.\d+)*)\b')
_MULTI_DECIMAL_RE   = re.compile(r'\b\d+(?:\.\d+)+\b')
_ORDINAL_RE         = re.compile(r'\b(\d+)(st|nd|rd|th)\b', re.IGNORECASE)
_INTEGER_RE         = re.compile(r'\b\d+\b')
_YEAR_RE            = re.compile(r'^(18|19|20)\d{2}$')

# ---------------------------------------------------------------------------
# Helper predicates
# ---------------------------------------------------------------------------

def _preceded_by_note(text: str, start: int) -> bool:
    """Return True if 'Note', 'Notes', or 'Level' immediately precede *start*."""
    lookback = text[max(0, start - 7):start].lower()
    return lookback.endswith(("note ", "notes ", "level "))


def _followed_by_unit(text: str, end: int) -> bool:
    """Return True if a measurement unit immediately follows position *end*."""
    tail = text[end:end + 15]
    return re.match(rf'^[\s-]*(?:{_UNITS_PATTERN})\b', tail, re.IGNORECASE) is not None


def _blank(text: str, start: int, end: int) -> str:
    """Replace the slice [start:end] with spaces (to prevent double-counting)."""
    return text[:start] + " " * (end - start) + text[end:]

# ---------------------------------------------------------------------------
# Text pre-processing
# ---------------------------------------------------------------------------

def preprocess_text(text: str) -> str:
    """
    Strip isolated numbers (likely page numbers) and join the remaining
    lines into a single string ready for NLP parsing.
    """
    filtered = [
        line.strip()
        for line in text.split("\n")
        if not re.fullmatch(r"\d+(?:\.\d+)*", line.strip())
    ]
    return " ".join(filtered)

# ---------------------------------------------------------------------------
# Reference extraction
# ---------------------------------------------------------------------------

def extract_references(
    text: str,
    worded_titles_pattern: re.Pattern,
    count_every_reference: bool,
) -> str:
    """
    Return a comma-separated string of every law/appendix reference found in
    *text*, sorted with appendices first then numerically.

    Extraction steps (in order, with already-matched spans blanked out):
        1. Appendix references  (e.g. ``Appendix B.2``)
        2. Letter-prefixed decimals  (e.g. ``A.4.3``)
        3. Multi-level decimals  (e.g. ``22.3.1``)
        4. Ordinal numbers  (e.g. ``29th`` → ``29``)
        5. Plain integers in sentences that mention "Law(s)"
        6. Worded titles  (e.g. ``See "Single Wicket"``)
    """
    text = preprocess_text(str(text))
    doc  = nlp(text)
    refs: list[str] = []

    for sent in doc.sents:
        buf = re.sub(r"^\d+", "", sent.text)   # strip leading rule number

        # 1. Appendix references
        for m in _APPENDIX_RE.finditer(buf):
            refs.append(m.group(1))
            buf = _blank(buf, m.start(1), m.end(1))

        # 2. Letter-prefixed decimals (e.g. A.4.3)
        for m in _LETTER_DECIMAL_RE.finditer(buf):
            s, e = m.start(1), m.end(1)
            if _preceded_by_note(buf, s) or _followed_by_unit(buf, e):
                continue
            refs.append(m.group(1))
            buf = _blank(buf, s, e)

        # 3. Multi-level decimals (e.g. 22.3.1)
        for m in _MULTI_DECIMAL_RE.finditer(buf):
            s, e = m.start(), m.end()
            if _preceded_by_note(buf, s) or _followed_by_unit(buf, e):
                continue
            refs.append(m.group())
            buf = _blank(buf, s, e)

        # 4. Ordinals (e.g. 29th → 29)
        for m in _ORDINAL_RE.finditer(buf):
            s, e = m.start(1), m.end(1)
            if _preceded_by_note(buf, s) or _followed_by_unit(buf, e):
                continue
            tail = buf[m.end():m.end() + 20].strip().lower()
            first_word = re.match(r"[a-z]+", tail)
            if first_word and first_word.group() in _MONTH_NAMES:
                continue
            refs.append(m.group(1))
            buf = _blank(buf, m.start(), m.end())

        # 5. Plain integers in "Law(s)" sentences
        if re.search(r"\bLaws?\b", buf, re.IGNORECASE):
            for m in _INTEGER_RE.finditer(buf):
                s, e = m.start(), m.end()
                num_str = m.group()

                if _YEAR_RE.fullmatch(num_str):
                    continue

                # Skip if adjacent to a decimal point
                left  = buf[s - 1] if s > 0 else ""
                # Advance past trailing punctuation to find the true next char
                tail_idx = e
                while tail_idx < len(buf) and buf[tail_idx] in ".,;: ":
                    tail_idx += 1
                right = buf[tail_idx] if tail_idx < len(buf) else ""
                if left == "." or right == ".":
                    continue

                if _preceded_by_note(buf, s) or _followed_by_unit(buf, e):
                    continue

                refs.append(num_str)

        # 6. Worded titles
        for m in worded_titles_pattern.finditer(buf):
            title = re.sub(r"^See\s+", "", m.group(), flags=re.IGNORECASE)
            title = re.sub(r'["""'']', "", title).upper().strip()
            refs.append(title)
            buf = _blank(buf, m.start(), m.end())

    def _sort_key(x: str):
        if re.fullmatch(r"\d+(?:\.\d+)*", x):
            return (1, [int(p) for p in x.split(".")])
        return (0, x)   # appendices / worded titles first

    if not count_every_reference:
        refs = list(set(refs))

    return ", ".join(sorted(refs, key=_sort_key))

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_year(year: str) -> None:
    """Read the enumeration CSV for *year*, extract citations, and save."""
    in_path  = f"./data/datasets/interdependency_networks/citation_tables/source/enumerated_text/{year}_enumerated_text.csv"
    out_path = f"./data/datasets/interdependency_networks/citation_tables/{year}_citation_table.csv"

    df = pd.read_csv(in_path)
    df["cited_laws"] = df["text"].progress_apply(
        lambda t: extract_references(t, _WORDED_TITLES_PATTERN, COUNT_EVERY_REFERENCE)
    )
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


def main() -> None:
    for year in YEARS:
        process_year(year)


if __name__ == "__main__":
    main()