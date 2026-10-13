#!/usr/bin/env python3
"""Normalise Nigerian Pidgin orthographic variants (paper Appendix A).

Applies the variant -> standard mapping in ``data/pidgin_orthography_map.tsv``
to transcripts. Multi-word variants ("throw way", "he been") take precedence
over single words, and matching is case-insensitive on whole words only.

The context-dependent homophones in Appendix B (``data/pidgin_homophones.tsv``)
are NOT rewritten here: in the paper they were resolved by scoring each
candidate with the Nigerian Pidgin 5-gram LM.

Note: the paper applies this normalisation to training transcripts only; test
set transcripts were left unnormalised.

Examples:
    echo "Dem people they talk say e better" | python scripts/normalize_pidgin.py
    python scripts/normalize_pidgin.py transcripts.txt > transcripts.norm.txt
"""

import argparse
import csv
import re
import sys
from pathlib import Path

DEFAULT_MAP = Path(__file__).resolve().parent.parent / "data" / "pidgin_orthography_map.tsv"


def _canon_apostrophes(s: str) -> str:
    return s.replace("’", "'")


def load_mapping(path: Path = DEFAULT_MAP) -> dict[str, str]:
    with open(path, encoding="utf-8") as f:
        return {
            _canon_apostrophes(row["variant"]).lower(): _canon_apostrophes(row["standard"])
            for row in csv.DictReader(f, delimiter="\t")
        }


class PidginNormalizer:
    def __init__(self, mapping: dict[str, str] | None = None):
        self.mapping = mapping if mapping is not None else load_mapping()
        # Longest variants first so "throw way" wins over "throw".
        variants = sorted(self.mapping, key=len, reverse=True)
        alternation = "|".join(re.escape(v).replace(r"\ ", r"\s+") for v in variants)
        self.pattern = re.compile(rf"(?<![\w'])(?:{alternation})(?![\w'])", re.IGNORECASE)

    def __call__(self, text: str) -> str:
        text = _canon_apostrophes(text)
        return self.pattern.sub(lambda m: self.mapping[" ".join(m.group(0).lower().split())], text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", nargs="?", help="Text file, one transcript per line (default: stdin)")
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAP, help="Variant mapping TSV")
    args = parser.parse_args()

    normalize = PidginNormalizer(load_mapping(args.mapping))
    stream = open(args.input, encoding="utf-8") if args.input else sys.stdin
    with stream:
        for line in stream:
            sys.stdout.write(normalize(line.rstrip("\n")) + "\n")


if __name__ == "__main__":
    main()
