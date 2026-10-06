# inspect_nagato_concepts_v1083.py
#
# Diagnose whether concepts are actually present in data/data-nagato.txt,
# and show corpus-local evidence around each concept.
#
# This script intentionally does not use outside knowledge.  It answers:
#   1) Is the concept present in the raw corpus?
#   2) How many times?
#   3) In what local contexts?
#
# This helps distinguish:
#   raw-corpus presence
#   vs.
#   ability to answer "<concept>とは" conversationally.

from __future__ import annotations

import argparse
import re
from pathlib import Path


DEFAULT_CONCEPTS = ["長門", "長門有希", "時間", "宇宙"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inspect concept occurrence/context in data-nagato.txt."
    )
    p.add_argument("--data", default="data/data-nagato.txt")
    p.add_argument(
        "--concepts",
        nargs="*",
        default=DEFAULT_CONCEPTS,
        help="Concept strings to inspect.",
    )
    p.add_argument(
        "--max-contexts",
        type=int,
        default=8,
        help="Maximum number of matching contexts printed per concept.",
    )
    return p.parse_args()


def resolve_data_path(value: str) -> Path:
    path = Path(value)
    if path.exists():
        return path
    if value == "data/data-nagato.txt":
        fallback = Path("data-nagato.txt")
        if fallback.exists():
            return fallback
    raise FileNotFoundError(f"Corpus not found: {path}")


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def sentence_like_units(text: str) -> list[str]:
    # Preserve Japanese sentence punctuation while making local contexts easy
    # to inspect.  Empty fragments are discarded.
    units = re.split(r"(?<=[。！？!?])|\n+", text)
    return [normalize_space(x) for x in units if normalize_space(x)]


def main() -> None:
    args = parse_args()
    path = resolve_data_path(args.data)
    text = path.read_text(encoding="utf-8")
    units = sentence_like_units(text)

    print("=" * 88)
    print(" LLM_TRY v10.8.3 Nagato Corpus Concept Inspection")
    print("=" * 88)
    print("Corpus     :", path)
    print("Characters :", len(text))
    print("Units      :", len(units))
    print()

    for concept in args.concepts:
        exact_count = text.count(concept)
        matching = [u for u in units if concept in u]

        print("-" * 88)
        print("Concept         :", concept)
        print("Exact count     :", exact_count)
        print("Matching units  :", len(matching))
        print("Raw-corpus known:", "YES" if exact_count >= 2 else "WEAK/NO")
        print()

        if not matching:
            print("  (no matching context)")
            print()
            continue

        for i, unit in enumerate(matching[: max(1, args.max_contexts)], 1):
            print(f"  [{i:02d}] {unit}")

        if len(matching) > args.max_contexts:
            print(f"  ... {len(matching) - args.max_contexts} more context(s)")
        print()

    print("=" * 88)
    print(" Interpretation")
    print("=" * 88)
    print("Exact count >= 2 means v10.8.2 allows the concept through the lexical")
    print("pre-generation gate.  It does NOT guarantee that the LM can answer")
    print("'<concept>とは' correctly.")
    print()
    print("If a concept is frequent but the chat answer is poor, the next step is")
    print("corpus-derived knowledge SFT or retrieval, not further gate relaxation.")


if __name__ == "__main__":
    main()
