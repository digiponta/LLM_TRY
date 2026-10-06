# build_nagato_knowledge_sft_v1084.py
#
# Build corpus-derived Knowledge SFT pairs from data/data-nagato.txt.
#
# Design:
#   - extract sentence-like units from the raw corpus
#   - identify concept-bearing declarative sentences
#   - create conservative question/answer pairs using only corpus text
#   - do not invent definitions that are not explicitly present
#
# This is intentionally corpus-grounded.  It converts already learned raw
# statements into a conversational retrieval/SFT shape.

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


DEFAULT_CONCEPTS = ["時間", "宇宙"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build corpus-derived Knowledge SFT from data-nagato.txt."
    )
    p.add_argument("--data", default="data/data-nagato.txt")
    p.add_argument(
        "--output",
        default="data/nagato_corpus_knowledge_v1084.jsonl",
    )
    p.add_argument(
        "--concepts",
        nargs="*",
        default=DEFAULT_CONCEPTS,
        help="Concepts to convert into grounded QA pairs.",
    )
    p.add_argument(
        "--max-per-concept",
        type=int,
        default=24,
        help="Maximum source statements retained for each concept.",
    )
    p.add_argument(
        "--min-answer-chars",
        type=int,
        default=12,
        help="Minimum answer length after cleaning.",
    )
    p.add_argument(
        "--max-answer-chars",
        type=int,
        default=180,
        help="Maximum answer length after cleaning.",
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
    units = re.split(r"(?<=[。！？!?])|\n+", text)
    return [normalize_space(x) for x in units if normalize_space(x)]


def clean_statement(unit: str) -> str:
    s = normalize_space(unit)
    s = re.sub(r"^[・＊*#>\-\s]+", "", s)
    return s.strip()


def statement_score(concept: str, statement: str) -> tuple[int, int, int]:
    """Prefer direct, compact, declarative concept statements."""
    direct = 0
    if statement.startswith(concept + "は"):
        direct += 4
    if statement.startswith(concept + "も"):
        direct += 3
    if concept + "の定義" in statement:
        direct += 5
    if concept + "とは" in statement:
        direct += 5
    if "。" in statement:
        direct += 1

    # Prefer medium length factual-looking statements.
    length = len(statement)
    compact = -abs(length - 55)

    # Penalize highly conversational / quoted / uncertain fragments.
    penalty = 0
    for marker in ("と思う", "かもしれ", "多分", "？", "?", "ごっこ", "ｗ", "笑"):
        if marker in statement:
            penalty -= 3

    return direct + penalty, compact, -length


def is_usable_statement(
    concept: str,
    statement: str,
    min_chars: int,
    max_chars: int,
) -> bool:
    if concept not in statement:
        return False
    if not (min_chars <= len(statement) <= max_chars):
        return False
    if statement.count(concept) > 4:
        return False
    if re.fullmatch(r".*[！？!?]\s*", statement):
        return False

    # Reject obvious meta/chat fragments that are poor knowledge targets.
    bad = (
        "長門ごっこ",
        "ネタ",
        "RT ",
        "http://",
        "https://",
    )
    if any(x in statement for x in bad):
        return False
    return True


def question_variants(concept: str) -> list[str]:
    return [
        f"{concept}とは",
        f"{concept}について教えて",
        f"{concept}を説明して",
    ]


def main() -> None:
    args = parse_args()
    path = resolve_data_path(args.data)
    output = Path(args.output)

    text = path.read_text(encoding="utf-8")
    units = sentence_like_units(text)

    rows: list[dict] = []
    per_concept_counts: Counter[str] = Counter()

    print("=" * 92)
    print(" LLM_TRY v10.8.4 Corpus-Derived Knowledge SFT Builder")
    print("=" * 92)
    print("Corpus :", path)
    print("Output :", output)
    print()

    for concept in args.concepts:
        candidates: list[str] = []
        seen: set[str] = set()

        for unit in units:
            statement = clean_statement(unit)
            if statement in seen:
                continue
            if not is_usable_statement(
                concept,
                statement,
                args.min_answer_chars,
                args.max_answer_chars,
            ):
                continue
            seen.add(statement)
            candidates.append(statement)

        candidates.sort(
            key=lambda s: statement_score(concept, s),
            reverse=True,
        )
        selected = candidates[: max(1, args.max_per_concept)]

        print("-" * 92)
        print("Concept         :", concept)
        print("Candidates      :", len(candidates))
        print("Selected source :", len(selected))

        # Each grounded statement receives multiple query forms.  The answer is
        # copied from the source corpus verbatim except whitespace normalization.
        for idx, statement in enumerate(selected):
            for question in question_variants(concept):
                rows.append({
                    "user": question,
                    "assistant": statement,
                    "source": "data-nagato.txt",
                    "concept": concept,
                    "source_rank": idx + 1,
                })
                per_concept_counts[concept] += 1

        for i, statement in enumerate(selected[:5], 1):
            print(f"  [{i:02d}] {statement}")
        print()

    # Deduplicate exact user/assistant pairs while preserving order.
    deduped: list[dict] = []
    pair_seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row["user"], row["assistant"])
        if key in pair_seen:
            continue
        pair_seen.add(key)
        deduped.append(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in deduped:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("=" * 92)
    print(" Build completed")
    print("=" * 92)
    print("QA pairs :", len(deduped))
    for concept in args.concepts:
        count = sum(1 for row in deduped if row["concept"] == concept)
        print(f"{concept:8s}: {count}")
    print("Saved    :", output)
    print()
    print("Important:")
    print("  Answers are derived only from statements found in data-nagato.txt.")
    print("  Review the generated JSONL before SFT because the source corpus may")
    print("  contain speculative, personal, fictional, or internally inconsistent")
    print("  statements.")
    print()
    print("Next:")
    print(
        "  python train_nagato_knowledge_sft_v1084.py "
        f"--data {output}"
    )


if __name__ == "__main__":
    main()
