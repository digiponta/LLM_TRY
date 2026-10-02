# build_nagato_knowledge_sft_v1085.py
#
# Build ONE canonical corpus-grounded answer per concept.
#
# v10.8.4 mapped one question (e.g. "時間とは") to many different answers,
# which creates conflicting supervision for a small LM.  v10.8.5 instead:
#   - ranks corpus statements
#   - selects a small set of complementary evidence sentences
#   - concatenates them into one canonical answer
#   - maps every paraphrased question for the concept to that SAME answer

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


DEFAULT_CONCEPTS = ["時間", "宇宙"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build non-conflicting corpus-derived Knowledge SFT."
    )
    p.add_argument("--data", default="data/data-nagato.txt")
    p.add_argument(
        "--output",
        default="data/nagato_corpus_knowledge_v1085.jsonl",
    )
    p.add_argument("--concepts", nargs="*", default=DEFAULT_CONCEPTS)
    p.add_argument(
        "--evidence-sentences",
        type=int,
        default=3,
        help="Number of top corpus statements merged into one canonical answer.",
    )
    p.add_argument("--min-answer-chars", type=int, default=12)
    p.add_argument("--max-source-chars", type=int, default=150)
    p.add_argument(
        "--max-canonical-chars",
        type=int,
        default=320,
        help="Maximum length of the merged canonical answer.",
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


def statement_score(concept: str, s: str) -> tuple[int, int, int]:
    score = 0

    # Strongly prefer definition-like/direct assertions.
    if s.startswith(concept + "は"):
        score += 10
    if s.startswith(concept + "も"):
        score += 6
    if concept + "の定義" in s:
        score += 12
    if concept + "とは" in s:
        score += 12

    # Prefer useful explanatory predicates.
    for marker in (
        "である", "です", "存在", "進行", "定義", "全体", "空間",
        "状態", "膨張", "運動", "観測", "世界",
    ):
        if marker in s:
            score += 1

    # Penalize speculative/meta/chat-like material.
    for marker in (
        "と思う", "かもしれ", "多分", "ごっこ", "ネタ", "？", "?",
        "興味深い", "ｗ", "笑", "RT ", "http://", "https://",
    ):
        if marker in s:
            score -= 5

    length = len(s)
    compact = -abs(length - 60)
    return score, compact, -length


def usable(concept: str, s: str, min_chars: int, max_chars: int) -> bool:
    if concept not in s:
        return False
    if not (min_chars <= len(s) <= max_chars):
        return False
    if s.count(concept) > 4:
        return False
    if s.endswith(("？", "?", "！", "!")):
        return False
    return True


def dedupe_near_identical(statements: list[str]) -> list[str]:
    result: list[str] = []
    for s in statements:
        compact = re.sub(r"[、。\s]", "", s)
        duplicate = False
        for old in result:
            old_compact = re.sub(r"[、。\s]", "", old)
            shorter = min(len(compact), len(old_compact))
            if shorter == 0:
                continue
            common_prefix = 0
            for a, b in zip(compact, old_compact):
                if a != b:
                    break
                common_prefix += 1
            if common_prefix / shorter >= 0.80:
                duplicate = True
                break
        if not duplicate:
            result.append(s)
    return result


def question_variants(concept: str) -> list[str]:
    return [
        f"{concept}とは",
        f"{concept}について教えて",
        f"{concept}を説明して",
        f"{concept}って何",
    ]


def make_canonical_answer(selected: list[str], max_chars: int) -> str:
    parts: list[str] = []
    total = 0
    for s in selected:
        s = s.strip()
        if not s.endswith("。"):
            s += "。"
        if total + len(s) > max_chars and parts:
            break
        parts.append(s)
        total += len(s)
    return "".join(parts)


def main() -> None:
    args = parse_args()
    corpus_path = resolve_data_path(args.data)
    output_path = Path(args.output)

    text = corpus_path.read_text(encoding="utf-8")
    units = sentence_like_units(text)

    rows: list[dict] = []

    print("=" * 92)
    print(" LLM_TRY v10.8.5 Non-Conflicting Corpus Knowledge Builder")
    print("=" * 92)
    print("Corpus :", corpus_path)
    print("Output :", output_path)
    print()

    for concept in args.concepts:
        candidates = []
        seen = set()

        for unit in units:
            s = clean_statement(unit)
            if s in seen:
                continue
            seen.add(s)
            if usable(
                concept,
                s,
                args.min_answer_chars,
                args.max_source_chars,
            ):
                candidates.append(s)

        candidates.sort(
            key=lambda s: statement_score(concept, s),
            reverse=True,
        )
        candidates = dedupe_near_identical(candidates)
        selected = candidates[: max(1, args.evidence_sentences)]
        answer = make_canonical_answer(
            selected,
            args.max_canonical_chars,
        )

        print("-" * 92)
        print("Concept          :", concept)
        print("Candidates       :", len(candidates))
        print("Evidence selected:", len(selected))
        print("Canonical answer :", answer)
        print()

        if not answer:
            print("[WARN] no canonical answer generated; skipping", concept)
            continue

        for question in question_variants(concept):
            rows.append({
                "user": question,
                "assistant": answer,
                "source": "data-nagato.txt",
                "concept": concept,
                "evidence": selected,
                "policy": "one-concept-one-canonical-answer",
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("=" * 92)
    print("Build completed")
    print("=" * 92)
    print("Concepts :", len({row['concept'] for row in rows}))
    print("QA pairs :", len(rows))
    print("Saved    :", output_path)
    print()
    print("Rule: every paraphrase of a concept maps to exactly one canonical answer.")
    print("Review the generated answers before training.")
    print()
    print("Next:")
    print(
        "  python train_nagato_knowledge_sft_v1085.py "
        f"--data {output_path}"
    )


if __name__ == "__main__":
    main()
