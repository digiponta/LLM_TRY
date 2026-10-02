# build_nagato_semantic_merge_v1089.py
#
# Auto-discover concepts from data-nagato.txt and build merged Semantic Memory.
#
# v10.8.9 removes the fixed ["時間", "宇宙"] concept list.
# Concepts are discovered conservatively from subject-like corpus patterns:
#   Xは...
#   Xも...
#   Xとは...
#   Xの定義は...
#
# The actual proposition extraction/merge logic is reused from v10.8.6/8.

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from build_nagato_semantic_merge_v1086 import (
    clean_statement,
    dedupe_props,
    extract_proposition,
    merge_propositions,
    proposition_score,
    question_variants,
    resolve_data_path,
    sentence_like_units,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Auto-discover concepts and build merged Semantic Memory."
    )
    p.add_argument("--data", default="data/data-nagato.txt")
    p.add_argument(
        "--output",
        default="data/nagato_semantic_merge_v1089.jsonl",
    )
    p.add_argument(
        "--propositions-output",
        default="data/nagato_semantic_propositions_v1089.jsonl",
    )
    p.add_argument("--min-occurrences", type=int, default=2)
    p.add_argument("--max-concepts", type=int, default=200)
    p.add_argument("--max-source-chars", type=int, default=180)
    p.add_argument("--max-propositions", type=int, default=8)
    p.add_argument("--max-answer-chars", type=int, default=280)
    return p.parse_args()


def plausible_concept(text: str) -> bool:
    t = text.strip(" 、。・()（）[]「」『』")
    if not (2 <= len(t) <= 24):
        return False
    if re.search(r"[\s、。！？!?：:；;,]", t):
        return False
    if re.fullmatch(r"[0-9０-９]+", t):
        return False

    # Exclude common grammatical/function words and obvious pronouns.
    stop = {
        "これ", "それ", "あれ", "ここ", "そこ", "ため", "もの", "こと",
        "私", "あなた", "貴方", "人間", "場合", "よう", "ところ",
    }
    return t not in stop


def discover_concepts(units: list[str], min_occurrences: int) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter()

    patterns = (
        r"^([^、。！？!?]{2,24})は[、,]?",
        r"^([^、。！？!?]{2,24})も[、,]?",
        r"^([^、。！？!?]{2,24})とは[、,]?",
        r"^([^、。！？!?]{2,24})の定義は[、,]?",
    )

    for unit in units:
        s = clean_statement(unit)
        for pattern in patterns:
            m = re.match(pattern, s)
            if not m:
                continue
            concept = m.group(1).strip()
            # Collapse accidental "XもY" captures only when a single leading
            # noun-like token can be recovered conservatively.
            concept = re.sub(r"(?:も|が|を|に|で)$", "", concept).strip()
            if plausible_concept(concept):
                counts[concept] += 1
            break

    result = [
        (concept, n)
        for concept, n in counts.items()
        if n >= min_occurrences
    ]
    result.sort(key=lambda x: (-x[1], x[0]))
    return result


def main() -> None:
    args = parse_args()
    corpus_path = resolve_data_path(args.data)
    text = corpus_path.read_text(encoding="utf-8")
    units = sentence_like_units(text)

    discovered = discover_concepts(units, args.min_occurrences)
    concepts = [c for c, _ in discovered[: args.max_concepts]]

    print("=" * 96)
    print(" LLM_TRY v10.8.9 Auto Semantic Memory Builder")
    print("=" * 96)
    print("Corpus            :", corpus_path)
    print("Discovered concepts:", len(discovered))
    print("Selected concepts :", len(concepts))
    print()
    print("Top concepts:")
    for concept, count in discovered[:30]:
        print(f"  {concept:<24} subject-patterns={count}")
    print()

    qa_rows = []
    prop_rows = []

    for concept in concepts:
        props = []
        for unit in units:
            s = clean_statement(unit)
            if concept not in s or len(s) > args.max_source_chars:
                continue
            p = extract_proposition(concept, s)
            if p is not None:
                props.append(p)

        props.sort(key=proposition_score, reverse=True)
        props = dedupe_props(props)
        selected = props[: max(1, args.max_propositions)]
        answer = merge_propositions(
            concept,
            selected,
            args.max_answer_chars,
        )
        if not answer:
            continue

        for p in selected:
            row = {
                "concept": concept,
                "subject": p.subject,
                "predicate": p.predicate,
                "object": p.object,
                "condition": p.condition,
                "context": p.context,
                "polarity": p.polarity,
                "source": p.source,
            }
            prop_rows.append(row)

        for q in question_variants(concept):
            qa_rows.append({
                "user": q,
                "assistant": answer,
                "source": "data-nagato.txt",
                "concept": concept,
                "policy": "auto-semantic-proposition-merge-v1089",
                "proposition_count": len(selected),
            })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in qa_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    pout = Path(args.propositions_output)
    with pout.open("w", encoding="utf-8") as f:
        for row in prop_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("=" * 96)
    print("Build completed")
    print("=" * 96)
    print("Concepts with answers:", len({r["concept"] for r in qa_rows}))
    print("QA pairs             :", len(qa_rows))
    print("Propositions         :", len(prop_rows))
    print("QA output            :", out)
    print("Prop output          :", pout)


if __name__ == "__main__":
    main()
