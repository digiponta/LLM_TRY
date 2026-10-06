# build_unified_semantic_memory_v1090.py
#
# LLM_TRY v10.9.0 Unified Semantic Memory Builder
#
# Merge:
#   1) auto-discovered corpus semantic memory (v10.8.9)
#   2) relation facts found in local JSONL files under data/
#
# A relation fact is any JSON object containing:
#   subject, relation, value
# plus optional:
#   condition, condition_predicate, condition_polarity, relation_context
#
# Example:
#   {"subject":"文学","relation":"includes","value":"数学", ...}
#
# Output is one retrieval record per concept.  Corpus-derived explanatory
# clauses and relation clauses are combined rather than treated as competing.

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


DEFAULT_CORPUS_MEMORY = "data/nagato_semantic_merge_v1089.jsonl"
DEFAULT_OUTPUT = "data/unified_semantic_memory_v1090.jsonl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge corpus semantic memory and relation facts."
    )
    p.add_argument("--corpus-memory", default=DEFAULT_CORPUS_MEMORY)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument(
        "--relation-file",
        action="append",
        default=[],
        help="Optional explicit JSONL relation file; may be repeated.",
    )
    return p.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def relation_fact_from_row(row: dict, source_path: Path) -> dict | None:
    subject = str(row.get("subject", "")).strip()
    relation = str(row.get("relation", "")).strip()
    value = str(row.get("value", "")).strip()
    if not subject or not relation or not value:
        return None

    return {
        "subject": subject,
        "relation": relation,
        "value": value,
        "condition": str(row.get("condition", "")).strip(),
        "condition_predicate": str(row.get("condition_predicate", "")).strip(),
        "condition_polarity": bool(row.get("condition_polarity", True)),
        "relation_context": str(row.get("relation_context", "")).strip(),
        "source_file": str(source_path),
    }


def discover_relation_facts(data_dir: Path, explicit: list[str]) -> list[dict]:
    paths: list[Path] = []

    for item in explicit:
        path = Path(item)
        if path.exists():
            paths.append(path)

    if data_dir.exists():
        for path in sorted(data_dir.rglob("*.jsonl")):
            if path not in paths:
                paths.append(path)

    facts = []
    seen = set()

    for path in paths:
        for row in load_jsonl(path):
            fact = relation_fact_from_row(row, path)
            if fact is None:
                continue
            key = (
                fact["subject"],
                fact["relation"],
                fact["value"],
                fact["condition"],
                fact["condition_predicate"],
                fact["condition_polarity"],
                fact["relation_context"],
            )
            if key in seen:
                continue
            seen.add(key)
            facts.append(fact)

    return facts


def render_relation(fact: dict) -> str:
    s = fact["subject"]
    r = fact["relation"]
    v = fact["value"]
    cond = fact["condition"]
    ctx = fact["relation_context"]
    polarity = fact["condition_polarity"]

    relation_templates = {
        "includes": f"{s}は、{v}を含む",
        "contains": f"{s}は、{v}を含む",
        "is_a": f"{s}は、{v}の一種である",
        "isa": f"{s}は、{v}の一種である",
        "part_of": f"{s}は、{v}の一部である",
        "has": f"{s}は、{v}を持つ",
        "uses": f"{s}は、{v}を使う",
        "related_to": f"{s}は、{v}と関係する",
        "causes": f"{s}は、{v}を引き起こす",
        "depends_on": f"{s}は、{v}に依存する",
    }

    clause = relation_templates.get(r, f"{s}は、{r}として{v}と関係する")

    if ctx:
        clause = f"{ctx}、{clause}"
    if cond:
        clause = f"{cond}の場合、{clause}"
    if not polarity and cond:
        clause = f"{cond}でない場合、" + clause.split("の場合、", 1)[-1]

    return clause.rstrip("。") + "。"


def question_variants(concept: str) -> list[str]:
    return [
        concept,
        f"{concept}とは",
        f"{concept}について教えて",
        f"{concept}を説明して",
        f"{concept}って何",
    ]


def main() -> None:
    args = parse_args()

    corpus_path = Path(args.corpus_memory)
    corpus_rows = load_jsonl(corpus_path)

    corpus_answer_by_concept: dict[str, str] = {}
    for row in corpus_rows:
        concept = str(row.get("concept", "")).strip()
        answer = str(row.get("assistant", "")).strip()
        if concept and answer and concept not in corpus_answer_by_concept:
            corpus_answer_by_concept[concept] = answer

    relation_facts = discover_relation_facts(
        Path(args.data_dir),
        args.relation_file,
    )

    relation_by_subject: dict[str, list[dict]] = defaultdict(list)
    for fact in relation_facts:
        relation_by_subject[fact["subject"]].append(fact)

    concepts = sorted(
        set(corpus_answer_by_concept) | set(relation_by_subject)
    )

    output_rows = []

    print("=" * 96)
    print(" LLM_TRY v10.9.0 Unified Semantic Memory Builder")
    print("=" * 96)
    print("Corpus memory concepts :", len(corpus_answer_by_concept))
    print("Relation facts found   :", len(relation_facts))
    print("Relation subjects      :", len(relation_by_subject))
    print("Unified concepts       :", len(concepts))
    print()

    for concept in concepts:
        parts: list[str] = []
        source_types: list[str] = []

        corpus_answer = corpus_answer_by_concept.get(concept, "")
        if corpus_answer:
            parts.append(corpus_answer)
            source_types.append("corpus")

        facts = relation_by_subject.get(concept, [])
        rendered_relations = []
        seen_clause = set()
        for fact in facts:
            clause = render_relation(fact)
            if clause in seen_clause:
                continue
            seen_clause.add(clause)
            rendered_relations.append(clause)

        if rendered_relations:
            parts.extend(rendered_relations)
            source_types.append("relation")

        answer = "".join(parts).strip()
        if not answer:
            continue

        print("-" * 96)
        print("Concept :", concept)
        print("Sources :", "+".join(source_types))
        print("Relations:", len(rendered_relations))
        print("Answer  :", answer[:400] + ("..." if len(answer) > 400 else ""))

        for q in question_variants(concept):
            output_rows.append({
                "user": q,
                "assistant": answer,
                "concept": concept,
                "source_types": source_types,
                "relation_count": len(rendered_relations),
                "policy": "unified-semantic-memory-v1090",
            })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print()
    print("=" * 96)
    print("Build completed")
    print("=" * 96)
    print("Output records :", len(output_rows))
    print("Saved          :", out)

    if not relation_facts:
        print()
        print("[INFO] No relation facts were found in data/*.jsonl.")
        print("       If your relation store is elsewhere, rerun with:")
        print("       --relation-file PATH_TO_RELATION_FACTS.jsonl")


if __name__ == "__main__":
    main()
