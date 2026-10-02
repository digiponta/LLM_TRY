# build_unified_semantic_memory_v1091.py
#
# LLM_TRY v10.9.1 Unified Semantic Memory Builder
#
# Improvements over v10.9.0:
#   - merge both v10.8.9 auto memory and v10.8.6 fallback memory
#   - render "definition" relation naturally
#   - deduplicate equivalent relation facts (prefer contextual form)
#   - index incoming relations on the relation value/object side
#
# Example:
#   文学 --includes--> 数学
#
# yields:
#   文学: 文学は、数学を含む。
#   数学: 数学は、文学に含まれる対象として関係する。
#
# This does NOT invent a definition of 数学; it only exposes the known relation.

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


DEFAULT_CORPUS_MEMORIES = [
    "data/nagato_semantic_merge_v1089.jsonl",
    "data/nagato_semantic_merge_v1086.jsonl",
]
DEFAULT_OUTPUT = "data/unified_semantic_memory_v1091.jsonl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge corpus semantic memory and bidirectional relation facts."
    )
    p.add_argument(
        "--corpus-memory",
        action="append",
        default=[],
        help="Semantic-memory JSONL; may be repeated.",
    )
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
        p = Path(item)
        if p.exists():
            paths.append(p)
    if data_dir.exists():
        for p in sorted(data_dir.rglob("*.jsonl")):
            if p not in paths:
                paths.append(p)

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


def relation_base_key(fact: dict) -> tuple[str, str, str, str, str, bool]:
    return (
        fact["subject"],
        fact["relation"],
        fact["value"],
        fact["condition"],
        fact["condition_predicate"],
        fact["condition_polarity"],
    )


def prefer_contextual_relation_facts(facts: list[dict]) -> list[dict]:
    """For equivalent facts, keep the contextual version when available."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for fact in facts:
        groups[relation_base_key(fact)].append(fact)

    result = []
    for group in groups.values():
        group.sort(
            key=lambda x: (
                bool(x["relation_context"]),
                len(x["relation_context"]),
            ),
            reverse=True,
        )
        result.append(group[0])
    return result


def natural_definition(subject: str, value: str) -> str:
    value = value.strip().rstrip("。")
    if value.endswith(("である", "です", "だ")):
        return f"{subject}は、{value}。"
    return f"{subject}は、{value}である。"


def render_outgoing_relation(fact: dict) -> str:
    s = fact["subject"]
    r = fact["relation"]
    v = fact["value"]
    cond = fact["condition"]
    ctx = fact["relation_context"]
    polarity = fact["condition_polarity"]

    if r == "definition":
        clause = natural_definition(s, v).rstrip("。")
    else:
        templates = {
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
        clause = templates.get(r, f"{s}は、{v}と{r}の関係にある")

    if ctx:
        clause = f"{ctx}、{clause}"
    if cond:
        if polarity:
            clause = f"{cond}の場合、{clause}"
        else:
            clause = f"{cond}でない場合、{clause}"
    return clause.rstrip("。") + "。"


def render_incoming_relation(fact: dict) -> str | None:
    s = fact["subject"]
    r = fact["relation"]
    v = fact["value"]

    # A definition does not become a useful reverse fact.
    if r == "definition":
        return None

    templates = {
        "includes": f"{v}は、{s}に含まれる対象として関係する。",
        "contains": f"{v}は、{s}に含まれる対象として関係する。",
        "is_a": f"{v}は、{s}の上位概念として関係する。",
        "isa": f"{v}は、{s}の上位概念として関係する。",
        "part_of": f"{v}は、{s}を一部として含む関係にある。",
        "has": f"{v}は、{s}が持つ対象として関係する。",
        "uses": f"{v}は、{s}が使う対象として関係する。",
        "related_to": f"{v}は、{s}と関係する。",
        "causes": f"{v}は、{s}によって引き起こされる対象として関係する。",
        "depends_on": f"{v}は、{s}が依存する対象として関係する。",
    }
    return templates.get(r, f"{v}は、{s}との{r}関係の対象である。")


def normalized_clause(text: str) -> str:
    t = text.lower()
    t = re.sub(r"[\s、。,.「」『』()（）]", "", t)
    # Context prefixes should not cause the same core relation to be repeated.
    for prefix in ("分類上", "一般に", "文脈上"):
        t = t.replace(prefix, "")
    return t


def append_unique(parts: list[str], clause: str, seen: set[str]) -> None:
    key = normalized_clause(clause)
    if not key or key in seen:
        return

    # Suppress a shorter duplicate already contained in a richer clause.
    for old in list(seen):
        if len(key) >= 8 and (key in old or old in key):
            if len(key) <= len(old):
                return

    seen.add(key)
    parts.append(clause)


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

    memory_paths = (
        [Path(x) for x in args.corpus_memory]
        if args.corpus_memory
        else [Path(x) for x in DEFAULT_CORPUS_MEMORIES]
    )

    corpus_answer_by_concept: dict[str, str] = {}
    corpus_source_by_concept: dict[str, str] = {}

    for path in memory_paths:
        for row in load_jsonl(path):
            concept = str(row.get("concept", "")).strip()
            answer = str(row.get("assistant", "")).strip()
            if concept and answer and concept not in corpus_answer_by_concept:
                corpus_answer_by_concept[concept] = answer
                corpus_source_by_concept[concept] = str(path)

    relation_facts = discover_relation_facts(
        Path(args.data_dir),
        args.relation_file,
    )
    relation_facts = prefer_contextual_relation_facts(relation_facts)

    outgoing: dict[str, list[dict]] = defaultdict(list)
    incoming: dict[str, list[dict]] = defaultdict(list)
    for fact in relation_facts:
        outgoing[fact["subject"]].append(fact)
        incoming[fact["value"]].append(fact)

    concepts = sorted(
        set(corpus_answer_by_concept)
        | set(outgoing)
        | set(incoming)
    )

    output_rows = []

    print("=" * 100)
    print(" LLM_TRY v10.9.1 Unified Semantic Memory Builder")
    print("=" * 100)
    print("Corpus memories       :", ", ".join(str(p) for p in memory_paths))
    print("Corpus memory concepts:", len(corpus_answer_by_concept))
    print("Relation facts        :", len(relation_facts))
    print("Outgoing subjects     :", len(outgoing))
    print("Incoming values       :", len(incoming))
    print("Unified concepts      :", len(concepts))
    print()

    for concept in concepts:
        parts: list[str] = []
        seen: set[str] = set()
        source_types: list[str] = []

        corpus_answer = corpus_answer_by_concept.get(concept, "")
        if corpus_answer:
            append_unique(parts, corpus_answer, seen)
            source_types.append("corpus")

        out_count = 0
        for fact in outgoing.get(concept, []):
            clause = render_outgoing_relation(fact)
            before = len(parts)
            append_unique(parts, clause, seen)
            if len(parts) > before:
                out_count += 1

        in_count = 0
        for fact in incoming.get(concept, []):
            clause = render_incoming_relation(fact)
            if not clause:
                continue
            before = len(parts)
            append_unique(parts, clause, seen)
            if len(parts) > before:
                in_count += 1

        if out_count or in_count:
            source_types.append("relation")

        answer = "".join(parts).strip()
        if not answer:
            continue

        print("-" * 100)
        print("Concept  :", concept)
        print("Sources  :", "+".join(source_types))
        print("Outgoing :", out_count)
        print("Incoming :", in_count)
        if concept in corpus_source_by_concept:
            print("Corpus   :", corpus_source_by_concept[concept])
        print("Answer   :", answer[:500] + ("..." if len(answer) > 500 else ""))

        for q in question_variants(concept):
            output_rows.append({
                "user": q,
                "assistant": answer,
                "concept": concept,
                "source_types": source_types,
                "outgoing_relation_count": out_count,
                "incoming_relation_count": in_count,
                "policy": "unified-semantic-memory-v1091",
            })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print()
    print("=" * 100)
    print("Build completed")
    print("=" * 100)
    print("Output records:", len(output_rows))
    print("Saved         :", out)


if __name__ == "__main__":
    main()
