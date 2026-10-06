# build_nagato_semantic_merge_v1086.py
#
# LLM_TRY v10.8.6 Semantic Merge Builder
#
# Purpose:
#   Treat multiple corpus statements about the same concept as a proposition set,
#   not as competing answers.
#
# Example:
#   X is Y.
#   X is Z.
# becomes:
#   X is Y, and is also Z.
#
# The builder extracts a lightweight proposition structure:
#   subject / predicate / object / condition / context / polarity
#
# It then groups compatible propositions and renders one merged canonical answer.
# Explicit negation or incompatible polarity is retained as a separate clause.

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


DEFAULT_CONCEPTS = ["時間", "宇宙"]


@dataclass(frozen=True)
class Proposition:
    subject: str
    predicate: str
    object: str
    condition: str = ""
    context: str = ""
    polarity: bool = True
    source: str = ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build semantically merged Knowledge SFT from data-nagato.txt."
    )
    p.add_argument("--data", default="data/data-nagato.txt")
    p.add_argument(
        "--output",
        default="data/nagato_semantic_merge_v1086.jsonl",
    )
    p.add_argument(
        "--propositions-output",
        default="data/nagato_semantic_propositions_v1086.jsonl",
    )
    p.add_argument("--concepts", nargs="*", default=DEFAULT_CONCEPTS)
    p.add_argument("--max-source-chars", type=int, default=180)
    p.add_argument("--max-propositions", type=int, default=8)
    p.add_argument("--max-answer-chars", type=int, default=280)
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


NEGATION_MARKERS = (
    "ではない", "でない", "しない", "存在しない", "ない",
    "無い", "なく", "ぬ",
)


def detect_polarity(text: str) -> bool:
    return not any(marker in text for marker in NEGATION_MARKERS)


def split_condition_context(text: str) -> tuple[str, str, str]:
    condition = ""
    context = ""
    body = text

    condition_patterns = (
        r"^(?P<cond>[^、。]{2,40}(?:場合|とき|時|なら|では|において))、(?P<body>.+)$",
        r"^(?P<cond>この理論では)\s*(?P<body>.+)$",
    )
    for pattern in condition_patterns:
        m = re.match(pattern, body)
        if m:
            condition = m.group("cond").strip()
            body = m.group("body").strip()
            break

    context_markers = (
        "観測上", "外から", "中では", "空間内", "時間軸上", "この世界では",
    )
    for marker in context_markers:
        if marker in body:
            context = marker
            break

    return body, condition, context


def extract_proposition(concept: str, statement: str) -> Proposition | None:
    body, condition, context = split_condition_context(statement)
    polarity = detect_polarity(body)

    # Direct subject-predicate patterns.
    patterns = (
        ("wa", rf"^(?P<s>{re.escape(concept)})は、?(?P<o>.+?)(?:。)?$"),
        ("mo", rf"^(?P<s>{re.escape(concept)})も、?(?P<o>.+?)(?:。)?$"),
        ("definition", rf"^(?P<s>{re.escape(concept)})の定義は、?(?P<o>.+?)(?:。)?$"),
        ("toha", rf"^(?P<s>{re.escape(concept)})とは、?(?P<o>.+?)(?:。)?$"),
    )

    for kind, pattern in patterns:
        m = re.match(pattern, body)
        if not m:
            continue
        obj = m.group("o").strip(" 。")
        if not obj:
            return None
        predicate = "is"
        if kind == "definition":
            predicate = "definition"
        elif "存在" in obj:
            predicate = "exists"
        elif "進行" in obj:
            predicate = "progresses"
        elif "膨張" in obj:
            predicate = "expands"
        elif "観測" in obj:
            predicate = "observed_as"

        # For "XもY" keep the original clause structure.  Rewriting it as
        # "XはY" can corrupt coordinated subjects such as "時間も空間も...".
        if kind == "mo":
            return Proposition(
                subject=concept,
                predicate="related_statement",
                object=body.rstrip("。"),
                condition=condition,
                context=context,
                polarity=polarity,
                source=statement,
            )

        return Proposition(
            subject=concept,
            predicate=predicate,
            object=obj,
            condition=condition,
            context=context,
            polarity=polarity,
            source=statement,
        )

    # Fallback: concept-bearing sentence becomes a contextual relation.
    if concept in body:
        return Proposition(
            subject=concept,
            predicate="related_statement",
            object=body.rstrip("。"),
            condition=condition,
            context=context,
            polarity=polarity,
            source=statement,
        )
    return None


def proposition_score(p: Proposition) -> tuple[int, int, int]:
    score = 0
    if p.predicate == "definition":
        score += 12
    elif p.predicate in ("is", "exists", "progresses", "expands", "observed_as"):
        score += 8
    else:
        score += 2

    if p.condition:
        score += 2
    if p.context:
        score += 1
    if not p.polarity:
        score += 1

    length = len(p.object)
    return score, -abs(length - 55), -length


def semantically_duplicate(a: Proposition, b: Proposition) -> bool:
    if a.subject != b.subject or a.polarity != b.polarity:
        return False
    aa = re.sub(r"[、。\s]", "", a.object)
    bb = re.sub(r"[、。\s]", "", b.object)
    if aa == bb:
        return True
    shorter = min(len(aa), len(bb))
    if shorter < 8:
        return False
    prefix = 0
    for x, y in zip(aa, bb):
        if x != y:
            break
        prefix += 1
    return prefix / shorter >= 0.80


def dedupe_props(props: list[Proposition]) -> list[Proposition]:
    result: list[Proposition] = []
    for p in props:
        if any(semantically_duplicate(p, old) for old in result):
            continue
        result.append(p)
    return result


def render_clause(p: Proposition) -> str:
    obj = p.object.strip().rstrip("。")
    if p.predicate == "definition":
        clause = f"{p.subject}の定義は、{obj}"
    elif p.predicate == "related_statement" and obj.startswith(p.subject):
        clause = obj
    else:
        clause = f"{p.subject}は、{obj}"

    if p.condition and not clause.startswith(p.condition):
        clause = f"{p.condition}、{clause}"
    if p.context and p.context not in clause:
        clause = f"{p.context}では、{clause}"

    # Polarity is metadata.  Do not synthesize Japanese negation here:
    # the source clause already carries its own wording, and appending
    # "ではない" can create broken forms such as "観測されないではない".
    return clause.rstrip("。") + "。"


def merge_propositions(
    concept: str,
    props: list[Proposition],
    max_chars: int,
) -> str:
    """Merge complementary propositions while preserving source wording."""
    clauses: list[str] = []
    used = 0

    for p in props:
        clause = render_clause(p)

        # Prefer concise semantic coverage over exhaustive dumping.
        if len(clauses) >= 6:
            break
        if used + len(clause) > max_chars and clauses:
            break

        clauses.append(clause)
        used += len(clause)

    if not clauses:
        return ""

    merged = clauses[0]
    for clause in clauses[1:]:
        if clause.startswith(concept + "は、"):
            tail = clause[len(concept + "は、"):]
            merged += f"また、{tail}"
        else:
            merged += f"また、{clause}"
    return merged


def question_variants(concept: str) -> list[str]:
    return [
        f"{concept}とは",
        f"{concept}について教えて",
        f"{concept}を説明して",
        f"{concept}って何",
    ]


def main() -> None:
    args = parse_args()
    corpus_path = resolve_data_path(args.data)
    output_path = Path(args.output)
    props_path = Path(args.propositions_output)

    text = corpus_path.read_text(encoding="utf-8")
    units = sentence_like_units(text)

    all_rows: list[dict] = []
    all_props: list[dict] = []

    print("=" * 96)
    print(" LLM_TRY v10.8.6 Semantic Proposition Merge Builder")
    print("=" * 96)
    print("Corpus :", corpus_path)
    print("Output :", output_path)
    print()

    for concept in args.concepts:
        props: list[Proposition] = []
        for unit in units:
            s = clean_statement(unit)
            if concept not in s:
                continue
            if len(s) > args.max_source_chars:
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

        print("-" * 96)
        print("Concept              :", concept)
        print("Extracted propositions:", len(props))
        print("Selected propositions :", len(selected))
        for i, p in enumerate(selected, 1):
            print(
                f"  [{i:02d}] pred={p.predicate:<18} "
                f"pol={str(p.polarity):<5} "
                f"cond={p.condition or '-'} "
                f"ctx={p.context or '-'}"
            )
            print("       ", p.object)
        print("Merged answer:")
        print(" ", answer)
        print()

        for p in selected:
            row = asdict(p)
            row["concept"] = concept
            all_props.append(row)

        if answer:
            for q in question_variants(concept):
                all_rows.append({
                    "user": q,
                    "assistant": answer,
                    "source": "data-nagato.txt",
                    "concept": concept,
                    "policy": "semantic-proposition-merge-v1086",
                    "proposition_count": len(selected),
                })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    props_path.parent.mkdir(parents=True, exist_ok=True)
    with props_path.open("w", encoding="utf-8") as f:
        for row in all_props:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("=" * 96)
    print("Build completed")
    print("=" * 96)
    print("QA pairs     :", len(all_rows))
    print("Propositions :", len(all_props))
    print("QA output    :", output_path)
    print("Prop output  :", props_path)
    print()
    print("Next:")
    print(
        "  python train_nagato_knowledge_sft_v1086.py "
        f"--data {output_path}"
    )


if __name__ == "__main__":
    main()
