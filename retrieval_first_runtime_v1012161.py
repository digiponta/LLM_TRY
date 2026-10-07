#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.12.16.1 Retrieval-First Runtime.

Runtime priority:
    Subject-Keyed Corpus Memory HIT
        -> full proposition retrieval
        -> relation metadata / function structure
    MISS
        -> existing Semantic Knowledge Architecture / model route

This module performs no training and never mutates model checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from function_semantic_decomposition_v101213 import (
    FunctionStructure,
    infer_action,
    infer_purpose,
    infer_target,
)
from semantic_role_generalization_v101210 import normalize
from subject_keyed_corpus_memory_v101216 import (
    CorpusMemoryRecord,
    lookup_subject,
)


@dataclass(frozen=True)
class RetrievalFirstResult:
    hit: bool
    subject: str
    answer: str
    relation: str
    records: tuple[CorpusMemoryRecord, ...]
    function_structure: FunctionStructure | None
    route: str
    provenance: str


def _char_bigrams(text: str) -> set[str]:
    compact = re.sub(r"[\s。、,.!?！？:：;；「」『』（）()\[\]{}]", "", normalize(text).lower())
    if not compact:
        return set()
    if len(compact) < 2:
        return {compact}
    return {compact[i:i + 2] for i in range(len(compact) - 1)}


def _query_terms(query: str, subject: str) -> str:
    value = normalize(query)
    value = value.replace(subject, "")
    value = re.sub(r"(とは|について|教えて|説明して|説明してください|って何|何ですか)", "", value)
    return value.strip()


def rank_records(
    records: list[CorpusMemoryRecord],
    *,
    subject: str,
    query: str = "",
) -> list[CorpusMemoryRecord]:
    """Rank exact-subject corpus records for compact Top-K retrieval.

    Ranking favors direct subject statements, query-term overlap, relation
    compatibility, and compact statements. Corpus order remains the final
    deterministic tie breaker.
    """
    q_terms = _query_terms(query, subject)
    q_bigrams = _char_bigrams(q_terms)

    scored = []
    for index, row in enumerate(records):
        statement = normalize(row.statement)
        score = 0.0

        if statement.startswith(f"{subject}は") or statement.startswith(f"{subject}が"):
            score += 3.0

        if "とは" in query and row.relation == "definition":
            score += 2.0
        elif "なぜ" in query and row.relation == "cause":
            score += 2.0
        elif ("使い方" in query or "方法" in query) and row.relation == "function":
            score += 2.0

        if q_bigrams:
            s_bigrams = _char_bigrams(statement)
            score += 4.0 * (len(q_bigrams & s_bigrams) / max(1, len(q_bigrams)))

        # Prefer concise evidence when relevance is otherwise similar.
        score -= min(len(statement), 240) / 2400.0
        scored.append((score, -index, row))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [row for _, _, row in scored]


def compose_records(records: list[CorpusMemoryRecord]) -> str:
    statements: list[str] = []
    for row in records:
        if row.statement not in statements:
            statements.append(row.statement)
    return " / ".join(statements)


def dominant_relation(records: list[CorpusMemoryRecord]) -> str:
    if not records:
        return ""
    # Deterministic first-source relation preserves corpus order.
    return records[0].relation


def resolve_subject(
    memory_path: str | Path,
    subject: str,
    *,
    query: str = "",
    top_k: int = 5,
) -> RetrievalFirstResult:
    target = normalize(subject)
    if not target:
        return RetrievalFirstResult(
            hit=False,
            subject="",
            answer="",
            relation="",
            records=(),
            function_structure=None,
            route="FALLBACK",
            provenance="none",
        )

    rows = lookup_subject(Path(memory_path), target)
    if not rows:
        return RetrievalFirstResult(
            hit=False,
            subject=target,
            answer="",
            relation="",
            records=(),
            function_structure=None,
            route="FALLBACK",
            provenance="none",
        )

    ranked_rows = rank_records(
        rows,
        subject=target,
        query=query,
    )
    selected_rows = ranked_rows[:max(1, int(top_k))]
    answer = compose_records(selected_rows)
    relation = dominant_relation(selected_rows)
    structure = None
    if relation == "function":
        structure = FunctionStructure(
            subject=target,
            action=infer_action(answer),
            target=infer_target(target, answer),
            purpose=infer_purpose(answer),
            answer=answer,
        )

    return RetrievalFirstResult(
        hit=True,
        subject=target,
        answer=answer,
        relation=relation,
        records=tuple(selected_rows),
        function_structure=structure,
        route="CORPUS_MEMORY",
        provenance=(
            "data-nagato.txt:subject-keyed-corpus-memory:"
            f"top-k={len(selected_rows)}/{len(rows)}:v10.15"
        ),
    )


def truth_allows_direct_retrieval(state: str) -> bool:
    """Return whether corpus memory may answer directly under Truth-State policy.

    FALSE / OUTDATED / CONTESTED must continue through the existing
    Truth-Aware Semantic Architecture. TRUE and UNVERIFIED may be retrieved
    directly with provenance/truth state exposed to the caller.
    """
    return str(state).strip().upper() not in {
        "FALSE", "OUTDATED", "CONTESTED",
    }
