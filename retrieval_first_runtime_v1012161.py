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

    answer = compose_records(rows)
    relation = dominant_relation(rows)
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
        records=tuple(rows),
        function_structure=structure,
        route="CORPUS_MEMORY",
        provenance="data-nagato.txt:subject-keyed-corpus-memory:v10.12.16",
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
