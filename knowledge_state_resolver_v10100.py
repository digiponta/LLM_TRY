#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_TRY v10.10.0 Knowledge State Resolver

Resolve an explicit concept query into exactly one knowledge state before
runtime routing.

Priority:
    TYPED
    CANONICAL
    UNIFIED
    INTERNALIZED
    RAW_CORPUS_ONLY
    UNKNOWN

RAW_CORPUS_ONLY is evidence that a token occurs in the raw corpus. It is NOT
sufficient evidence that the model can answer correctly.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Mapping

from knowledge_provenance_v10102 import (
    KnowledgeProvenance,
    provenance_for_state,
)

from internalized_knowledge_v10100 import (
    internalized_concept_for_focus,
    load_internalized_concepts,
)
from subject_keyed_proposition_v1090 import (
    compose_subject_from_index,
)
from typed_subject_proposition_v10100 import (
    typed_query_lookup,
    typed_subject_rows,
)


KNOWLEDGE_STATES = (
    "TYPED",
    "CANONICAL",
    "UNIFIED",
    "INTERNALIZED",
    "RAW_CORPUS_ONLY",
    "UNKNOWN",
    "NON_CONCEPT",
)


@dataclass(frozen=True)
class KnowledgeState:
    state: str
    focus: str = ""
    answer: str = ""
    predicate_type: str = ""
    reason: str = ""
    provenance: KnowledgeProvenance | None = None

    @property
    def is_retrieval(self) -> bool:
        return self.state in {"TYPED", "CANONICAL", "UNIFIED"}

    @property
    def permits_model_generation(self) -> bool:
        return self.state in {"INTERNALIZED", "NON_CONCEPT"}


def extract_query_focus(question: str) -> str:
    q = str(question).strip()

    # Conversational greetings must never be split as "<concept> + は".
    # Example: "こんにちは" previously matched the generic trailing-"は"
    # pattern as focus="こんにち".
    non_concept_patterns = (
        "こんにちは",
        "こんばんは",
        "おはよう",
        "おはようございます",
        "はじめまして",
        "お疲れさま",
        "お疲れ様",
        "ありがとう",
        "ありがとうございます",
        "さようなら",
        "またね",
        "やあ",
        "hello",
        "hi",
    )
    q_lower = q.lower()
    if any(
        q_lower == phrase.lower()
        or q_lower.startswith(phrase.lower())
        for phrase in non_concept_patterns
    ):
        return ""

    patterns = (
        r"^(.+?)(?:とは)$",
        r"^(.+?)(?:って何)$",
        r"^(.+?)(?:について教えて)$",
        r"^(.+?)(?:を説明して)$",
        r"^(.+?)(?:を簡単に説明して)$",
        r"^([^\s。、！？?]{1,24})は$",
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, q)
        if match:
            return match.group(1).strip()
    return ""


def unified_lookup_row(
    knowledge_path: Path,
    focus: str,
) -> dict[str, object] | None:
    if not focus or not knowledge_path.exists():
        return None

    for raw in knowledge_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        concept = str(row.get("concept", "")).strip()
        answer = str(row.get("assistant", "")).strip()
        if concept == focus and answer:
            return row
    return None


def unified_lookup(
    knowledge_path: Path,
    focus: str,
) -> str:
    row = unified_lookup_row(knowledge_path, focus)
    if row is None:
        return ""
    return str(row.get("assistant", "")).strip()


def raw_corpus_occurrences(
    corpus_path: Path,
    focus: str,
) -> int:
    if not focus or not corpus_path.exists():
        return 0
    try:
        text = corpus_path.read_text(encoding="utf-8")
    except OSError:
        return 0
    return text.count(focus)


def resolve_knowledge_state(
    question: str,
    *,
    typed_index_path: Path,
    subject_index_path: Path,
    unified_path: Path,
    learning_log: Path,
    learning_state: Path,
    raw_corpus_path: Path,
    canonical_definitions: Mapping[str, str],
    raw_min_occurrences: int = 2,
) -> KnowledgeState:
    """Resolve one explicit concept query to one knowledge state."""

    # Typed questions such as "GPUの性質は" do not necessarily match the
    # generic concept-query grammar, so typed detection must run first.
    typed_hit = typed_query_lookup(typed_index_path, question)
    if typed_hit is not None:
        subject, predicate_type, answer = typed_hit
        return KnowledgeState(
            state="TYPED",
            focus=subject,
            answer=answer,
            predicate_type=predicate_type,
            reason="typed predicate query hit",
            provenance=provenance_for_state(
                "TYPED",
                source="typed-subject-proposition",
                origin=str(typed_index_path),
                evidence=f"predicate_type={predicate_type}",
            ),
        )

    focus = extract_query_focus(question)
    if not focus:
        return KnowledgeState(
            state="NON_CONCEPT",
            reason="no explicit concept query",
            provenance=provenance_for_state(
                "NON_CONCEPT",
                source="runtime-conversation",
                origin="chat input",
                evidence="no explicit concept query",
            ),
        )

    # Generic subject-keyed propositions are also typed semantic knowledge.
    subject_answer = compose_subject_from_index(subject_index_path, focus)
    if subject_answer:
        rows = typed_subject_rows(typed_index_path, focus)
        predicate_types = []
        for row in rows:
            if row.predicate_type not in predicate_types:
                predicate_types.append(row.predicate_type)
        return KnowledgeState(
            state="TYPED",
            focus=focus,
            answer=subject_answer,
            predicate_type=",".join(predicate_types),
            reason="subject-keyed semantic compose hit",
            provenance=provenance_for_state(
                "TYPED",
                source="subject-keyed-proposition",
                origin=str(subject_index_path),
                evidence=(
                    "predicate_types="
                    + ",".join(predicate_types)
                ),
            ),
        )

    canonical = canonical_definitions.get(focus.lower(), "")
    if canonical:
        return KnowledgeState(
            state="CANONICAL",
            focus=focus,
            answer=canonical,
            reason="canonical definition hit",
            provenance=provenance_for_state(
                "CANONICAL",
                source="canonical-definition",
                origin="chat.py:CANONICAL_DEFINITIONS",
                evidence=f"canonical key={focus.lower()}",
            ),
        )

    unified_row = unified_lookup_row(unified_path, focus)
    if unified_row is not None:
        unified = str(unified_row.get("assistant", "")).strip()
        metadata = {}
        for key in (
            "semantic_schema",
            "atomic_count",
            "predicate_types",
        ):
            if key in unified_row:
                metadata[key] = str(unified_row.get(key))
        return KnowledgeState(
            state="UNIFIED",
            focus=focus,
            answer=unified,
            reason="unified semantic memory hit",
            provenance=provenance_for_state(
                "UNIFIED",
                source=str(
                    unified_row.get("source", "unified-semantic-memory")
                ),
                origin=str(unified_path),
                timestamp=str(unified_row.get("updated_at", "")),
                fingerprint=str(unified_row.get("fingerprint", "")),
                evidence="concept row matched unified semantic memory",
                metadata=metadata,
            ),
        )

    concepts = load_internalized_concepts(
        learning_log,
        learning_state,
    )
    internalized = internalized_concept_for_focus(concepts, focus)
    if internalized is not None:
        latest_fingerprint = (
            internalized.fingerprints[-1]
            if internalized.fingerprints
            else ""
        )
        return KnowledgeState(
            state="INTERNALIZED",
            focus=internalized.concept,
            reason=(
                "trained fingerprint evidence present; "
                f"trained_pairs={internalized.trained_pairs}"
            ),
            provenance=provenance_for_state(
                "INTERNALIZED",
                source=internalized.latest_source,
                origin=str(learning_log),
                timestamp=internalized.latest_timestamp,
                fingerprint=latest_fingerprint,
                evidence=(
                    "trained fingerprint evidence; "
                    f"trained_pairs={internalized.trained_pairs}"
                ),
                metadata={
                    "latest_question": internalized.latest_question,
                    "trained_pairs": str(internalized.trained_pairs),
                    "sources": ",".join(internalized.sources),
                },
            ),
        )

    occurrences = raw_corpus_occurrences(raw_corpus_path, focus)
    if occurrences >= max(1, raw_min_occurrences):
        return KnowledgeState(
            state="RAW_CORPUS_ONLY",
            focus=focus,
            reason=f"raw corpus occurrences={occurrences}",
            provenance=provenance_for_state(
                "RAW_CORPUS_ONLY",
                source="raw-corpus",
                origin=str(raw_corpus_path),
                evidence=f"occurrences={occurrences}",
            ),
        )

    return KnowledgeState(
        state="UNKNOWN",
        focus=focus,
        reason="no validated knowledge source",
        provenance=provenance_for_state(
            "UNKNOWN",
            source="none",
            origin="",
            evidence="no validated knowledge source",
        ),
    )
