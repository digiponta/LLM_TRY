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

    @property
    def is_retrieval(self) -> bool:
        return self.state in {"TYPED", "CANONICAL", "UNIFIED"}

    @property
    def permits_model_generation(self) -> bool:
        return self.state in {"INTERNALIZED", "NON_CONCEPT"}


def extract_query_focus(question: str) -> str:
    q = str(question).strip()
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


def unified_lookup(
    knowledge_path: Path,
    focus: str,
) -> str:
    if not focus or not knowledge_path.exists():
        return ""

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
            return answer
    return ""


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
    focus = extract_query_focus(question)
    if not focus:
        return KnowledgeState(
            state="NON_CONCEPT",
            reason="no explicit concept query",
        )

    # Typed query has the strongest explicit semantic intent.
    typed_hit = typed_query_lookup(typed_index_path, question)
    if typed_hit is not None:
        subject, predicate_type, answer = typed_hit
        return KnowledgeState(
            state="TYPED",
            focus=subject,
            answer=answer,
            predicate_type=predicate_type,
            reason="typed predicate query hit",
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
        )

    canonical = canonical_definitions.get(focus.lower(), "")
    if canonical:
        return KnowledgeState(
            state="CANONICAL",
            focus=focus,
            answer=canonical,
            reason="canonical definition hit",
        )

    unified = unified_lookup(unified_path, focus)
    if unified:
        return KnowledgeState(
            state="UNIFIED",
            focus=focus,
            answer=unified,
            reason="unified semantic memory hit",
        )

    concepts = load_internalized_concepts(
        learning_log,
        learning_state,
    )
    internalized = internalized_concept_for_focus(concepts, focus)
    if internalized is not None:
        return KnowledgeState(
            state="INTERNALIZED",
            focus=internalized.concept,
            reason=(
                "trained fingerprint evidence present; "
                f"trained_pairs={internalized.trained_pairs}"
            ),
        )

    occurrences = raw_corpus_occurrences(raw_corpus_path, focus)
    if occurrences >= max(1, raw_min_occurrences):
        return KnowledgeState(
            state="RAW_CORPUS_ONLY",
            focus=focus,
            reason=f"raw corpus occurrences={occurrences}",
        )

    return KnowledgeState(
        state="UNKNOWN",
        focus=focus,
        reason="no validated knowledge source",
    )
