#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_TRY v10.11.1 Semantic Knowledge Snapshot / Inspection

Inspect one concept across every semantic knowledge layer without mutating it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from semantic_proposition_v1090 import Proposition, load_propositions
from subject_keyed_proposition_v1090 import compose_subject_from_index
from typed_subject_proposition_v10100 import (
    TypedSubjectStatement,
    typed_subject_rows,
)
from unified_semantic_bridge_v1090 import load_unified_rows
from internalized_knowledge_v10100 import (
    InternalizedConcept,
    internalized_concept_for_focus,
    load_internalized_concepts,
)
from semantic_knowledge_architecture_v10110 import (
    SemanticKnowledgeArchitecture,
    SemanticKnowledgeResult,
)


@dataclass(frozen=True)
class SemanticKnowledgeSnapshot:
    concept: str
    query: str
    atomic_propositions: tuple[Proposition, ...]
    subject_answer: str
    typed_rows: tuple[TypedSubjectStatement, ...]
    unified_row: Mapping[str, object] | None
    internalized: InternalizedConcept | None
    result: SemanticKnowledgeResult

    @property
    def proposition_count(self) -> int:
        return len(self.atomic_propositions)

    @property
    def typed_count(self) -> int:
        return len(self.typed_rows)

    @property
    def unified_present(self) -> bool:
        return self.unified_row is not None

    @property
    def internalized_present(self) -> bool:
        return self.internalized is not None

    @property
    def truth_state(self) -> str:
        return self.result.truth_state or ""

    @property
    def final_action(self) -> str:
        return self.result.action


def snapshot_semantic_knowledge(
    architecture: SemanticKnowledgeArchitecture,
    concept: str,
    *,
    query: str | None = None,
) -> SemanticKnowledgeSnapshot:
    concept = str(concept).strip()
    if not concept:
        raise ValueError("concept must be non-empty")

    config = architecture.config
    inspect_query = (
        str(query).strip()
        if query is not None and str(query).strip()
        else f"{concept}とは"
    )

    atomic = tuple(
        item
        for item in load_propositions(config.proposition_path)
        if item.subject == concept
    )

    subject_answer = compose_subject_from_index(
        config.subject_index_path,
        concept,
    )

    typed = tuple(
        typed_subject_rows(
            config.typed_index_path,
            concept,
        )
    )

    unified_row = None
    for row in load_unified_rows(config.unified_path):
        if str(row.get("concept", "")).strip() == concept:
            unified_row = row
            break

    internalized = internalized_concept_for_focus(
        load_internalized_concepts(
            config.learning_log,
            config.learning_state,
        ),
        concept,
    )

    result = architecture.resolve(inspect_query)

    return SemanticKnowledgeSnapshot(
        concept=concept,
        query=inspect_query,
        atomic_propositions=atomic,
        subject_answer=subject_answer,
        typed_rows=typed,
        unified_row=unified_row,
        internalized=internalized,
        result=result,
    )
