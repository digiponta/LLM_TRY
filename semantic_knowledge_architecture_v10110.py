#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_TRY v10.11.0 Semantic Knowledge Architecture

Single facade over:
  Atomic Proposition
  Subject / Typed Index
  Unified Semantic Memory
  Internalized Knowledge
  Knowledge State Resolver
  Provenance
  Dispatcher
  Truth State Overlay

The lower-level modules remain independently testable. Runtime clients should
prefer this facade so routing policy exists in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Mapping

from knowledge_state_resolver_v10100 import (
    KnowledgeState,
    resolve_knowledge_state,
)
from knowledge_state_dispatcher_v10101 import (
    DispatchResult,
    dispatch_knowledge_state,
)
from truth_state_v10103 import (
    TruthRecord,
    effective_truth_record,
)
from truth_aware_dispatch_v10103 import (
    TruthDispatchResult,
    apply_truth_policy,
)
from semantic_proposition_v1090 import (
    Proposition,
    add_statement,
    load_propositions,
)
from subject_keyed_proposition_v1090 import (
    compose_subject_from_index,
    sync_subject_index,
)
from typed_subject_proposition_v10100 import (
    sync_typed_index,
    typed_subject_rows,
)
from unified_semantic_bridge_v1090 import (
    load_unified_rows,
    sync_all_propositions,
    sync_subject_from_propositions,
)
from internalized_knowledge_v10100 import (
    InternalizedConcept,
    internalized_concept_for_focus,
    load_internalized_concepts,
)


@dataclass(frozen=True)
class SemanticKnowledgeConfig:
    proposition_path: Path
    subject_index_path: Path
    typed_index_path: Path
    unified_path: Path
    learning_log: Path
    learning_state: Path
    raw_corpus_path: Path
    truth_store_path: Path
    canonical_definitions: Mapping[str, str]
    checkpoint_fingerprints: frozenset[str] | None = None


@dataclass(frozen=True)
class SemanticKnowledgeResult:
    query: str
    knowledge_state: KnowledgeState
    base_dispatch: DispatchResult
    dispatch: DispatchResult
    truth: TruthRecord | None = None
    truth_result: TruthDispatchResult | None = None
    routed_query: str = ""
    bare_concept_routed: bool = False
    bare_unknown_blocked: bool = False
    bare_focus: str = ""

    @property
    def action(self) -> str:
        return self.dispatch.action

    @property
    def answer(self) -> str:
        return self.dispatch.answer

    @property
    def focus(self) -> str:
        return self.dispatch.focus

    @property
    def state(self) -> str:
        return self.dispatch.state

    @property
    def truth_state(self) -> str:
        return self.truth.state if self.truth is not None else ""

    @property
    def provenance(self):
        return self.dispatch.provenance


@dataclass(frozen=True)
class SemanticKnowledgeSyncResult:
    atomic_count: int
    subject_index_count: int
    typed_index_count: int
    unified_subject_count: int


@dataclass(frozen=True)
class SemanticKnowledgeSnapshot:
    concept: str
    query: str
    atomic_propositions: tuple[Proposition, ...]
    subject_answer: str
    typed_rows: tuple[object, ...]
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


class SemanticKnowledgeArchitecture:
    """Single runtime/control surface for semantic knowledge."""

    BARE_CONVERSATIONAL_ALLOWLIST = {
        "長門",
        "長門有希",
        "hello",
        "hi",
    }

    def __init__(self, config: SemanticKnowledgeConfig):
        self.config = config

    @staticmethod
    def _bare_concept_focus(query: str) -> str:
        q = str(query).strip()
        if not q or len(q) > 24:
            return ""
        if re.search(r"[\s。、！？!?？,:：;；]", q):
            return ""
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_+.#\-]{1,23}", q):
            return q
        if re.fullmatch(r"[一-龯々ァ-ヶー]{2,24}", q):
            return q
        return ""

    def _validated_semantic_concept_exists(self, concept: str) -> bool:
        if not concept:
            return False

        if compose_subject_from_index(
            self.config.subject_index_path,
            concept,
        ):
            return True

        if self.config.canonical_definitions.get(concept.lower(), ""):
            return True

        for row in load_unified_rows(self.config.unified_path):
            if str(row.get("concept", "")).strip() == concept:
                if str(row.get("assistant", "")).strip():
                    return True

        internalized = internalized_concept_for_focus(
            load_internalized_concepts(
                self.config.learning_log,
                self.config.learning_state,
            ),
            concept,
        )
        return internalized is not None

    def canonicalize_bare_semantic_query(
        self,
        query: str,
    ) -> tuple[str, bool]:
        focus = self._bare_concept_focus(query)
        if not focus:
            return query, False
        if not self._validated_semantic_concept_exists(focus):
            return query, False
        return f"{focus}とは", True

    def resolve(
        self,
        query: str,
    ) -> SemanticKnowledgeResult:
        bare_focus = self._bare_concept_focus(query)
        routed_query, bare_concept_routed = (
            self.canonicalize_bare_semantic_query(query)
        )

        bare_unknown_blocked = (
            bool(bare_focus)
            and not bare_concept_routed
            and bare_focus.lower()
            not in {
                item.lower()
                for item in self.BARE_CONVERSATIONAL_ALLOWLIST
            }
        )

        if bare_unknown_blocked:
            state = KnowledgeState(
                state="UNKNOWN",
                focus=bare_focus,
                reason=(
                    "bare concept lacks validated semantic "
                    "or conversational evidence"
                ),
            )
        else:
            state = resolve_knowledge_state(
                routed_query,
            typed_index_path=self.config.typed_index_path,
            subject_index_path=self.config.subject_index_path,
            unified_path=self.config.unified_path,
            learning_log=self.config.learning_log,
            learning_state=self.config.learning_state,
            raw_corpus_path=self.config.raw_corpus_path,
                canonical_definitions=self.config.canonical_definitions,
                checkpoint_fingerprints=(
                    set(self.config.checkpoint_fingerprints)
                    if self.config.checkpoint_fingerprints is not None
                    else None
                ),
            )
        base_dispatch = dispatch_knowledge_state(state)
        dispatch = base_dispatch
        truth = None
        truth_result = None

        # UNKNOWN has no concept evidence to annotate. NON_CONCEPT has no
        # semantic concept either. All concept-bearing states use the overlay.
        if dispatch.focus and dispatch.state != "UNKNOWN":
            truth = effective_truth_record(
                self.config.truth_store_path,
                dispatch.focus,
            )
            truth_result = apply_truth_policy(
                dispatch,
                truth,
            )
            dispatch = truth_result.dispatch

        return SemanticKnowledgeResult(
            query=query,
            knowledge_state=state,
            base_dispatch=base_dispatch,
            dispatch=dispatch,
            truth=truth,
            truth_result=truth_result,
            routed_query=routed_query,
            bare_concept_routed=bare_concept_routed,
            bare_unknown_blocked=bare_unknown_blocked,
            bare_focus=bare_focus,
        )

    def teach_proposition(
        self,
        statement: str,
    ) -> list[Proposition]:
        added = add_statement(
            self.config.proposition_path,
            statement,
        )
        if not added:
            return []

        sync_subject_index(
            self.config.proposition_path,
            self.config.subject_index_path,
        )
        sync_typed_index(
            self.config.proposition_path,
            self.config.typed_index_path,
        )
        sync_subject_from_propositions(
            self.config.proposition_path,
            self.config.unified_path,
            added[0].subject,
        )
        return added

    def sync_all(self) -> SemanticKnowledgeSyncResult:
        propositions = load_propositions(
            self.config.proposition_path
        )
        subject_count = sync_subject_index(
            self.config.proposition_path,
            self.config.subject_index_path,
        )
        typed_count = sync_typed_index(
            self.config.proposition_path,
            self.config.typed_index_path,
        )
        unified_count = sync_all_propositions(
            self.config.proposition_path,
            self.config.unified_path,
        )
        return SemanticKnowledgeSyncResult(
            atomic_count=len(propositions),
            subject_index_count=subject_count,
            typed_index_count=typed_count,
            unified_subject_count=unified_count,
        )

    def status(self) -> dict[str, object]:
        return {
            "architecture": "Semantic Knowledge Architecture",
            "version": "v10.12.1",
            "proposition_path": str(self.config.proposition_path),
            "subject_index_path": str(self.config.subject_index_path),
            "typed_index_path": str(self.config.typed_index_path),
            "unified_path": str(self.config.unified_path),
            "learning_log": str(self.config.learning_log),
            "learning_state": str(self.config.learning_state),
            "raw_corpus_path": str(self.config.raw_corpus_path),
            "truth_store_path": str(self.config.truth_store_path),
            "checkpoint_fingerprint_count": (
                len(self.config.checkpoint_fingerprints)
                if self.config.checkpoint_fingerprints is not None
                else -1
            ),
            "layers": (
                "proposition",
                "subject-index",
                "typed-index",
                "unified-memory",
                "internalized",
                "provenance",
                "truth-state",
                "resolver",
                "dispatcher",
            ),
        }
