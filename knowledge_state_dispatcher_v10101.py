#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_TRY v10.10.1 Knowledge State Dispatcher

Convert a resolved KnowledgeState into exactly one runtime action.

Actions:
    RETRIEVE  - return deterministic answer from validated knowledge source
    GENERATE  - allow model generation
    BLOCK     - do not generate; route to knowledge teaching/review
"""

from __future__ import annotations

from dataclasses import dataclass

from knowledge_state_resolver_v10100 import KnowledgeState
from knowledge_provenance_v10102 import KnowledgeProvenance


DISPATCH_ACTIONS = (
    "RETRIEVE",
    "GENERATE",
    "BLOCK",
)


@dataclass(frozen=True)
class DispatchResult:
    action: str
    state: str
    focus: str = ""
    answer: str = ""
    route: str = ""
    predicate_type: str = ""
    reason: str = ""
    provenance: KnowledgeProvenance | None = None

    @property
    def is_terminal(self) -> bool:
        return self.action in {"RETRIEVE", "BLOCK"}


def dispatch_knowledge_state(
    state: KnowledgeState,
) -> DispatchResult:
    """Map one resolved knowledge state to one runtime action."""
    if state.state == "TYPED":
        return DispatchResult(
            action="RETRIEVE",
            state=state.state,
            focus=state.focus,
            answer=state.answer,
            route="typed semantic retrieval",
            predicate_type=state.predicate_type,
            reason=state.reason,
            provenance=state.provenance,
        )

    if state.state == "CANONICAL":
        return DispatchResult(
            action="RETRIEVE",
            state=state.state,
            focus=state.focus,
            answer=state.answer,
            route="canonical definition retrieval",
            reason=state.reason,
            provenance=state.provenance,
        )

    if state.state == "UNIFIED":
        return DispatchResult(
            action="RETRIEVE",
            state=state.state,
            focus=state.focus,
            answer=state.answer,
            route="unified semantic retrieval",
            reason=state.reason,
            provenance=state.provenance,
        )

    if state.state == "INTERNALIZED":
        return DispatchResult(
            action="GENERATE",
            state=state.state,
            focus=state.focus,
            route="internalized model generation",
            reason=state.reason,
            provenance=state.provenance,
        )

    if state.state == "INTERNALIZED_STALE":
        return DispatchResult(
            action="BLOCK",
            state=state.state,
            focus=state.focus,
            route="internalized checkpoint stale",
            reason=state.reason,
            provenance=state.provenance,
        )

    if state.state == "NON_CONCEPT":
        return DispatchResult(
            action="GENERATE",
            state=state.state,
            route="normal model generation",
            reason=state.reason,
            provenance=state.provenance,
        )

    if state.state == "RAW_CORPUS_ONLY":
        return DispatchResult(
            action="BLOCK",
            state=state.state,
            focus=state.focus,
            route="raw corpus review",
            reason=state.reason,
            provenance=state.provenance,
        )

    if state.state == "UNKNOWN":
        return DispatchResult(
            action="BLOCK",
            state=state.state,
            focus=state.focus,
            route="unknown knowledge",
            reason=state.reason,
            provenance=state.provenance,
        )

    raise ValueError(f"unsupported knowledge state: {state.state!r}")
