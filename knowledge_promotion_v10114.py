#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_TRY v10.11.5 Knowledge Promotion Pipeline

Promote a pending UNKNOWN_KNOWLEDGE request into validated Semantic Knowledge.

Pipeline:
    UNKNOWN queue
      -> explicit human teaching
      -> structural proposition validation
      -> Proposition / Subject / Typed / Unified synchronization
      -> queue promotion record
      -> bare semantic routing enabled

Truth State is intentionally NOT promoted to TRUE automatically.
Newly promoted knowledge remains UNVERIFIED unless truth is explicitly set.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from semantic_proposition_v1090 import (
    Proposition,
    decompose_statement,
    normalize_fragment,
)
from semantic_knowledge_architecture_v10110 import (
    SemanticKnowledgeArchitecture,
    SemanticKnowledgeResult,
)
from knowledge_queue_lifecycle_v10115 import (
    mark_concept_promoted,
    pending_request_for_concept,
    pending_requests,
)


@dataclass(frozen=True)
class PromotionResult:
    concept: str
    statement: str
    promoted: bool
    propositions: tuple[Proposition, ...] = ()
    queue_resolved: int = 0
    reason: str = ""
    post_result: SemanticKnowledgeResult | None = None


def pending_knowledge_requests(
    queue_path: Path,
) -> list[dict]:
    """Backward-compatible alias for the consolidated lifecycle manager."""
    return pending_requests(queue_path)


def validate_promotion_statement(
    concept: str,
    statement: str,
) -> tuple[bool, str, tuple[Proposition, ...]]:
    concept = normalize_fragment(concept)
    propositions = tuple(decompose_statement(statement))

    if not concept:
        return False, "concept must be non-empty", ()

    if not propositions:
        return (
            False,
            "statement must use semantic proposition form: XはYである",
            (),
        )

    wrong_subjects = sorted({
        item.subject
        for item in propositions
        if item.subject != concept
    })
    if wrong_subjects:
        return (
            False,
            "statement subject does not match promotion concept: "
            + ",".join(wrong_subjects),
            (),
        )

    return True, "validated semantic proposition", propositions


def promote_knowledge(
    architecture: SemanticKnowledgeArchitecture,
    queue_path: Path,
    concept: str,
    statement: str,
    *,
    require_pending: bool = True,
) -> PromotionResult:
    concept = normalize_fragment(concept)
    statement = str(statement).strip()

    valid, reason, propositions = validate_promotion_statement(
        concept,
        statement,
    )
    if not valid:
        return PromotionResult(
            concept=concept,
            statement=statement,
            promoted=False,
            reason=reason,
        )

    pending = pending_request_for_concept(queue_path, concept)
    if require_pending and pending is None:
        return PromotionResult(
            concept=concept,
            statement=statement,
            promoted=False,
            reason="no pending UNKNOWN_KNOWLEDGE request for concept",
        )

    added = architecture.teach_proposition(statement)
    if not added:
        return PromotionResult(
            concept=concept,
            statement=statement,
            promoted=False,
            reason="semantic proposition teaching rejected",
        )

    post = architecture.resolve(concept)
    if not post.bare_concept_routed:
        return PromotionResult(
            concept=concept,
            statement=statement,
            promoted=False,
            propositions=tuple(added),
            reason="promotion stored but bare semantic routing not enabled",
            post_result=post,
        )

    resolved = mark_concept_promoted(
        queue_path,
        concept,
        statement=statement,
        post_state=post.state,
        source="knowledge-promotion-v10.11.5",
    )

    return PromotionResult(
        concept=concept,
        statement=statement,
        promoted=True,
        propositions=tuple(added),
        queue_resolved=resolved,
        reason="knowledge promoted to validated semantic layer",
        post_result=post,
    )
