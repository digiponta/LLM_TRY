#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_TRY v10.11.4 Knowledge Promotion Pipeline

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
import json
import time
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


@dataclass(frozen=True)
class PromotionResult:
    concept: str
    statement: str
    promoted: bool
    propositions: tuple[Proposition, ...] = ()
    queue_resolved: int = 0
    reason: str = ""
    post_result: SemanticKnowledgeResult | None = None


def normalize_queue_text(text: str) -> str:
    return " ".join(str(text).strip().split()).lower()


def _matches_concept_request(user_text: str, concept: str) -> bool:
    user = normalize_queue_text(user_text)
    concept_norm = normalize_queue_text(concept)
    return user in {
        concept_norm,
        normalize_queue_text(f"{concept}とは"),
        normalize_queue_text(f"{concept}について教えて"),
    }


def pending_knowledge_requests(
    queue_path: Path,
) -> list[dict]:
    if not queue_path.exists():
        return []

    rows: list[dict] = []
    for raw in queue_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if str(row.get("resolution", "")) != "UNKNOWN_KNOWLEDGE":
            continue
        if str(row.get("status", "pending")) != "pending":
            continue
        rows.append(row)
    return rows


def pending_request_for_concept(
    queue_path: Path,
    concept: str,
) -> dict | None:
    for row in pending_knowledge_requests(queue_path):
        if _matches_concept_request(
            str(row.get("user", "")),
            concept,
        ):
            return row
    return None


def mark_promoted_requests(
    queue_path: Path,
    concept: str,
    statement: str,
    post_state: str,
) -> int:
    if not queue_path.exists():
        return 0

    changed = 0
    output: list[dict] = []
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    for raw in queue_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue

        if (
            str(row.get("resolution", "")) == "UNKNOWN_KNOWLEDGE"
            and str(row.get("status", "pending")) == "pending"
            and _matches_concept_request(
                str(row.get("user", "")),
                concept,
            )
        ):
            row["status"] = "promoted"
            row["promoted_at"] = now
            row["promoted_concept"] = concept
            row["promoted_statement"] = statement
            row["post_knowledge_state"] = post_state
            changed += 1

        output.append(row)

    if changed:
        queue_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False) + "\n"
                for row in output
            ),
            encoding="utf-8",
        )

    return changed


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

    resolved = mark_promoted_requests(
        queue_path,
        concept,
        statement,
        post.state,
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
