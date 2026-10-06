#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.12.14 Two-Pass Function Semantic Resolver.

Pass 1:
    Generate ordinary subject knowledge from the current production model.

Pass 2:
    Extract action / target / purpose only from that generated evidence, then
    ask the same model to compose a concise function-oriented answer.

No teacher answer, HOLDOUT answer, or gold function slots are used at runtime.
No model weights are modified.
"""

from __future__ import annotations

from dataclasses import dataclass

from function_semantic_decomposition_v101213 import (
    FunctionStructure,
    infer_action,
    infer_purpose,
    infer_target,
    normalize,
)


@dataclass(frozen=True)
class TwoPassFunctionResult:
    subject: str
    evidence: str
    structure: FunctionStructure
    compose_prompt: str
    answer: str
    used_fallback: bool


def extract_structure_from_evidence(
    subject: str,
    evidence: str,
) -> FunctionStructure:
    """Derive function slots only from model-generated evidence."""
    subject_n = normalize(subject)
    evidence_n = normalize(evidence)
    if not subject_n:
        raise ValueError("subject must be non-empty")
    if not evidence_n:
        raise ValueError("evidence must be non-empty")

    return FunctionStructure(
        subject=subject_n,
        action=infer_action(evidence_n),
        target=infer_target(subject_n, evidence_n),
        purpose=infer_purpose(evidence_n),
        answer=evidence_n,
    )


def function_evidence_query(subject: str) -> str:
    """Pass-1 query: intentionally ordinary and structure-free."""
    subject_n = normalize(subject)
    if not subject_n:
        raise ValueError("subject must be non-empty")
    return f"{subject_n}とは"


def function_compose_prompt(
    subject: str,
    evidence: str,
    structure: FunctionStructure,
) -> str:
    """Pass-2 prompt built only from Pass-1 evidence and extracted slots."""
    subject_n = normalize(subject)
    evidence_n = normalize(evidence)
    return (
        f"主語: {subject_n}\n"
        "意味役割: function\n"
        f"action: {structure.action}\n"
        f"target: {structure.target}\n"
        f"purpose: {structure.purpose}\n"
        f"根拠: {evidence_n}\n"
        "要求: 根拠に含まれる情報だけを使い、対象が何をするものかを"
        "一文で簡潔に説明してください。根拠にない内容は追加しないでください"
    )


def structure_is_informative(structure: FunctionStructure) -> bool:
    """At least one slot must contain information beyond generic defaults."""
    return any((
        structure.action != "function",
        structure.target != "unspecified",
        structure.purpose != "unspecified",
    ))


def deterministic_function_render(
    subject: str,
    evidence: str,
    structure: FunctionStructure,
) -> str:
    """Conservative fallback when Pass-2 generation is unusable.

    Prefer the generated evidence verbatim when the inferred structure is too
    weak.  Otherwise render only extracted information; never invent slots.
    """
    subject_n = normalize(subject)
    evidence_n = normalize(evidence)

    if not structure_is_informative(structure):
        return evidence_n

    parts: list[str] = []
    if structure.action != "function":
        parts.append(f"action={structure.action}")
    if structure.target != "unspecified":
        parts.append(f"target={structure.target}")
    if structure.purpose != "unspecified":
        parts.append(f"purpose={structure.purpose}")

    if not parts:
        return evidence_n

    return f"{subject_n}の機能は、" + "、".join(parts) + f"。根拠: {evidence_n}"


def relation_uses_two_pass(relation: str) -> bool:
    """Only function is routed to the v10.12.14 resolver."""
    return normalize(relation).lower() == "function"
