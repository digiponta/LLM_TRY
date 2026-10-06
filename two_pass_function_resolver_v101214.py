#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.12.16 Subject-Keyed Corpus Memory + Two-Pass Function Resolver.

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
from pathlib import Path

from subject_keyed_corpus_memory_v101216 import compose_subject_evidence
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
    evidence_queries: tuple[str, ...] = ()
    evidence_items: tuple[str, ...] = ()
    memory_evidence: str = ""
    memory_hit: bool = False


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
    """Backward-compatible primary Pass-1 query."""
    return function_evidence_queries(subject)[0]


def function_evidence_queries(subject: str) -> tuple[str, ...]:
    """Leakage-free evidence probes.

    These prompts mention only the subject and the generic notion of role or
    behavior. They never include teacher answers or gold function slots.
    """
    subject_n = normalize(subject)
    if not subject_n:
        raise ValueError("subject must be non-empty")
    return (
        f"{subject_n}とは",
        f"{subject_n}は何をするものですか",
        f"{subject_n}の役割は何ですか",
    )


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


def resolve_two_pass_function(
    subject: str,
    generate,
    malformed=None,
    corpus_memory: str | Path | None = None,
) -> TwoPassFunctionResult:
    """Run leakage-free two-pass resolution with an injected generator.

    generate(query) must return text from the current model. malformed
    optionally accepts generated text and returns True for unusable output.
    """
    queries = function_evidence_queries(subject)
    evidence_items: list[str] = []

    memory_evidence = ""
    if corpus_memory is not None:
        memory_evidence = normalize(
            compose_subject_evidence(Path(corpus_memory), subject)
        )
        if memory_evidence:
            evidence_items.append(memory_evidence)

    # Generated probes remain useful as secondary evidence, but exact corpus
    # memory is the primary subject-keyed source when present.
    for query in queries:
        generated = normalize(generate(query))
        if not generated:
            continue
        if malformed is not None and malformed(generated):
            continue
        if generated not in evidence_items:
            evidence_items.append(generated)

    if not evidence_items:
        raise RuntimeError("Pass-1 evidence generation returned no usable text")

    evidence = " / ".join(evidence_items)
    structure = extract_structure_from_evidence(subject, evidence)
    compose_prompt = function_compose_prompt(subject, evidence, structure)
    composed = normalize(generate(compose_prompt))

    bad = not composed
    if malformed is not None and composed:
        bad = bool(malformed(composed))

    if bad:
        answer = deterministic_function_render(subject, evidence, structure)
        used_fallback = True
    else:
        answer = composed
        used_fallback = False

    return TwoPassFunctionResult(
        subject=normalize(subject),
        evidence=evidence,
        structure=structure,
        compose_prompt=compose_prompt,
        answer=answer,
        used_fallback=used_fallback,
        evidence_queries=queries,
        evidence_items=tuple(evidence_items),
        memory_evidence=memory_evidence,
        memory_hit=bool(memory_evidence),
    )
