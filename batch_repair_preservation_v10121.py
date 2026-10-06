#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.12.1 Batch Repair Preservation Gate support."""

from __future__ import annotations

from dataclasses import dataclass
import json
import time
from pathlib import Path

from internalized_knowledge_v10100 import (
    InternalizedRecord,
    TRUSTED_SOURCES,
    extract_definition_focus,
    pair_fingerprint,
)


@dataclass(frozen=True)
class PreservationProbe:
    concept: str
    question: str
    teacher_answer: str
    fingerprint: str
    generated_answer: str
    semantic: float
    lexical: float
    required: float
    contradiction: bool
    passed: bool
    reason: str


@dataclass(frozen=True)
class PreservationSummary:
    protected: int
    passed: int
    failed: int

    @property
    def ok(self) -> bool:
        return self.failed == 0


def protected_internalized_records(
    learning_log: Path,
    checkpoint_fingerprints: set[str] | frozenset[str],
    excluded_fingerprints: set[str] | frozenset[str],
    excluded_concepts: set[str] | frozenset[str] = frozenset(),
) -> list[InternalizedRecord]:
    """Build concept-level pre-repair protected INTERNALIZED baseline.

    Repair targets are excluded by concept, not only by one teaching
    fingerprint. For the remaining concepts, keep the latest checkpoint-bound
    trusted record, matching normal INTERNALIZED runtime semantics.
    """
    if not learning_log.exists():
        return []

    bound = {str(x) for x in checkpoint_fingerprints}
    excluded = {str(x) for x in excluded_fingerprints}
    excluded_names = {
        str(x).strip().lower()
        for x in excluded_concepts
        if str(x).strip()
    }

    latest_by_concept: dict[str, InternalizedRecord] = {}

    for raw in learning_log.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue

        source = str(row.get("source", "")).strip()
        if source not in TRUSTED_SOURCES:
            continue

        question = str(row.get("user", "")).strip()
        teacher = str(row.get("assistant", "")).strip()
        if not question or not teacher:
            continue

        fp = pair_fingerprint(question, teacher)
        if fp not in bound or fp in excluded:
            continue

        concept = extract_definition_focus(question)
        if not concept:
            continue

        key = concept.strip().lower()
        if key in excluded_names:
            continue

        # Assignment in log order deliberately keeps the latest trusted row.
        latest_by_concept[key] = InternalizedRecord(
            concept=concept,
            question=question,
            teacher_answer=teacher,
            fingerprint=fp,
            source=source,
            timestamp=str(row.get("timestamp", "")).strip(),
        )

    return list(latest_by_concept.values())


def append_preservation_audit(
    path: Path,
    *,
    batch_fingerprints: tuple[str, ...],
    probes: list[PreservationProbe],
) -> PreservationSummary:
    passed = sum(1 for probe in probes if probe.passed)
    failed = len(probes) - passed
    summary = PreservationSummary(
        protected=len(probes),
        passed=passed,
        failed=failed,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "version": "v10.12.1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "batch_fingerprints": list(batch_fingerprints),
        "protected": summary.protected,
        "passed": summary.passed,
        "failed": summary.failed,
        "status": "PASS" if summary.ok else "FAIL",
        "probes": [
            {
                "concept": probe.concept,
                "question": probe.question,
                "fingerprint": probe.fingerprint,
                "generated_answer": probe.generated_answer,
                "semantic": probe.semantic,
                "lexical": probe.lexical,
                "required": probe.required,
                "contradiction": probe.contradiction,
                "passed": probe.passed,
                "reason": probe.reason,
            }
            for probe in probes
        ],
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return summary
