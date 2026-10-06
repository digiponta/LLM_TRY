#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.12.3 Candidate Repair Quality Gate support."""

from __future__ import annotations

from dataclasses import dataclass
import json
import time
from pathlib import Path


@dataclass(frozen=True)
class RepairQualityProbe:
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
class RepairQualitySummary:
    targets: int
    passed: int
    failed: int

    @property
    def ok(self) -> bool:
        return self.failed == 0 and self.targets > 0


def append_repair_quality_audit(
    path: Path,
    *,
    probes: list[RepairQualityProbe],
) -> RepairQualitySummary:
    passed = sum(1 for probe in probes if probe.passed)
    failed = len(probes) - passed
    summary = RepairQualitySummary(
        targets=len(probes),
        passed=passed,
        failed=failed,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "version": "v10.12.3",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "targets": summary.targets,
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
