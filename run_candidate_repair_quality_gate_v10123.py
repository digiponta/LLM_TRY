#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.12.3 Candidate Repair Quality Gate Regression."""

from __future__ import annotations

import tempfile
from pathlib import Path

from candidate_repair_quality_v10123 import (
    RepairQualityProbe,
    append_repair_quality_audit,
)


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (
        f" : {detail}" if detail else ""
    ))
    if not ok:
        raise AssertionError(name)


def main() -> None:
    print("=" * 108)
    print(" LLM_TRY v10.12.3 Candidate Repair Quality Gate Regression")
    print("=" * 108)

    with tempfile.TemporaryDirectory() as td:
        audit = Path(td) / "repair_quality.jsonl"

        passing = [
            RepairQualityProbe(
                concept="文学",
                question="文学とは",
                teacher_answer="文学とは、言語を用いた芸術",
                fingerprint="fp-literature",
                generated_answer="文学とは、言語を用いた芸術",
                semantic=1.0,
                lexical=1.0,
                required=1.0,
                contradiction=False,
                passed=True,
                reason="composite teacher fidelity passed",
            ),
            RepairQualityProbe(
                concept="架空装置",
                question="架空装置とは",
                teacher_answer="架空装置は実験用装置である。",
                fingerprint="fp-device",
                generated_answer="架空装置は実験用装置である。",
                semantic=1.0,
                lexical=1.0,
                required=1.0,
                contradiction=False,
                passed=True,
                reason="composite teacher fidelity passed",
            ),
        ]

        summary = append_repair_quality_audit(
            audit,
            probes=passing,
        )
        check(
            "all-repair-targets-pass",
            summary.ok
            and summary.targets == 2
            and summary.passed == 2
            and summary.failed == 0,
            str(summary),
        )

        failing = list(passing)
        failing[1] = RepairQualityProbe(
            concept="架空装置",
            question="架空装置とは",
            teacher_answer="架空装置は実験用装置である。",
            fingerprint="fp-device",
            generated_answer="架空装置は未知の装置である。",
            semantic=0.95,
            lexical=0.30,
            required=0.0,
            contradiction=False,
            passed=False,
            reason="composite teacher fidelity failed",
        )
        summary = append_repair_quality_audit(
            audit,
            probes=failing,
        )
        check(
            "single-target-failure-blocks-promotion",
            not summary.ok
            and summary.targets == 2
            and summary.passed == 1
            and summary.failed == 1,
            str(summary),
        )

        summary = append_repair_quality_audit(
            audit,
            probes=[],
        )
        check(
            "empty-target-set-fails-closed",
            not summary.ok
            and summary.targets == 0
            and summary.failed == 0,
            str(summary),
        )

    print()
    print("All-target pass       : PASS")
    print("Single-target failure : PASS")
    print("Empty-set fail-closed : PASS")
    print("STATUS                : CANDIDATE_REPAIR_QUALITY_GATE_PASS")


if __name__ == "__main__":
    main()
