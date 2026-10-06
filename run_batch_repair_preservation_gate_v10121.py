#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.12.1 Batch Repair Preservation Gate Regression."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from batch_repair_preservation_v10121 import (
    PreservationProbe,
    append_preservation_audit,
    protected_internalized_records,
)
from internalized_knowledge_v10100 import pair_fingerprint


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (
        f" : {detail}" if detail else ""
    ))
    if not ok:
        raise AssertionError(name)


def main() -> None:
    print("=" * 108)
    print(" LLM_TRY v10.12.1 Batch Repair Preservation Gate Regression")
    print("=" * 108)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        log = root / "chat_history.jsonl"
        audit = root / "preservation.jsonl"

        rows = [
            {
                "user": "量子センサーとは",
                "assistant": "量子センサーは高感度計測を行うセンサーである。",
                "source": "chat-manual",
            },
            {
                "user": "数学とは",
                "assistant": "数学は数や構造を論理的に扱う学問である。",
                "source": "chat-manual",
            },
            {
                "user": "文学とは",
                "assistant": "文学は言語による芸術を探究する学問である。",
                "source": "chat-manual",
            },
        ]
        log.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )

        fps = {
            pair_fingerprint(row["user"], row["assistant"])
            for row in rows
        }
        repair_fp = pair_fingerprint(
            rows[2]["user"],
            rows[2]["assistant"],
        )

        protected = protected_internalized_records(
            log,
            fps,
            {repair_fp},
        )
        check(
            "repair-target-excluded",
            len(protected) == 2
            and {r.concept for r in protected}
            == {"量子センサー", "数学"},
            str([r.concept for r in protected]),
        )

        passing = [
            PreservationProbe(
                concept=record.concept,
                question=record.question,
                teacher_answer=record.teacher_answer,
                fingerprint=record.fingerprint,
                generated_answer=record.teacher_answer,
                semantic=1.0,
                lexical=1.0,
                required=1.0,
                contradiction=False,
                passed=True,
                reason="composite teacher fidelity passed",
            )
            for record in protected
        ]
        summary = append_preservation_audit(
            audit,
            batch_fingerprints=(repair_fp,),
            probes=passing,
        )
        check(
            "all-protected-pass",
            summary.ok
            and summary.protected == 2
            and summary.passed == 2
            and summary.failed == 0,
            str(summary),
        )

        failing = list(passing)
        failing[0] = PreservationProbe(
            concept=failing[0].concept,
            question=failing[0].question,
            teacher_answer=failing[0].teacher_answer,
            fingerprint=failing[0].fingerprint,
            generated_answer="壊れた回答",
            semantic=0.91,
            lexical=0.10,
            required=0.0,
            contradiction=False,
            passed=False,
            reason="composite teacher fidelity failed",
        )
        summary = append_preservation_audit(
            audit,
            batch_fingerprints=(repair_fp,),
            probes=failing,
        )
        check(
            "single-regression-fails-batch",
            not summary.ok
            and summary.failed == 1
            and summary.passed == 1,
            str(summary),
        )

    print()
    print("Protected-set selection : PASS")
    print("All-pass batch gate      : PASS")
    print("Regression detection     : PASS")
    print("STATUS                   : BATCH_REPAIR_PRESERVATION_GATE_PASS")


if __name__ == "__main__":
    main()
