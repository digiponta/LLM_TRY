#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.11.9 Internalized Verification Loop Regression."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from chat import mark_pair_for_retraining, pair_fingerprint
from internalized_verification_v10119 import (
    active_verifications,
    mark_retrain,
    mark_verified,
    upsert_unstable,
    verification_summary,
)


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (
        f" : {detail}" if detail else ""
    ))
    if not ok:
        raise AssertionError(name)


def main() -> None:
    print("=" * 104)
    print(" LLM_TRY v10.11.9 Internalized Verification Loop Regression")
    print("=" * 104)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        verification_path = root / "verification.jsonl"
        state_path = root / "chat_learning_state.json"

        question = "量子センサーとは"
        teacher = (
            "量子センサーは、量子的性質を利用して"
            "高感度計測を行うセンサーである。"
        )
        candidate = (
            "量子センサーは、量子的な役割と研究を学習して"
            "研究を行う研究を行う。"
        )
        fp = pair_fingerprint(question, teacher)

        state_path.write_text(
            json.dumps(
                {"trained_fingerprints": [fp]},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        created = upsert_unstable(
            verification_path,
            concept="量子センサー",
            question=question,
            teacher_answer=teacher,
            candidate_answer=candidate,
            reason=(
                "composite teacher fidelity: "
                "lexical 0.407 < 0.450; "
                "required 0.333 < 0.500"
            ),
            semantic=0.938,
            lexical=0.407,
            required=0.333,
            contradiction=False,
            missing_terms=("量子的性質", "高感度計測"),
            fingerprint=fp,
        )
        summary = verification_summary(verification_path)
        check(
            "unstable-task-created",
            created
            and summary.pending == 1
            and summary.total == 1,
            str(summary),
        )

        reactivated = mark_pair_for_retraining(
            state_path,
            question,
            teacher,
        )
        moved = mark_retrain(verification_path, fp)
        state = json.loads(state_path.read_text(encoding="utf-8"))
        check(
            "trusted-pair-reactivated",
            reactivated
            and fp not in state.get("trained_fingerprints", []),
            str(state),
        )
        summary = verification_summary(verification_path)
        check(
            "verification-retrain-state",
            moved == 1
            and summary.retrain == 1
            and summary.pending == 0,
            str(summary),
        )

        rows = active_verifications(verification_path)
        check(
            "active-task-visible",
            len(rows) == 1
            and rows[0].get("status") == "retrain"
            and rows[0].get("attempts") == 1,
            str(rows),
        )

        # Simulate /train consuming the trusted pair again.
        state_path.write_text(
            json.dumps(
                {"trained_fingerprints": [fp]},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        verified = mark_verified(
            verification_path,
            "量子センサー",
            fp,
        )
        summary = verification_summary(verification_path)
        check(
            "verification-resolved-after-pass",
            verified == 1
            and summary.verified == 1
            and summary.retrain == 0
            and not active_verifications(verification_path),
            str(summary),
        )

    print()
    print("Unstable task creation : PASS")
    print("Trusted reactivation   : PASS")
    print("Retrain lifecycle      : PASS")
    print("Verification resolution: PASS")
    print("STATUS                 : INTERNALIZED_VERIFICATION_LOOP_PASS")


if __name__ == "__main__":
    main()
