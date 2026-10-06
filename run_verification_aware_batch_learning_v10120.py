#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.12.0 Verification-Aware Batch Learning Regression."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from chat import mark_pair_for_retraining, pair_fingerprint
from internalized_verification_v10119 import (
    batch_repair_plan,
    mark_batch_retrain,
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
    print("=" * 108)
    print(" LLM_TRY v10.12.0 Verification-Aware Batch Learning Regression")
    print("=" * 108)

    teachers = [
        (
            "量子センサー",
            "量子センサーとは",
            "量子センサーは量子的性質を利用して高感度計測を行うセンサーである。",
        ),
        (
            "宇宙",
            "宇宙とは",
            "宇宙は時間と空間およびその中の物質とエネルギーの総体である。",
        ),
        (
            "数学",
            "数学とは",
            "数学は数・量・構造・空間などを論理的に扱う学問である。",
        ),
    ]

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        verification = root / "verification.jsonl"
        state = root / "chat_learning_state.json"

        fingerprints = [
            pair_fingerprint(question, teacher)
            for _, question, teacher in teachers
        ]
        state.write_text(
            json.dumps(
                {"trained_fingerprints": fingerprints},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        for index, (concept, question, teacher) in enumerate(
            teachers,
            1,
        ):
            upsert_unstable(
                verification,
                concept=concept,
                question=question,
                teacher_answer=teacher,
                candidate_answer=f"bad candidate {index}",
                reason="composite teacher fidelity failed",
                semantic=0.94,
                lexical=0.30,
                required=0.25,
                contradiction=False,
                missing_terms=("required-a", "required-b"),
                fingerprint=fingerprints[index - 1],
            )

        plan = batch_repair_plan(verification)
        check(
            "batch-plan-collects-all-active",
            plan.count == 3
            and len(plan.fingerprints) == 3
            and set(plan.concepts)
            == {"量子センサー", "宇宙", "数学"},
            str(plan),
        )

        reactivated = 0
        for row in plan.tasks:
            if mark_pair_for_retraining(
                state,
                str(row["question"]),
                str(row["teacher_answer"]),
            ):
                reactivated += 1
        changed = mark_batch_retrain(
            verification,
            set(plan.fingerprints),
        )

        state_data = json.loads(
            state.read_text(encoding="utf-8")
        )
        check(
            "all-repair-pairs-reactivated",
            reactivated == 3
            and not state_data.get("trained_fingerprints"),
            str(state_data),
        )
        summary = verification_summary(verification)
        check(
            "all-tasks-enter-retrain",
            changed == 3
            and summary.retrain == 3
            and summary.pending == 0,
            str(summary),
        )

        # Simulate one batch training run consuming all three pairs.
        state.write_text(
            json.dumps(
                {"trained_fingerprints": fingerprints},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        for concept, _, teacher in teachers:
            question = next(
                q for c, q, t in teachers
                if c == concept and t == teacher
            )
            fp = pair_fingerprint(question, teacher)
            check(
                f"resolve-{concept}",
                mark_verified(
                    verification,
                    concept,
                    fp,
                ) == 1,
            )

        summary = verification_summary(verification)
        check(
            "batch-verification-resolved",
            summary.verified == 3
            and summary.retrain == 0
            and summary.pending == 0,
            str(summary),
        )

    print()
    print("Batch collection      : PASS")
    print("Bulk reactivation     : PASS")
    print("Single batch lifecycle: PASS")
    print("Per-concept resolution: PASS")
    print("STATUS                : VERIFICATION_AWARE_BATCH_LEARNING_PASS")


if __name__ == "__main__":
    main()
