#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.11.5 Knowledge Queue Lifecycle Regression."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from knowledge_queue_lifecycle_v10115 import (
    consolidate_legacy_resolved,
    lifecycle_summary,
    mark_concept_promoted,
    mark_concept_verified,
    revoke_concept_verification,
)


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (
        f" : {detail}" if detail else ""
    ))
    if not ok:
        raise AssertionError(name)


def main() -> None:
    print("=" * 104)
    print(" LLM_TRY v10.11.5 Knowledge Queue Lifecycle Regression")
    print("=" * 104)

    with tempfile.TemporaryDirectory() as td:
        queue = Path(td) / "knowledge_queue.jsonl"
        rows = [
            {
                "resolution": "UNKNOWN_KNOWLEDGE",
                "user": "時間",
                "reason": "bare unknown",
                "timestamp": "2026-10-05T21:00:00+0900",
            },
            {
                "resolution": "UNKNOWN_KNOWLEDGE",
                "user": "時間とは",
                "reason": "raw corpus only",
                "timestamp": "2026-10-05T21:01:00+0900",
            },
            {
                "resolution": "UNKNOWN_KNOWLEDGE",
                "user": "時間とは、",
                "reason": "legacy variant",
                "timestamp": "2026-10-05T21:02:00+0900",
                "status": "resolved",
            },
        ]
        queue.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )

        before = lifecycle_summary(queue)
        check(
            "initial-summary",
            before.pending == 2
            and before.legacy_resolved == 1,
            str(before),
        )

        promoted = mark_concept_promoted(
            queue,
            "時間",
            statement="時間は、出来事の順序と間隔を表す概念である。",
            post_state="TYPED",
        )
        check("promotion-count", promoted == 2, str(promoted))

        migrated = consolidate_legacy_resolved(queue)
        check("legacy-consolidation", migrated == 1, str(migrated))

        after_promote = lifecycle_summary(queue)
        check(
            "promoted-summary",
            after_promote.pending == 0
            and after_promote.promoted == 3
            and after_promote.legacy_resolved == 0,
            str(after_promote),
        )

        verified = mark_concept_verified(
            queue,
            "時間",
            truth_source="chat-manual-truth",
        )
        check("verify-count", verified == 3, str(verified))

        after_verify = lifecycle_summary(queue)
        check(
            "verified-summary",
            after_verify.verified == 3
            and after_verify.promoted == 0,
            str(after_verify),
        )

        revoked = revoke_concept_verification(
            queue,
            "時間",
            new_truth_state="FALSE",
        )
        check("revoke-count", revoked == 3, str(revoked))

        final = lifecycle_summary(queue)
        check(
            "revoked-summary",
            final.promoted == 3
            and final.verified == 0
            and final.pending == 0,
            str(final),
        )

    print()
    print("pending -> promoted : PASS")
    print("legacy consolidation: PASS")
    print("promoted -> verified: PASS")
    print("verification revoke : PASS")
    print("STATUS              : KNOWLEDGE_QUEUE_LIFECYCLE_PASS")


if __name__ == "__main__":
    main()
