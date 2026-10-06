#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.10.3 Truth State Store Regression."""

from __future__ import annotations

import tempfile
from pathlib import Path

from truth_state_v10103 import (
    TRUTH_STATES,
    effective_truth_record,
    load_truth_records,
    upsert_truth_record,
)


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (
        f" : {detail}" if detail else ""
    ))
    if not ok:
        raise AssertionError(name)


def main() -> None:
    print("=" * 96)
    print(" LLM_TRY v10.10.3 Truth State Store Regression")
    print("=" * 96)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "truth.jsonl"

        for state in TRUTH_STATES:
            upsert_truth_record(
                path,
                f"concept-{state.lower()}",
                state,
                reason=f"reason-{state.lower()}",
                correction=(
                    f"correction-{state.lower()}"
                    if state in {"FALSE", "OUTDATED"}
                    else ""
                ),
                source="test",
                updated_at="2026-10-05T21:00:00+0900",
            )

        records = load_truth_records(path)
        check("five-states", len(records) == 5, str(records))
        check(
            "state-set",
            {record.state for record in records} == set(TRUTH_STATES),
            str(records),
        )

        upsert_truth_record(
            path,
            "concept-false",
            "TRUE",
            source="test-update",
            updated_at="2026-10-05T22:00:00+0900",
        )
        records = load_truth_records(path)
        updated = next(r for r in records if r.concept == "concept-false")
        check("upsert-replaces", updated.state == "TRUE", str(updated))
        check(
            "upsert-no-duplicate",
            sum(1 for r in records if r.concept == "concept-false") == 1,
            str(records),
        )

        default = effective_truth_record(path, "missing-concept")
        check(
            "default-unverified",
            default.state == "UNVERIFIED"
            and default.source == "truth-default",
            str(default),
        )

    print()
    print("State persistence    : PASS")
    print("Upsert semantics     : PASS")
    print("Default UNVERIFIED   : PASS")
    print("STATUS               : TRUTH_STATE_STORE_PASS")


if __name__ == "__main__":
    main()
