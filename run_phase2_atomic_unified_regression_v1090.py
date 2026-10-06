#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.9.0 Phase 2 Unified Semantic Memory Regression."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from semantic_proposition_v1090 import add_statement
from unified_semantic_bridge_v1090 import (
    load_unified_rows,
    sync_all_propositions,
    sync_subject_from_propositions,
)


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (
        f" : {detail}" if detail else ""
    ))
    if not ok:
        raise AssertionError(name)


def main() -> None:
    print("=" * 92)
    print(" LLM_TRY v10.9.0 Phase 2 Atomic -> Unified Semantic Memory Regression")
    print("=" * 92)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        props = root / "props.jsonl"
        unified = root / "unified.jsonl"

        # Preserve an unrelated existing semantic-memory row.
        unified.write_text(
            json.dumps(
                {
                    "concept": "宇宙",
                    "assistant": "既存の宇宙知識。",
                    "source": "legacy",
                },
                ensure_ascii=False,
            ) + "\n",
            encoding="utf-8",
        )

        add_statement(props, "架空装置は高速である。")
        row1 = sync_subject_from_propositions(
            props, unified, "架空装置"
        )
        check("first-sync-created", row1 is not None, str(row1))

        rows = load_unified_rows(unified)
        device = next(
            row for row in rows
            if row.get("concept") == "架空装置"
        )
        check(
            "first-sync-answer",
            device.get("assistant") == "架空装置は、高速である。",
            str(device),
        )
        check(
            "legacy-row-preserved",
            any(row.get("concept") == "宇宙" for row in rows),
            str(rows),
        )

        add_statement(props, "架空装置は低消費電力である。")
        row2 = sync_subject_from_propositions(
            props, unified, "架空装置"
        )
        check("second-sync-updated", row2 is not None, str(row2))

        rows = load_unified_rows(unified)
        device_rows = [
            row for row in rows
            if row.get("concept") == "架空装置"
        ]
        check(
            "single-row-per-concept",
            len(device_rows) == 1,
            str(device_rows),
        )
        check(
            "composed-unified-answer",
            device_rows[0].get("assistant")
            == "架空装置は、高速であり、低消費電力である。",
            str(device_rows[0]),
        )
        check(
            "provenance",
            device_rows[0].get("source") == "atomic-proposition"
            and device_rows[0].get("atomic_count") == 2,
            str(device_rows[0]),
        )

        add_statement(props, "別装置は小型である。")
        count = sync_all_propositions(props, unified)
        check("sync-all-count", count == 2, str(count))

        rows = load_unified_rows(unified)
        check(
            "sync-all-second-subject",
            any(
                row.get("concept") == "別装置"
                and row.get("assistant") == "別装置は、小型である。"
                for row in rows
            ),
            str(rows),
        )

    print()
    print("Atomic proposition source : PASS")
    print("Unified memory upsert      : PASS")
    print("Existing rows preserved    : PASS")
    print("Duplicate concept collapse : PASS")
    print("Provenance                 : PASS")
    print("STATUS                     : PHASE2_ATOMIC_UNIFIED_PASS")


if __name__ == "__main__":
    main()
