#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.9.0 Subject-Keyed End-to-End Semantic Regression."""

from __future__ import annotations

import tempfile
from pathlib import Path

from semantic_proposition_v1090 import add_statement
from subject_keyed_proposition_v1090 import (
    compose_subject_from_index,
    subject_mapping,
    sync_subject_index,
)
from unified_semantic_bridge_v1090 import (
    load_unified_rows,
    sync_subject_from_propositions,
)


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (
        f" : {detail}" if detail else ""
    ))
    if not ok:
        raise AssertionError(name)


def main() -> None:
    print("=" * 96)
    print(" LLM_TRY v10.9.0 Subject-Keyed End-to-End Semantic Regression")
    print("=" * 96)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        props = root / "atomic.jsonl"
        index = root / "subject_index.jsonl"
        unified = root / "unified.jsonl"

        add_statement(props, "架空装置は高速である。")
        add_statement(props, "架空装置は低消費電力である。")

        count = sync_subject_index(props, index)
        check("subject-index-count", count == 2, str(count))

        mappings = subject_mapping(index, "架空装置")
        check(
            "subject-keyed-form",
            mappings == [
                "架空装置 => 架空装置は高速である。",
                "架空装置 => 架空装置は低消費電力である。",
            ],
            str(mappings),
        )

        composed = compose_subject_from_index(index, "架空装置")
        check(
            "subject-index-compose",
            composed == "架空装置は、高速であり、低消費電力である。",
            composed,
        )

        synced = sync_subject_from_propositions(
            props,
            unified,
            "架空装置",
        )
        check("unified-sync", synced is not None, str(synced))

        rows = load_unified_rows(unified)
        check(
            "unified-answer-equals-index-compose",
            len(rows) == 1
            and rows[0].get("assistant") == composed,
            str(rows),
        )

    print()
    print("Atomic source              : PASS")
    print("Subject-keyed index        : PASS")
    print("Subject => full statement  : PASS")
    print("Multi-predicate compose    : PASS")
    print("Unified memory consistency : PASS")
    print("STATUS                     : SUBJECT_KEYED_END_TO_END_PASS")


if __name__ == "__main__":
    main()
