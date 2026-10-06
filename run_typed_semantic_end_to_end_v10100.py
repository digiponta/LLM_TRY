#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.10.0 Typed Semantic End-to-End Regression."""

from __future__ import annotations

import tempfile
from pathlib import Path

from semantic_proposition_v1090 import add_statement
from subject_keyed_proposition_v1090 import (
    compose_subject_from_index,
    sync_subject_index,
)
from typed_subject_proposition_v10100 import (
    sync_typed_index,
    typed_subject_mapping,
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
    print("=" * 100)
    print(" LLM_TRY v10.10.0 Typed Semantic End-to-End Regression")
    print("=" * 100)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        props = root / "atomic.jsonl"
        subject_index = root / "subject.jsonl"
        typed_index = root / "typed.jsonl"
        unified = root / "unified.jsonl"

        add_statement(props, "GPUは高速である。")
        add_statement(props, "GPUは並列計算が得意である。")
        add_statement(props, "GPUはCUDAを利用可能である。")

        subject_count = sync_subject_index(props, subject_index)
        typed_count = sync_typed_index(props, typed_index)

        check("subject-index-count", subject_count == 3, str(subject_count))
        check("typed-index-count", typed_count == 3, str(typed_count))

        typed = typed_subject_mapping(typed_index, "GPU")
        check(
            "typed-paths",
            typed == [
                "GPU => property => GPUは高速である。",
                "GPU => capability => GPUは並列計算が得意である。",
                "GPU => relation => GPUはCUDAを利用可能である。",
            ],
            str(typed),
        )

        composed = compose_subject_from_index(subject_index, "GPU")
        check(
            "composed-answer",
            composed
            == "GPUは、高速であり、並列計算が得意であり、CUDAを利用可能である。",
            composed,
        )

        synced = sync_subject_from_propositions(props, unified, "GPU")
        check("unified-sync", synced is not None, str(synced))

        rows = load_unified_rows(unified)
        check("unified-single-row", len(rows) == 1, str(rows))
        row = rows[0]
        check(
            "unified-answer-stable",
            row.get("assistant") == composed,
            str(row),
        )
        check(
            "unified-predicate-types",
            row.get("predicate_types")
            == ["property", "capability", "relation"],
            str(row),
        )
        check(
            "unified-schema",
            row.get("semantic_schema")
            == "subject-predicate-type-statement-v10.10.0",
            str(row),
        )

    print()
    print("Atomic propositions      : PASS")
    print("Subject index            : PASS")
    print("Typed subject index      : PASS")
    print("Unified typed provenance : PASS")
    print("Runtime answer stability : PASS")
    print("STATUS                   : TYPED_SEMANTIC_END_TO_END_PASS")


if __name__ == "__main__":
    main()
