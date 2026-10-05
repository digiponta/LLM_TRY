#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.10.0 Typed Query End-to-End Semantic Regression."""

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
    typed_query_lookup,
)


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (
        f" : {detail}" if detail else ""
    ))
    if not ok:
        raise AssertionError(name)


def main() -> None:
    print("=" * 100)
    print(" LLM_TRY v10.10.0 Typed Query End-to-End Regression")
    print("=" * 100)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        props = root / "atomic.jsonl"
        subject_index = root / "subject.jsonl"
        typed_index = root / "typed.jsonl"

        for statement in (
            "GPUは高速である。",
            "GPUは低消費電力である。",
            "GPUは並列計算が得意である。",
            "GPUはCUDAを利用可能である。",
        ):
            add_statement(props, statement)

        sync_subject_index(props, subject_index)
        sync_typed_index(props, typed_index)

        full = compose_subject_from_index(subject_index, "GPU")
        check(
            "full-compose-preserved",
            full == (
                "GPUは、高速であり、低消費電力であり、"
                "並列計算が得意であり、CUDAを利用可能である。"
            ),
            full,
        )

        property_hit = typed_query_lookup(typed_index, "GPUの性質は")
        check(
            "property-query",
            property_hit == (
                "GPU",
                "property",
                "GPUは、高速であり、低消費電力である。",
            ),
            str(property_hit),
        )

        capability_hit = typed_query_lookup(
            typed_index,
            "GPUは何が得意",
        )
        check(
            "capability-query",
            capability_hit == (
                "GPU",
                "capability",
                "GPUは、並列計算が得意である。",
            ),
            str(capability_hit),
        )

        relation_hit = typed_query_lookup(
            typed_index,
            "GPUとCUDAの関係は",
        )
        check(
            "relation-query",
            relation_hit == (
                "GPU",
                "relation",
                "GPUは、CUDAを利用可能である。",
            ),
            str(relation_hit),
        )

        # Generic subject question has no explicit typed cue and must remain
        # on the existing full-compose path.
        generic_hit = typed_query_lookup(
            typed_index,
            "GPUについて教えて",
        )
        check(
            "generic-falls-through",
            generic_hit is None,
            str(generic_hit),
        )

    print()
    print("Full compose preserved : PASS")
    print("Property routing       : PASS")
    print("Capability routing     : PASS")
    print("Relation routing       : PASS")
    print("Generic fallback       : PASS")
    print("STATUS                 : TYPED_QUERY_END_TO_END_PASS")


if __name__ == "__main__":
    main()
