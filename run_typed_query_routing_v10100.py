#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.10.0 Typed Predicate Query Routing Regression."""

from __future__ import annotations

import tempfile
from pathlib import Path

from semantic_proposition_v1090 import add_statement
from typed_subject_proposition_v10100 import (
    detect_query_predicate_type,
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
    print("=" * 96)
    print(" LLM_TRY v10.10.0 Typed Predicate Query Routing Regression")
    print("=" * 96)

    check(
        "detect-property",
        detect_query_predicate_type("GPUの性質は") == "property",
        str(detect_query_predicate_type("GPUの性質は")),
    )
    check(
        "detect-capability",
        detect_query_predicate_type("GPUは何が得意") == "capability",
        str(detect_query_predicate_type("GPUは何が得意")),
    )
    check(
        "detect-relation",
        detect_query_predicate_type("GPUとCUDAの関係は") == "relation",
        str(detect_query_predicate_type("GPUとCUDAの関係は")),
    )
    check(
        "detect-definition",
        detect_query_predicate_type("GPUとは") == "definition",
        str(detect_query_predicate_type("GPUとは")),
    )

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        props = root / "atomic.jsonl"
        typed = root / "typed.jsonl"

        add_statement(props, "GPUは高速である。")
        add_statement(props, "GPUは低消費電力である。")
        add_statement(props, "GPUは並列計算が得意である。")
        add_statement(props, "GPUはCUDAを利用可能である。")
        add_statement(
            props,
            "GPUは多数の演算を並列に実行する処理装置である。",
        )
        sync_typed_index(props, typed)

        property_hit = typed_query_lookup(typed, "GPUの性質は")
        check(
            "property-filter",
            property_hit == (
                "GPU",
                "property",
                "GPUは、高速であり、低消費電力である。",
            ),
            str(property_hit),
        )

        capability_hit = typed_query_lookup(typed, "GPUは何が得意")
        check(
            "capability-filter",
            capability_hit == (
                "GPU",
                "capability",
                "GPUは、並列計算が得意である。",
            ),
            str(capability_hit),
        )

        relation_hit = typed_query_lookup(typed, "GPUとCUDAの関係は")
        check(
            "relation-filter",
            relation_hit == (
                "GPU",
                "relation",
                "GPUは、CUDAを利用可能である。",
            ),
            str(relation_hit),
        )

        definition_hit = typed_query_lookup(typed, "GPUとは")
        check(
            "definition-filter",
            definition_hit == (
                "GPU",
                "definition",
                "GPUは、多数の演算を並列に実行する処理装置である。",
            ),
            str(definition_hit),
        )

        miss = typed_query_lookup(typed, "GPUについて教えて")
        check("untyped-question-miss", miss is None, str(miss))

    print()
    print("Predicate intent detection : PASS")
    print("Typed filtering            : PASS")
    print("Typed answer composition   : PASS")
    print("Fallback isolation         : PASS")
    print("STATUS                     : TYPED_QUERY_ROUTING_PASS")


if __name__ == "__main__":
    main()
