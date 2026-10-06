#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.10.0 Typed Subject Proposition Regression."""

from __future__ import annotations

import tempfile
from pathlib import Path

from semantic_proposition_v1090 import add_statement
from typed_subject_proposition_v10100 import (
    classify_predicate_type,
    load_typed_index,
    sync_typed_index,
    typed_index_dict,
    typed_subject_mapping,
)


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (
        f" : {detail}" if detail else ""
    ))
    if not ok:
        raise AssertionError(name)


def main() -> None:
    print("=" * 96)
    print(" LLM_TRY v10.10.0 Subject => Predicate Type => Statement Regression")
    print("=" * 96)

    check(
        "classify-property",
        classify_predicate_type("高速") == "property",
        classify_predicate_type("高速"),
    )
    check(
        "classify-capability",
        classify_predicate_type("並列計算が得意") == "capability",
        classify_predicate_type("並列計算が得意"),
    )
    check(
        "classify-relation",
        classify_predicate_type("CUDAを利用可能") == "relation",
        classify_predicate_type("CUDAを利用可能"),
    )
    check(
        "classify-definition",
        classify_predicate_type("多数の演算を並列に実行する処理装置")
        == "definition",
        classify_predicate_type(
            "多数の演算を並列に実行する処理装置"
        ),
    )

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        props = root / "atomic.jsonl"
        typed = root / "typed.jsonl"

        add_statement(props, "GPUは高速である。")
        add_statement(props, "GPUは並列計算が得意である。")
        add_statement(props, "GPUはCUDAを利用可能である。")
        add_statement(
            props,
            "GPUは多数の演算を並列に実行する処理装置である。",
        )

        count = sync_typed_index(props, typed)
        check("typed-index-count", count == 4, str(count))

        rows = load_typed_index(typed)
        check("typed-load-count", len(rows) == 4, str(rows))

        mapping = typed_subject_mapping(typed, "GPU")
        check(
            "typed-subject-mapping",
            mapping == [
                "GPU => property => GPUは高速である。",
                "GPU => capability => GPUは並列計算が得意である。",
                "GPU => relation => GPUはCUDAを利用可能である。",
                (
                    "GPU => definition => "
                    "GPUは多数の演算を並列に実行する処理装置である。"
                ),
            ],
            str(mapping),
        )

        grouped = typed_index_dict(typed)
        check(
            "all-four-types-present",
            set(grouped["GPU"].keys())
            == {"property", "capability", "relation", "definition"},
            str(grouped),
        )

    print()
    print("Predicate classification : PASS")
    print("Typed subject index      : PASS")
    print("Four predicate types     : PASS")
    print("Subject => Type => Text  : PASS")
    print("STATUS                   : TYPED_SUBJECT_PROPOSITION_PASS")


if __name__ == "__main__":
    main()
