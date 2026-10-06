#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.9.0 Subject-Keyed Proposition Regression."""

from __future__ import annotations

import tempfile
from pathlib import Path

from semantic_proposition_v1090 import add_statement
from subject_keyed_proposition_v1090 import (
    load_subject_index,
    subject_index_dict,
    subject_mapping,
    sync_subject_index,
)


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (
        f" : {detail}" if detail else ""
    ))
    if not ok:
        raise AssertionError(name)


def main() -> None:
    print("=" * 92)
    print(" LLM_TRY v10.9.0 Subject-Keyed Semantic Proposition Regression")
    print("=" * 92)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        props = root / "props.jsonl"
        index = root / "subject_index.jsonl"

        add_statement(props, "GPUは高速である。")
        add_statement(props, "GPUは並列計算が得意である。")
        add_statement(props, "CPUは汎用処理が得意である。")

        count = sync_subject_index(props, index)
        check("index-count", count == 3, str(count))

        rows = load_subject_index(index)
        check("load-count", len(rows) == 3, str(rows))

        gpu = subject_mapping(index, "GPU")
        check(
            "gpu-keyed-mapping",
            gpu == [
                "GPU => GPUは高速である。",
                "GPU => GPUは並列計算が得意である。",
            ],
            str(gpu),
        )

        cpu = subject_mapping(index, "CPU")
        check(
            "cpu-keyed-mapping",
            cpu == [
                "CPU => CPUは汎用処理が得意である。",
            ],
            str(cpu),
        )

        grouped = subject_index_dict(index)
        check(
            "grouped-subject-count",
            len(grouped) == 2,
            str(grouped),
        )

        # Compound input must still become one indexed row per atomic fact.
        add_statement(
            props,
            "架空装置は、高速であり、低消費電力である。",
        )
        count = sync_subject_index(props, index)
        device = subject_mapping(index, "架空装置")
        check("compound-index-count", count == 5, str(count))
        check(
            "compound-two-keyed-statements",
            device == [
                "架空装置 => 架空装置は高速である。",
                "架空装置 => 架空装置は低消費電力である。",
            ],
            str(device),
        )

    print()
    print("Atomic -> keyed statement : PASS")
    print("Subject grouping           : PASS")
    print("Compound decomposition     : PASS")
    print("Derived index rebuild      : PASS")
    print("STATUS                     : SUBJECT_KEYED_PROPOSITION_PASS")


if __name__ == "__main__":
    main()
