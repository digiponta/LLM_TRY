#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.12.16 Subject-Keyed Corpus Memory regression."""

from __future__ import annotations

import tempfile
from pathlib import Path

from subject_keyed_corpus_memory_v101216 import (
    CorpusMemoryRecord,
    compose_subject_evidence,
    lookup_subject,
    save_memory,
    subject_mapping,
)


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" : {detail}" if detail else ""))
    if not ok:
        raise AssertionError(name)


def main():
    print("=" * 116)
    print(" LLM_TRY v10.12.16 Subject-Keyed Corpus Memory Regression")
    print("=" * 116)

    rows = [
        CorpusMemoryRecord(
            subject="GPU",
            statement="GPUは高速である。",
            relation="property",
            object_description="高速である",
            source_text="GPUは高速である。",
            source="test",
        ),
        CorpusMemoryRecord(
            subject="CPU",
            statement="CPUは命令を実行する。",
            relation="function",
            object_description="命令を実行する",
            source_text="CPUは命令を実行する。",
            source="test",
        ),
    ]

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "memory.jsonl"
        save_memory(path, rows)

        gpu = lookup_subject(path, "GPU")
        check("exact-subject-lookup", len(gpu) == 1)
        check("full-proposition-preserved", gpu[0].statement == "GPUは高速である。")
        check(
            "canonical-mapping",
            subject_mapping(path, "GPU") == ["GPU => GPUは高速である。"],
            str(subject_mapping(path, "GPU")),
        )
        check(
            "composed-evidence",
            compose_subject_evidence(path, "CPU") == "CPUは命令を実行する。",
        )
        check("unknown-subject-empty", lookup_subject(path, "UNKNOWN") == [])

    print()
    print("Exact lookup        : PASS")
    print("Full proposition    : PASS")
    print("Canonical mapping   : PASS")
    print("STATUS              : SUBJECT_KEYED_CORPUS_MEMORY_PASS")


if __name__ == "__main__":
    main()
