#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.12.16.1 Retrieval-First Runtime regression."""

from __future__ import annotations

import tempfile
from pathlib import Path

from retrieval_first_runtime_v1012161 import resolve_subject
from subject_keyed_corpus_memory_v101216 import CorpusMemoryRecord, save_memory


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" : {detail}" if detail else ""))
    if not ok:
        raise AssertionError(name)


def main():
    print("=" * 116)
    print(" LLM_TRY v10.12.16.1 Retrieval-First Runtime Regression")
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

        gpu = resolve_subject(path, "GPU")
        check("memory-hit", gpu.hit)
        check("retrieval-route", gpu.route == "CORPUS_MEMORY", gpu.route)
        check("full-proposition", gpu.answer == "GPUは高速である。", gpu.answer)
        check("property-relation", gpu.relation == "property")

        cpu = resolve_subject(path, "CPU")
        check("function-hit", cpu.hit)
        check("function-structure-present", cpu.function_structure is not None)
        check("function-action", cpu.function_structure.action == "execute",
              cpu.function_structure.action)
        check("function-target", cpu.function_structure.target == "命令",
              cpu.function_structure.target)

        miss = resolve_subject(path, "UNKNOWN")
        check("memory-miss", not miss.hit)
        check("fallback-route", miss.route == "FALLBACK", miss.route)

    print()
    print("Memory HIT          : PASS")
    print("Full Proposition    : PASS")
    print("Function structure  : PASS")
    print("MISS fallback       : PASS")
    print("Model retraining    : NONE")
    print("STATUS              : RETRIEVAL_FIRST_RUNTIME_PASS")


if __name__ == "__main__":
    main()
