#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.13.0 Semantic Knowledge Runtime stable regression."""

from __future__ import annotations

import tempfile
from pathlib import Path

from retrieval_first_runtime_v1012161 import resolve_subject
from subject_keyed_corpus_memory_v101216 import CorpusMemoryRecord, save_memory
from semantic_knowledge_architecture_v10110 import (
    SemanticKnowledgeArchitecture,
    SemanticKnowledgeConfig,
)


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" : {detail}" if detail else ""))
    if not ok:
        raise AssertionError(name)


def main():
    print("=" * 116)
    print(" LLM_TRY v10.13.0 Semantic Knowledge Runtime Stable Regression")
    print("=" * 116)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        memory = root / "memory.jsonl"
        save_memory(memory, [
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
        ])

        gpu = resolve_subject(memory, "GPU")
        check("retrieval-first-hit", gpu.hit)
        check("retrieval-full-proposition", gpu.answer == "GPUは高速である。")
        check("retrieval-provenance", "subject-keyed-corpus-memory" in gpu.provenance)

        cpu = resolve_subject(memory, "CPU")
        check("function-structure", cpu.function_structure is not None)
        check("function-action", cpu.function_structure.action == "execute")
        check("function-target", cpu.function_structure.target == "命令")

        miss = resolve_subject(memory, "UNKNOWN")
        check("unknown-fallback", not miss.hit and miss.route == "FALLBACK")

        config = SemanticKnowledgeConfig(
            proposition_path=root / "p.jsonl",
            subject_index_path=root / "s.jsonl",
            typed_index_path=root / "t.jsonl",
            unified_path=root / "u.jsonl",
            learning_log=root / "l.jsonl",
            learning_state=root / "ls.json",
            raw_corpus_path=root / "raw.txt",
            truth_store_path=root / "truth.jsonl",
            canonical_definitions={},
            checkpoint_fingerprints=frozenset(),
        )
        status = SemanticKnowledgeArchitecture(config).status()
        check("stable-version", status["version"] == "v10.13.0", str(status["version"]))
        layers = set(status["layers"])
        check("truth-layer", "truth-state" in layers)
        check("corpus-memory-layer", "subject-keyed-corpus-memory" in layers)
        check("retrieval-first-layer", "retrieval-first-runtime" in layers)
        check("resolver-layer", "resolver" in layers)
        check("dispatcher-layer", "dispatcher" in layers)

    print()
    print("Retrieval-first runtime : PASS")
    print("Function structure      : PASS")
    print("Unknown fallback        : PASS")
    print("Semantic architecture   : PASS")
    print("Stable version          : v10.13.0")
    print("STATUS                  : SEMANTIC_KNOWLEDGE_RUNTIME_STABLE_PASS")


if __name__ == "__main__":
    main()
