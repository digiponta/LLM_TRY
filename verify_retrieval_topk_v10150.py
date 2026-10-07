#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.15 Retrieval Top-K regression."""

from pathlib import Path
import tempfile

from retrieval_first_runtime_v1012161 import resolve_subject
from subject_keyed_corpus_memory_v101216 import CorpusMemoryRecord, save_memory


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" : {detail}" if detail else ""))
    if not ok:
        raise AssertionError(name)


def main():
    print("=" * 100)
    print(" LLM_TRY v10.15 Retrieval Top-K Regression")
    print("=" * 100)

    with tempfile.TemporaryDirectory() as td:
        memory = Path(td) / "memory.jsonl"
        rows = [
            CorpusMemoryRecord("宇宙", "宇宙は全体として膨張している。", "property", "全体として膨張している", "", "test"),
            CorpusMemoryRecord("宇宙", "全体としての宇宙は始まりと終わりを持たない。", "property", "始まりと終わりを持たない", "", "test"),
            CorpusMemoryRecord("宇宙", "宇宙は観測者によって異なる見え方をする。", "property", "観測者によって異なる見え方をする", "", "test"),
            CorpusMemoryRecord("宇宙", "時空間上の宇宙は均一と不均一が混在する。", "property", "均一と不均一が混在する", "", "test"),
            CorpusMemoryRecord("宇宙", "宇宙は膨張して平坦になる。", "property", "膨張して平坦になる", "", "test"),
            CorpusMemoryRecord("宇宙", "宇宙は一つの量子状態として見ることもできる。", "property", "量子状態", "", "test"),
        ]
        save_memory(memory, rows)

        result = resolve_subject(memory, "宇宙", query="宇宙とは", top_k=3)
        check("hit", result.hit)
        check("top-k-count", len(result.records) == 3, str(len(result.records)))
        check("answer-count", result.answer.count(" / ") == 2, result.answer)
        check("provenance-top-k", "top-k=3/6" in result.provenance, result.provenance)
        check("direct-subject-preferred", result.records[0].statement.startswith("宇宙は"), result.records[0].statement)

        one = resolve_subject(memory, "宇宙", query="宇宙とは", top_k=1)
        check("top-1-count", len(one.records) == 1)

    print()
    print("STATUS : RETRIEVAL_TOP_K_PASS")


if __name__ == "__main__":
    main()
