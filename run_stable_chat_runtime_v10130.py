#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.13.0 Stable Chat Runtime Contract Regression.

This regression validates the CURRENT runtime contract. It intentionally does
not reuse historical v10.5 expectations that predate Retrieval-First Runtime.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from chat import input_quality_check
from retrieval_first_runtime_v1012161 import (
    resolve_subject,
    truth_allows_direct_retrieval,
)
from subject_keyed_corpus_memory_v101216 import (
    CorpusMemoryRecord,
    save_memory,
)
from truth_state_v10103 import (
    effective_truth_record,
    upsert_truth_record,
)


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" : {detail}" if detail else ""))
    if not ok:
        raise AssertionError(name)


def main() -> None:
    print("=" * 116)
    print(" LLM_TRY v10.13.0 Stable Chat Runtime Contract Regression")
    print("=" * 116)

    # Input contract: well-formed concept queries are accepted; malformed input
    # remains rejected before routing.
    ok, reason = input_quality_check("CPUとは")
    check("input-valid-concept-query", ok, reason)

    ok, reason = input_quality_check("とは")
    check("input-subjectless-rejected", not ok, reason)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        memory = root / "memory.jsonl"
        truth = root / "truth.jsonl"

        save_memory(memory, [
            CorpusMemoryRecord(
                subject="宇宙",
                statement="宇宙は全体として膨張している。",
                relation="property",
                object_description="全体として膨張している",
                source_text="宇宙は全体として膨張している。",
                source="data-nagato.txt",
            ),
            CorpusMemoryRecord(
                subject="CPU",
                statement="CPUは命令を実行する。",
                relation="function",
                object_description="命令を実行する",
                source_text="CPUは命令を実行する。",
                source="data-nagato.txt",
            ),
        ])

        # Current contract: source-grounded corpus knowledge is retrievable.
        universe = resolve_subject(memory, "宇宙")
        check("corpus-known-hit", universe.hit)
        check(
            "corpus-known-full-proposition",
            universe.answer == "宇宙は全体として膨張している。",
            universe.answer,
        )

        # Unknown knowledge must still miss and fall back to the semantic/model
        # route rather than being hallucinated from corpus frequency.
        unknown = resolve_subject(memory, "未教示概念")
        check("unknown-memory-miss", not unknown.hit)
        check("unknown-fallback-route", unknown.route == "FALLBACK", unknown.route)

        # Function decomposition is part of the current stable runtime.
        cpu = resolve_subject(memory, "CPU")
        fs = cpu.function_structure
        check("function-structure-present", fs is not None)
        check("function-action", fs is not None and fs.action == "execute",
              fs.action if fs else "none")
        check("function-target", fs is not None and fs.target == "命令",
              fs.target if fs else "none")

        # Truth-State contract: default UNVERIFIED may be retrieved directly,
        # while FALSE / OUTDATED / CONTESTED must bypass direct corpus return.
        default_truth = effective_truth_record(truth, "宇宙")
        check("truth-default-unverified", default_truth.state == "UNVERIFIED")
        check(
            "truth-unverified-allows-retrieval",
            truth_allows_direct_retrieval(default_truth.state),
        )

        for state in ("FALSE", "OUTDATED", "CONTESTED"):
            record = upsert_truth_record(
                truth,
                "宇宙",
                state,
                reason="stable regression",
                source="test",
            )
            check(
                f"truth-{state.lower()}-bypasses-direct-retrieval",
                not truth_allows_direct_retrieval(record.state),
            )

        true_record = upsert_truth_record(
            truth,
            "宇宙",
            "TRUE",
            reason="stable regression",
            source="test",
        )
        check(
            "truth-true-allows-retrieval",
            truth_allows_direct_retrieval(true_record.state),
        )

    print()
    print("Input quality         : PASS")
    print("Corpus retrieval      : PASS")
    print("Unknown fallback      : PASS")
    print("Function structure    : PASS")
    print("Truth-State policy    : PASS")
    print("Historical v10.5 assumptions required : NO")
    print("STATUS                : STABLE_CHAT_RUNTIME_CONTRACT_PASS")


if __name__ == "__main__":
    main()
