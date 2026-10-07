#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.16.4 Raw-Input Conditional Candidate Guard regression."""

from pathlib import Path
from tempfile import TemporaryDirectory

from conditional_semantic_v10161 import (
    add_conditional_statement,
    pending_conditional_candidates,
    queue_conditional_candidate,
)

def check(label: str, ok: bool, detail: str = "") -> int:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" : {detail}" if detail else ""))
    return 1 if ok else 0

def main() -> int:
    print("=" * 104)
    print(" LLM_TRY v10.16.4 Raw-Input Conditional Candidate Guard Regression")
    print("=" * 104)
    passed = 0
    total = 0

    with TemporaryDirectory() as td:
        root = Path(td)
        candidates = root / "candidates.jsonl"
        props = root / "conditional.jsonl"

        total += 1
        queued, row = queue_conditional_candidate(
            candidates, "高温のCPUは停止する", proposition_path=props
        )
        passed += check("declarative-candidate", queued and row is not None)

        for query in (
            "CPUが高温のときは？",
            "高温の場合CPUはどうなる?",
            "低温時のCPUは？",
            "CPUは高温の場合どうなる？",
        ):
            total += 1
            q, qrow = queue_conditional_candidate(
                candidates, query, proposition_path=props
            )
            passed += check(f"question-not-candidate:{query}", not q and qrow is None)

        add_conditional_statement(props, "CPUは、高温の場合、停止する。")
        total += 1
        dup, duprow = queue_conditional_candidate(
            candidates, "CPUは、高温の場合、停止する。", proposition_path=props
        )
        passed += check("stored-fact-not-requeued", not dup and duprow is not None)

        total += 1
        pending = pending_conditional_candidates(candidates)
        passed += check("pending-count-stable", len(pending) == 1, f"pending={len(pending)}")

    print("-" * 104)
    print(f"Passed : {passed}/{total}")
    print(f"Failed : {total - passed}/{total}")
    status = "RAW_CONDITIONAL_CANDIDATE_GUARD_V10164_PASS" if passed == total else "RAW_CONDITIONAL_CANDIDATE_GUARD_V10164_FAIL"
    print(f"STATUS : {status}")
    return 0 if passed == total else 1

if __name__ == "__main__":
    raise SystemExit(main())
