#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.16.3 Conditional Candidate Lifecycle regression."""

from pathlib import Path
from tempfile import TemporaryDirectory

from conditional_semantic_v10161 import (
    approve_conditional_candidate,
    answer_conditional_query,
    pending_conditional_candidates,
    prepare_conditional_sleep_pairs,
    queue_conditional_candidate,
)

def check(label: str, ok: bool, detail: str = "") -> int:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" : {detail}" if detail else ""))
    return 1 if ok else 0

def main() -> int:
    print("=" * 104)
    print(" LLM_TRY v10.16.3 Conditional Candidate Lifecycle Regression")
    print("=" * 104)
    passed = 0
    total = 0

    with TemporaryDirectory() as td:
        root = Path(td)
        candidates = root / "candidates.jsonl"
        props = root / "conditional.jsonl"
        learning_log = root / "learning.jsonl"
        learning_state = root / "state.json"
        learning_state.write_text('{"trained_fingerprints": []}\n', encoding="utf-8")

        total += 1
        queued, row = queue_conditional_candidate(candidates, "高温のCPUは停止する")
        passed += check("auto-candidate-high-temp", queued and row is not None, row.render() if row else "")

        total += 1
        queued2, _ = queue_conditional_candidate(candidates, "高温のCPUは停止する")
        passed += check("candidate-deduplicate", not queued2)

        total += 1
        queued3, row3 = queue_conditional_candidate(candidates, "低温のCPUは正常に動作する")
        passed += check("auto-candidate-low-temp", queued3 and row3 is not None, row3.render() if row3 else "")

        total += 1
        pending = pending_conditional_candidates(candidates)
        passed += check("pending-count-2", len(pending) == 2, f"pending={len(pending)}")

        total += 1
        approved = approve_conditional_candidate(candidates, props, index=1)
        passed += check("approve-one", approved == 1, f"approved={approved}")

        total += 1
        _, answer = answer_conditional_query(props, "CPUが高温のときは？")
        passed += check("retrieval-after-approval", answer == "CPUは、高温の場合、停止する。", answer)

        total += 1
        approved_rest = approve_conditional_candidate(candidates, props, index=None)
        passed += check("approve-rest", approved_rest == 1, f"approved={approved_rest}")

        total += 1
        queued_sleep = prepare_conditional_sleep_pairs(props, learning_log, learning_state)
        passed += check("sleep-pairs-queued", queued_sleep == 2, f"queued={queued_sleep}")

        total += 1
        queued_sleep_again = prepare_conditional_sleep_pairs(props, learning_log, learning_state)
        passed += check("sleep-pairs-deduplicated", queued_sleep_again == 0, f"queued={queued_sleep_again}")

    print("-" * 104)
    print(f"Passed : {passed}/{total}")
    print(f"Failed : {total - passed}/{total}")
    status = "CONDITIONAL_CANDIDATE_LIFECYCLE_V10163_PASS" if passed == total else "CONDITIONAL_CANDIDATE_LIFECYCLE_V10163_FAIL"
    print(f"STATUS : {status}")
    return 0 if passed == total else 1

if __name__ == "__main__":
    raise SystemExit(main())
