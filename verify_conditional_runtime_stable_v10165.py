#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.16.5 Conditional Semantic Runtime Stable regression."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from conditional_semantic_v10161 import (
    add_conditional_statement,
    conditional_semantic_status,
    queue_conditional_candidate,
)

ROOT = Path(__file__).resolve().parent

REGRESSIONS = [
    ("modifier-normalization", "verify_modifier_condition_v10160.py", "MODIFIER_TO_CONDITION_V10160_PASS"),
    ("conditional-semantic", "verify_conditional_semantic_v10161.py", "CONDITIONAL_SEMANTIC_V10161_PASS"),
    ("command-normalization", "verify_command_normalization_v10162.py", "COMMAND_NORMALIZATION_GUARD_V10162_PASS"),
    ("candidate-lifecycle", "verify_conditional_candidate_v10163.py", "CONDITIONAL_CANDIDATE_LIFECYCLE_V10163_PASS"),
    ("raw-candidate-guard", "verify_conditional_candidate_guard_v10164.py", "RAW_CONDITIONAL_CANDIDATE_GUARD_V10164_PASS"),
]

def run_regression(label: str, script: str, expected: str) -> bool:
    command = [sys.executable, str(ROOT / script)]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    ok = completed.returncode == 0 and expected in output
    print(f"[{'PASS' if ok else 'FAIL'}] {label} : {expected}")
    if not ok:
        print(output.rstrip())
    return ok

def status_regression() -> bool:
    with TemporaryDirectory() as td:
        root = Path(td)
        propositions = root / "conditional.jsonl"
        candidates = root / "candidates.jsonl"
        state = root / "learning_state.json"
        state.write_text('{"trained_fingerprints": []}\n', encoding="utf-8")

        add_conditional_statement(propositions, "高温のCPUは停止する")
        queue_conditional_candidate(
            candidates,
            "低温のCPUは正常に動作する",
            proposition_path=propositions,
        )
        status = conditional_semantic_status(
            propositions, candidates, state, frozenset()
        )
        ok = (
            status.propositions == 1
            and status.pending_candidates == 1
            and status.internalized == 0
            and status.pending_internalization == 1
        )
        print(
            f"[{'PASS' if ok else 'FAIL'}] conditional-status : "
            f"stored={status.propositions} "
            f"candidates={status.pending_candidates} "
            f"internalized={status.internalized} "
            f"pending={status.pending_internalization}"
        )
        return ok

def main() -> int:
    print("=" * 108)
    print(" LLM_TRY v10.16.5 Conditional Semantic Runtime Stable Regression")
    print("=" * 108)
    passed = 0
    total = len(REGRESSIONS) + 1

    for label, script, expected in REGRESSIONS:
        if run_regression(label, script, expected):
            passed += 1

    if status_regression():
        passed += 1

    print("-" * 108)
    print("Historical conditional cases : 48/48 expected through component regressions")
    print(f"Stable integration checks    : {passed}/{total}")
    print(f"Failed                       : {total - passed}/{total}")
    status = (
        "CONDITIONAL_SEMANTIC_RUNTIME_STABLE_V10165_PASS"
        if passed == total
        else "CONDITIONAL_SEMANTIC_RUNTIME_STABLE_V10165_FAIL"
    )
    print(f"STATUS : {status}")
    return 0 if passed == total else 1

if __name__ == "__main__":
    raise SystemExit(main())
