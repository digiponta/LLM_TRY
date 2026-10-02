# run_full_regression_v1051.py
#
# LLM_TRY v10.5.1 Stable Candidate full regression runner.
#
# Runs the major non-destructive regressions accumulated through v10.5 and
# reports one consolidated PASS/FAIL summary.

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass


@dataclass
class TestCase:
    name: str
    command: list[str]


TESTS = [
    TestCase(
        "known-false-rejection-v96",
        [sys.executable, "eval_known_false_rejection_v96.py"],
    ),
    TestCase(
        "integrated-known-unknown-v97",
        [sys.executable, "eval_integrated_gate_v97.py"],
    ),
    TestCase(
        "multiturn-history-v101",
        [sys.executable, "eval_multiturn_history_v101.py"],
    ),
    TestCase(
        "unknown-teaching-loop-v102",
        [sys.executable, "eval_unknown_teaching_loop_v102.py"],
    ),
    TestCase(
        "trained-concept-promotion-v1021",
        [sys.executable, "eval_trained_concept_promotion_v1021.py"],
    ),
    TestCase(
        "concept-query-promotion-v1022",
        [sys.executable, "eval_trained_concept_promotion_v1022.py"],
    ),
    TestCase(
        "single-pair-training-gate-v1023",
        [sys.executable, "eval_single_pair_training_gate_v1023.py"],
    ),
    TestCase(
        "persistent-checkpoint-v1024",
        [sys.executable, "eval_persistent_checkpoint_v1024.py"],
    ),
    TestCase(
        "multiconcept-incremental-v104",
        [sys.executable, "eval_multiconcept_incremental_v104.py"],
    ),
    TestCase(
        "bare-concept-gate-v105",
        [sys.executable, "eval_bare_concept_gate_v105.py"],
    ),
]


def main() -> None:
    print("=" * 96)
    print(" LLM_TRY v10.5.1 Full Regression / Stable Candidate")
    print("=" * 96)

    passed = 0
    results: list[tuple[str, int]] = []

    for index, test in enumerate(TESTS, 1):
        print()
        print("-" * 96)
        print(f"[{index}/{len(TESTS)}] {test.name}")
        print("-" * 96)

        proc = subprocess.run(test.command, check=False)
        results.append((test.name, proc.returncode))

        if proc.returncode == 0:
            passed += 1
            print(f"[SUITE PASS] {test.name}")
        else:
            print(f"[SUITE FAIL] {test.name} exit={proc.returncode}")

    print()
    print("=" * 96)
    print(" Full Regression Summary")
    print("=" * 96)
    for name, code in results:
        print(f"[{'PASS' if code == 0 else 'FAIL'}] {name}")

    print()
    print(f"Passed            : {passed}/{len(TESTS)}")
    print(f"Failed            : {len(TESTS) - passed}/{len(TESTS)}")
    print(
        "Stable candidate : "
        + ("PASS" if passed == len(TESTS) else "FAIL")
    )

    raise SystemExit(0 if passed == len(TESTS) else 1)


if __name__ == "__main__":
    main()
