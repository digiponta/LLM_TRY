#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.12.12.2 raw+semantic promotion verification."""

from __future__ import annotations

import subprocess
import sys


def main() -> None:
    print("=" * 116)
    print(" LLM_TRY v10.12.12.2 Raw+Semantic Candidate Promotion Gate - Verification")
    print("=" * 116)

    steps = [
        (
            "Regression",
            [sys.executable, "run_raw_semantic_promotion_gate_v1012122.py"],
        ),
        (
            "Candidate gate",
            [sys.executable, "promote_raw_semantic_candidate_v1012122.py"],
        ),
    ]

    for name, command in steps:
        print()
        print("=" * 116)
        print(" " + name)
        print("=" * 116)
        print("Command:", " ".join(command))
        result = subprocess.run(command)
        if result.returncode != 0:
            print(f"[FAIL] {name} exit_code={result.returncode}")
            raise SystemExit(result.returncode)
        print(f"[PASS] {name}")

    print()
    print("=" * 116)
    print(" FINAL RESULT")
    print("=" * 116)
    print("Regression      : PASS")
    print("Candidate gate  : PASS")
    print("Promotion       : NOT REQUESTED")
    print("STATUS          : RAW_SEMANTIC_PROMOTION_GATE_FULL_PASS")


if __name__ == "__main__":
    main()
