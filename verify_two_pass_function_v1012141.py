#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.12.14.1 full Multi-Probe Two-Pass Function verification."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

STEPS = [
    ("Resolver Regression", [sys.executable, "run_two_pass_function_v101214.py"]),
    ("Resolver Evaluation", [sys.executable, "evaluate_two_pass_function_v101214.py"]),
]


def run(name, command):
    print()
    print("=" * 116)
    print(" " + name)
    print("=" * 116)
    print("Command:", " ".join(command))
    print()
    result = subprocess.run(command)
    if result.returncode:
        print(f"[FAIL] {name} exit_code={result.returncode}")
        raise SystemExit(result.returncode)
    print(f"[PASS] {name}")


def main():
    print("=" * 116)
    print(" LLM_TRY v10.12.14.1 Multi-Probe Two-Pass Function Resolver - Full Verification")
    print("=" * 116)

    required = [
        "data/nagato_corpus_semantic_holdout_seen_v101212.jsonl",
        "data/nagato_corpus_semantic_holdout_unseen_v101212.jsonl",
        "model/model-gpu-v1.6.2-online.pt",
        "model/tokenizer-v0.7-bpe.json",
    ]
    missing = [x for x in required if not Path(x).exists()]
    if missing:
        print("[FAIL] missing:", missing)
        raise SystemExit(2)

    for name, command in STEPS:
        run(name, command)

    print()
    print("=" * 116)
    print(" FINAL RESULT")
    print("=" * 116)
    print("Resolver regression : PASS")
    print("Resolver evaluation : PASS")
    print("Model retraining    : NONE")
    print("Production mutation : NONE")
    print("STATUS              : MULTI_PROBE_TWO_PASS_FUNCTION_FULL_PASS")


if __name__ == "__main__":
    main()
