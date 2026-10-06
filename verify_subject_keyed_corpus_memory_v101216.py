#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.12.16 full verification."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

STEPS = [
    ("Corpus Memory Regression", [sys.executable, "run_subject_keyed_corpus_memory_v101216.py"]),
    ("Corpus Memory Build", [sys.executable, "build_subject_keyed_corpus_memory_v101216.py"]),
    ("Corpus Memory Evaluation", [sys.executable, "evaluate_subject_keyed_corpus_memory_v101216.py"]),
    ("Semantic Dataset Build", [sys.executable, "build_corpus_semantic_v101212.py"]),
    ("Repeated Semantic Training", [sys.executable, "train_corpus_semantic_v101212.py"]),
    ("Semantic Evaluation", [sys.executable, "evaluate_corpus_semantic_v101212.py"]),
    ("Memory-backed Function Evaluation", [sys.executable, "evaluate_two_pass_function_v101214.py"]),
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
    print(" LLM_TRY v10.12.16 Subject-Keyed Corpus Memory - Full Verification")
    print("=" * 116)

    required = [
        "data/data-nagato.txt",
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
    print("Corpus memory       : PASS")
    print("Semantic training   : PASS")
    print("Semantic evaluation : PASS")
    print("Function resolver   : PASS")
    print("STATUS              : SUBJECT_KEYED_CORPUS_MEMORY_FULL_PASS")


if __name__ == "__main__":
    main()
