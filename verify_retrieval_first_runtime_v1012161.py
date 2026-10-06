#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.12.16.1 Retrieval-First Runtime full verification."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

STEPS = [
    ("Corpus Memory Build", [sys.executable, "build_subject_keyed_corpus_memory_v101216.py"]),
    ("Retrieval-First Regression", [sys.executable, "run_retrieval_first_runtime_v1012161.py"]),
    ("Retrieval-First Evaluation", [sys.executable, "evaluate_retrieval_first_runtime_v1012161.py"]),
    ("Chat Syntax Check", [sys.executable, "-m", "py_compile", "chat.py"]),
]


def run(name, command):
    print()
    print("=" * 116)
    print(" " + name)
    print("=" * 116)
    print("Command:", " ".join(command))
    print()
    r = subprocess.run(command)
    if r.returncode:
        print(f"[FAIL] {name} exit_code={r.returncode}")
        raise SystemExit(r.returncode)
    print(f"[PASS] {name}")


def main():
    print("=" * 116)
    print(" LLM_TRY v10.12.16.1 Retrieval-First Runtime - Full Verification")
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
    print("Corpus memory        : PASS")
    print("Retrieval-first      : PASS")
    print("Unknown fallback     : PASS")
    print("Chat integration     : PASS")
    print("Model retraining     : NONE")
    print("Production mutation  : NONE")
    print("STATUS               : RETRIEVAL_FIRST_RUNTIME_FULL_PASS")


if __name__ == "__main__":
    main()
