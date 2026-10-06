#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.13.0 Semantic Knowledge Runtime Stable verification."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

STEPS = [
    ("Stable Architecture Regression", [sys.executable, "run_semantic_runtime_stable_v10130.py"]),
    ("Retrieval-First Full Verification", [sys.executable, "verify_retrieval_first_runtime_v1012161.py"]),
    ("Stable Chat Runtime Contract", [sys.executable, "run_stable_chat_runtime_v10130.py"]),
    (
        "Runtime Syntax Check",
        [
            sys.executable,
            "-m",
            "py_compile",
            "chat.py",
            "semantic_knowledge_architecture_v10110.py",
            "subject_keyed_corpus_memory_v101216.py",
            "retrieval_first_runtime_v1012161.py",
        ],
    ),
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
    print(" LLM_TRY v10.13.0 Semantic Knowledge Runtime Stable Release - Verification")
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
    print("Semantic architecture : PASS")
    print("Retrieval-first       : PASS")
    print("Subject corpus memory : PASS")
    print("Unknown fallback      : PASS")
    print("Stable chat runtime   : PASS")
    print("Truth/runtime syntax  : PASS")
    print("Model retraining      : NONE")
    print("Production mutation   : NONE")
    print("STATUS                : SEMANTIC_KNOWLEDGE_RUNTIME_STABLE_FULL_PASS")


if __name__ == "__main__":
    main()
