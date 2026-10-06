#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.12.15 Subject-to-Proposition full verification."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

STEPS = [
    ("Subject Mapping Regression", [sys.executable, "run_subject_to_proposition_v101215.py"]),
    ("Dataset Build", [sys.executable, "build_corpus_semantic_v101212.py"]),
    ("Semantic Training", [sys.executable, "train_corpus_semantic_v101212.py"]),
    ("Semantic Evaluation", [sys.executable, "evaluate_corpus_semantic_v101212.py"]),
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
    print(" LLM_TRY v10.12.15 Subject-to-Proposition + Semantic Training - Full Verification")
    print("=" * 116)

    req = [
        "data/data-nagato.txt",
        "model/model-gpu-v1.6.2-online.pt",
        "model/tokenizer-v0.7-bpe.json",
    ]
    missing = [x for x in req if not Path(x).exists()]
    if missing:
        print("[FAIL] missing:", missing)
        raise SystemExit(2)

    for name, command in STEPS:
        run(name, command)

    print()
    print("=" * 116)
    print(" FINAL RESULT")
    print("=" * 116)
    print("Subject mapping    : PASS")
    print("Dataset build      : PASS")
    print("Semantic training  : PASS")
    print("Semantic evaluation: PASS")
    print("STATUS             : SUBJECT_TO_PROPOSITION_FULL_PASS")


if __name__ == "__main__":
    main()
