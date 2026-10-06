#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.12.9 end-to-end verification runner.

Runs:
1. semantic QA regression
2. semantic QA candidate training
3. unseen holdout semantic QA evaluation

Stops immediately on failure.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


STEPS = [
    (
        "Regression",
        [sys.executable, "run_semantic_qa_generalization_v10129.py"],
    ),
    (
        "Semantic QA Training",
        [sys.executable, "train_semantic_qa_generalization_v10129.py"],
    ),
    (
        "Holdout Evaluation",
        [sys.executable, "evaluate_semantic_qa_generalization_v10129.py"],
    ),
]


def run_step(name: str, command: list[str]) -> None:
    print()
    print("=" * 112)
    print(f" {name}")
    print("=" * 112)
    print("Command:", " ".join(command))
    print()

    completed = subprocess.run(command)

    if completed.returncode != 0:
        print()
        print(f"[FAIL] {name} exit_code={completed.returncode}")
        raise SystemExit(completed.returncode)

    print()
    print(f"[PASS] {name}")


def main() -> None:
    print("=" * 112)
    print(" LLM_TRY v10.12.9 Semantic QA Generalization - Full Verification")
    print("=" * 112)

    required = [
        "run_semantic_qa_generalization_v10129.py",
        "train_semantic_qa_generalization_v10129.py",
        "evaluate_semantic_qa_generalization_v10129.py",
        "data/nagato_qa_train_v10128.jsonl",
        "data/nagato_qa_holdout_v10128.jsonl",
        "model/model-gpu-v1.6.2-online.pt",
        "model/tokenizer-v0.7-bpe.json",
    ]

    missing = [path for path in required if not Path(path).exists()]
    if missing:
        print("[FAIL] Missing required files:")
        for path in missing:
            print("  -", path)
        raise SystemExit(2)

    for name, command in STEPS:
        run_step(name, command)

    print()
    print("=" * 112)
    print(" FINAL RESULT")
    print("=" * 112)
    print("Regression            : PASS")
    print("Semantic QA training  : PASS")
    print("Holdout evaluation    : PASS")
    print("STATUS                : SEMANTIC_QA_FULL_VERIFICATION_PASS")


if __name__ == "__main__":
    main()
