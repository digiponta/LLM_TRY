#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.12.10 end-to-end semantic role verification."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


STEPS = [
    ("Regression", [sys.executable, "run_preservation_aware_semantic_role_v1012101.py"]),
    ("Dataset Build", [sys.executable, "build_semantic_role_split_v101210.py"]),
    ("Semantic Role Training", [sys.executable, "train_semantic_role_generalization_v101210.py"]),
    ("Semantic Role Evaluation", [sys.executable, "evaluate_semantic_role_generalization_v101210.py"]),
]


def run_step(name: str, command: list[str]) -> None:
    print()
    print("=" * 116)
    print(f" {name}")
    print("=" * 116)
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
    print("=" * 116)
    print(" LLM_TRY v10.12.10.1 Preservation-Aware Semantic Role - Full Verification")
    print("=" * 116)

    required = [
        "data/data-nagato.txt",
        "model/model-gpu-v1.6.2-online.pt",
        "model/tokenizer-v0.7-bpe.json",
    ]
    missing = [x for x in required if not Path(x).exists()]
    if missing:
        print("[FAIL] Missing required files:")
        for x in missing:
            print("  -", x)
        raise SystemExit(2)

    for name, command in STEPS:
        run_step(name, command)

    print()
    print("=" * 116)
    print(" FINAL RESULT")
    print("=" * 116)
    print("Regression              : PASS")
    print("Dataset build           : PASS")
    print("Semantic role training  : PASS")
    print("Semantic role evaluation: PASS")
    print("STATUS                  : PRESERVATION_AWARE_SEMANTIC_ROLE_FULL_PASS")


if __name__ == "__main__":
    main()
