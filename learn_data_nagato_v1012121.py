#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.12.12.1 One-command data-nagato learning pipeline.

Pipeline:
  production
    -> timestamped raw continued-pretraining candidate
    -> corpus-to-semantic dataset
    -> semantic training on top of raw candidate
    -> end-to-end semantic evaluation against production

Production is never overwritten by this script.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PRODUCTION = "model/model-gpu-v1.6.2-online.pt"
RAW_CANDIDATE = "model/model-gpu-v1.6.2-online-nagato-candidate.pt"
FINAL_CANDIDATE = "model/model-gpu-v1.6.2-online-nagato-semantic-candidate.pt"


STEPS = [
    (
        "Raw data-nagato continued pretraining",
        [
            sys.executable,
            "train_nagato_moderate_v10124.py",
            "--base-model", PRODUCTION,
            "--output", RAW_CANDIDATE,
        ],
    ),
    (
        "Raw knowledge-gain evaluation",
        [
            sys.executable,
            "evaluate_nagato_knowledge_gain_v10126.py",
            "--after", RAW_CANDIDATE,
        ],
        False,
    ),
    (
        "Corpus-to-Semantic dataset build",
        [
            sys.executable,
            "build_corpus_semantic_v101212.py",
        ],
    ),
    (
        "Corpus-to-Semantic training",
        [
            sys.executable,
            "train_corpus_semantic_v101212.py",
            "--base-model", RAW_CANDIDATE,
            "--output", FINAL_CANDIDATE,
        ],
    ),
    (
        "Final semantic evaluation",
        [
            sys.executable,
            "evaluate_corpus_semantic_v101212.py",
            "--before", PRODUCTION,
            "--after", FINAL_CANDIDATE,
            "--report", "results/data_nagato_learning_v1012121.json",
        ],
    ),
]


def run_step(name: str, command: list[str], required: bool = True) -> int:
    print()
    print("=" * 116)
    print(" " + name)
    print("=" * 116)
    print("Command:", " ".join(command))
    print()
    result = subprocess.run(command)
    if result.returncode != 0:
        label = "FAIL" if required else "WARN"
        print(f"[{label}] {name} exit_code={result.returncode}")
        if required:
            raise SystemExit(result.returncode)
    else:
        print(f"[PASS] {name}")
    return result.returncode


def main() -> None:
    print("=" * 116)
    print(" LLM_TRY v10.12.12.1 data-nagato Full Learning Pipeline")
    print("=" * 116)

    required_files = [
        "data/data-nagato.txt",
        PRODUCTION,
        "model/tokenizer-v0.7-bpe.json",
    ]
    missing = [x for x in required_files if not Path(x).exists()]
    if missing:
        print("[FAIL] Missing required files:")
        for item in missing:
            print("  -", item)
        raise SystemExit(2)

    raw_eval_status = 0
    for step in STEPS:
        if len(step) == 2:
            name, command = step
            required = True
        else:
            name, command, required = step
        status = run_step(name, command, required=required)
        if name == "Raw knowledge-gain evaluation":
            raw_eval_status = status

    print()
    print("=" * 116)
    print(" FINAL RESULT")
    print("=" * 116)
    print("Raw pretraining       : PASS")
    print(
        "Raw knowledge gain    : "
        + ("PASS" if raw_eval_status == 0 else "QA threshold WARN")
    )
    print("Semantic dataset      : PASS")
    print("Semantic training     : PASS")
    print("Final semantic eval   : PASS")
    print("Production overwritten: NO")
    print("Final candidate       :", FINAL_CANDIDATE)
    print("STATUS                : DATA_NAGATO_FULL_LEARNING_PASS")
    print()
    print("The final candidate has NOT been promoted automatically.")


if __name__ == "__main__":
    main()
