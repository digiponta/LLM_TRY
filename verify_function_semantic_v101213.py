#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.12.13 full function semantic verification."""

from __future__ import annotations
import subprocess,sys
from pathlib import Path

STEPS=[
 ("Regression",[sys.executable,"run_function_semantic_v101213.py"]),
 ("Function Training",[sys.executable,"train_function_semantic_v101213.py"]),
 ("Function Evaluation",[sys.executable,"evaluate_function_semantic_v101213.py"]),
]

def run(name,cmd):
    print()
    print("="*116); print(" "+name); print("="*116)
    print("Command:"," ".join(cmd)); print()
    r=subprocess.run(cmd)
    if r.returncode:
        print(f"[FAIL] {name} exit_code={r.returncode}")
        raise SystemExit(r.returncode)
    print(f"[PASS] {name}")

def main():
    print("="*116)
    print(" LLM_TRY v10.12.13 Function Semantic Decomposition - Full Verification")
    print("="*116)
    req=[
        "data/nagato_corpus_semantic_train_v101212.jsonl",
        "data/nagato_corpus_semantic_holdout_seen_v101212.jsonl",
        "data/nagato_corpus_semantic_holdout_unseen_v101212.jsonl",
        "model/model-gpu-v1.6.2-online.pt",
        "model/tokenizer-v0.7-bpe.json",
    ]
    missing=[x for x in req if not Path(x).exists()]
    if missing:
        print("[FAIL] missing:",missing)
        raise SystemExit(2)
    for name,cmd in STEPS:
        run(name,cmd)
    print()
    print("="*116); print(" FINAL RESULT"); print("="*116)
    print("Regression          : PASS")
    print("Function training   : PASS")
    print("Function evaluation : PASS")
    print("STATUS              : FUNCTION_SEMANTIC_FULL_PASS")

if __name__=="__main__":
    main()
