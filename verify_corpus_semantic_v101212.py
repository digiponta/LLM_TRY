#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.12.12 full corpus-to-semantic verification."""

from __future__ import annotations
import subprocess,sys
from pathlib import Path

STEPS=[
 ("Regression",[sys.executable,"run_corpus_semantic_v101212.py"]),
 ("Dataset Build",[sys.executable,"build_corpus_semantic_v101212.py"]),
 ("Semantic Training",[sys.executable,"train_corpus_semantic_v101212.py"]),
 ("Semantic Evaluation",[sys.executable,"evaluate_corpus_semantic_v101212.py"]),
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
    print(" LLM_TRY v10.12.12 Corpus-to-Semantic Knowledge - Full Verification")
    print("="*116)
    req=["data/data-nagato.txt","model/model-gpu-v1.6.2-online.pt","model/tokenizer-v0.7-bpe.json"]
    missing=[x for x in req if not Path(x).exists()]
    if missing:
        print("[FAIL] missing:",missing); raise SystemExit(2)
    for name,cmd in STEPS: run(name,cmd)
    print()
    print("="*116); print(" FINAL RESULT"); print("="*116)
    print("Regression        : PASS")
    print("Dataset build     : PASS")
    print("Semantic training : PASS")
    print("Semantic eval     : PASS")
    print("STATUS            : CORPUS_TO_SEMANTIC_FULL_PASS")

if __name__=="__main__":
    main()
