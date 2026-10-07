#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.15 integrated verification."""

from pathlib import Path
import json
import subprocess
import sys


def run(name, cmd):
    print()
    print("=" * 100)
    print(" " + name)
    print("=" * 100)
    result = subprocess.run(cmd)
    if result.returncode:
        raise SystemExit(result.returncode)
    print(f"[PASS] {name}")


def main():
    run("Retrieval Top-K", [sys.executable, "verify_retrieval_topk_v10150.py"])
    run("Daily Conversation Routing", [sys.executable, "verify_daily_conversation_v10140.py"])
    run("Daily Conversation SFT Dataset", [sys.executable, "build_daily_conversation_sft_v10150.py"])

    path = Path("data/daily_conversation_sft_v10150.jsonl")
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(rows) < 50:
        raise AssertionError(f"daily SFT data too small: {len(rows)}")
    if not all(row.get("source") == "daily-conversation-sft" for row in rows):
        raise AssertionError("unexpected Daily Conversation SFT source")

    print()
    print("=" * 100)
    print(" FINAL RESULT")
    print("=" * 100)
    print("Retrieval Top-K       : PASS")
    print("Daily Conversation    : PASS")
    print("Daily SFT dataset     : PASS")
    print("Daily SFT pairs       :", len(rows))
    print("STATUS                : V10_15_TOPK_DAILY_SFT_READY")


if __name__ == "__main__":
    main()
