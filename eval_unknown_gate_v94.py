# eval_unknown_gate_v94.py
#
# Evaluate ChatGPT-style unknown routing for known vs unknown prompts.
# This script does NOT train "未学習です" into the language model.
# It runs chat.py in normal gate-enabled mode and scores whether the gate
# preserves known prompts while rejecting unknown paraphrases.

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

KNOWN = [
    "あなたは誰ですか",
    "名前を教えてください",
    "自己紹介してください",
    "AIとは",
    "AIについて教えて",
    "LLMとは",
    "LLMって何",
    "CUDAとは",
    "CUDAって何",
    "量子力学とは",
    "量子力学って何",
]


def load_unknown(path: Path):
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        q = str(row.get("user", "")).strip()
        if q:
            rows.append(q)
    return rows


def run_chat(model: str, prompt: str) -> str:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    proc = subprocess.run(
        [
            "python",
            "-X",
            "utf8",
            "chat.py",
            "--model",
            model,
            "--temperature",
            "0",
        ],
        input=prompt + "\nexit\n",
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="strict",
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"chat.py failed for prompt={prompt!r}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return proc.stdout


def extract_answer(text: str) -> str:
    marker = "AI> "
    idx = text.find(marker)
    if idx < 0:
        return ""
    tail = text[idx + len(marker):]
    return tail.splitlines()[0].strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="model/model-llm-try-nagato-chat-v94.pt")
    p.add_argument("--unknown-data", default="data/nagato_unknown_paraphrase.jsonl")
    args = p.parse_args()

    unknown = load_unknown(Path(args.unknown_data))

    print("=" * 80)
    print(" LLM_TRY v9.5 Pre-generation Unknown Concept Gate Benchmark")
    print("=" * 80)
    print("Model        :", args.model)
    print("Known prompts:", len(KNOWN))
    print("Unknown      :", len(unknown))
    print()

    known_ok = 0
    for q in KNOWN:
        out = run_chat(args.model, q)
        ans = extract_answer(out)
        ok = bool(ans) and ans != "未学習です"
        known_ok += int(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] KNOWN   {q} -> {ans or '<EMPTY>'}")

    print()
    unknown_ok = 0
    for q in unknown:
        out = run_chat(args.model, q)
        ans = extract_answer(out)
        ok = ans == "未学習です"
        unknown_ok += int(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] UNKNOWN {q} -> {ans or '<EMPTY>'}")

    print()
    print("Summary")
    print("-" * 80)
    print(f"Known preservation : {known_ok}/{len(KNOWN)} = {known_ok/len(KNOWN)*100:.1f}%")
    print(f"Unknown rejection  : {unknown_ok}/{len(unknown)} = {unknown_ok/len(unknown)*100:.1f}%")
    balanced = 0.5 * (known_ok/len(KNOWN) + unknown_ok/len(unknown))
    print(f"Balanced accuracy  : {balanced*100:.1f}%")


if __name__ == "__main__":
    main()
