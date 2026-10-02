# eval_known_false_rejection_v96.py
#
# Diagnose false rejection on known prompts by comparing:
#   A) normal gate-enabled chat
#   B) the same model with --no-unknown-rejection
#
# The goal is to distinguish a model-generation failure from a gate failure.

from __future__ import annotations

import argparse
import os
import subprocess

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


def run_chat(model: str, prompt: str, reject: bool) -> str:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    cmd = [
        "python", "-X", "utf8",
        "chat.py",
        "--model", model,
        "--temperature", "0",
    ]
    if not reject:
        cmd.append("--no-unknown-rejection")

    proc = subprocess.run(
        cmd,
        input=prompt + "\nexit\n",
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="strict",
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"chat.py failed for prompt={prompt!r}, reject={reject}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return proc.stdout


def line_after_prefix(text: str, prefix: str) -> str:
    for line in text.splitlines():
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    return ""


def gate_line(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("[gate="):
            return line.strip()
    return ""


def candidate_line(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("[candidate="):
            return line[len("[candidate="):-1] if line.endswith("]") else line
    return ""


def classify(gated_answer: str, raw_answer: str) -> str:
    if gated_answer != "未学習です":
        return "PASS"
    if raw_answer and raw_answer != "未学習です":
        return "FALSE_REJECT"
    return "MODEL_FAIL"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="model/model-llm-try-nagato-chat-v94.pt")
    args = p.parse_args()

    print("=" * 92)
    print(" LLM_TRY v9.6 Known False-Rejection Diagnostic")
    print("=" * 92)
    print("Model:", args.model)
    print()

    counts = {"PASS": 0, "FALSE_REJECT": 0, "MODEL_FAIL": 0}

    for q in KNOWN:
        gated = run_chat(args.model, q, reject=True)
        raw = run_chat(args.model, q, reject=False)

        gated_answer = line_after_prefix(gated, "AI> ")
        raw_answer = line_after_prefix(raw, "AI> ")
        status = classify(gated_answer, raw_answer)
        counts[status] += 1

        print(f"[{status}] {q}")
        print(f"  gated     : {gated_answer or '<EMPTY>'}")
        print(f"  no-reject : {raw_answer or '<EMPTY>'}")

        cand = candidate_line(gated)
        diag = gate_line(gated)
        if cand:
            print(f"  candidate : {cand}")
        if diag:
            print(f"  gate      : {diag}")
        print()

    total = len(KNOWN)
    print("Summary")
    print("-" * 92)
    print(f"PASS         : {counts['PASS']}/{total}")
    print(f"FALSE_REJECT : {counts['FALSE_REJECT']}/{total}")
    print(f"MODEL_FAIL   : {counts['MODEL_FAIL']}/{total}")


if __name__ == "__main__":
    main()
