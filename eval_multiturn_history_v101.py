# eval_multiturn_history_v101.py
#
# Regression for v10.1 history-contamination behavior.
# Verifies that a sequence of accepted persona turns does not falsely reject
# a later known technical definition when those prior turns are not selected
# into the actual generation prompt.

from __future__ import annotations

import argparse
import os
import subprocess


SEQUENCE = [
    ("あなたは誰ですか", True),
    ("名前を教えてください", True),
    ("自己紹介してください", True),
    ("AIとは", True),
    ("LLMって何", True),
    ("CUDAとは", True),
    ("量子力学とは", True),
    ("宇宙とは", False),
]


def run_session(model: str) -> str:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    payload = "\n".join(q for q, _ in SEQUENCE) + "\nexit\n"

    proc = subprocess.run(
        [
            "python", "-X", "utf8",
            "chat.py",
            "--model", model,
            "--temperature", "0",
        ],
        input=payload,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="strict",
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"chat.py failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc.stdout


def extract_answers(text: str) -> list[str]:
    answers = []
    for line in text.splitlines():
        pos = line.find("AI> ")
        if pos >= 0:
            answers.append(line[pos + 4:].strip())
    return answers


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="model/model-llm-try-nagato-chat-v94.pt")
    args = p.parse_args()

    out = run_session(args.model)
    answers = extract_answers(out)

    print("=" * 92)
    print(" LLM_TRY v10.1 Multi-turn History Contamination Regression")
    print("=" * 92)
    print("Model:", args.model)
    print()

    passed = 0
    if len(answers) != len(SEQUENCE):
        print(f"[FAIL] parsed answers: expected={len(SEQUENCE)} actual={len(answers)}")
    else:
        for (q, should_be_known), ans in zip(SEQUENCE, answers):
            ok = (ans != "未学習です") if should_be_known else (ans == "未学習です")
            passed += int(ok)
            print(f"[{'PASS' if ok else 'FAIL'}] {q} -> {ans or '<EMPTY>'}")

    total = len(SEQUENCE)
    print()
    print("Summary")
    print("-" * 92)
    print(f"Passed            : {passed}/{total}")
    print(f"Failed            : {total-passed}/{total}")
    print(f"Regression status : {'PASS' if passed == total else 'FAIL'}")

    raise SystemExit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
