# eval_integrated_gate_v97.py
#
# LLM_TRY v9.7 integrated regression for:
#   - known concept preservation
#   - unknown paraphrase rejection
#   - final balanced accuracy
#
# Uses gate-enabled chat.py only.

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


def load_unknown(path: Path) -> list[str]:
    rows: list[str] = []
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
            "python", "-X", "utf8",
            "chat.py",
            "--model", model,
            "--temperature", "0",
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
    for line in text.splitlines():
        pos = line.find("AI> ")
        if pos >= 0:
            return line[pos + 4:].strip()
    return ""


def extract_gate(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("[gate="):
            return line.strip()
    return ""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="model/model-llm-try-nagato-chat-v94.pt")
    p.add_argument(
        "--unknown-data",
        default="data/nagato_unknown_paraphrase.jsonl",
    )
    args = p.parse_args()

    unknown = load_unknown(Path(args.unknown_data))

    print("=" * 92)
    print(" LLM_TRY v9.7 Integrated Known/Unknown Regression")
    print("=" * 92)
    print("Model          :", args.model)
    print("Known prompts  :", len(KNOWN))
    print("Unknown prompts:", len(unknown))
    print()

    known_ok = 0
    parse_fail = 0

    for q in KNOWN:
        out = run_chat(args.model, q)
        ans = extract_answer(out)
        if not ans:
            parse_fail += 1
            ok = False
            status = "PARSE_FAIL"
        else:
            ok = ans != "未学習です"
            status = "PASS" if ok else "FAIL"
        known_ok += int(ok)
        print(f"[{status}] KNOWN   {q} -> {ans or '<EMPTY>'}")
        if not ok:
            diag = extract_gate(out)
            if diag:
                print(f"         {diag}")

    print()
    unknown_ok = 0

    for q in unknown:
        out = run_chat(args.model, q)
        ans = extract_answer(out)
        if not ans:
            parse_fail += 1
            ok = False
            status = "PARSE_FAIL"
        else:
            ok = ans == "未学習です"
            status = "PASS" if ok else "FAIL"
        unknown_ok += int(ok)
        print(f"[{status}] UNKNOWN {q} -> {ans or '<EMPTY>'}")
        if not ok:
            diag = extract_gate(out)
            if diag:
                print(f"         {diag}")

    known_rate = known_ok / len(KNOWN) if KNOWN else 0.0
    unknown_rate = unknown_ok / len(unknown) if unknown else 0.0
    balanced = 0.5 * (known_rate + unknown_rate)

    print()
    print("Summary")
    print("-" * 92)
    print(f"Known preservation : {known_ok}/{len(KNOWN)} = {known_rate*100:.1f}%")
    print(f"Unknown rejection  : {unknown_ok}/{len(unknown)} = {unknown_rate*100:.1f}%")
    print(f"Balanced accuracy  : {balanced*100:.1f}%")
    print(f"Parse failures     : {parse_fail}")

    all_pass = (
        known_ok == len(KNOWN)
        and unknown_ok == len(unknown)
        and parse_fail == 0
    )
    print(f"Regression status  : {'PASS' if all_pass else 'FAIL'}")

    raise SystemExit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
