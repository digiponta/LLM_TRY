# eval_adaptive_preservation_v103.py
#
# LLM_TRY v10.3 regression:
# Verify that adaptive learning persists without destroying the stable v10.x
# baseline and without promoting unrelated unknown concepts.

from __future__ import annotations

import os
import subprocess

BASELINE_KNOWN = [
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

LEARNED = {
    "宇宙とは": "宇宙は、物質・エネルギー・時空を含む世界全体を指す。",
}

STILL_UNKNOWN = [
    "ブラックホールとは",
    "相対性理論とは",
    "化学とは",
]


def run_chat(prompt: str) -> str:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        ["python", "-X", "utf8", "chat.py", "--temperature", "0"],
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
    print("=" * 96)
    print(" LLM_TRY v10.4.2 Adaptive Preservation Regression")
    print("=" * 96)

    passed = 0
    total = 0

    print()
    print("A) Stable baseline preservation")
    print("-" * 96)
    for q in BASELINE_KNOWN:
        total += 1
        out = run_chat(q)
        ans = extract_answer(out)
        ok = bool(ans) and ans != "未学習です"
        passed += int(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {q} -> {ans or '<EMPTY>'}")
        if not ok:
            gate = extract_gate(out)
            if gate:
                print(f"       {gate}")

    print()
    print("B) Learned concept persistence")
    print("-" * 96)
    for q, expected in LEARNED.items():
        total += 1
        out = run_chat(q)
        ans = extract_answer(out)
        gate = extract_gate(out)
        ok = ans == expected and "gate=KNOWN" in gate and "resolution=ACCEPT" in gate
        passed += int(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {q} -> {ans or '<EMPTY>'}")
        if gate:
            print(f"       {gate}")

    print()
    print("C) Unrelated unknown isolation")
    print("-" * 96)
    for q in STILL_UNKNOWN:
        total += 1
        out = run_chat(q)
        ans = extract_answer(out)
        ok = ans == "未学習です"
        passed += int(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {q} -> {ans or '<EMPTY>'}")
        if not ok:
            gate = extract_gate(out)
            if gate:
                print(f"       {gate}")

    print()
    print("Summary")
    print("-" * 96)
    print(f"Passed            : {passed}/{total}")
    print(f"Failed            : {total - passed}/{total}")
    print("Regression status : " + ("PASS" if passed == total else "FAIL"))

    raise SystemExit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
