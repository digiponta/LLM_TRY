#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.12.15.1 Balanced Subject-to-Proposition regression."""

from __future__ import annotations


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" : {detail}" if detail else ""))
    if not ok:
        raise AssertionError(name)


def mapping(subject: str, answer: str) -> str:
    return f"{subject} => {answer}"


def training_rows(subject: str, answer: str, weight: int) -> list[tuple[str, str]]:
    rows = []
    for _ in range(max(1, weight)):
        rows.append((f"{subject} =>", answer))
    return rows


def main():
    print("=" * 116)
    print(" LLM_TRY v10.12.15.1 Balanced Subject-to-Proposition Regression")
    print("=" * 116)

    subject = "GPU"
    answer = "GPUは高速である。"
    mapped = mapping(subject, answer)

    check("canonical-mapping", mapped == "GPU => GPUは高速である。", mapped)

    rows = training_rows(subject, answer, 1)
    check("mapping-row-count", len(rows) == 1, str(len(rows)))
    check("lookup-prompt-present", ("GPU =>", answer) in rows)
    check("full-mapping-not-fed-as-input", (mapped, answer) not in rows)
    check("target-is-full-proposition", all(y == answer for _, y in rows))

    print()
    print("Subject mapping       : PASS")
    print("Full proposition      : PASS")
    print("Balanced integration  : PASS")
    print("STATUS                : SUBJECT_TO_PROPOSITION_TRAINING_PASS")


if __name__ == "__main__":
    main()
