#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.9.0 Runtime Normalization / Bare Concept Regression."""

from chat import normalize_runtime_input, pre_generation_unknown_concept


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" : {detail}" if detail else ""))
    if not ok:
        raise AssertionError(name)


def main():
    print("=" * 88)
    print(" LLM_TRY v10.9.0 Runtime Normalization / Bare Concept Regression")
    print("=" * 88)

    check(
        "persona-trailing-comma",
        normalize_runtime_input("貴方は、") == "貴方は",
        normalize_runtime_input("貴方は、"),
    )
    check(
        "definition-trailing-question",
        normalize_runtime_input("GPUとは？") == "GPUとは",
        normalize_runtime_input("GPUとは？"),
    )
    check(
        "internal-punctuation-preserved",
        normalize_runtime_input("CPUとGPU、どちら") == "CPUとGPU、どちら",
        normalize_runtime_input("CPUとGPU、どちら"),
    )

    unknown, focus = pre_generation_unknown_concept("時間")
    check("bare-time-unknown", unknown and focus == "時間", f"{unknown=}, {focus=}")

    unknown, focus = pre_generation_unknown_concept("CPU")
    check("bare-cpu-known", (not unknown) and focus == "CPU", f"{unknown=}, {focus=}")

    unknown, focus = pre_generation_unknown_concept("貴方は")
    check("persona-not-knowledge-unknown", not unknown, f"{unknown=}, {focus=}")

    print()
    print("Surface normalization : PASS")
    print("Bare concept strictness: PASS")
    print("Persona preservation   : PASS")
    print("STATUS                 : RUNTIME_GATE_FIX_PASS")


if __name__ == "__main__":
    main()
