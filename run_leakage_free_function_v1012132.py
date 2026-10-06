#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.12.13.2 Leakage-Free Function Inference regression."""

from __future__ import annotations

from semantic_role_generalization_v101210 import RoleProposition
from function_semantic_decomposition_v101213 import (
    function_inference_prompt,
    function_inference_slot_prompt,
    function_structure_prompt,
)


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" : {detail}" if detail else ""))
    if not ok:
        raise AssertionError(name)


def main():
    print("="*116)
    print(" LLM_TRY v10.12.13.2 Leakage-Free Function Inference Regression")
    print("="*116)

    teacher = "CPUは命令を実行する。"
    subject = "CPU"

    inference = function_inference_prompt(subject)
    compact = function_inference_slot_prompt(subject)

    check("subject-present", subject in inference)
    check("gold-answer-absent", teacher not in inference)
    check("gold-action-hidden", "action: ?" in inference)
    check("gold-target-hidden", "target: ?" in inference)
    check("gold-purpose-hidden", "purpose: ?" in inference)

    check("compact-action-hidden", "function.action=?" in compact)
    check("compact-target-hidden", "function.target=?" in compact)
    check("compact-purpose-hidden", "function.purpose=?" in compact)

    item = RoleProposition(
        subject=subject,
        relation="function",
        object_description="命令を実行する",
        question="CPUとは",
        answer=teacher,
    )
    oracle = function_structure_prompt(item)
    check("oracle-is-diagnostic-only", "action: execute" in oracle)
    check("inference-differs-from-oracle", inference != oracle)

    print()
    print("Teacher leakage      : NONE")
    print("Unknown slots        : PRESERVED")
    print("Oracle separation    : PASS")
    print("STATUS               : LEAKAGE_FREE_FUNCTION_INFERENCE_PASS")


if __name__=="__main__":
    main()
