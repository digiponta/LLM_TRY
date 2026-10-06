#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.12.13 Function Semantic Decomposition regression."""

from __future__ import annotations

from semantic_role_generalization_v101210 import RoleProposition
from function_semantic_decomposition_v101213 import (
    decompose_function,
    infer_action,
    infer_target,
    infer_purpose,
    function_structure_prompt,
    function_slot_prompt,
)


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" : {detail}" if detail else ""))
    if not ok:
        raise AssertionError(name)


def main():
    print("="*116)
    print(" LLM_TRY v10.12.13 Function Semantic Decomposition Regression")
    print("="*116)

    check("action-execute", infer_action("CPUは命令を実行する。")=="execute")
    check("action-process", infer_action("装置は情報を処理する。")=="process")
    check("action-use", infer_action("機構は資源を利用する。")=="use")

    check("target-object-before-verb", infer_target("CPU","CPUは命令を実行する。")=="命令",
          infer_target("CPU","CPUは命令を実行する。"))
    check("target-verb-before-object", infer_target("CPU","CPUは実行している命令を保持する。")=="命令",
          infer_target("CPU","CPUは実行している命令を保持する。"))
    check("purpose-computation", infer_purpose("CPUは命令を実行する。")=="computation")

    item=RoleProposition(
        subject="CPU",
        relation="function",
        object_description="命令を実行する",
        question="CPUとは",
        answer="CPUは命令を実行する。",
    )
    fs=decompose_function(item)
    check("decompose-subject",fs.subject=="CPU")
    check("decompose-action",fs.action=="execute")
    check("decompose-target",fs.target=="命令")
    check("structured-prompt","action: execute" in function_structure_prompt(item))
    check("slot-prompt","function.target=命令" in function_slot_prompt(item))

    print()
    print("Action extraction   : PASS")
    print("Target extraction   : PASS")
    print("Purpose extraction  : PASS")
    print("Structured prompts  : PASS")
    print("STATUS              : FUNCTION_SEMANTIC_DECOMPOSITION_PASS")


if __name__=="__main__":
    main()
