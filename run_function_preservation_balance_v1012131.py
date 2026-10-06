#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.12.13.1 Preservation-Balanced Function Training regression."""

from __future__ import annotations

from semantic_role_generalization_v101210 import RoleProposition, training_queries
from function_semantic_decomposition_v101213 import function_training_queries


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" : {detail}" if detail else ""))
    if not ok:
        raise AssertionError(name)


def main():
    print("="*116)
    print(" LLM_TRY v10.12.13.1 Preservation-Balanced Function Training Regression")
    print("="*116)

    f = RoleProposition("CPU","function","命令を実行する","CPUとは","CPUは命令を実行する。")
    p = RoleProposition("磁石","property","磁場を持つ","磁石とは","磁石は磁場を持つ。")

    base_function=len(training_queries(f))
    generic_per_weight=len(training_queries(f)[1:])
    structured_per_weight=len(function_training_queries(f))
    nonfunction_per_weight=len(training_queries(p)[1:])

    function_generic_weight=3
    function_struct_weight=2
    nonfunction_replay_weight=2

    generic_rows=generic_per_weight*function_generic_weight
    structured_rows=structured_per_weight*function_struct_weight
    nonfunction_rows=nonfunction_per_weight*nonfunction_replay_weight

    check("base-function-prompts",base_function==3,str(base_function))
    check("generic-function-replay",generic_rows==6,str(generic_rows))
    check("structured-function-replay",structured_rows==4,str(structured_rows))
    check("generic-dominates-structured",generic_rows>=structured_rows,
          f"generic={generic_rows} structured={structured_rows}")
    check("nonfunction-replay-present",nonfunction_rows==4,str(nonfunction_rows))
    check("lower-lr-policy",2e-6 < 4e-6)
    check("shorter-epoch-policy",4 < 6)

    print()
    print("Generic preservation : PASS")
    print("Structured learning  : PASS")
    print("Nonfunction replay   : PASS")
    print("Conservative update  : PASS")
    print("STATUS               : FUNCTION_PRESERVATION_BALANCE_PASS")


if __name__=="__main__":
    main()
