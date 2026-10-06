#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.12.14 Two-Pass Function Resolver regression."""

from __future__ import annotations

from two_pass_function_resolver_v101214 import (
    extract_structure_from_evidence,
    function_compose_prompt,
    function_evidence_query,
    relation_uses_two_pass,
    resolve_two_pass_function,
)


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" : {detail}" if detail else ""))
    if not ok:
        raise AssertionError(name)


def main():
    print("=" * 116)
    print(" LLM_TRY v10.12.14 Two-Pass Function Resolver Regression")
    print("=" * 116)

    subject = "CPU"
    evidence = "CPUは命令を実行する。"

    check("pass1-query", function_evidence_query(subject) == "CPUとは")

    structure = extract_structure_from_evidence(subject, evidence)
    check("extract-action", structure.action == "execute", structure.action)
    check("extract-target", structure.target == "命令", structure.target)
    check("extract-purpose", structure.purpose == "computation", structure.purpose)

    prompt = function_compose_prompt(subject, evidence, structure)
    check("evidence-present", evidence in prompt)
    check("action-present", "action: execute" in prompt)
    check("target-present", "target: 命令" in prompt)
    check("purpose-present", "purpose: computation" in prompt)

    calls = []
    def fake_generate(query):
        calls.append(query)
        if len(calls) == 1:
            return evidence
        return "CPUは命令を実行する機能を持つ。"

    result = resolve_two_pass_function(subject, fake_generate)
    check("two-pass-call-count", len(calls) == 2, str(len(calls)))
    check("pass1-no-gold-slots", "action:" not in calls[0])
    check("pass2-uses-pass1-evidence", evidence in calls[1])
    check("answer-returned", bool(result.answer))
    check("no-fallback", not result.used_fallback)

    check("route-function", relation_uses_two_pass("function"))
    check("route-property-unchanged", not relation_uses_two_pass("property"))
    check("route-definition-unchanged", not relation_uses_two_pass("definition"))
    check("route-cause-unchanged", not relation_uses_two_pass("cause"))
    check("route-comparison-unchanged", not relation_uses_two_pass("comparison"))

    print()
    print("Pass-1 evidence      : PASS")
    print("Slot extraction      : PASS")
    print("Pass-2 composition   : PASS")
    print("Function-only route  : PASS")
    print("Weight modification  : NONE")
    print("STATUS               : TWO_PASS_FUNCTION_RESOLVER_PASS")


if __name__ == "__main__":
    main()
