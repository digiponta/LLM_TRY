#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.12.12.2 raw+semantic promotion gate regression."""

from __future__ import annotations


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" : {detail}" if detail else ""))
    if not ok:
        raise AssertionError(name)


def gate(
    binding: bool,
    metadata: bool,
    internalized_ok: bool,
    persona: bool,
    plain_gain: float,
    semantic_gain: float,
    seen_gain: float,
    unseen_gain: float,
    improved: int,
    regressed: int,
) -> bool:
    return all((
        binding,
        metadata,
        internalized_ok,
        persona,
        plain_gain >= 0.005,
        semantic_gain >= 0.010 and improved > regressed,
        seen_gain >= 0.010 and improved > regressed,
        unseen_gain >= 0.000 and improved >= regressed,
    ))


def main() -> None:
    print("=" * 116)
    print(" LLM_TRY v10.12.12.2 Raw+Semantic Promotion Gate Regression")
    print("=" * 116)

    check(
        "accept-valid-candidate",
        gate(True, True, True, True, 0.016, 0.021, 0.020, 0.023, 19, 1),
    )
    check(
        "reject-binding-loss",
        not gate(False, True, True, True, 0.016, 0.021, 0.020, 0.023, 19, 1),
    )
    check(
        "reject-metadata-loss",
        not gate(True, False, True, True, 0.016, 0.021, 0.020, 0.023, 19, 1),
    )
    check(
        "reject-retention-loss",
        not gate(True, True, False, True, 0.016, 0.021, 0.020, 0.023, 19, 1),
    )
    check(
        "reject-persona-loss",
        not gate(True, True, True, False, 0.016, 0.021, 0.020, 0.023, 19, 1),
    )
    check(
        "reject-plain-gain-loss",
        not gate(True, True, True, True, 0.004, 0.021, 0.020, 0.023, 19, 1),
    )
    check(
        "reject-semantic-gain-loss",
        not gate(True, True, True, True, 0.016, 0.009, 0.020, 0.023, 19, 1),
    )
    check(
        "reject-seen-loss",
        not gate(True, True, True, True, 0.016, 0.021, 0.009, 0.023, 19, 1),
    )
    check(
        "reject-unseen-regression",
        not gate(True, True, True, True, 0.016, 0.021, 0.020, -0.001, 19, 1),
    )

    print()
    print("Binding gate       : PASS")
    print("Metadata gate      : PASS")
    print("Retention gate     : PASS")
    print("Persona gate       : PASS")
    print("QA/semantic gates  : PASS")
    print("Seen/unseen gates  : PASS")
    print("STATUS             : RAW_SEMANTIC_PROMOTION_GATE_PASS")


if __name__ == "__main__":
    main()
