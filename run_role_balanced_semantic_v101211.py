#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.12.11 Role-Balanced Semantic Training regression."""

from __future__ import annotations

from semantic_role_generalization_v101210 import RoleProposition
from train_semantic_role_generalization_v101210 import (
    balance_role_items,
    role_counts,
)


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" : {detail}" if detail else ""))
    if not ok:
        raise AssertionError(name)


def item(subject: str, relation: str) -> RoleProposition:
    return RoleProposition(
        subject=subject,
        relation=relation,
        object_description=f"{subject}-{relation}",
        question=f"{subject}とは",
        answer=f"{subject}は、{relation}。",
    )


def main() -> None:
    print("=" * 116)
    print(" LLM_TRY v10.12.11 Role-Balanced Semantic Training Regression")
    print("=" * 116)

    rows = [
        item("P1", "property"),
        item("P2", "property"),
        item("P3", "property"),
        item("P4", "property"),
        item("F1", "function"),
        item("D1", "definition"),
        item("D2", "definition"),
    ]

    before = role_counts(rows)
    balanced, before2, after = balance_role_items(rows, max_multiplier=4)

    check("role-counts-stable", before == before2, str(before2))
    check("majority-kept", after["property"] == 4, str(after))
    check("function-oversampled", after["function"] == 4, str(after))
    check("definition-oversampled", after["definition"] == 4, str(after))
    check("balanced-total", len(balanced) == 12, str(len(balanced)))

    capped, _, capped_after = balance_role_items(rows, max_multiplier=2)
    check("multiplier-cap-applied", capped_after["function"] == 2, str(capped_after))
    check("deterministic-order", [x.subject for x in balanced] == [x.subject for x in balance_role_items(rows, max_multiplier=4)[0]])

    print()
    print("Minority-role oversampling : PASS")
    print("Multiplier cap             : PASS")
    print("Determinism                : PASS")
    print("STATUS                     : ROLE_BALANCED_SEMANTIC_TRAINING_PASS")


if __name__ == "__main__":
    main()
