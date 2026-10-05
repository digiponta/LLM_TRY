#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.11.7 Internalized Fidelity Gate Regression."""

from __future__ import annotations

from chat import apply_internalized_fidelity_policy


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (
        f" : {detail}" if detail else ""
    ))
    if not ok:
        raise AssertionError(name)


def main() -> None:
    print("=" * 96)
    print(" LLM_TRY v10.11.7 Internalized Fidelity Gate Regression")
    print("=" * 96)

    accepted, reason = apply_internalized_fidelity_policy(
        accepted=True,
        fidelity=0.88,
        threshold=0.90,
    )
    check(
        "low-fidelity-rejected",
        not accepted
        and "internalized teacher fidelity" in reason,
        reason,
    )

    accepted, reason = apply_internalized_fidelity_policy(
        accepted=True,
        fidelity=0.93,
        threshold=0.90,
    )
    check(
        "high-fidelity-accepted",
        accepted
        and reason == "internalized teacher fidelity passed",
        reason,
    )

    accepted, reason = apply_internalized_fidelity_policy(
        accepted=False,
        fidelity=0.99,
        threshold=0.90,
    )
    check(
        "prior-gate-failure-preserved",
        not accepted and reason == "",
        reason,
    )

    print()
    print("Low fidelity block : PASS")
    print("High fidelity pass : PASS")
    print("Prior gate preserve: PASS")
    print("STATUS             : INTERNALIZED_FIDELITY_GATE_PASS")


if __name__ == "__main__":
    main()
