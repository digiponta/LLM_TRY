#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.10.3 Truth-Aware Dispatch Regression."""

from knowledge_state_dispatcher_v10101 import DispatchResult
from truth_aware_dispatch_v10103 import apply_truth_policy
from truth_state_v10103 import TruthRecord


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (
        f" : {detail}" if detail else ""
    ))
    if not ok:
        raise AssertionError(name)


def base(action: str = "RETRIEVE") -> DispatchResult:
    return DispatchResult(
        action=action,
        state="UNIFIED" if action == "RETRIEVE" else "INTERNALIZED",
        focus="宇宙",
        answer=("旧回答" if action == "RETRIEVE" else ""),
        route="test-route",
        reason="test",
    )


def main() -> None:
    print("=" * 96)
    print(" LLM_TRY v10.10.3 Truth-Aware Dispatch Regression")
    print("=" * 96)

    true_result = apply_truth_policy(
        base(),
        TruthRecord("宇宙", "TRUE"),
    )
    check(
        "true-preserves",
        true_result.dispatch.action == "RETRIEVE"
        and true_result.dispatch.answer == "旧回答"
        and not true_result.warning,
        str(true_result),
    )

    unverified = apply_truth_policy(
        base(),
        TruthRecord("宇宙", "UNVERIFIED"),
    )
    check(
        "unverified-warns",
        unverified.dispatch.action == "RETRIEVE"
        and "unverified" in unverified.warning,
        str(unverified),
    )

    contested = apply_truth_policy(
        base(),
        TruthRecord("宇宙", "CONTESTED", reason="複数説あり"),
    )
    check(
        "contested-warns",
        contested.dispatch.action == "RETRIEVE"
        and "contested" in contested.warning,
        str(contested),
    )

    false_corrected = apply_truth_policy(
        base(),
        TruthRecord(
            "宇宙",
            "FALSE",
            correction="修正回答",
        ),
    )
    check(
        "false-correction",
        false_corrected.dispatch.action == "RETRIEVE"
        and false_corrected.dispatch.answer == "修正回答"
        and false_corrected.correction_applied,
        str(false_corrected),
    )

    false_blocked = apply_truth_policy(
        base(),
        TruthRecord("宇宙", "FALSE"),
    )
    check(
        "false-block",
        false_blocked.dispatch.action == "BLOCK",
        str(false_blocked),
    )

    outdated_corrected = apply_truth_policy(
        base(),
        TruthRecord(
            "宇宙",
            "OUTDATED",
            correction="最新版",
        ),
    )
    check(
        "outdated-correction",
        outdated_corrected.dispatch.action == "RETRIEVE"
        and outdated_corrected.dispatch.answer == "最新版",
        str(outdated_corrected),
    )

    internalized_false = apply_truth_policy(
        base(action="GENERATE"),
        TruthRecord(
            "宇宙",
            "FALSE",
            correction="安全な修正",
        ),
    )
    check(
        "internalized-false-converts-to-retrieval",
        internalized_false.dispatch.action == "RETRIEVE"
        and internalized_false.dispatch.answer == "安全な修正",
        str(internalized_false),
    )

    print()
    print("TRUE preserve         : PASS")
    print("UNVERIFIED warning    : PASS")
    print("CONTESTED warning     : PASS")
    print("FALSE correction/block: PASS")
    print("OUTDATED correction   : PASS")
    print("Internalized overlay  : PASS")
    print("STATUS                : TRUTH_AWARE_DISPATCH_PASS")


if __name__ == "__main__":
    main()
