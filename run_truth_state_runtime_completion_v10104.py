#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.10.4 Truth-State Runtime Completion Regression."""

from knowledge_state_dispatcher_v10101 import DispatchResult
from truth_aware_dispatch_v10103 import apply_truth_policy
from truth_state_v10103 import TruthRecord


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (
        f" : {detail}" if detail else ""
    ))
    if not ok:
        raise AssertionError(name)


def dispatch(
    *,
    action: str,
    state: str,
    focus: str,
    answer: str = "",
    route: str = "base-route",
) -> DispatchResult:
    return DispatchResult(
        action=action,
        state=state,
        focus=focus,
        answer=answer,
        route=route,
        reason="base reason",
    )


def main() -> None:
    print("=" * 104)
    print(" LLM_TRY v10.10.4 Truth-State Runtime Completion Regression")
    print("=" * 104)

    base_retrieval = dispatch(
        action="RETRIEVE",
        state="CANONICAL",
        focus="CPU",
        answer="CPUは中央処理装置である。",
    )

    true_result = apply_truth_policy(
        base_retrieval,
        TruthRecord("CPU", "TRUE"),
    )
    check(
        "true-preserve",
        true_result.dispatch.action == "RETRIEVE"
        and true_result.dispatch.answer == base_retrieval.answer
        and true_result.runtime_status == "PASS"
        and not true_result.warning,
        str(true_result),
    )

    unverified_result = apply_truth_policy(
        base_retrieval,
        TruthRecord("CPU", "UNVERIFIED"),
    )
    check(
        "unverified-warning",
        unverified_result.dispatch.action == "RETRIEVE"
        and unverified_result.runtime_status == "WARN"
        and "unverified" in unverified_result.warning,
        str(unverified_result),
    )

    contested_result = apply_truth_policy(
        base_retrieval,
        TruthRecord(
            "CPU",
            "CONTESTED",
            reason="複数の見解がある",
        ),
    )
    check(
        "contested-warning",
        contested_result.dispatch.action == "RETRIEVE"
        and contested_result.runtime_status == "WARN"
        and "contested" in contested_result.warning,
        str(contested_result),
    )

    false_corrected = apply_truth_policy(
        base_retrieval,
        TruthRecord(
            "CPU",
            "FALSE",
            correction="CPUは命令を実行する中央処理装置である。",
        ),
    )
    check(
        "false-correction",
        false_corrected.dispatch.action == "RETRIEVE"
        and false_corrected.dispatch.route == "truth-state correction"
        and false_corrected.correction_applied
        and false_corrected.runtime_status == "CORRECTED",
        str(false_corrected),
    )

    false_block = apply_truth_policy(
        base_retrieval,
        TruthRecord("CPU", "FALSE"),
    )
    check(
        "false-block",
        false_block.dispatch.action == "BLOCK"
        and false_block.runtime_status == "BLOCK"
        and bool(false_block.user_message),
        str(false_block),
    )

    outdated_corrected = apply_truth_policy(
        base_retrieval,
        TruthRecord(
            "CPU",
            "OUTDATED",
            correction="CPUの最新版説明。",
        ),
    )
    check(
        "outdated-correction",
        outdated_corrected.dispatch.action == "RETRIEVE"
        and outdated_corrected.correction_applied
        and outdated_corrected.runtime_status == "CORRECTED",
        str(outdated_corrected),
    )

    outdated_block = apply_truth_policy(
        base_retrieval,
        TruthRecord("CPU", "OUTDATED"),
    )
    check(
        "outdated-block",
        outdated_block.dispatch.action == "BLOCK"
        and outdated_block.runtime_status == "BLOCK"
        and bool(outdated_block.user_message),
        str(outdated_block),
    )

    internalized = dispatch(
        action="GENERATE",
        state="INTERNALIZED",
        focus="量子センサー",
    )
    internalized_false = apply_truth_policy(
        internalized,
        TruthRecord(
            "量子センサー",
            "FALSE",
            correction="量子センサーの修正説明。",
        ),
    )
    check(
        "internalized-false-no-generation",
        internalized_false.dispatch.action == "RETRIEVE"
        and internalized_false.dispatch.route == "truth-state correction"
        and internalized_false.runtime_status == "CORRECTED",
        str(internalized_false),
    )

    raw = dispatch(
        action="BLOCK",
        state="RAW_CORPUS_ONLY",
        focus="時間",
    )
    raw_true = apply_truth_policy(
        raw,
        TruthRecord("時間", "TRUE"),
    )
    check(
        "raw-remains-blocked-even-true",
        raw_true.dispatch.action == "BLOCK"
        and raw_true.runtime_status == "PASS",
        str(raw_true),
    )

    raw_unverified = apply_truth_policy(
        raw,
        TruthRecord("時間", "UNVERIFIED"),
    )
    check(
        "raw-unverified-remains-blocked",
        raw_unverified.dispatch.action == "BLOCK"
        and raw_unverified.runtime_status == "WARN",
        str(raw_unverified),
    )

    print()
    print("TRUE preserve                  : PASS")
    print("UNVERIFIED warning             : PASS")
    print("CONTESTED warning              : PASS")
    print("FALSE correction/block         : PASS")
    print("OUTDATED correction/block      : PASS")
    print("Internalized FALSE interception: PASS")
    print("RAW block preservation         : PASS")
    print("STATUS                         : TRUTH_STATE_RUNTIME_COMPLETION_PASS")


if __name__ == "__main__":
    main()
