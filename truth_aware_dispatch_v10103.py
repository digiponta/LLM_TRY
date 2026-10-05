#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_TRY v10.10.3 Truth-Aware Dispatch Policy.
"""

from __future__ import annotations

from dataclasses import dataclass

from knowledge_state_dispatcher_v10101 import DispatchResult
from truth_state_v10103 import TruthRecord


@dataclass(frozen=True)
class TruthDispatchResult:
    dispatch: DispatchResult
    truth: TruthRecord
    warning: str = ""
    correction_applied: bool = False


def apply_truth_policy(
    dispatch: DispatchResult,
    truth: TruthRecord,
) -> TruthDispatchResult:
    """Apply truth policy without deleting the underlying knowledge."""
    state = truth.state

    if state == "TRUE":
        return TruthDispatchResult(
            dispatch=dispatch,
            truth=truth,
        )

    if state == "UNVERIFIED":
        return TruthDispatchResult(
            dispatch=dispatch,
            truth=truth,
            warning="knowledge is unverified",
        )

    if state == "CONTESTED":
        return TruthDispatchResult(
            dispatch=dispatch,
            truth=truth,
            warning=(
                "knowledge is contested"
                + (f": {truth.reason}" if truth.reason else "")
            ),
        )

    if state in {"FALSE", "OUTDATED"}:
        label = "false" if state == "FALSE" else "outdated"
        if truth.correction:
            corrected = DispatchResult(
                action="RETRIEVE",
                state=dispatch.state,
                focus=dispatch.focus,
                answer=truth.correction,
                route="truth-state correction",
                predicate_type=dispatch.predicate_type,
                reason=(
                    f"{label} knowledge replaced by stored correction"
                ),
                provenance=dispatch.provenance,
            )
            return TruthDispatchResult(
                dispatch=corrected,
                truth=truth,
                warning=(
                    f"original knowledge is {label}; "
                    "stored correction applied"
                ),
                correction_applied=True,
            )

        blocked = DispatchResult(
            action="BLOCK",
            state=dispatch.state,
            focus=dispatch.focus,
            answer="",
            route="truth-state block",
            predicate_type=dispatch.predicate_type,
            reason=f"{label} knowledge has no correction",
            provenance=dispatch.provenance,
        )
        return TruthDispatchResult(
            dispatch=blocked,
            truth=truth,
            warning=f"knowledge is {label}",
        )

    raise ValueError(f"unsupported truth state: {state!r}")
