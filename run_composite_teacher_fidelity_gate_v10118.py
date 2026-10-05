#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.11.8 Composite Teacher Fidelity Gate Regression."""

from __future__ import annotations

from chat import (
    CompositeFidelityResult,
    _polarity_contradiction,
    _required_content_coverage,
    _teacher_lexical_coverage,
    apply_composite_internalized_fidelity_policy,
)


TEACHER = (
    "量子センサーは、量子的性質を利用して"
    "高感度計測を行うセンサーである。"
)
BAD = (
    "量子センサーは、量子的な役割と研究を学習して"
    "研究を行う研究を行う。"
)
GOOD = (
    "量子センサーは、量子的性質を利用して"
    "高感度の計測を行うセンサーである。"
)


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (
        f" : {detail}" if detail else ""
    ))
    if not ok:
        raise AssertionError(name)


def main() -> None:
    print("=" * 104)
    print(" LLM_TRY v10.11.8 Composite Teacher Fidelity Gate Regression")
    print("=" * 104)

    bad_lex = _teacher_lexical_coverage(BAD, TEACHER)
    bad_req, required, matched = _required_content_coverage(
        BAD,
        TEACHER,
        concept="量子センサー",
    )
    bad = CompositeFidelityResult(
        semantic_similarity=0.938,
        lexical_coverage=bad_lex,
        required_coverage=bad_req,
        contradiction=False,
        required_terms=required,
        matched_terms=matched,
    )
    accepted, reason = apply_composite_internalized_fidelity_policy(
        accepted=True,
        result=bad,
        semantic_threshold=0.90,
        lexical_threshold=0.45,
        required_threshold=0.50,
    )
    check(
        "high-semantic-bad-content-rejected",
        not accepted
        and (
            "lexical" in reason
            or "required" in reason
        ),
        (
            f"sem={bad.semantic_similarity:.3f} "
            f"lex={bad.lexical_coverage:.3f} "
            f"req={bad.required_coverage:.3f} "
            f"required={required} matched={matched} "
            f"reason={reason}"
        ),
    )

    good_lex = _teacher_lexical_coverage(GOOD, TEACHER)
    good_req, good_required, good_matched = (
        _required_content_coverage(
            GOOD,
            TEACHER,
            concept="量子センサー",
        )
    )
    good = CompositeFidelityResult(
        semantic_similarity=0.96,
        lexical_coverage=good_lex,
        required_coverage=good_req,
        contradiction=False,
        required_terms=good_required,
        matched_terms=good_matched,
    )
    accepted, reason = apply_composite_internalized_fidelity_policy(
        accepted=True,
        result=good,
        semantic_threshold=0.90,
        lexical_threshold=0.45,
        required_threshold=0.50,
    )
    check(
        "faithful-paraphrase-accepted",
        accepted,
        (
            f"sem={good.semantic_similarity:.3f} "
            f"lex={good.lexical_coverage:.3f} "
            f"req={good.required_coverage:.3f} "
            f"reason={reason}"
        ),
    )

    contradiction = _polarity_contradiction(
        "量子センサーは高感度計測を行わないセンサーである。",
        TEACHER,
    )
    contradictory = CompositeFidelityResult(
        semantic_similarity=0.97,
        lexical_coverage=0.80,
        required_coverage=0.75,
        contradiction=contradiction,
    )
    accepted, reason = apply_composite_internalized_fidelity_policy(
        accepted=True,
        result=contradictory,
        semantic_threshold=0.90,
        lexical_threshold=0.45,
        required_threshold=0.50,
    )
    check(
        "contradiction-rejected",
        contradiction
        and not accepted
        and "contradiction" in reason,
        reason,
    )

    prior_failed, prior_reason = (
        apply_composite_internalized_fidelity_policy(
            accepted=False,
            result=good,
            semantic_threshold=0.90,
            lexical_threshold=0.45,
            required_threshold=0.50,
        )
    )
    check(
        "prior-gate-failure-preserved",
        not prior_failed and prior_reason == "",
        prior_reason,
    )

    print()
    print("High semantic bad content : PASS")
    print("Faithful paraphrase       : PASS")
    print("Contradiction detection   : PASS")
    print("Prior gate preservation   : PASS")
    print("STATUS                    : COMPOSITE_TEACHER_FIDELITY_GATE_PASS")


if __name__ == "__main__":
    main()
