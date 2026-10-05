#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.10.0 Knowledge State Runtime Policy Regression."""

from knowledge_state_resolver_v10100 import KnowledgeState


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (
        f" : {detail}" if detail else ""
    ))
    if not ok:
        raise AssertionError(name)


def main() -> None:
    print("=" * 96)
    print(" LLM_TRY v10.10.0 Knowledge State Runtime Policy Regression")
    print("=" * 96)

    states = {
        "TYPED": KnowledgeState("TYPED"),
        "CANONICAL": KnowledgeState("CANONICAL"),
        "UNIFIED": KnowledgeState("UNIFIED"),
        "INTERNALIZED": KnowledgeState("INTERNALIZED"),
        "RAW_CORPUS_ONLY": KnowledgeState("RAW_CORPUS_ONLY"),
        "UNKNOWN": KnowledgeState("UNKNOWN"),
        "NON_CONCEPT": KnowledgeState("NON_CONCEPT"),
    }

    check(
        "retrieval-states",
        all(states[name].is_retrieval for name in (
            "TYPED", "CANONICAL", "UNIFIED"
        )),
    )
    check(
        "internalized-generates",
        states["INTERNALIZED"].permits_model_generation,
    )
    check(
        "non-concept-generates",
        states["NON_CONCEPT"].permits_model_generation,
    )
    check(
        "raw-blocked",
        not states["RAW_CORPUS_ONLY"].permits_model_generation,
    )
    check(
        "unknown-blocked",
        not states["UNKNOWN"].permits_model_generation,
    )

    print()
    print("Retrieval policy       : PASS")
    print("Internalized policy    : PASS")
    print("Raw corpus block       : PASS")
    print("Unknown block          : PASS")
    print("STATUS                 : KNOWLEDGE_STATE_RUNTIME_POLICY_PASS")


if __name__ == "__main__":
    main()
