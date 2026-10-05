#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.10.1 Knowledge State Dispatcher Regression."""

from knowledge_state_dispatcher_v10101 import dispatch_knowledge_state
from knowledge_state_resolver_v10100 import KnowledgeState


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (
        f" : {detail}" if detail else ""
    ))
    if not ok:
        raise AssertionError(name)


def main() -> None:
    print("=" * 96)
    print(" LLM_TRY v10.10.1 Knowledge State Dispatcher Regression")
    print("=" * 96)

    typed = dispatch_knowledge_state(
        KnowledgeState(
            "TYPED",
            focus="GPU",
            answer="GPUは高速である。",
            predicate_type="property",
        )
    )
    check(
        "typed-retrieve",
        typed.action == "RETRIEVE"
        and typed.answer == "GPUは高速である。"
        and typed.predicate_type == "property",
        str(typed),
    )

    canonical = dispatch_knowledge_state(
        KnowledgeState(
            "CANONICAL",
            focus="CPU",
            answer="CPUは中央処理装置である。",
        )
    )
    check("canonical-retrieve", canonical.action == "RETRIEVE", str(canonical))

    unified = dispatch_knowledge_state(
        KnowledgeState(
            "UNIFIED",
            focus="宇宙",
            answer="宇宙は全体である。",
        )
    )
    check("unified-retrieve", unified.action == "RETRIEVE", str(unified))

    internalized = dispatch_knowledge_state(
        KnowledgeState("INTERNALIZED", focus="量子センサー")
    )
    check(
        "internalized-generate",
        internalized.action == "GENERATE",
        str(internalized),
    )

    raw = dispatch_knowledge_state(
        KnowledgeState("RAW_CORPUS_ONLY", focus="時間")
    )
    check("raw-block", raw.action == "BLOCK", str(raw))

    unknown = dispatch_knowledge_state(
        KnowledgeState("UNKNOWN", focus="未学習架空概念")
    )
    check("unknown-block", unknown.action == "BLOCK", str(unknown))

    normal = dispatch_knowledge_state(
        KnowledgeState("NON_CONCEPT")
    )
    check(
        "non-concept-generate",
        normal.action == "GENERATE",
        str(normal),
    )

    print()
    print("Retrieval dispatch     : PASS")
    print("Internalized dispatch  : PASS")
    print("Raw corpus block       : PASS")
    print("Unknown block          : PASS")
    print("Normal chat generation : PASS")
    print("STATUS                 : KNOWLEDGE_STATE_DISPATCHER_PASS")


if __name__ == "__main__":
    main()
