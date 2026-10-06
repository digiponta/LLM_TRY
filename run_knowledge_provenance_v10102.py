#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.10.2 Provenance Model Regression."""

from knowledge_provenance_v10102 import (
    RETRIEVAL_PRIORITY,
    provenance_for_state,
)


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (
        f" : {detail}" if detail else ""
    ))
    if not ok:
        raise AssertionError(name)


def main() -> None:
    print("=" * 96)
    print(" LLM_TRY v10.10.2 Knowledge Provenance Regression")
    print("=" * 96)

    expected = {
        "TYPED": 10,
        "CANONICAL": 20,
        "UNIFIED": 30,
        "INTERNALIZED": 40,
        "RAW_CORPUS_ONLY": 50,
        "UNKNOWN": 60,
        "NON_CONCEPT": 70,
    }
    check("priority-map", RETRIEVAL_PRIORITY == expected, str(RETRIEVAL_PRIORITY))

    p = provenance_for_state(
        "INTERNALIZED",
        source="chat-manual",
        origin="data/chat_history.jsonl",
        timestamp="2026-10-05T12:00:00+0900",
        fingerprint="abc123",
        evidence="trained fingerprint evidence",
        metadata={"trained_pairs": "1"},
    )
    check("source", p.source == "chat-manual", str(p))
    check("origin", p.origin.endswith("chat_history.jsonl"), str(p))
    check("timestamp", bool(p.timestamp), str(p))
    check("fingerprint", p.fingerprint == "abc123", str(p))
    check("priority", p.retrieval_priority == 40, str(p))
    check("metadata", p.metadata.get("trained_pairs") == "1", str(p))

    print()
    print("Priority model      : PASS")
    print("Source tracking     : PASS")
    print("Origin tracking     : PASS")
    print("Fingerprint tracking: PASS")
    print("STATUS              : KNOWLEDGE_PROVENANCE_PASS")


if __name__ == "__main__":
    main()
