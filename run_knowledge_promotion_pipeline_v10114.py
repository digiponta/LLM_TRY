#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.11.4 Knowledge Promotion Pipeline Regression."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from knowledge_promotion_v10114 import (
    pending_knowledge_requests,
    promote_knowledge,
)
from semantic_knowledge_architecture_v10110 import (
    SemanticKnowledgeArchitecture,
    SemanticKnowledgeConfig,
)


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (
        f" : {detail}" if detail else ""
    ))
    if not ok:
        raise AssertionError(name)


def main() -> None:
    print("=" * 104)
    print(" LLM_TRY v10.11.4 Knowledge Promotion Pipeline Regression")
    print("=" * 104)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        queue = root / "knowledge_queue.jsonl"
        config = SemanticKnowledgeConfig(
            proposition_path=root / "atomic.jsonl",
            subject_index_path=root / "subject.jsonl",
            typed_index_path=root / "typed.jsonl",
            unified_path=root / "unified.jsonl",
            learning_log=root / "chat_history.jsonl",
            learning_state=root / "chat_learning_state.json",
            raw_corpus_path=root / "corpus.txt",
            truth_store_path=root / "truth.jsonl",
            canonical_definitions={},
        )
        architecture = SemanticKnowledgeArchitecture(config)

        queue.write_text(
            json.dumps(
                {
                    "fingerprint": "test-time",
                    "resolution": "UNKNOWN_KNOWLEDGE",
                    "action": "retrieve/teach",
                    "user": "時間",
                    "candidate_answer": "",
                    "reason": "bare concept lacks validated semantic evidence",
                    "timestamp": "2026-10-05T21:00:00+0900",
                },
                ensure_ascii=False,
            ) + "\n",
            encoding="utf-8",
        )

        before = architecture.resolve("時間")
        check(
            "before-blocked",
            before.bare_unknown_blocked
            and before.state == "UNKNOWN"
            and before.action == "BLOCK",
            str(before),
        )

        bad = promote_knowledge(
            architecture,
            queue,
            "時間",
            "宇宙は、広いものである。",
        )
        check(
            "subject-validation",
            not bad.promoted
            and "subject does not match" in bad.reason,
            str(bad),
        )

        result = promote_knowledge(
            architecture,
            queue,
            "時間",
            "時間は、出来事の順序と間隔を表す概念である。",
        )
        check(
            "promotion-success",
            result.promoted
            and result.queue_resolved == 1
            and len(result.propositions) == 1,
            str(result),
        )

        after = architecture.resolve("時間")
        check(
            "bare-route-enabled",
            after.bare_concept_routed
            and not after.bare_unknown_blocked
            and after.state == "TYPED"
            and after.action == "RETRIEVE",
            str(after),
        )

        check(
            "truth-default-unverified",
            after.truth_state == "UNVERIFIED",
            str(after),
        )

        pending = pending_knowledge_requests(queue)
        check(
            "queue-resolved",
            len(pending) == 0,
            str(pending),
        )

        no_queue = promote_knowledge(
            architecture,
            queue,
            "架空概念",
            "架空概念は、試験用の概念である。",
        )
        check(
            "pending-required",
            not no_queue.promoted
            and "no pending" in no_queue.reason,
            str(no_queue),
        )

    print()
    print("Unknown intake        : PASS")
    print("Structural validation : PASS")
    print("Semantic promotion    : PASS")
    print("Bare routing enable   : PASS")
    print("Truth stays unverified: PASS")
    print("Queue resolution      : PASS")
    print("STATUS                : KNOWLEDGE_PROMOTION_PIPELINE_PASS")


if __name__ == "__main__":
    main()
