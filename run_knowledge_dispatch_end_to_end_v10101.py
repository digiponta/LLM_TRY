#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.10.1 Resolver -> Dispatcher End-to-End Regression."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from internalized_knowledge_v10100 import pair_fingerprint
from knowledge_state_dispatcher_v10101 import dispatch_knowledge_state
from knowledge_state_resolver_v10100 import resolve_knowledge_state
from semantic_proposition_v1090 import add_statement
from subject_keyed_proposition_v1090 import sync_subject_index
from typed_subject_proposition_v10100 import sync_typed_index


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (
        f" : {detail}" if detail else ""
    ))
    if not ok:
        raise AssertionError(name)


def main() -> None:
    print("=" * 100)
    print(" LLM_TRY v10.10.1 Resolver -> Dispatcher End-to-End Regression")
    print("=" * 100)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        props = root / "atomic.jsonl"
        subject = root / "subject.jsonl"
        typed = root / "typed.jsonl"
        unified = root / "unified.jsonl"
        log = root / "chat_history.jsonl"
        state = root / "chat_learning_state.json"
        corpus = root / "corpus.txt"

        add_statement(props, "GPUは高速である。")
        add_statement(props, "GPUは並列計算が得意である。")
        sync_subject_index(props, subject)
        sync_typed_index(props, typed)

        unified.write_text(
            json.dumps(
                {
                    "concept": "宇宙",
                    "assistant": "宇宙は、時空と物質を含む全体である。",
                },
                ensure_ascii=False,
            ) + "\n",
            encoding="utf-8",
        )

        iq = "量子センサーとは"
        ia = "量子センサーは、高感度計測を行うセンサーである。"
        log.write_text(
            json.dumps(
                {
                    "user": iq,
                    "assistant": ia,
                    "source": "chat-manual",
                },
                ensure_ascii=False,
            ) + "\n",
            encoding="utf-8",
        )
        state.write_text(
            json.dumps(
                {
                    "trained_fingerprints": [
                        pair_fingerprint(iq, ia),
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        corpus.write_text(
            "時間について考える。時間とは何か。時間の流れ。\n",
            encoding="utf-8",
        )

        canonical = {
            "cpu": "CPUは、命令を実行する中央処理装置である。",
        }

        common = dict(
            typed_index_path=typed,
            subject_index_path=subject,
            unified_path=unified,
            learning_log=log,
            learning_state=state,
            raw_corpus_path=corpus,
            canonical_definitions=canonical,
        )

        cases = (
            ("GPUの性質は", "TYPED", "RETRIEVE"),
            ("CPUとは", "CANONICAL", "RETRIEVE"),
            ("宇宙とは", "UNIFIED", "RETRIEVE"),
            ("量子センサーとは", "INTERNALIZED", "GENERATE"),
            ("時間とは", "RAW_CORPUS_ONLY", "BLOCK"),
            ("未学習架空概念とは", "UNKNOWN", "BLOCK"),
            ("こんにちは", "NON_CONCEPT", "GENERATE"),
        )

        for query, expected_state, expected_action in cases:
            resolved = resolve_knowledge_state(query, **common)
            dispatched = dispatch_knowledge_state(resolved)
            check(
                f"{expected_state}:{expected_action}",
                resolved.state == expected_state
                and dispatched.state == expected_state
                and dispatched.action == expected_action,
                f"resolved={resolved}, dispatched={dispatched}",
            )

    print()
    print("Resolver priority       : PASS")
    print("Dispatcher action map   : PASS")
    print("Retrieval isolation     : PASS")
    print("Generation isolation    : PASS")
    print("Blocked state isolation : PASS")
    print("STATUS                  : KNOWLEDGE_DISPATCH_END_TO_END_PASS")


if __name__ == "__main__":
    main()
