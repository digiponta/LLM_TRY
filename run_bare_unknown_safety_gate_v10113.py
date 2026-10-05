#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.11.3 Bare Unknown Safety Gate Regression."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from internalized_knowledge_v10100 import pair_fingerprint
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
    print(" LLM_TRY v10.11.3 Bare Unknown Safety Gate Regression")
    print("=" * 104)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        config = SemanticKnowledgeConfig(
            proposition_path=root / "atomic.jsonl",
            subject_index_path=root / "subject.jsonl",
            typed_index_path=root / "typed.jsonl",
            unified_path=root / "unified.jsonl",
            learning_log=root / "chat_history.jsonl",
            learning_state=root / "chat_learning_state.json",
            raw_corpus_path=root / "corpus.txt",
            truth_store_path=root / "truth.jsonl",
            canonical_definitions={
                "cpu": "CPUは、命令を実行する中央処理装置である。",
            },
        )
        architecture = SemanticKnowledgeArchitecture(config)

        architecture.teach_proposition("GPUは高速である。")

        config.unified_path.write_text(
            config.unified_path.read_text(encoding="utf-8")
            + json.dumps(
                {
                    "concept": "宇宙",
                    "assistant": "宇宙は全体である。",
                    "source": "test-unified",
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        iq = "量子センサーとは"
        ia = "量子センサーは、高感度計測を行うセンサーである。"
        fp = pair_fingerprint(iq, ia)
        config.learning_log.write_text(
            json.dumps(
                {
                    "user": iq,
                    "assistant": ia,
                    "source": "chat-manual",
                    "timestamp": "2026-10-05T20:25:54+0900",
                },
                ensure_ascii=False,
            ) + "\n",
            encoding="utf-8",
        )
        config.learning_state.write_text(
            json.dumps(
                {"trained_fingerprints": [fp]},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        config.raw_corpus_path.write_text(
            "時間。時間。時間。未知語。未知語。\n",
            encoding="utf-8",
        )

        for query, expected_state in (
            ("GPU", "TYPED"),
            ("CPU", "CANONICAL"),
            ("宇宙", "UNIFIED"),
            ("量子センサー", "INTERNALIZED"),
        ):
            result = architecture.resolve(query)
            check(
                f"known-bare:{query}",
                result.bare_concept_routed
                and not result.bare_unknown_blocked
                and result.state == expected_state,
                str(result),
            )

        nagato = architecture.resolve("長門")
        check(
            "persona-preserved",
            not nagato.bare_concept_routed
            and not nagato.bare_unknown_blocked
            and nagato.state == "NON_CONCEPT"
            and nagato.action == "GENERATE",
            str(nagato),
        )

        time_bare = architecture.resolve("時間")
        check(
            "raw-bare-blocked",
            time_bare.bare_unknown_blocked
            and time_bare.bare_focus == "時間"
            and time_bare.state == "UNKNOWN"
            and time_bare.action == "BLOCK",
            str(time_bare),
        )

        unknown = architecture.resolve("架空概念")
        check(
            "unknown-bare-blocked",
            unknown.bare_unknown_blocked
            and unknown.state == "UNKNOWN"
            and unknown.action == "BLOCK",
            str(unknown),
        )

        explicit_time = architecture.resolve("時間とは")
        check(
            "explicit-raw-path-preserved",
            not explicit_time.bare_unknown_blocked
            and explicit_time.state == "RAW_CORPUS_ONLY"
            and explicit_time.action == "BLOCK",
            str(explicit_time),
        )

        hello = architecture.resolve("hello")
        check(
            "hello-preserved",
            not hello.bare_unknown_blocked
            and hello.state == "NON_CONCEPT"
            and hello.action == "GENERATE",
            str(hello),
        )

    print()
    print("Validated bare routing : PASS")
    print("Persona preservation   : PASS")
    print("Raw bare safety block  : PASS")
    print("Unknown bare block     : PASS")
    print("Explicit raw path      : PASS")
    print("Greeting preservation  : PASS")
    print("STATUS                 : BARE_UNKNOWN_SAFETY_GATE_PASS")


if __name__ == "__main__":
    main()
