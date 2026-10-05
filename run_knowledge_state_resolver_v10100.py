#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.10.0 Knowledge State Resolver Regression."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from internalized_knowledge_v10100 import pair_fingerprint
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
    print("=" * 96)
    print(" LLM_TRY v10.10.0 Knowledge State Resolver Regression")
    print("=" * 96)

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
                    "source": "test",
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

        typed_state = resolve_knowledge_state("GPUの性質は", **common)
        check("typed", typed_state.state == "TYPED", str(typed_state))

        canonical_state = resolve_knowledge_state("CPUとは", **common)
        check(
            "canonical",
            canonical_state.state == "CANONICAL",
            str(canonical_state),
        )

        unified_state = resolve_knowledge_state("宇宙とは", **common)
        check(
            "unified",
            unified_state.state == "UNIFIED",
            str(unified_state),
        )

        internalized_state = resolve_knowledge_state(
            "量子センサーとは",
            **common,
        )
        check(
            "internalized",
            internalized_state.state == "INTERNALIZED",
            str(internalized_state),
        )

        raw_state = resolve_knowledge_state("時間とは", **common)
        check(
            "raw-corpus-only",
            raw_state.state == "RAW_CORPUS_ONLY",
            str(raw_state),
        )
        check(
            "raw-does-not-generate",
            not raw_state.permits_model_generation,
            str(raw_state),
        )

        unknown_state = resolve_knowledge_state("未学習架空概念とは", **common)
        check(
            "unknown",
            unknown_state.state == "UNKNOWN",
            str(unknown_state),
        )

        non_concept = resolve_knowledge_state("こんにちは", **common)
        check(
            "non-concept",
            non_concept.state == "NON_CONCEPT",
            str(non_concept),
        )

    print()
    print("Typed state          : PASS")
    print("Canonical state      : PASS")
    print("Unified state        : PASS")
    print("Internalized state   : PASS")
    print("Raw-corpus isolation : PASS")
    print("Unknown state        : PASS")
    print("STATUS               : KNOWLEDGE_STATE_RESOLVER_PASS")


if __name__ == "__main__":
    main()
