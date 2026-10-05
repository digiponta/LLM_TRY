#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.10.2 Provenance End-to-End Regression."""

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
    print(" LLM_TRY v10.10.2 Provenance End-to-End Regression")
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
        sync_subject_index(props, subject)
        sync_typed_index(props, typed)

        unified.write_text(
            json.dumps(
                {
                    "concept": "宇宙",
                    "assistant": "宇宙は、時空と物質を含む全体である。",
                    "source": "atomic-proposition",
                    "updated_at": "2026-10-05T12:00:00+0900",
                    "semantic_schema": "subject-predicate-type-statement-v10.10.0",
                    "atomic_count": 2,
                },
                ensure_ascii=False,
            ) + "\n",
            encoding="utf-8",
        )

        iq = "量子センサーとは"
        ia = "量子センサーは、高感度計測を行うセンサーである。"
        fp = pair_fingerprint(iq, ia)
        log.write_text(
            json.dumps(
                {
                    "user": iq,
                    "assistant": ia,
                    "source": "chat-manual",
                    "timestamp": "2026-10-05T13:00:00+0900",
                },
                ensure_ascii=False,
            ) + "\n",
            encoding="utf-8",
        )
        state.write_text(
            json.dumps(
                {"trained_fingerprints": [fp]},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        corpus.write_text(
            "時間について考える。時間とは何か。時間の流れ。\n",
            encoding="utf-8",
        )

        common = dict(
            typed_index_path=typed,
            subject_index_path=subject,
            unified_path=unified,
            learning_log=log,
            learning_state=state,
            raw_corpus_path=corpus,
            canonical_definitions={
                "cpu": "CPUは、命令を実行する中央処理装置である。",
            },
        )

        typed_state = resolve_knowledge_state("GPUの性質は", **common)
        check(
            "typed-source",
            typed_state.provenance is not None
            and typed_state.provenance.source == "typed-subject-proposition"
            and typed_state.provenance.retrieval_priority == 10,
            str(typed_state),
        )

        canonical_state = resolve_knowledge_state("CPUとは", **common)
        check(
            "canonical-source",
            canonical_state.provenance is not None
            and canonical_state.provenance.source == "canonical-definition"
            and canonical_state.provenance.retrieval_priority == 20,
            str(canonical_state),
        )

        unified_state = resolve_knowledge_state("宇宙とは", **common)
        check(
            "unified-source",
            unified_state.provenance is not None
            and unified_state.provenance.source == "atomic-proposition"
            and unified_state.provenance.timestamp
            == "2026-10-05T12:00:00+0900"
            and unified_state.provenance.metadata.get("atomic_count") == "2",
            str(unified_state),
        )

        internalized_state = resolve_knowledge_state(
            "量子センサーとは",
            **common,
        )
        check(
            "internalized-fingerprint",
            internalized_state.provenance is not None
            and internalized_state.provenance.fingerprint == fp
            and internalized_state.provenance.timestamp
            == "2026-10-05T13:00:00+0900"
            and internalized_state.provenance.retrieval_priority == 40,
            str(internalized_state),
        )

        raw_state = resolve_knowledge_state("時間とは", **common)
        check(
            "raw-origin",
            raw_state.provenance is not None
            and raw_state.provenance.source == "raw-corpus"
            and raw_state.provenance.retrieval_priority == 50,
            str(raw_state),
        )

        unknown_state = resolve_knowledge_state(
            "未学習架空概念とは",
            **common,
        )
        check(
            "unknown-source",
            unknown_state.provenance is not None
            and unknown_state.provenance.source == "none"
            and unknown_state.provenance.retrieval_priority == 60,
            str(unknown_state),
        )

        normal_state = resolve_knowledge_state("こんにちは", **common)
        check(
            "normal-source",
            normal_state.provenance is not None
            and normal_state.provenance.source == "runtime-conversation"
            and normal_state.provenance.retrieval_priority == 70,
            str(normal_state),
        )

        dispatched = dispatch_knowledge_state(internalized_state)
        check(
            "dispatcher-preserves-provenance",
            dispatched.provenance == internalized_state.provenance,
            str(dispatched),
        )

    print()
    print("Typed provenance        : PASS")
    print("Canonical provenance    : PASS")
    print("Unified provenance      : PASS")
    print("Internalized provenance : PASS")
    print("Raw/unknown provenance  : PASS")
    print("Dispatcher propagation  : PASS")
    print("STATUS                  : KNOWLEDGE_PROVENANCE_END_TO_END_PASS")


if __name__ == "__main__":
    main()
