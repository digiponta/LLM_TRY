#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.11.1 Semantic Knowledge Snapshot Regression."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from internalized_knowledge_v10100 import pair_fingerprint
from semantic_knowledge_architecture_v10110 import (
    SemanticKnowledgeArchitecture,
    SemanticKnowledgeConfig,
)
from semantic_knowledge_snapshot_v10111 import snapshot_semantic_knowledge
from truth_state_v10103 import upsert_truth_record


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (
        f" : {detail}" if detail else ""
    ))
    if not ok:
        raise AssertionError(name)


def main() -> None:
    print("=" * 104)
    print(" LLM_TRY v10.11.1 Semantic Knowledge Snapshot Regression")
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
                "cpu": "CPUは中央処理装置である。",
            },
        )
        architecture = SemanticKnowledgeArchitecture(config)

        architecture.teach_proposition("GPUは高速である。")
        architecture.teach_proposition("GPUは並列計算が得意である。")
        gpu = snapshot_semantic_knowledge(architecture, "GPU")

        check("gpu-propositions", gpu.proposition_count == 2, str(gpu))
        check("gpu-subject-answer", bool(gpu.subject_answer), str(gpu))
        check("gpu-typed", gpu.typed_count == 2, str(gpu))
        check("gpu-unified", gpu.unified_present, str(gpu))
        check(
            "gpu-result",
            gpu.result.knowledge_state.state == "TYPED"
            and gpu.final_action == "RETRIEVE",
            str(gpu),
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

        upsert_truth_record(
            config.truth_store_path,
            "量子センサー",
            "FALSE",
            correction="量子センサーの修正説明。",
            source="test",
        )

        quantum = snapshot_semantic_knowledge(
            architecture,
            "量子センサー",
        )
        check(
            "internalized-present",
            quantum.internalized_present,
            str(quantum),
        )
        check(
            "truth-visible",
            quantum.truth_state == "FALSE",
            str(quantum),
        )
        check(
            "final-correction",
            quantum.final_action == "RETRIEVE"
            and quantum.result.answer == "量子センサーの修正説明。",
            str(quantum),
        )
        check(
            "provenance-visible",
            quantum.result.provenance is not None
            and quantum.result.provenance.fingerprint == fp,
            str(quantum),
        )

    print()
    print("Proposition inspection : PASS")
    print("Typed inspection       : PASS")
    print("Unified inspection     : PASS")
    print("Internalized inspection: PASS")
    print("Provenance inspection  : PASS")
    print("Truth inspection       : PASS")
    print("Final dispatch         : PASS")
    print("STATUS                 : SEMANTIC_KNOWLEDGE_SNAPSHOT_PASS")


if __name__ == "__main__":
    main()
