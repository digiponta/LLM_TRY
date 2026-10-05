#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.11.0 Semantic Knowledge Architecture Regression."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from internalized_knowledge_v10100 import pair_fingerprint
from semantic_knowledge_architecture_v10110 import (
    SemanticKnowledgeArchitecture,
    SemanticKnowledgeConfig,
)
from truth_state_v10103 import upsert_truth_record


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (
        f" : {detail}" if detail else ""
    ))
    if not ok:
        raise AssertionError(name)


def main() -> None:
    print("=" * 104)
    print(" LLM_TRY v10.11.0 Semantic Knowledge Architecture Regression")
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

        added = architecture.teach_proposition("GPUは高速である。")
        check("proposition-teach", len(added) == 1, str(added))

        gpu = architecture.resolve("GPUの性質は")
        check(
            "typed-route",
            gpu.state == "TYPED"
            and gpu.action == "RETRIEVE"
            and gpu.truth_state == "UNVERIFIED",
            str(gpu),
        )

        cpu = architecture.resolve("CPUとは")
        check(
            "canonical-route",
            cpu.state == "CANONICAL"
            and cpu.action == "RETRIEVE",
            str(cpu),
        )

        config.unified_path.write_text(
            json.dumps(
                {
                    "concept": "宇宙",
                    "assistant": "宇宙は、時空と物質を含む全体である。",
                    "source": "test-unified",
                },
                ensure_ascii=False,
            ) + "\n",
            encoding="utf-8",
        )
        universe = architecture.resolve("宇宙とは")
        check(
            "unified-route",
            universe.state == "UNIFIED"
            and universe.action == "RETRIEVE",
            str(universe),
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
                    "timestamp": "2026-10-05T20:00:00+0900",
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

        internalized = architecture.resolve(iq)
        check(
            "internalized-route",
            internalized.state == "INTERNALIZED"
            and internalized.action == "GENERATE",
            str(internalized),
        )

        upsert_truth_record(
            config.truth_store_path,
            "量子センサー",
            "FALSE",
            correction="量子センサーの修正説明。",
            source="test",
        )
        corrected = architecture.resolve(iq)
        check(
            "truth-correction",
            corrected.state == "INTERNALIZED"
            and corrected.action == "RETRIEVE"
            and corrected.answer == "量子センサーの修正説明。"
            and corrected.truth_result is not None
            and corrected.truth_result.correction_applied,
            str(corrected),
        )

        config.raw_corpus_path.write_text(
            "時間。時間とは何か。時間の流れ。\n",
            encoding="utf-8",
        )
        raw = architecture.resolve("時間とは")
        check(
            "raw-block",
            raw.state == "RAW_CORPUS_ONLY"
            and raw.action == "BLOCK",
            str(raw),
        )

        sync = architecture.sync_all()
        check(
            "sync-all",
            sync.atomic_count == 1
            and sync.subject_index_count == 1
            and sync.typed_index_count == 1
            and sync.unified_subject_count == 1,
            str(sync),
        )

        status = architecture.status()
        check(
            "architecture-status",
            status.get("version") == "v10.11.0"
            and "truth-state" in status.get("layers", ()),
            str(status),
        )

    print()
    print("Proposition layer     : PASS")
    print("Typed/Unified routing : PASS")
    print("Internalized routing  : PASS")
    print("Truth overlay         : PASS")
    print("Raw isolation         : PASS")
    print("Full synchronization  : PASS")
    print("STATUS                : SEMANTIC_KNOWLEDGE_ARCHITECTURE_PASS")


if __name__ == "__main__":
    main()
