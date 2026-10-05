#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.11.2 Bare Concept Semantic Routing Regression."""

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
    print(" LLM_TRY v10.11.2 Bare Concept Semantic Routing Regression")
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
        architecture.teach_proposition("GPUは並列計算が得意である。")

        config.unified_path.write_text(
            config.unified_path.read_text(encoding="utf-8")
            + json.dumps(
                {
                    "concept": "宇宙",
                    "assistant": "宇宙は、時空と物質を含む全体である。",
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
            "時間。時間。時間。長門。長門。\n",
            encoding="utf-8",
        )

        gpu = architecture.resolve("GPU")
        check(
            "gpu-bare-routed",
            gpu.bare_concept_routed
            and gpu.routed_query == "GPUとは"
            and gpu.state == "TYPED"
            and gpu.action == "RETRIEVE",
            str(gpu),
        )

        cpu = architecture.resolve("CPU")
        check(
            "cpu-bare-routed",
            cpu.bare_concept_routed
            and cpu.routed_query == "CPUとは"
            and cpu.state == "CANONICAL"
            and cpu.action == "RETRIEVE",
            str(cpu),
        )

        universe = architecture.resolve("宇宙")
        check(
            "universe-bare-routed",
            universe.bare_concept_routed
            and universe.routed_query == "宇宙とは"
            and universe.state == "UNIFIED"
            and universe.action == "RETRIEVE",
            str(universe),
        )

        quantum = architecture.resolve("量子センサー")
        check(
            "internalized-bare-routed",
            quantum.bare_concept_routed
            and quantum.routed_query == "量子センサーとは"
            and quantum.state == "INTERNALIZED"
            and quantum.action == "GENERATE",
            str(quantum),
        )

        nagato = architecture.resolve("長門")
        check(
            "persona-not-promoted",
            not nagato.bare_concept_routed
            and nagato.routed_query == "長門"
            and nagato.state == "NON_CONCEPT"
            and nagato.action == "GENERATE",
            str(nagato),
        )

        raw = architecture.resolve("時間")
        check(
            "raw-corpus-not-promoted",
            not raw.bare_concept_routed
            and raw.routed_query == "時間"
            and raw.state == "NON_CONCEPT",
            str(raw),
        )

        unknown = architecture.resolve("架空概念")
        check(
            "unknown-bare-preserves-normal-path",
            not unknown.bare_concept_routed
            and unknown.state == "NON_CONCEPT",
            str(unknown),
        )

    print()
    print("TYPED bare route        : PASS")
    print("CANONICAL bare route    : PASS")
    print("UNIFIED bare route      : PASS")
    print("INTERNALIZED bare route : PASS")
    print("Persona preservation    : PASS")
    print("Raw corpus isolation    : PASS")
    print("Unknown preservation    : PASS")
    print("STATUS                  : BARE_CONCEPT_SEMANTIC_ROUTING_PASS")


if __name__ == "__main__":
    main()
