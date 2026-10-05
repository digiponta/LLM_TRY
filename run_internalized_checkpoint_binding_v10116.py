#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.11.6 Internalized Checkpoint Binding Regression."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

from internalized_knowledge_v10100 import pair_fingerprint
from model import LanguageModel
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


def make_config(
    root: Path,
    checkpoint_fingerprints: frozenset[str] | None,
) -> SemanticKnowledgeConfig:
    return SemanticKnowledgeConfig(
        proposition_path=root / "atomic.jsonl",
        subject_index_path=root / "subject.jsonl",
        typed_index_path=root / "typed.jsonl",
        unified_path=root / "unified.jsonl",
        learning_log=root / "chat_history.jsonl",
        learning_state=root / "chat_learning_state.json",
        raw_corpus_path=root / "corpus.txt",
        truth_store_path=root / "truth.jsonl",
        canonical_definitions={},
        checkpoint_fingerprints=checkpoint_fingerprints,
    )


def main() -> None:
    print("=" * 104)
    print(" LLM_TRY v10.11.6 Internalized Checkpoint Binding Regression")
    print("=" * 104)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)

        question = "量子センサーとは"
        answer = "量子センサーは、量子的性質を利用して高感度計測を行うセンサーである。"
        fp = pair_fingerprint(question, answer)

        (root / "chat_history.jsonl").write_text(
            json.dumps(
                {
                    "user": question,
                    "assistant": answer,
                    "source": "chat-manual",
                    "timestamp": "2026-10-05T20:25:54+0900",
                },
                ensure_ascii=False,
            ) + "\n",
            encoding="utf-8",
        )
        (root / "chat_learning_state.json").write_text(
            json.dumps(
                {"trained_fingerprints": [fp]},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (root / "corpus.txt").write_text(
            "量子センサー。量子センサー。\n",
            encoding="utf-8",
        )

        stale_arch = SemanticKnowledgeArchitecture(
            make_config(root, frozenset())
        )
        stale = stale_arch.resolve("量子センサー")
        check(
            "stale-detected",
            stale.bare_concept_routed
            and stale.state == "INTERNALIZED_STALE"
            and stale.action == "BLOCK",
            str(stale),
        )
        check(
            "stale-provenance",
            stale.provenance is not None
            and stale.provenance.metadata.get("checkpoint_bound") == "False",
            stale.provenance.compact() if stale.provenance else "",
        )

        current_arch = SemanticKnowledgeArchitecture(
            make_config(root, frozenset({fp}))
        )
        current = current_arch.resolve("量子センサー")
        check(
            "current-bound",
            current.bare_concept_routed
            and current.state == "INTERNALIZED"
            and current.action == "GENERATE",
            str(current),
        )
        check(
            "current-provenance",
            current.provenance is not None
            and current.provenance.metadata.get("checkpoint_bound") == "True",
            current.provenance.compact() if current.provenance else "",
        )

        legacy_arch = SemanticKnowledgeArchitecture(
            make_config(root, None)
        )
        legacy = legacy_arch.resolve("量子センサー")
        check(
            "legacy-compatible",
            legacy.state == "INTERNALIZED"
            and legacy.action == "GENERATE",
            str(legacy),
        )

        model = LanguageModel(
            vocab_size=32,
            d_model=16,
            num_layers=1,
            hidden_dim=32,
            num_heads=4,
            context_length=16,
        )
        checkpoint_path = root / "bound.pt"
        model.save_checkpoint(
            str(checkpoint_path),
            epoch=1,
            loss=1.0,
            metadata={
                "knowledge_binding_version": "v10.11.6",
                "trained_fingerprints": [fp],
            },
        )
        _, loaded = LanguageModel.load_checkpoint(
            str(checkpoint_path),
            device=torch.device("cpu"),
        )
        metadata = loaded.get("metadata", {})
        check(
            "checkpoint-metadata-roundtrip",
            metadata.get("knowledge_binding_version") == "v10.11.6"
            and fp in metadata.get("trained_fingerprints", []),
            str(metadata),
        )

    print()
    print("Stale detection       : PASS")
    print("Stale generation block: PASS")
    print("Current binding       : PASS")
    print("Legacy compatibility  : PASS")
    print("Metadata roundtrip    : PASS")
    print("STATUS                : INTERNALIZED_CHECKPOINT_BINDING_PASS")


if __name__ == "__main__":
    main()
