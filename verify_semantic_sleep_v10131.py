#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Regression for LLM_TRY v10.13.1 Semantic Sleep Learning."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

from semantic_sleep_learning_v10131 import (
    bootstrap_nagato_semantic_memory,
    load_nagato_semantic_memory,
    prepare_semantic_sleep_pairs,
    semantic_sleep_status,
)
from subject_keyed_corpus_memory_v101216 import (
    CorpusMemoryRecord,
    save_memory,
)
from unified_semantic_bridge_v1090 import (
    load_unified_rows,
    save_unified_rows,
)


def check(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)
    print(f"[PASS] {name}")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        raw = root / "data-nagato.txt"
        corpus = root / "corpus.jsonl"
        semantic = root / "nagato-semantic.jsonl"
        unified = root / "unified.jsonl"
        log = root / "chat_history.jsonl"
        state = root / "chat_learning_state.json"

        raw.write_text("dummy\n", encoding="utf-8")
        save_memory(
            corpus,
            [
                CorpusMemoryRecord(
                    subject="長門",
                    statement="長門は情報統合思念体に関係する。",
                    relation="definition",
                    object_description="",
                    source_text="長門は情報統合思念体に関係する。",
                    source=str(raw),
                ),
                CorpusMemoryRecord(
                    subject="長門",
                    statement="長門は読書を好む。",
                    relation="property",
                    object_description="",
                    source_text="長門は読書を好む。",
                    source=str(raw),
                ),
                CorpusMemoryRecord(
                    subject="宇宙",
                    statement="宇宙は時間と空間を含む。",
                    relation="definition",
                    object_description="",
                    source_text="宇宙は時間と空間を含む。",
                    source=str(raw),
                ),
            ],
        )
        save_unified_rows(
            unified,
            [
                {
                    "concept": "長門",
                    "assistant": "manual preserved",
                    "source": "chat-manual",
                }
            ],
        )

        sync = bootstrap_nagato_semantic_memory(
            raw,
            corpus,
            semantic,
            unified,
        )
        check("semantic-memory-subjects", sync.memory_subjects == 2)
        check("unified-preserve-existing", sync.unified_preserved == 1)
        check("unified-add-missing", sync.unified_added == 1)

        semantic_rows = load_nagato_semantic_memory(semantic)
        check("semantic-memory-written", len(semantic_rows) == 2)
        check(
            "full-proposition-composed",
            "読書を好む" in next(
                row["assistant"]
                for row in semantic_rows
                if row["concept"] == "長門"
            ),
        )

        queued, status = prepare_semantic_sleep_pairs(
            semantic,
            log,
            state,
            frozenset(),
        )
        check("sleep-pairs-queued", queued == 2)
        check("sleep-pending-before-training", status.pending == 2)

        rows = [
            json.loads(line)
            for line in log.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        check(
            "sleep-source-tagged",
            all(row.get("source") == "semantic-sleep" for row in rows),
        )

        first_fp = semantic_rows[0]["fingerprint"]
        state.write_text(
            json.dumps(
                {
                    "version": "v1.6.2",
                    "trained_fingerprints": [first_fp],
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        status = semantic_sleep_status(
            semantic,
            state,
            {first_fp},
        )
        check("checkpoint-bound-internalized", status.internalized == 1)
        check("checkpoint-bound-pending", status.pending == 1)

    print("STATUS : SEMANTIC_SLEEP_V10131_PASS")


if __name__ == "__main__":
    main()
