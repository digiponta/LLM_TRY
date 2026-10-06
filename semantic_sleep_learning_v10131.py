#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.13.1 Semantic Sleep Learning.

Phase A (memory):
    data-nagato subject-keyed corpus memory
      -> dedicated Nagato Semantic Memory
      -> missing subjects synchronized into Unified Semantic Memory

Phase B (/sleep):
    pending Semantic Memory entries
      -> trusted semantic-sleep QA pairs
      -> existing incremental trainer
      -> checkpoint-bound INTERNALIZED knowledge

The memory phase is immediate and does not mutate model weights.
The sleep phase is explicit and only queues pairs not already consumed by the
shared learning-state fingerprints.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Iterable

from internalized_knowledge_v10100 import (
    load_trained_fingerprints,
    pair_fingerprint,
)
from subject_keyed_corpus_memory_v101216 import load_memory
from unified_semantic_bridge_v1090 import (
    load_unified_rows,
    save_unified_rows,
)

DEFAULT_NAGATO_SEMANTIC_MEMORY = "data/nagato_semantic_memory_v10131.jsonl"


@dataclass(frozen=True)
class SemanticSleepStatus:
    memory_subjects: int
    internalized: int
    pending: int


@dataclass(frozen=True)
class SemanticMemorySync:
    corpus_records: int
    memory_subjects: int
    unified_added: int
    unified_preserved: int


def _group_corpus(memory_path: Path) -> list[dict]:
    rows = load_memory(memory_path)
    grouped: dict[str, dict] = {}
    order: list[str] = []

    for row in rows:
        if row.subject not in grouped:
            grouped[row.subject] = {
                "concept": row.subject,
                "statements": [],
                "relations": [],
                "sources": [],
            }
            order.append(row.subject)

        item = grouped[row.subject]
        if row.statement not in item["statements"]:
            item["statements"].append(row.statement)
        if row.relation and row.relation not in item["relations"]:
            item["relations"].append(row.relation)
        if row.source and row.source not in item["sources"]:
            item["sources"].append(row.source)

    out = []
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    for subject in order:
        item = grouped[subject]
        answer = " / ".join(item["statements"])
        question = f"{subject}とは"
        out.append({
            "concept": subject,
            "assistant": answer,
            "question": question,
            "source": "data-nagato-semantic-memory",
            "provenance": "data/data-nagato.txt",
            "relations": item["relations"],
            "source_files": item["sources"],
            "fingerprint": pair_fingerprint(question, answer),
            "updated_at": now,
            "version": "v10.13.1",
        })
    return out


def save_nagato_semantic_memory(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_nagato_semantic_memory(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid Nagato Semantic Memory line {line_no}: {exc}"
            ) from exc
        if isinstance(row, dict):
            out.append(row)
    return out


def bootstrap_nagato_semantic_memory(
    corpus_memory_path: Path,
    semantic_memory_path: Path,
    unified_path: Path,
) -> SemanticMemorySync:
    rows = _group_corpus(corpus_memory_path)
    save_nagato_semantic_memory(semantic_memory_path, rows)

    unified = load_unified_rows(unified_path)
    by_concept = {
        str(row.get("concept", "")).strip(): row
        for row in unified
        if str(row.get("concept", "")).strip()
    }

    added = 0
    preserved = 0
    for row in rows:
        concept = str(row["concept"])
        if concept in by_concept:
            # Never overwrite promoted/manual/atomic semantic knowledge.
            preserved += 1
            continue
        unified_row = {
            "concept": concept,
            "assistant": row["assistant"],
            "source": "data-nagato-semantic-memory",
            "provenance": row["provenance"],
            "relations": row["relations"],
            "semantic_schema": "subject-full-proposition-v10.13.1",
            "updated_at": row["updated_at"],
        }
        unified.append(unified_row)
        by_concept[concept] = unified_row
        added += 1

    save_unified_rows(unified_path, unified)
    return SemanticMemorySync(
        corpus_records=len(load_memory(corpus_memory_path)),
        memory_subjects=len(rows),
        unified_added=added,
        unified_preserved=preserved,
    )


def semantic_sleep_status(
    semantic_memory_path: Path,
    learning_state_path: Path,
) -> SemanticSleepStatus:
    rows = load_nagato_semantic_memory(semantic_memory_path)
    trained = load_trained_fingerprints(learning_state_path)
    internalized = sum(
        1
        for row in rows
        if str(row.get("fingerprint", "")) in trained
    )
    return SemanticSleepStatus(
        memory_subjects=len(rows),
        internalized=internalized,
        pending=max(0, len(rows) - internalized),
    )


def prepare_semantic_sleep_pairs(
    semantic_memory_path: Path,
    learning_log_path: Path,
    learning_state_path: Path,
) -> tuple[int, SemanticSleepStatus]:
    rows = load_nagato_semantic_memory(semantic_memory_path)
    trained = load_trained_fingerprints(learning_state_path)

    existing = set()
    if learning_log_path.exists():
        for raw in learning_log_path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            try:
                old = json.loads(raw)
            except json.JSONDecodeError:
                continue
            user = str(old.get("user", "")).strip()
            answer = str(old.get("assistant", "")).strip()
            if user and answer:
                existing.add(pair_fingerprint(user, answer))

    learning_log_path.parent.mkdir(parents=True, exist_ok=True)
    queued = 0
    with learning_log_path.open("a", encoding="utf-8") as handle:
        for row in rows:
            question = str(row.get("question", "")).strip()
            answer = str(row.get("assistant", "")).strip()
            fingerprint = str(row.get("fingerprint", "")).strip()
            if not question or not answer or not fingerprint:
                continue
            if fingerprint in trained or fingerprint in existing:
                continue
            payload = {
                "user": question,
                "assistant": answer,
                "source": "semantic-sleep",
                "semantic_origin": "data-nagato-semantic-memory",
                "fingerprint": fingerprint,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            existing.add(fingerprint)
            queued += 1

    return queued, semantic_sleep_status(
        semantic_memory_path,
        learning_state_path,
    )
