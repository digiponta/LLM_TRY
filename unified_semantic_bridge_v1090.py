#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_TRY v10.9.0 Phase 2
Atomic Proposition Store -> Unified Semantic Memory bridge.

The unified semantic-memory format remains backward compatible with chat.py:
    {"concept": "...", "assistant": "..."}

Additional provenance fields are optional and ignored by the existing lookup.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

from semantic_proposition_v1090 import compose_subject, load_propositions
from typed_subject_proposition_v10100 import classify_predicate_type


def load_unified_rows(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []

    rows: List[Dict[str, object]] = []
    for line_no, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        raw = raw.strip()
        if not raw:
            continue
        try:
            item = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid unified semantic JSON line {line_no}: {exc}"
            ) from exc
        if not isinstance(item, dict):
            raise ValueError(
                f"invalid unified semantic row {line_no}: {item!r}"
            )
        rows.append(item)
    return rows


def save_unified_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def upsert_unified_concept(
    path: Path,
    concept: str,
    assistant: str,
    *,
    source: str = "atomic-proposition",
    atomic_count: int | None = None,
    predicate_types: List[str] | None = None,
) -> Dict[str, object]:
    concept = str(concept).strip()
    assistant = str(assistant).strip()
    if not concept or not assistant:
        raise ValueError("concept and assistant must be non-empty")

    rows = load_unified_rows(path)
    replacement: Dict[str, object] = {
        "concept": concept,
        "assistant": assistant,
        "source": source,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if atomic_count is not None:
        replacement["atomic_count"] = int(atomic_count)
    if predicate_types:
        replacement["predicate_types"] = list(dict.fromkeys(predicate_types))
        replacement["semantic_schema"] = (
            "subject-predicate-type-statement-v10.10.0"
        )

    out: List[Dict[str, object]] = []
    replaced = False
    for row in rows:
        if str(row.get("concept", "")).strip() == concept:
            if not replaced:
                out.append(replacement)
                replaced = True
            # Drop duplicate rows for the same concept while preserving
            # every other concept unchanged.
            continue
        out.append(row)

    if not replaced:
        out.append(replacement)

    save_unified_rows(path, out)
    return replacement


def sync_subject_from_propositions(
    proposition_path: Path,
    unified_path: Path,
    subject: str,
) -> Dict[str, object] | None:
    subject = str(subject).strip()
    if not subject:
        return None

    propositions = [
        p for p in load_propositions(proposition_path)
        if p.subject == subject
    ]
    if not propositions:
        return None

    assistant = compose_subject(proposition_path, subject)
    if not assistant:
        return None

    predicate_types: List[str] = []
    for proposition in propositions:
        predicate_type = classify_predicate_type(proposition.value)
        if predicate_type not in predicate_types:
            predicate_types.append(predicate_type)

    return upsert_unified_concept(
        unified_path,
        subject,
        assistant,
        source="atomic-proposition",
        atomic_count=len(propositions),
        predicate_types=predicate_types,
    )


def sync_all_propositions(
    proposition_path: Path,
    unified_path: Path,
) -> int:
    subjects: List[str] = []
    for proposition in load_propositions(proposition_path):
        if proposition.subject not in subjects:
            subjects.append(proposition.subject)

    synced = 0
    for subject in subjects:
        if sync_subject_from_propositions(
            proposition_path,
            unified_path,
            subject,
        ) is not None:
            synced += 1
    return synced
