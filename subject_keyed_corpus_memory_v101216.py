#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.12.16 Subject-Keyed Corpus Memory.

Builds a persistent exact-subject knowledge store from the source-grounded
Corpus-to-Semantic dataset representation.

Canonical mapping:
    subject => full proposition
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Iterable

from semantic_role_generalization_v101210 import normalize


DEFAULT_MEMORY = "data/subject_keyed_corpus_memory_v101216.jsonl"


@dataclass(frozen=True)
class CorpusMemoryRecord:
    subject: str
    statement: str
    relation: str
    object_description: str
    source_text: str
    source: str

    @property
    def mapping(self) -> str:
        return f"{self.subject} => {self.statement}"


def normalize_record(row: dict) -> CorpusMemoryRecord:
    subject = normalize(row.get("subject", ""))
    statement = normalize(row.get("answer", row.get("statement", "")))
    relation = normalize(row.get("relation", ""))
    object_description = normalize(row.get("object_description", ""))
    source_text = normalize(row.get("source_text", statement))
    source = str(row.get("source", "")).strip()
    if not subject or not statement:
        raise ValueError(f"invalid corpus memory row: {row!r}")
    return CorpusMemoryRecord(
        subject=subject,
        statement=statement,
        relation=relation,
        object_description=object_description,
        source_text=source_text,
        source=source,
    )


def dedupe_records(rows: Iterable[CorpusMemoryRecord]) -> list[CorpusMemoryRecord]:
    out = []
    seen = set()
    for row in rows:
        key = (row.subject, row.statement)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def save_memory(path: Path, rows: Iterable[CorpusMemoryRecord]) -> None:
    records = dedupe_records(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in records:
            payload = asdict(row)
            payload["mapping"] = row.mapping
            payload["version"] = "v10.12.16"
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_memory(path: Path) -> list[CorpusMemoryRecord]:
    if not path.exists():
        return []
    rows = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            item = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid corpus memory JSON line {line_no}: {exc}") from exc
        rows.append(normalize_record(item))
    return dedupe_records(rows)


def lookup_subject(path: Path, subject: str) -> list[CorpusMemoryRecord]:
    target = normalize(subject)
    return [row for row in load_memory(path) if row.subject == target]


def compose_subject_evidence(path: Path, subject: str) -> str:
    rows = lookup_subject(path, subject)
    statements = []
    for row in rows:
        if row.statement not in statements:
            statements.append(row.statement)
    return " / ".join(statements)


def subject_mapping(path: Path, subject: str) -> list[str]:
    return [row.mapping for row in lookup_subject(path, subject)]
