#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_TRY v10.9.0 Subject-Keyed Semantic Proposition Index

Canonical interpretation:
    "GPUは高速である。"
        ->
    subject   = "GPU"
    statement = "GPUは高速である。"

Conceptually:
    GPU => GPUは高速である。

The atomic proposition store remains the source of truth.  This module builds
and maintains a derived subject-keyed index for direct subject lookup.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Dict, Iterable, List

from semantic_proposition_v1090 import (
    Proposition,
    compose_propositions,
    load_propositions,
    normalize_fragment,
)


@dataclass(frozen=True)
class SubjectStatement:
    subject: str
    statement: str
    value: str

    def normalized(self) -> "SubjectStatement":
        subject = normalize_fragment(self.subject)
        value = normalize_fragment(self.value)
        return SubjectStatement(
            subject=subject,
            statement=render_atomic_statement(subject, value),
            value=value,
        )


def render_atomic_statement(subject: str, value: str) -> str:
    subject = normalize_fragment(subject)
    value = normalize_fragment(value)
    if not subject or not value:
        return ""
    return f"{subject}は{value}である。"


def proposition_to_subject_statement(
    proposition: Proposition,
) -> SubjectStatement:
    p = proposition.normalized()
    return SubjectStatement(
        subject=p.subject,
        statement=render_atomic_statement(p.subject, p.value),
        value=p.value,
    )


def build_subject_index(
    propositions: Iterable[Proposition],
) -> List[SubjectStatement]:
    rows: List[SubjectStatement] = []
    seen = set()

    for proposition in propositions:
        row = proposition_to_subject_statement(proposition)
        key = (row.subject, row.statement)
        if not row.subject or not row.statement or key in seen:
            continue
        seen.add(key)
        rows.append(row)

    return rows


def load_subject_index(path: Path) -> List[SubjectStatement]:
    if not path.exists():
        return []

    rows: List[SubjectStatement] = []
    seen = set()

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
                f"invalid subject index JSON line {line_no}: {exc}"
            ) from exc

        subject = normalize_fragment(item.get("subject", ""))
        value = normalize_fragment(item.get("value", ""))
        statement = str(item.get("statement", "")).strip()

        if not statement and subject and value:
            statement = render_atomic_statement(subject, value)

        if not subject or not statement:
            raise ValueError(
                f"invalid subject index line {line_no}: {item!r}"
            )

        if not value:
            # Older rows may omit value. Keep lookup compatibility, but value
            # is required for newly generated rows.
            value = statement

        row = SubjectStatement(subject, statement, value)
        key = (row.subject, row.statement)
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)

    return rows


def save_subject_index(
    path: Path,
    rows: Iterable[SubjectStatement],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    normalized: List[SubjectStatement] = []
    seen = set()
    for item in rows:
        row = item.normalized()
        key = (row.subject, row.statement)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(row)

    with path.open("w", encoding="utf-8") as f:
        for row in normalized:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def sync_subject_index(
    proposition_path: Path,
    index_path: Path,
) -> int:
    """Rebuild the complete derived index from the atomic source of truth."""
    rows = build_subject_index(load_propositions(proposition_path))
    save_subject_index(index_path, rows)
    return len(rows)


def subject_statements(
    index_path: Path,
    subject: str,
) -> List[SubjectStatement]:
    target = normalize_fragment(subject)
    return [
        row for row in load_subject_index(index_path)
        if row.subject == target
    ]


def subject_mapping(
    index_path: Path,
    subject: str,
) -> List[str]:
    """Return conceptual 'subject => statement' strings."""
    rows = subject_statements(index_path, subject)
    return [
        f"{row.subject} => {row.statement}"
        for row in rows
    ]


def subject_index_dict(
    index_path: Path,
) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    for row in load_subject_index(index_path):
        result.setdefault(row.subject, []).append(row.statement)
    return result


def compose_subject_from_index(
    index_path: Path,
    subject: str,
) -> str:
    """Compose a subject's indexed atomic statements into one canonical answer."""
    rows = subject_statements(index_path, subject)
    propositions = [
        Proposition(row.subject, row.value)
        for row in rows
    ]
    return compose_propositions(propositions, subject=subject)
