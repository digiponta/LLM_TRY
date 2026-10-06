#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_TRY v10.9.0 Semantic Proposition Compose / Decompose

Imported from the validated LLM_SEM v0.18.0 design, adapted for LLM_TRY.

Canonical atomic representation:
    Proposition(subject="X", value="Y")

Bidirectional normalization:
    XはYである。
    XはZである。
        -> Xは、Yであり、Zである。

    Xは、Yであり、Zである。
        -> Proposition("X","Y"), Proposition("X","Z")

This store is independent from model weights and the existing unified semantic
memory. It is intended as a normalization layer that can later feed the
existing LLM_TRY knowledge/relation pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


TRAILING = "。．.!！?？"
LEADING_SEPARATORS = "、，, "


@dataclass(frozen=True)
class Proposition:
    subject: str
    value: str

    def normalized(self) -> "Proposition":
        return Proposition(
            normalize_fragment(self.subject),
            normalize_fragment(self.value),
        )


def normalize_fragment(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).strip()
    return text.strip(TRAILING).strip()


def decompose_statement(text: str) -> List[Proposition]:
    """Parse the canonical Japanese copular proposition grammar.

    Supported:
      XはYである
      Xは、Yである。
      Xは、Yであり、Zである。
      XはYでありZである
    """
    normalized = unicodedata.normalize("NFKC", str(text)).strip()
    normalized = normalized.rstrip(TRAILING).strip()

    if "は" not in normalized:
        return []

    subject, body = normalized.split("は", 1)
    subject = normalize_fragment(subject)
    body = body.lstrip(LEADING_SEPARATORS).strip()

    if not subject or not body or not body.endswith("である"):
        return []

    body = body[:-len("である")].strip()
    if not body:
        return []

    raw_parts = re.split(r"\s*であり\s*(?:、|，|,)?\s*", body)
    values: List[str] = []
    for raw in raw_parts:
        value = normalize_fragment(raw.lstrip(LEADING_SEPARATORS))
        if value and value not in values:
            values.append(value)

    return [Proposition(subject, value) for value in values]


def compose_propositions(
    propositions: Sequence[Proposition],
    subject: str | None = None,
) -> str:
    rows = [p.normalized() for p in propositions]

    if subject is not None:
        target = normalize_fragment(subject)
        rows = [p for p in rows if p.subject == target]

    if not rows:
        return ""

    subjects = {p.subject for p in rows}
    if len(subjects) != 1:
        raise ValueError(
            "compose_propositions requires exactly one subject"
        )

    subject_value = rows[0].subject
    values: List[str] = []
    for row in rows:
        if row.value not in values:
            values.append(row.value)

    if len(values) == 1:
        return f"{subject_value}は、{values[0]}である。"

    return (
        f"{subject_value}は、"
        + "であり、".join(values[:-1])
        + f"であり、{values[-1]}である。"
    )


def merge_propositions(
    existing: Iterable[Proposition],
    incoming: Iterable[Proposition],
) -> List[Proposition]:
    result: List[Proposition] = []
    seen = set()

    for item in list(existing) + list(incoming):
        p = item.normalized()
        key = (p.subject, p.value)
        if not p.subject or not p.value or key in seen:
            continue
        seen.add(key)
        result.append(p)

    return result


def group_by_subject(
    propositions: Iterable[Proposition],
) -> Dict[str, List[Proposition]]:
    grouped: Dict[str, List[Proposition]] = {}
    for item in propositions:
        p = item.normalized()
        grouped.setdefault(p.subject, []).append(p)
    return grouped


def load_propositions(path: Path) -> List[Proposition]:
    if not path.exists():
        return []

    rows: List[Proposition] = []
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
                f"invalid proposition JSON line {line_no}: {exc}"
            ) from exc

        subject = normalize_fragment(item.get("subject", ""))
        value = normalize_fragment(item.get("value", ""))
        if not subject or not value:
            raise ValueError(
                f"invalid proposition line {line_no}: {item!r}"
            )
        rows.append(Proposition(subject, value))

    return merge_propositions([], rows)


def save_propositions(path: Path, propositions: Iterable[Proposition]) -> None:
    rows = merge_propositions([], propositions)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in rows:
            f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")


def add_statement(path: Path, statement: str) -> List[Proposition]:
    incoming = decompose_statement(statement)
    if not incoming:
        return []

    current = load_propositions(path)
    save_propositions(path, merge_propositions(current, incoming))
    return incoming


def compose_subject(path: Path, subject: str) -> str:
    return compose_propositions(
        load_propositions(path),
        subject=subject,
    )
