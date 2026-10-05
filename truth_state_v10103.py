#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_TRY v10.10.3 Truth State Overlay

Truth state is stored separately from model weights and semantic knowledge.
False/contested/outdated knowledge is retained for auditability rather than
being silently deleted.

States:
    TRUE
    FALSE
    UNVERIFIED
    CONTESTED
    OUTDATED
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import time
from pathlib import Path
from typing import Iterable, List


TRUTH_STATES = (
    "TRUE",
    "FALSE",
    "UNVERIFIED",
    "CONTESTED",
    "OUTDATED",
)


@dataclass(frozen=True)
class TruthRecord:
    concept: str
    state: str
    reason: str = ""
    correction: str = ""
    source: str = "manual"
    updated_at: str = ""

    def normalized(self) -> "TruthRecord":
        concept = str(self.concept).strip()
        state = normalize_truth_state(self.state)
        return TruthRecord(
            concept=concept,
            state=state,
            reason=str(self.reason).strip(),
            correction=str(self.correction).strip(),
            source=str(self.source).strip() or "manual",
            updated_at=str(self.updated_at).strip(),
        )


def normalize_truth_state(value: str) -> str:
    state = str(value).strip().upper()
    if state not in TRUTH_STATES:
        raise ValueError(
            f"invalid truth state {value!r}; expected one of {TRUTH_STATES}"
        )
    return state


def load_truth_records(path: Path) -> List[TruthRecord]:
    if not path.exists():
        return []

    records: List[TruthRecord] = []
    seen = set()
    for line_no, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        if not raw.strip():
            continue
        try:
            item = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid truth-state JSON line {line_no}: {exc}"
            ) from exc

        record = TruthRecord(
            concept=str(item.get("concept", "")),
            state=str(item.get("state", "UNVERIFIED")),
            reason=str(item.get("reason", "")),
            correction=str(item.get("correction", "")),
            source=str(item.get("source", "manual")),
            updated_at=str(item.get("updated_at", "")),
        ).normalized()
        if not record.concept:
            continue

        key = record.concept.lower()
        if key in seen:
            # Keep the last row if legacy duplicate rows exist.
            records = [
                r for r in records
                if r.concept.lower() != key
            ]
        seen.add(key)
        records.append(record)

    return records


def save_truth_records(path: Path, records: Iterable[TruthRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized: List[TruthRecord] = []
    index: dict[str, int] = {}

    for raw_record in records:
        record = raw_record.normalized()
        if not record.concept:
            continue
        key = record.concept.lower()
        if key in index:
            normalized[index[key]] = record
        else:
            index[key] = len(normalized)
            normalized.append(record)

    with path.open("w", encoding="utf-8") as f:
        for record in normalized:
            f.write(
                json.dumps(asdict(record), ensure_ascii=False)
                + "\n"
            )


def truth_record_for_concept(
    records: Iterable[TruthRecord],
    concept: str,
) -> TruthRecord | None:
    target = str(concept).strip().lower()
    if not target:
        return None
    for record in records:
        if record.concept.lower() == target:
            return record
    return None


def get_truth_record(
    path: Path,
    concept: str,
) -> TruthRecord | None:
    return truth_record_for_concept(
        load_truth_records(path),
        concept,
    )


def upsert_truth_record(
    path: Path,
    concept: str,
    state: str,
    *,
    reason: str = "",
    correction: str = "",
    source: str = "manual",
    updated_at: str = "",
) -> TruthRecord:
    concept = str(concept).strip()
    if not concept:
        raise ValueError("concept must be non-empty")

    record = TruthRecord(
        concept=concept,
        state=normalize_truth_state(state),
        reason=reason,
        correction=correction,
        source=source,
        updated_at=(
            updated_at
            or time.strftime("%Y-%m-%dT%H:%M:%S%z")
        ),
    ).normalized()

    records = load_truth_records(path)
    out: List[TruthRecord] = []
    replaced = False
    for old in records:
        if old.concept.lower() == concept.lower():
            if not replaced:
                out.append(record)
                replaced = True
            continue
        out.append(old)
    if not replaced:
        out.append(record)

    save_truth_records(path, out)
    return record


def effective_truth_record(
    path: Path,
    concept: str,
) -> TruthRecord:
    """Return explicit state or conservative UNVERIFIED default."""
    explicit = get_truth_record(path, concept)
    if explicit is not None:
        return explicit

    return TruthRecord(
        concept=str(concept).strip(),
        state="UNVERIFIED",
        reason="no explicit truth-state record",
        source="truth-default",
        updated_at="",
    )
