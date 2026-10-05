#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_TRY v10.11.5 Knowledge Queue Lifecycle Consolidation

Canonical UNKNOWN_KNOWLEDGE lifecycle:
    pending -> promoted -> verified

Legacy "resolved" rows are preserved for audit but are no longer written by
the semantic knowledge runtime.

Verification means explicit Truth State TRUE. If TRUE is later revoked, a
verified row returns to promoted.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
import time
from pathlib import Path
from typing import Iterable


ACTIVE_STATUSES = ("pending", "promoted", "verified")
LEGACY_STATUSES = ("resolved",)


@dataclass(frozen=True)
class QueueLifecycleSummary:
    pending: int = 0
    promoted: int = 0
    verified: int = 0
    legacy_resolved: int = 0
    other: int = 0

    @property
    def total(self) -> int:
        return (
            self.pending
            + self.promoted
            + self.verified
            + self.legacy_resolved
            + self.other
        )


def normalize_queue_text(text: str) -> str:
    return " ".join(str(text).strip().split()).lower()


def extract_queue_concept(user_text: str) -> str:
    q = str(user_text).strip().rstrip("。、，, ")
    patterns = (
        r"^(.+?)(?:とは)$",
        r"^(.+?)(?:って何)$",
        r"^(.+?)(?:について教えて)$",
        r"^(.+?)(?:を説明して)$",
        r"^(.+?)(?:を簡単に説明して)$",
        r"^(.+?)(?:は)$",
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, q)
        if match:
            return match.group(1).strip()
    if q and not re.search(r"[\s。、！？!?？,:：;；]", q) and len(q) <= 40:
        return q
    return ""


def row_concept(row: dict) -> str:
    promoted = str(row.get("promoted_concept", "")).strip()
    if promoted:
        return promoted
    verified = str(row.get("verified_concept", "")).strip()
    if verified:
        return verified
    return extract_queue_concept(str(row.get("user", "")))


def matches_concept(row: dict, concept: str) -> bool:
    return normalize_queue_text(row_concept(row)) == normalize_queue_text(concept)


def load_queue_rows(queue_path: Path) -> list[dict]:
    if not queue_path.exists():
        return []
    rows: list[dict] = []
    for raw in queue_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if str(row.get("resolution", "")) != "UNKNOWN_KNOWLEDGE":
            continue
        if not str(row.get("status", "")).strip():
            row["status"] = "pending"
        rows.append(row)
    return rows


def save_queue_rows(queue_path: Path, rows: Iterable[dict]) -> None:
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def lifecycle_summary(queue_path: Path) -> QueueLifecycleSummary:
    counts = {
        "pending": 0,
        "promoted": 0,
        "verified": 0,
        "legacy_resolved": 0,
        "other": 0,
    }
    for row in load_queue_rows(queue_path):
        status = str(row.get("status", "pending"))
        if status in ACTIVE_STATUSES:
            counts[status] += 1
        elif status == "resolved":
            counts["legacy_resolved"] += 1
        else:
            counts["other"] += 1
    return QueueLifecycleSummary(**counts)


def rows_with_status(
    queue_path: Path,
    status: str,
) -> list[dict]:
    return [
        row for row in load_queue_rows(queue_path)
        if str(row.get("status", "pending")) == status
    ]


def pending_requests(queue_path: Path) -> list[dict]:
    return rows_with_status(queue_path, "pending")


def pending_request_for_concept(
    queue_path: Path,
    concept: str,
) -> dict | None:
    for row in pending_requests(queue_path):
        if matches_concept(row, concept):
            return row
    return None


def mark_concept_promoted(
    queue_path: Path,
    concept: str,
    *,
    statement: str = "",
    post_state: str = "",
    source: str = "knowledge-promotion",
) -> int:
    rows = load_queue_rows(queue_path)
    if not rows:
        return 0

    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    changed = 0
    for row in rows:
        status = str(row.get("status", "pending"))
        if status != "pending" or not matches_concept(row, concept):
            continue
        row["status"] = "promoted"
        row["promoted_at"] = now
        row["promoted_concept"] = concept
        if statement:
            row["promoted_statement"] = statement
        if post_state:
            row["post_knowledge_state"] = post_state
        row["lifecycle_source"] = source
        changed += 1

    if changed:
        save_queue_rows(queue_path, rows)
    return changed


def mark_concept_verified(
    queue_path: Path,
    concept: str,
    *,
    truth_source: str = "",
) -> int:
    rows = load_queue_rows(queue_path)
    if not rows:
        return 0

    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    changed = 0
    for row in rows:
        status = str(row.get("status", "pending"))
        if status != "promoted" or not matches_concept(row, concept):
            continue
        row["status"] = "verified"
        row["verified_at"] = now
        row["verified_concept"] = concept
        row["verification_truth_state"] = "TRUE"
        if truth_source:
            row["verification_source"] = truth_source
        changed += 1

    if changed:
        save_queue_rows(queue_path, rows)
    return changed


def revoke_concept_verification(
    queue_path: Path,
    concept: str,
    *,
    new_truth_state: str,
) -> int:
    rows = load_queue_rows(queue_path)
    if not rows:
        return 0

    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    changed = 0
    for row in rows:
        status = str(row.get("status", "pending"))
        if status != "verified" or not matches_concept(row, concept):
            continue
        row["status"] = "promoted"
        row["verification_revoked_at"] = now
        row["verification_truth_state"] = new_truth_state
        changed += 1

    if changed:
        save_queue_rows(queue_path, rows)
    return changed
