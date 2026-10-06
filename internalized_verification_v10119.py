#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.11.9 Internalized Verification Loop."""

from __future__ import annotations

from dataclasses import dataclass
import json
import time
from pathlib import Path


PENDING = "pending"
RETRAIN = "retrain"
VERIFIED = "verified"
FAILED = "failed"


@dataclass(frozen=True)
class VerificationSummary:
    pending: int = 0
    retrain: int = 0
    verified: int = 0
    failed: int = 0
    total: int = 0


@dataclass(frozen=True)
class BatchRepairPlan:
    tasks: tuple[dict, ...] = ()
    fingerprints: tuple[str, ...] = ()
    concepts: tuple[str, ...] = ()

    @property
    def count(self) -> int:
        return len(self.tasks)


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _save(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def upsert_unstable(
    path: Path,
    *,
    concept: str,
    question: str,
    teacher_answer: str,
    candidate_answer: str,
    reason: str,
    semantic: float,
    lexical: float,
    required: float,
    contradiction: bool,
    missing_terms: tuple[str, ...],
    fingerprint: str,
) -> bool:
    rows = _load(path)
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    for row in rows:
        if (
            str(row.get("concept", "")) == concept
            and str(row.get("fingerprint", "")) == fingerprint
            and str(row.get("status", "")) in {PENDING, RETRAIN}
        ):
            row.update({
                "question": question,
                "teacher_answer": teacher_answer,
                "candidate_answer": candidate_answer,
                "reason": reason,
                "semantic": semantic,
                "lexical": lexical,
                "required": required,
                "contradiction": contradiction,
                "missing_terms": list(missing_terms),
                "last_failed_at": now,
                "attempts": int(row.get("attempts", 1)) + 1,
                "status": PENDING,
            })
            _save(path, rows)
            return False

    rows.append({
        "concept": concept,
        "question": question,
        "teacher_answer": teacher_answer,
        "candidate_answer": candidate_answer,
        "reason": reason,
        "semantic": semantic,
        "lexical": lexical,
        "required": required,
        "contradiction": contradiction,
        "missing_terms": list(missing_terms),
        "fingerprint": fingerprint,
        "status": PENDING,
        "attempts": 1,
        "created_at": now,
        "last_failed_at": now,
    })
    _save(path, rows)
    return True


def mark_retrain(path: Path, fingerprint: str) -> int:
    rows = _load(path)
    changed = 0
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    for row in rows:
        if (
            str(row.get("fingerprint", "")) == fingerprint
            and str(row.get("status", "")) == PENDING
        ):
            row["status"] = RETRAIN
            row["retrain_at"] = now
            changed += 1
    if changed:
        _save(path, rows)
    return changed


def mark_verified(path: Path, concept: str, fingerprint: str) -> int:
    rows = _load(path)
    changed = 0
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    for row in rows:
        if (
            str(row.get("concept", "")) == concept
            and str(row.get("fingerprint", "")) == fingerprint
            and str(row.get("status", "")) in {PENDING, RETRAIN}
        ):
            row["status"] = VERIFIED
            row["verified_at"] = now
            changed += 1
    if changed:
        _save(path, rows)
    return changed


def verification_summary(path: Path) -> VerificationSummary:
    rows = _load(path)
    counts = {PENDING: 0, RETRAIN: 0, VERIFIED: 0, FAILED: 0}
    for row in rows:
        status = str(row.get("status", ""))
        if status in counts:
            counts[status] += 1
    return VerificationSummary(
        pending=counts[PENDING],
        retrain=counts[RETRAIN],
        verified=counts[VERIFIED],
        failed=counts[FAILED],
        total=len(rows),
    )


def active_verifications(path: Path) -> list[dict]:
    return [
        row for row in _load(path)
        if str(row.get("status", "")) in {PENDING, RETRAIN}
    ]


def batch_repair_plan(path: Path) -> BatchRepairPlan:
    """Return all active verification tasks that require repair training."""
    tasks = tuple(
        row for row in _load(path)
        if str(row.get("status", "")) in {PENDING, RETRAIN}
        and str(row.get("question", "")).strip()
        and str(row.get("teacher_answer", "")).strip()
        and str(row.get("fingerprint", "")).strip()
    )
    fingerprints = tuple(
        dict.fromkeys(str(row["fingerprint"]) for row in tasks)
    )
    concepts = tuple(
        dict.fromkeys(str(row.get("concept", "")) for row in tasks)
    )
    return BatchRepairPlan(
        tasks=tasks,
        fingerprints=fingerprints,
        concepts=concepts,
    )


def mark_batch_retrain(path: Path, fingerprints: set[str]) -> int:
    if not fingerprints:
        return 0
    rows = _load(path)
    changed = 0
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    for row in rows:
        if (
            str(row.get("fingerprint", "")) in fingerprints
            and str(row.get("status", "")) in {PENDING, RETRAIN}
        ):
            if str(row.get("status", "")) != RETRAIN:
                changed += 1
            row["status"] = RETRAIN
            row["batch_retrain_at"] = now
    if changed:
        _save(path, rows)
    return changed


def mark_batch_failed(
    path: Path,
    fingerprints: set[str],
    reason: str,
) -> int:
    if not fingerprints:
        return 0
    rows = _load(path)
    changed = 0
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    for row in rows:
        if (
            str(row.get("fingerprint", "")) in fingerprints
            and str(row.get("status", "")) in {PENDING, RETRAIN}
        ):
            row["status"] = FAILED
            row["failed_at"] = now
            row["failure_reason"] = reason
            changed += 1
    if changed:
        _save(path, rows)
    return changed
