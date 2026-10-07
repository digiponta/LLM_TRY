#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_TRY v10.16.1 Conditional Semantic Proposition

Condition-aware storage and retrieval for Japanese conditional knowledge.
"""

from __future__ import annotations
from dataclasses import asdict, dataclass
import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable
import time

from internalized_knowledge_v10100 import pair_fingerprint, load_trained_fingerprints

from modifier_condition_normalization_v10160 import (
    classify_modifier,
    normalize_modifier_condition,
)

TRAILING = "、，,。．.!！?？:：;；"

@dataclass(frozen=True)
class ConditionalProposition:
    subject: str
    condition: str
    predicate: str
    condition_polarity: bool = True

    def normalized(self) -> "ConditionalProposition":
        return ConditionalProposition(
            subject=_surface(self.subject),
            condition=_normalize_condition(self.condition),
            predicate=_surface(self.predicate),
            condition_polarity=bool(self.condition_polarity),
        )

    def render(self) -> str:
        row = self.normalized()
        return f"{row.subject}は、{row.condition}の場合、{row.predicate}。"

@dataclass(frozen=True)
class ConditionalSemanticStatus:
    propositions: int
    pending_candidates: int
    internalized: int
    pending_internalization: int


@dataclass(frozen=True)
class ConditionalQuery:
    original: str
    subject: str
    condition: str
    normalized: str
    matched: bool
    reason: str = ""

def _surface(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text)).strip()
    return value.rstrip(TRAILING).strip()

def _normalize_condition(text: str) -> str:
    value = _surface(text)
    if value.endswith("時") and len(value) > 1:
        base = value[:-1].strip()
        is_condition, _ = classify_modifier(base)
        if is_condition:
            return base
    if value.endswith("のとき"):
        value = value[:-len("のとき")].strip()
    if value.endswith("の場合"):
        value = value[:-len("の場合")].strip()
    return value

def parse_conditional_statement(text: str) -> ConditionalProposition | None:
    source = _surface(text)
    if not source:
        return None

    m = re.fullmatch(
        r"\s*(?P<subject>.+?)\s*は\s*[、,]?\s*"
        r"(?P<condition>.+?)\s*の場合\s*[、,]?\s*"
        r"(?P<predicate>.+?)\s*",
        source,
    )
    if m:
        condition = _normalize_condition(m.group("condition"))
        is_condition, _ = classify_modifier(condition)
        if not is_condition:
            return None
        return ConditionalProposition(
            subject=_surface(m.group("subject")),
            condition=condition,
            predicate=_surface(m.group("predicate")),
        ).normalized()

    normalized = normalize_modifier_condition(source)
    if not normalized.transformed:
        return None

    m = re.fullmatch(
        r"(?P<subject>.+?)は、(?P<condition>.+?)の場合、(?P<predicate>.+)",
        normalized.normalized,
    )
    if not m:
        return None
    return ConditionalProposition(
        subject=_surface(m.group("subject")),
        condition=_normalize_condition(m.group("condition")),
        predicate=_surface(m.group("predicate")),
    ).normalized()

def parse_conditional_query(text: str) -> ConditionalQuery:
    original = str(text)
    source = _surface(original)
    patterns = (
        (re.compile(r"^(?P<condition>.+?)の場合[、,]?\s*(?P<subject>.+?)(?:は)?(?:どうなる|どうなりますか|どうなるの|どうなります|何が起きる|何が起こる)$"), "condition-first question"),
        (re.compile(r"^(?P<subject>.+?)が(?P<condition>.+?)(?:のとき|の時|時)(?:は)?$"), "subject-state question"),
        (re.compile(r"^(?P<condition>.+?)(?:時|の場合)の(?P<subject>.+?)(?:は)?$"), "condition-modified subject query"),
        (re.compile(r"^(?P<subject>.+?)は[、,]?(?P<condition>.+?)の場合[、,]?(?:どうなる|どうなりますか|どうなるの|何が起きる|何が起こる)$"), "canonical conditional question"),
    )
    for pattern, reason in patterns:
        match = pattern.fullmatch(source)
        if not match:
            continue
        subject = _surface(match.group("subject"))
        condition = _normalize_condition(match.group("condition"))
        is_condition, condition_reason = classify_modifier(condition)
        if not subject or not condition or not is_condition:
            continue
        return ConditionalQuery(
            original=original,
            subject=subject,
            condition=condition,
            normalized=f"{subject}は、{condition}の場合、どうなる",
            matched=True,
            reason=f"{reason}; {condition_reason}",
        )
    return ConditionalQuery(
        original=original,
        subject="",
        condition="",
        normalized=source,
        matched=False,
        reason="no conditional query pattern",
    )

def load_conditional_propositions(path: Path) -> list[ConditionalProposition]:
    if not path.exists():
        return []
    rows: list[ConditionalProposition] = []
    seen = set()
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            item = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid conditional proposition JSON line {line_no}: {exc}") from exc
        row = ConditionalProposition(
            subject=str(item.get("subject", "")),
            condition=str(item.get("condition", "")),
            predicate=str(item.get("predicate", "")),
            condition_polarity=bool(item.get("condition_polarity", True)),
        ).normalized()
        if not row.subject or not row.condition or not row.predicate:
            raise ValueError(f"invalid conditional proposition line {line_no}: {item!r}")
        key = (row.subject, row.condition, row.predicate, row.condition_polarity)
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    return rows

def save_conditional_propositions(path: Path, rows: Iterable[ConditionalProposition]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized: list[ConditionalProposition] = []
    seen = set()
    for item in rows:
        row = item.normalized()
        key = (row.subject, row.condition, row.predicate, row.condition_polarity)
        if not row.subject or not row.condition or not row.predicate or key in seen:
            continue
        seen.add(key)
        normalized.append(row)
    with path.open("w", encoding="utf-8") as f:
        for row in normalized:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")

def add_conditional_statement(path: Path, statement: str) -> ConditionalProposition | None:
    incoming = parse_conditional_statement(statement)
    if incoming is None:
        return None
    current = load_conditional_propositions(path)
    save_conditional_propositions(path, [*current, incoming])
    return incoming

def lookup_conditional(path: Path, subject: str, condition: str) -> list[ConditionalProposition]:
    target_subject = _surface(subject)
    target_condition = _normalize_condition(condition)
    return [
        row for row in load_conditional_propositions(path)
        if row.subject == target_subject
        and row.condition == target_condition
        and row.condition_polarity
    ]

def answer_conditional_query(path: Path, query: str) -> tuple[ConditionalQuery, str]:
    parsed = parse_conditional_query(query)
    if not parsed.matched:
        return parsed, ""
    rows = lookup_conditional(path, parsed.subject, parsed.condition)
    if not rows:
        return parsed, ""
    predicates: list[str] = []
    for row in rows:
        if row.predicate not in predicates:
            predicates.append(row.predicate)
    body = "、".join(predicates)
    answer = f"{parsed.subject}は、{parsed.condition}の場合、{body}。"
    return parsed, answer


DEFAULT_CONDITIONAL_CANDIDATE_QUEUE = "data/conditional_candidates_v10163.jsonl"


def load_conditional_candidates(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            item = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def queue_conditional_candidate(
    path: Path,
    statement: str,
    proposition_path: Path | None = None,
) -> tuple[bool, ConditionalProposition | None]:
    # A conditional question must never become a declarative knowledge
    # candidate, even if another normalization step could make it look like
    # "subject は condition の場合 predicate".
    query = parse_conditional_query(statement)
    if query.matched:
        return False, None

    row = parse_conditional_statement(statement)
    if row is None:
        return False, None

    if proposition_path is not None:
        for stored in load_conditional_propositions(proposition_path):
            if (
                stored.subject == row.subject
                and stored.condition == row.condition
                and stored.predicate == row.predicate
                and stored.condition_polarity == row.condition_polarity
            ):
                return False, row

    existing = load_conditional_candidates(path)
    key = (row.subject, row.condition, row.predicate)
    for item in existing:
        old = (
            str(item.get("subject", "")),
            str(item.get("condition", "")),
            str(item.get("predicate", "")),
        )
        if old == key and str(item.get("status", "pending")) == "pending":
            return False, row

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "subject": row.subject,
        "condition": row.condition,
        "predicate": row.predicate,
        "condition_polarity": row.condition_polarity,
        "canonical": row.render(),
        "source_statement": _surface(statement),
        "status": "pending",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "version": "v10.16.3",
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return True, row


def pending_conditional_candidates(path: Path) -> list[dict]:
    return [
        row for row in load_conditional_candidates(path)
        if str(row.get("status", "pending")) == "pending"
    ]


def approve_conditional_candidate(
    candidate_path: Path,
    proposition_path: Path,
    index: int | None = None,
) -> int:
    rows = load_conditional_candidates(candidate_path)
    pending_positions = [
        i for i, row in enumerate(rows)
        if str(row.get("status", "pending")) == "pending"
    ]
    if index is not None:
        if index < 1 or index > len(pending_positions):
            return 0
        selected = {pending_positions[index - 1]}
    else:
        selected = set(pending_positions)

    approved = 0
    for i in selected:
        row = rows[i]
        statement = str(row.get("canonical", "")).strip()
        if add_conditional_statement(proposition_path, statement) is None:
            continue
        row["status"] = "approved"
        row["approved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        approved += 1

    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    with candidate_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return approved


def prepare_conditional_sleep_pairs(
    proposition_path: Path,
    learning_log_path: Path,
    learning_state_path: Path,
) -> int:
    trained = load_trained_fingerprints(learning_state_path)
    existing: set[str] = set()
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

    queued = 0
    learning_log_path.parent.mkdir(parents=True, exist_ok=True)
    with learning_log_path.open("a", encoding="utf-8") as handle:
        for row in load_conditional_propositions(proposition_path):
            question = f"{row.condition}の場合{row.subject}はどうなる?"
            answer = row.render()
            fp = pair_fingerprint(question, answer)
            if fp in trained or fp in existing:
                continue
            payload = {
                "user": question,
                "assistant": answer,
                "source": "conditional-semantic-sleep",
                "semantic_origin": "conditional-propositions-v10.16.3",
                "fingerprint": fp,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            existing.add(fp)
            queued += 1
    return queued


def conditional_semantic_status(
    proposition_path: Path,
    candidate_path: Path,
    learning_state_path: Path,
    checkpoint_fingerprints: set[str] | frozenset[str] | None = None,
) -> ConditionalSemanticStatus:
    rows = load_conditional_propositions(proposition_path)
    candidates = pending_conditional_candidates(candidate_path)
    trained = load_trained_fingerprints(learning_state_path)
    effective = trained
    if checkpoint_fingerprints is not None:
        effective = trained.intersection(
            {str(value) for value in checkpoint_fingerprints}
        )

    internalized = 0
    for row in rows:
        question = f"{row.condition}の場合{row.subject}はどうなる?"
        answer = row.render()
        if pair_fingerprint(question, answer) in effective:
            internalized += 1

    return ConditionalSemanticStatus(
        propositions=len(rows),
        pending_candidates=len(candidates),
        internalized=internalized,
        pending_internalization=max(0, len(rows) - internalized),
    )
