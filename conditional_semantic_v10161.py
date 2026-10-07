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
