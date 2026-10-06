#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_TRY v10.10.0 Typed Subject-Keyed Semantic Proposition

Derived representation:
    Subject => Predicate Type => Statement

Example:
    GPU => property   => GPUは高速である。
    GPU => capability => GPUは並列計算が得意である。
    GPU => relation   => GPUはCUDAを利用可能である。

The atomic proposition store remains the source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List

from semantic_proposition_v1090 import (
    Proposition,
    load_propositions,
    normalize_fragment,
)
from subject_keyed_proposition_v1090 import render_atomic_statement


PREDICATE_TYPES = (
    "definition",
    "property",
    "capability",
    "relation",
)


@dataclass(frozen=True)
class TypedSubjectStatement:
    subject: str
    predicate_type: str
    statement: str
    value: str

    def normalized(self) -> "TypedSubjectStatement":
        subject = normalize_fragment(self.subject)
        value = normalize_fragment(self.value)
        predicate_type = normalize_predicate_type(self.predicate_type)
        return TypedSubjectStatement(
            subject=subject,
            predicate_type=predicate_type,
            statement=render_atomic_statement(subject, value),
            value=value,
        )


def normalize_predicate_type(value: str) -> str:
    value = str(value).strip().lower()
    if value not in PREDICATE_TYPES:
        return "property"
    return value


def classify_predicate_type(value: str) -> str:
    """Classify an atomic predicate with conservative deterministic rules.

    Precedence is important:
      relation -> capability -> definition -> property

    The rules are intentionally transparent and auditable.  They are not
    intended to be a general Japanese semantic parser.
    """
    text = normalize_fragment(value)
    compact = re.sub(r"\s+", "", text)

    relation_markers = (
        "を利用可能",
        "を利用する",
        "を使用する",
        "に依存する",
        "に接続する",
        "を含む",
        "に属する",
        "と関係する",
        "から成る",
        "で構成される",
    )
    if any(marker in compact for marker in relation_markers):
        return "relation"

    capability_markers = (
        "が得意",
        "を実行できる",
        "ができる",
        "できる",
        "が可能",
        "可能である",
        "に対応",
        "を処理できる",
        "を生成できる",
        "を計算できる",
    )
    if any(marker in compact for marker in capability_markers):
        return "capability"

    definition_endings = (
        "装置",
        "処理装置",
        "言語",
        "モデル",
        "技術",
        "基盤",
        "システム",
        "方式",
        "手法",
        "概念",
        "分野",
        "生物",
    )
    definition_markers = (
        "の一種",
        "一種の",
        "を指す",
        "と呼ばれる",
    )
    if (
        any(compact.endswith(ending) for ending in definition_endings)
        or any(marker in compact for marker in definition_markers)
    ):
        return "definition"

    return "property"


def proposition_to_typed_statement(
    proposition: Proposition,
) -> TypedSubjectStatement:
    p = proposition.normalized()
    return TypedSubjectStatement(
        subject=p.subject,
        predicate_type=classify_predicate_type(p.value),
        statement=render_atomic_statement(p.subject, p.value),
        value=p.value,
    )


def build_typed_index(
    propositions: Iterable[Proposition],
) -> List[TypedSubjectStatement]:
    rows: List[TypedSubjectStatement] = []
    seen = set()

    for proposition in propositions:
        row = proposition_to_typed_statement(proposition)
        key = (
            row.subject,
            row.predicate_type,
            row.statement,
        )
        if not row.subject or not row.statement or key in seen:
            continue
        seen.add(key)
        rows.append(row)

    return rows


def load_typed_index(path: Path) -> List[TypedSubjectStatement]:
    if not path.exists():
        return []

    rows: List[TypedSubjectStatement] = []
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
                f"invalid typed index JSON line {line_no}: {exc}"
            ) from exc

        subject = normalize_fragment(item.get("subject", ""))
        value = normalize_fragment(item.get("value", ""))
        predicate_type = normalize_predicate_type(
            item.get("predicate_type", "property")
        )
        statement = str(item.get("statement", "")).strip()
        if not statement and subject and value:
            statement = render_atomic_statement(subject, value)

        if not subject or not value or not statement:
            raise ValueError(
                f"invalid typed index line {line_no}: {item!r}"
            )

        row = TypedSubjectStatement(
            subject=subject,
            predicate_type=predicate_type,
            statement=statement,
            value=value,
        )
        key = (row.subject, row.predicate_type, row.statement)
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)

    return rows


def save_typed_index(
    path: Path,
    rows: Iterable[TypedSubjectStatement],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    normalized: List[TypedSubjectStatement] = []
    seen = set()
    for item in rows:
        row = item.normalized()
        key = (row.subject, row.predicate_type, row.statement)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(row)

    with path.open("w", encoding="utf-8") as f:
        for row in normalized:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def sync_typed_index(
    proposition_path: Path,
    typed_index_path: Path,
) -> int:
    rows = build_typed_index(load_propositions(proposition_path))
    save_typed_index(typed_index_path, rows)
    return len(rows)


def typed_subject_rows(
    typed_index_path: Path,
    subject: str,
) -> List[TypedSubjectStatement]:
    target = normalize_fragment(subject)
    return [
        row for row in load_typed_index(typed_index_path)
        if row.subject == target
    ]


def typed_subject_mapping(
    typed_index_path: Path,
    subject: str,
) -> List[str]:
    rows = typed_subject_rows(typed_index_path, subject)
    return [
        f"{row.subject} => {row.predicate_type} => {row.statement}"
        for row in rows
    ]


def typed_index_dict(
    typed_index_path: Path,
) -> Dict[str, Dict[str, List[str]]]:
    result: Dict[str, Dict[str, List[str]]] = {}
    for row in load_typed_index(typed_index_path):
        by_type = result.setdefault(row.subject, {})
        by_type.setdefault(row.predicate_type, []).append(row.statement)
    return result


def detect_query_predicate_type(question: str) -> str | None:
    """Detect which predicate type a question requests.

    This is intentionally conservative. A type is returned only when an
    explicit Japanese cue is present.
    """
    q = re.sub(r"\s+", "", str(question).strip())

    relation_cues = (
        "の関係",
        "との関係",
        "とどう関係",
        "何を利用",
        "何に依存",
        "何と接続",
        "何を含む",
    )
    if any(cue in q for cue in relation_cues):
        return "relation"

    capability_cues = (
        "何が得意",
        "何を得意",
        "何ができる",
        "何をできる",
        "できること",
        "の能力",
        "の機能",
    )
    if any(cue in q for cue in capability_cues):
        return "capability"

    property_cues = (
        "の性質",
        "の特徴",
        "どんな性質",
        "どんな特徴",
        "特徴は",
        "性質は",
    )
    if any(cue in q for cue in property_cues):
        return "property"

    definition_cues = (
        "とは",
        "の定義",
        "を定義",
        "って何",
    )
    if any(cue in q for cue in definition_cues):
        return "definition"

    return None


def detect_typed_subject(
    typed_index_path: Path,
    question: str,
) -> str:
    """Find the longest indexed subject explicitly mentioned in the question."""
    q = str(question).strip()
    subjects = sorted(
        {row.subject for row in load_typed_index(typed_index_path)},
        key=len,
        reverse=True,
    )
    for subject in subjects:
        if subject and subject in q:
            return subject
    return ""


def typed_rows_by_type(
    typed_index_path: Path,
    subject: str,
    predicate_type: str,
) -> List[TypedSubjectStatement]:
    target_type = normalize_predicate_type(predicate_type)
    return [
        row for row in typed_subject_rows(typed_index_path, subject)
        if row.predicate_type == target_type
    ]


def compose_typed_rows(
    rows: Iterable[TypedSubjectStatement],
) -> str:
    items = list(rows)
    if not items:
        return ""

    subject = items[0].subject
    values: List[str] = []
    for row in items:
        if row.subject != subject:
            raise ValueError("compose_typed_rows requires one subject")
        if row.value not in values:
            values.append(row.value)

    if len(values) == 1:
        return f"{subject}は、{values[0]}である。"

    return (
        f"{subject}は、"
        + "であり、".join(values[:-1])
        + f"であり、{values[-1]}である。"
    )


def typed_query_lookup(
    typed_index_path: Path,
    question: str,
) -> tuple[str, str, str] | None:
    """Return (subject, predicate_type, composed answer) for typed questions."""
    predicate_type = detect_query_predicate_type(question)
    if predicate_type is None:
        return None

    subject = detect_typed_subject(typed_index_path, question)
    if not subject:
        return None

    rows = typed_rows_by_type(
        typed_index_path,
        subject,
        predicate_type,
    )
    answer = compose_typed_rows(rows)
    if not answer:
        return None

    return subject, predicate_type, answer
