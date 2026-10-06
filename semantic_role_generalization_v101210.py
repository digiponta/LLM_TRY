#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.12.10 Semantic Role Generalization utilities."""

from __future__ import annotations

from dataclasses import dataclass
import re


RELATIONS = ("definition", "property", "function", "cause", "comparison")


@dataclass(frozen=True)
class RoleProposition:
    subject: str
    relation: str
    object_description: str
    question: str
    answer: str


def normalize(text: str) -> str:
    return " ".join(str(text).strip().split())


def infer_relation(subject: str, answer: str) -> str:
    text = normalize(answer)

    # Specific structural cues first.
    if re.search(r"(より|同じ|異な|違い|似て|類似|比較)", text):
        return "comparison"
    if re.search(r"(ため|ので|から|によって|原因|起因|ゆえ)", text):
        return "cause"
    if re.search(r"(使う|使って|使用|働く|働いて|実行|処理|行う|役割|仲介|利用)", text):
        return "function"
    if re.search(r"(と言われ|と呼ば|である|であり|とは)", text):
        return "definition"
    return "property"


def decompose_role(subject: str, question: str, answer: str) -> RoleProposition:
    subject_n = normalize(subject)
    question_n = normalize(question)
    answer_n = normalize(answer)
    if not subject_n or not question_n or not answer_n:
        raise ValueError("subject/question/answer must be non-empty")

    relation = infer_relation(subject_n, answer_n)
    object_description = answer_n
    for prefix in (f"{subject_n}は、", f"{subject_n}は"):
        if answer_n.startswith(prefix):
            object_description = answer_n[len(prefix):].strip()
            break
    if object_description.endswith("。"):
        object_description = object_description[:-1].strip()
    if not object_description:
        raise ValueError("object_description became empty")

    return RoleProposition(
        subject=subject_n,
        relation=relation,
        object_description=object_description,
        question=question_n,
        answer=answer_n,
    )


def semantic_role_prompt(item: RoleProposition) -> str:
    return (
        f"主語: {item.subject}\n"
        f"意味役割: {item.relation}\n"
        "要求: 学習済みの内容だけを使って、この役割に対応する説明を簡潔に答えてください"
    )


def compact_role_prompt(item: RoleProposition) -> str:
    return f"subject={item.subject} relation={item.relation} -> ?"


def training_queries(item: RoleProposition) -> tuple[str, str, str]:
    return (item.question, semantic_role_prompt(item), compact_role_prompt(item))
