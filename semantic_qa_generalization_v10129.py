#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.12.9 Semantic QA generalization utilities."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SemanticQA:
    concept: str
    relation: str
    object_description: str
    question: str
    answer: str


def normalize(text: str) -> str:
    return " ".join(str(text).strip().split())


def decompose_definition(concept: str, question: str, answer: str) -> SemanticQA:
    concept_n = normalize(concept)
    question_n = normalize(question)
    answer_n = normalize(answer)

    if not concept_n or not question_n or not answer_n:
        raise ValueError("concept/question/answer must be non-empty")

    relation = "definition"
    prefix_candidates = (
        f"{concept_n}は、",
        f"{concept_n}は",
    )
    object_description = answer_n
    for prefix in prefix_candidates:
        if answer_n.startswith(prefix):
            object_description = answer_n[len(prefix):].strip()
            break

    if object_description.endswith("。"):
        object_description = object_description[:-1].strip()

    if not object_description:
        raise ValueError("object_description became empty")

    return SemanticQA(
        concept=concept_n,
        relation=relation,
        object_description=object_description,
        question=question_n,
        answer=answer_n,
    )


def semantic_prompt(item: SemanticQA) -> str:
    return (
        f"概念: {item.concept}\n"
        f"関係: {item.relation}\n"
        "要求: この概念について、学習済みの内容だけを使って簡潔に説明してください"
    )


def relation_prompt(item: SemanticQA) -> str:
    return (
        f"主語={item.concept} / 関係={item.relation}\n"
        "この関係に対応する説明を回答してください"
    )


def training_queries(item: SemanticQA) -> tuple[str, str, str]:
    return (
        item.question,
        semantic_prompt(item),
        relation_prompt(item),
    )
