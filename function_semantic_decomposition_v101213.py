#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.12.13 Function Semantic Decomposition utilities."""

from __future__ import annotations

from dataclasses import dataclass
import re

from semantic_role_generalization_v101210 import RoleProposition, normalize


ACTION_PATTERNS = (
    ("execute", r"実行"),
    ("process", r"処理"),
    ("use", r"利用|使用|使う|使って|使われ"),
    ("operate", r"働く|動作"),
    ("perform", r"行う|行い"),
    ("mediate", r"仲介"),
)


@dataclass(frozen=True)
class FunctionStructure:
    subject: str
    action: str
    target: str
    purpose: str
    answer: str


def infer_action(answer: str) -> str:
    text = normalize(answer)
    for action, pattern in ACTION_PATTERNS:
        if re.search(pattern, text):
            return action
    return "function"


def infer_target(subject: str, answer: str) -> str:
    text = normalize(answer)

    # Object-before-verb patterns: "命令を実行", "情報を処理".
    match = re.search(
        r"([一-龯々ァ-ヶーA-Za-z0-9・]{1,24})を(?:同時に)?(?:実行|処理|利用|使用|仲介)",
        text,
    )
    if match:
        return match.group(1)

    # Verb-before-object patterns: "実行している命令".
    match = re.search(
        r"(?:実行|処理|利用|使用|動作)して(?:いる|いた)?"
        r"([一-龯々ァ-ヶーA-Za-z0-9・]{1,24})",
        text,
    )
    if match:
        return match.group(1)

    # Function/purpose phrases often leave the object after "使われる".
    match = re.search(
        r"(?:使われる|使用される)(?:ための|対象の)?"
        r"([一-龯々ァ-ヶーA-Za-z0-9・]{1,24})",
        text,
    )
    if match:
        return match.group(1)

    return "unspecified"


def infer_purpose(answer: str) -> str:
    text = normalize(answer)

    match = re.search(r"(.{2,50}?)(?:ために|ための|為に|為の)", text)
    if match:
        purpose = match.group(1)
        purpose = re.sub(r"^.*?[、,]", "", purpose).strip()
        return purpose[-40:] if purpose else "unspecified"

    if re.search(r"命令.*実行", text):
        return "computation"
    if re.search(r"情報.*処理", text):
        return "information_processing"
    return "unspecified"


def decompose_function(item: RoleProposition) -> FunctionStructure:
    if item.relation != "function":
        raise ValueError(f"Expected function relation, got: {item.relation}")
    return FunctionStructure(
        subject=item.subject,
        action=infer_action(item.answer),
        target=infer_target(item.subject, item.answer),
        purpose=infer_purpose(item.answer),
        answer=item.answer,
    )


def function_structure_prompt(item: RoleProposition) -> str:
    f = decompose_function(item)
    return (
        f"主語: {f.subject}\n"
        "意味役割: function\n"
        f"action: {f.action}\n"
        f"target: {f.target}\n"
        f"purpose: {f.purpose}\n"
        "要求: この機能構造に対応する学習済みの説明を簡潔に答えてください"
    )


def function_slot_prompt(item: RoleProposition) -> str:
    f = decompose_function(item)
    return (
        f"subject={f.subject} function.action={f.action} "
        f"function.target={f.target} function.purpose={f.purpose} -> ?"
    )


def function_training_queries(item: RoleProposition) -> tuple[str, str]:
    return (function_structure_prompt(item), function_slot_prompt(item))
