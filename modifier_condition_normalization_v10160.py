#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_TRY v10.16 Modifier-to-Condition Normalization

Conservative Japanese surface normalizer.

Goal:
    修飾語 + 主語 + 述語
        -> 主語 + 修飾語 + の場合 + 述語

Example:
    高温のCPUは停止する
        -> CPUは、高温の場合、停止する

Only condition-like modifiers are rewritten. Attribute/ownership/relation
phrases are preserved to avoid semantic corruption.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


TRAILING = "、，,。．.!！?？:：;；"


@dataclass(frozen=True)
class ModifierConditionResult:
    original: str
    normalized: str
    transformed: bool
    modifier: str = ""
    subject: str = ""
    predicate: str = ""
    reason: str = ""


# Explicit condition/state vocabulary. This is intentionally conservative.
_CONDITION_EXACT = {
    "高温", "低温", "常温",
    "高負荷", "低負荷", "過負荷",
    "空腹", "満腹",
    "雨天", "晴天", "曇天", "降雨時", "降雪時",
    "夜間", "昼間",
    "異常", "正常",
    "緊急時", "停止時", "起動時", "終了時",
    "実行時", "動作時", "待機時",
    "接続時", "切断時", "通信時",
    "学習時", "推論時", "訓練時",
    "高圧", "低圧", "高電圧", "低電圧",
    "高湿度", "低湿度",
}

# Markers that strongly indicate a temporal/state/environment condition.
_CONDITION_SUFFIXES = (
    "時", "中", "時点", "状態", "環境", "状況", "条件",
)

# Phrases that should not be promoted into conditions merely because they use の.
_RELATION_OR_ATTRIBUTE_HINTS = (
    "日本", "東京", "文学", "数学", "CPU", "GPU", "AI", "LLM", "CUDA",
)

# Canonical form produced by this module.
_CANONICAL_RE = re.compile(
    r"^\s*(?P<subject>.+?)\s*は\s*[、,]?\s*"
    r"(?P<modifier>.+?)\s*の場合\s*[、,]?\s*"
    r"(?P<predicate>.+?)\s*$"
)

# Primary input form is parsed in two stages:
#   (modifier の subject) は predicate
# The left side may itself contain の (e.g. 雨の日の道路), so candidate
# boundaries are evaluated from right to left and only a proven condition wins.
_SUBJECT_PREDICATE_RE = re.compile(
    r"^\s*(?P<left>.+?)\s*は\s*(?P<predicate>.+?)\s*$"
)


def _surface(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text)).strip()
    return value.rstrip(TRAILING).strip()


def classify_modifier(modifier: str) -> tuple[bool, str]:
    """Return (is_condition, reason).

    Rules intentionally prefer false negatives over false positives.
    """
    m = _surface(modifier)
    compact = re.sub(r"\s+", "", m)

    if not compact:
        return False, "empty modifier"

    if compact.endswith("場合"):
        return True, "explicit 場合 condition"

    if compact in _CONDITION_EXACT:
        return True, "condition lexicon"

    if any(compact.endswith(suffix) for suffix in _CONDITION_SUFFIXES):
        return True, "condition suffix"

    # Common compositional condition phrases.
    if re.search(r"(雨|雪|晴|曇).*(日|天|時)$", compact):
        return True, "weather condition"

    if re.search(r"(高|低|過)(温|負荷|圧|電圧|湿度)$", compact):
        return True, "scalar state condition"

    # State nouns such as 空腹/満腹/異常/正常 are handled above. Avoid broad
    # adjective conversion because e.g. 赤い車 should remain an attribute.
    if compact in _RELATION_OR_ATTRIBUTE_HINTS:
        return False, "relation/ownership hint"

    if re.search(r"(色|型|製|式|版|用)$", compact):
        return False, "attribute/relation suffix"

    return False, "modifier not proven conditional"


def normalize_modifier_condition(text: str) -> ModifierConditionResult:
    """Normalize one sentence into canonical condition order when safe."""
    original = str(text)
    source = _surface(original)
    if not source:
        return ModifierConditionResult(
            original=original,
            normalized="",
            transformed=False,
            reason="empty input",
        )

    canonical = _CANONICAL_RE.match(source)
    if canonical:
        subject = _surface(canonical.group("subject"))
        modifier = _surface(canonical.group("modifier"))
        predicate = _surface(canonical.group("predicate"))
        normalized = f"{subject}は、{modifier}の場合、{predicate}"
        return ModifierConditionResult(
            original=original,
            normalized=normalized,
            transformed=False,
            modifier=modifier,
            subject=subject,
            predicate=predicate,
            reason="already canonical",
        )

    match = _SUBJECT_PREDICATE_RE.match(source)
    if not match:
        return ModifierConditionResult(
            original=original,
            normalized=source,
            transformed=False,
            reason="unsupported surface pattern",
        )

    left = _surface(match.group("left"))
    predicate = _surface(match.group("predicate"))
    if "の" not in left:
        return ModifierConditionResult(
            original=original,
            normalized=source,
            transformed=False,
            predicate=predicate,
            reason="no modifier boundary",
        )

    modifier = ""
    subject = ""
    reason = "modifier not proven conditional"

    # Prefer the rightmost boundary that yields a condition-like modifier.
    # This correctly parses 雨の日の道路 as modifier=雨の日, subject=道路.
    boundaries = [i for i, ch in enumerate(left) if ch == "の"]
    for boundary in reversed(boundaries):
        candidate_modifier = _surface(left[:boundary])
        candidate_subject = _surface(left[boundary + 1:])
        if not candidate_modifier or not candidate_subject:
            continue
        is_condition, candidate_reason = classify_modifier(
            candidate_modifier
        )
        if is_condition:
            modifier = candidate_modifier
            subject = candidate_subject
            reason = candidate_reason
            break

    if not modifier or not subject or not predicate:
        return ModifierConditionResult(
            original=original,
            normalized=source,
            transformed=False,
            modifier=modifier,
            subject=subject,
            predicate=predicate,
            reason=reason,
        )

    normalized = f"{subject}は、{modifier}の場合、{predicate}"
    return ModifierConditionResult(
        original=original,
        normalized=normalized,
        transformed=(normalized != source),
        modifier=modifier,
        subject=subject,
        predicate=predicate,
        reason=reason,
    )


def normalize_modifier_condition_text(text: str) -> str:
    """String-only convenience API for runtime use."""
    return normalize_modifier_condition(text).normalized
