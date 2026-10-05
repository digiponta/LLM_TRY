#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_TRY v10.10.2 Knowledge Provenance / Source Tracking

Provenance describes why a knowledge state was selected. It is metadata only:
it does not alter the answer text or gate thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


RETRIEVAL_PRIORITY = {
    "TYPED": 10,
    "CANONICAL": 20,
    "UNIFIED": 30,
    "INTERNALIZED": 40,
    "INTERNALIZED_STALE": 45,
    "RAW_CORPUS_ONLY": 50,
    "UNKNOWN": 60,
    "NON_CONCEPT": 70,
}


@dataclass(frozen=True)
class KnowledgeProvenance:
    source: str = ""
    origin: str = ""
    timestamp: str = ""
    fingerprint: str = ""
    retrieval_priority: int = 0
    evidence: str = ""
    metadata: Mapping[str, str] = field(default_factory=dict)

    def compact(self) -> str:
        parts = [
            f"source={self.source or '-'}",
            f"origin={self.origin or '-'}",
            f"priority={self.retrieval_priority}",
        ]
        if self.timestamp:
            parts.append(f"timestamp={self.timestamp}")
        if self.fingerprint:
            parts.append(f"fingerprint={self.fingerprint}")
        if self.evidence:
            parts.append(f"evidence={self.evidence}")
        return ", ".join(parts)


def provenance_for_state(
    state: str,
    *,
    source: str = "",
    origin: str = "",
    timestamp: str = "",
    fingerprint: str = "",
    evidence: str = "",
    metadata: Mapping[str, str] | None = None,
) -> KnowledgeProvenance:
    return KnowledgeProvenance(
        source=source,
        origin=origin,
        timestamp=timestamp,
        fingerprint=fingerprint,
        retrieval_priority=RETRIEVAL_PRIORITY.get(state, 999),
        evidence=evidence,
        metadata=dict(metadata or {}),
    )
