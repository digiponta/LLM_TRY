#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.10.0 Internalized Registry Consolidation Regression."""

from __future__ import annotations

from internalized_knowledge_v10100 import (
    InternalizedRecord,
    consolidate_internalized_records,
    internalized_concept_for_focus,
)


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (
        f" : {detail}" if detail else ""
    ))
    if not ok:
        raise AssertionError(name)


def rec(
    concept: str,
    question: str,
    answer: str,
    fp: str,
    source: str,
) -> InternalizedRecord:
    return InternalizedRecord(
        concept=concept,
        question=question,
        teacher_answer=answer,
        fingerprint=fp,
        source=source,
    )


def main() -> None:
    print("=" * 100)
    print(" LLM_TRY v10.10.0 Internalized Registry Consolidation Regression")
    print("=" * 100)

    records = [
        rec("宇宙", "宇宙とは", "宇宙は...", "u1", "chat-manual"),
        rec("文学", "文学とは", "文学はA", "l1", "chat-manual"),
        rec("架空装置", "架空装置とは", "架空装置はA", "k1", "chat-manual"),
        rec("架空装置", "架空装置とは", "架空装置はB", "k2", "chat-recovery"),
        rec("文学", "文学とは", "文学はB", "l2", "chat-manual"),
        rec("文学", "文学とは", "文学はC", "l3", "chat-approved"),
        rec("文学", "文学とは", "文学はD", "l4", "chat-manual"),
        # Duplicate fingerprint must not increase trained_pairs.
        rec("文学", "文学とは", "文学はD", "l4", "chat-manual"),
        rec("量子センサー", "量子センサーとは", "量子センサーは...", "q1", "chat-manual"),
    ]

    concepts = consolidate_internalized_records(records)

    check("concept-count", len(concepts) == 4, str(concepts))

    literature = internalized_concept_for_focus(concepts, "文学")
    check("literature-present", literature is not None, str(literature))
    check(
        "literature-trained-pairs",
        literature is not None and literature.trained_pairs == 4,
        str(literature),
    )
    check(
        "literature-latest",
        literature is not None
        and literature.latest_teacher_answer == "文学はD",
        str(literature),
    )
    check(
        "literature-sources",
        literature is not None
        and literature.sources == (
            "chat-manual",
            "chat-approved",
        ),
        str(literature),
    )

    device = internalized_concept_for_focus(concepts, "架空装置")
    check(
        "device-trained-pairs",
        device is not None and device.trained_pairs == 2,
        str(device),
    )
    check(
        "device-source-history",
        device is not None
        and device.sources == (
            "chat-manual",
            "chat-recovery",
        ),
        str(device),
    )

    quantum = internalized_concept_for_focus(concepts, "量子センサー")
    check(
        "quantum-single-pair",
        quantum is not None and quantum.trained_pairs == 1,
        str(quantum),
    )

    print()
    print("Concept grouping        : PASS")
    print("Fingerprint dedup       : PASS")
    print("Training count          : PASS")
    print("Latest pair selection   : PASS")
    print("Source history          : PASS")
    print("STATUS                  : INTERNALIZED_CONSOLIDATION_PASS")


if __name__ == "__main__":
    main()
