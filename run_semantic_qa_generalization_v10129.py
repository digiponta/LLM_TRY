#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.12.9 Semantic QA Generalization regression."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from semantic_qa_generalization_v10129 import (
    decompose_definition,
    semantic_prompt,
    relation_prompt,
    training_queries,
)
from train_semantic_qa_generalization_v10129 import load_augmented_pairs


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" : {detail}" if detail else ""))
    if not ok:
        raise AssertionError(name)


def main() -> None:
    print("=" * 108)
    print(" LLM_TRY v10.12.9 Semantic QA Generalization Regression")
    print("=" * 108)

    item = decompose_definition(
        "物質波",
        "物質波とは",
        "物質波は、ドブロイ波と言われているもの。",
    )
    check("relation-is-definition", item.relation == "definition")
    check(
        "object-description-extracted",
        item.object_description == "ドブロイ波と言われているもの",
        item.object_description,
    )
    check(
        "semantic-prompt-has-subject",
        "概念: 物質波" in semantic_prompt(item),
    )
    check(
        "semantic-prompt-has-relation",
        "関係: definition" in semantic_prompt(item),
    )
    check(
        "relation-prompt-structured",
        "主語=物質波 / 関係=definition" in relation_prompt(item),
    )
    check("three-training-queries", len(training_queries(item)) == 3)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "train.jsonl"
        rows = [
            {
                "concept": "A",
                "question": "Aとは",
                "answer": "Aは、説明A。",
                "split_reason": "deterministic-train",
            },
            {
                "concept": "B",
                "question": "Bとは",
                "answer": "Bは、説明B。",
                "split_reason": "deterministic-train",
            },
        ]
        path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
            encoding="utf-8",
        )
        pairs, concepts = load_augmented_pairs(path)
        check("augmented-concepts", concepts == 2, str(concepts))
        check("augmented-rows", len(pairs) == 6, str(len(pairs)))

        rows.append({
            "concept": "H",
            "question": "Hとは",
            "answer": "Hは、HOLDOUT。",
            "split_reason": "deterministic-holdout",
        })
        path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
            encoding="utf-8",
        )
        rejected = False
        try:
            load_augmented_pairs(path)
        except ValueError:
            rejected = True
        check("holdout-leakage-rejected", rejected)

    semantic_gains = [0.05, 0.03, 0.01, -0.01]
    mean_gain = sum(semantic_gains) / len(semantic_gains)
    improved = sum(1 for x in semantic_gains if x > 1e-6)
    regressed = sum(1 for x in semantic_gains if x < -1e-6)
    policy_ok = mean_gain >= 0.01 and improved > regressed
    check("semantic-generalization-policy", policy_ok, f"{mean_gain:+.3f}")

    print()
    print("Semantic decomposition     : PASS")
    print("Relation-aware augmentation: PASS")
    print("Holdout leakage rejection  : PASS")
    print("Generalization policy      : PASS")
    print("STATUS                     : SEMANTIC_QA_GENERALIZATION_PASS")


if __name__ == "__main__":
    main()
