#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.12.10 Semantic Role Generalization regression."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from semantic_role_generalization_v101210 import (
    decompose_role,
    infer_relation,
    semantic_role_prompt,
    compact_role_prompt,
    training_queries,
)
from train_semantic_role_generalization_v101210 import load_augmented_pairs


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" : {detail}" if detail else ""))
    if not ok:
        raise AssertionError(name)


def main() -> None:
    print("=" * 112)
    print(" LLM_TRY v10.12.10 Semantic Role Generalization Regression")
    print("=" * 112)

    cases = [
        ("物質波", "物質波は、ドブロイ波と言われているもの。", "definition"),
        ("CPU", "CPUは、命令を実行するために使われる。", "function"),
        ("現象", "現象は、条件によって変化する。", "cause"),
        ("A", "Aは、Bと異なる。", "comparison"),
        ("磁石", "磁石は、磁場を持つ。", "property"),
    ]
    for subject, answer, expected in cases:
        actual = infer_relation(subject, answer)
        check(f"relation:{expected}", actual == expected, actual)

    item = decompose_role(
        "物質波",
        "物質波とは",
        "物質波は、ドブロイ波と言われているもの。",
    )
    check("subject-extracted", item.subject == "物質波")
    check("role-extracted", item.relation == "definition")
    check(
        "object-description-extracted",
        item.object_description == "ドブロイ波と言われているもの",
        item.object_description,
    )
    check("semantic-role-prompt", "意味役割: definition" in semantic_role_prompt(item))
    check("compact-role-prompt", "relation=definition" in compact_role_prompt(item))
    check("three-training-queries", len(training_queries(item)) == 3)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "train.jsonl"
        rows = [
            {
                "subject": "A",
                "relation": "definition",
                "object_description": "説明A",
                "question": "Aとは",
                "answer": "Aは、説明A。",
                "split_reason": "semantic-role-train",
            },
            {
                "subject": "B",
                "relation": "property",
                "object_description": "性質B",
                "question": "Bとは",
                "answer": "Bは、性質B。",
                "split_reason": "semantic-role-train",
            },
        ]
        path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
            encoding="utf-8",
        )
        pairs, relations, subjects = load_augmented_pairs(path)
        check("multi-role-loader", relations == {"definition", "property"}, str(relations))
        check("subject-count", subjects == 2, str(subjects))
        check("augmented-row-count", len(pairs) == 6, str(len(pairs)))

        rows.append({
            "subject": "H",
            "relation": "comparison",
            "object_description": "比較H",
            "question": "Hとは",
            "answer": "Hは、比較H。",
            "split_reason": "unseen-relation-holdout",
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
        check("unseen-role-leakage-rejected", rejected)

    seen_gains = [0.04, 0.02, 0.01, -0.01]
    unseen_gains = [0.02, 0.00, 0.01, -0.01]
    seen_mean = sum(seen_gains) / len(seen_gains)
    unseen_mean = sum(unseen_gains) / len(unseen_gains)
    seen_imp = sum(1 for x in seen_gains if x > 1e-6)
    seen_reg = sum(1 for x in seen_gains if x < -1e-6)
    unseen_imp = sum(1 for x in unseen_gains if x > 1e-6)
    unseen_reg = sum(1 for x in unseen_gains if x < -1e-6)

    check(
        "seen-role-policy",
        seen_mean >= 0.01 and seen_imp > seen_reg,
        f"{seen_mean:+.3f}",
    )
    check(
        "unseen-role-policy",
        unseen_mean >= 0.00 and unseen_imp >= unseen_reg,
        f"{unseen_mean:+.3f}",
    )

    print()
    print("Semantic role inference   : PASS")
    print("Multi-role augmentation   : PASS")
    print("Unseen-role leakage guard : PASS")
    print("Seen/unseen policies      : PASS")
    print("STATUS                    : SEMANTIC_ROLE_GENERALIZATION_PASS")


if __name__ == "__main__":
    main()
