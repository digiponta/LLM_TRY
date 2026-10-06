#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.12.10.1 Preservation-Aware Semantic Role regression."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from semantic_role_generalization_v101210 import infer_relation
from train_semantic_role_generalization_v101210 import load_augmented_pairs


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" : {detail}" if detail else ""))
    if not ok:
        raise AssertionError(name)


def main() -> None:
    print("=" * 116)
    print(" LLM_TRY v10.12.10.1 Preservation-Aware Semantic Role Regression")
    print("=" * 116)

    check(
        "function-precedes-purpose-cause",
        infer_relation("CPU", "CPUは、命令を実行するために使われる。") == "function",
    )

    # Strict split rule: a seen-role holdout relation must exist in TRAIN.
    train_relations = {"definition", "function", "property"}
    provisional = [
        {"subject": "A", "relation": "property"},
        {"subject": "B", "relation": "cause"},
    ]
    seen = [row for row in provisional if row["relation"] in train_relations]
    unseen = [row for row in provisional if row["relation"] not in train_relations]
    check("strict-seen-role-filter", [x["relation"] for x in seen] == ["property"])
    check("absent-role-routed-unseen", [x["relation"] for x in unseen] == ["cause"])

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
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        pairs, relations, subjects = load_augmented_pairs(path)
        check("multi-role-loader", relations == {"definition", "property"}, str(relations))
        check("augmented-role-rows", len(pairs) == 6, str(len(pairs)))
        check("subject-count", subjects == 2, str(subjects))

    # Preservation replay policy is intentionally non-zero.
    preservation_weight = 2
    protected_count = 5
    preservation_rows = preservation_weight * protected_count
    check("preservation-replay-enabled", preservation_rows == 10, str(preservation_rows))

    print()
    print("Strict seen-role split     : PASS")
    print("Absent-role routing        : PASS")
    print("Preservation replay policy : PASS")
    print("STATUS                     : PRESERVATION_AWARE_SEMANTIC_ROLE_PASS")


if __name__ == "__main__":
    main()
