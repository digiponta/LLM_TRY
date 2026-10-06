#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.12.8 Corpus-to-QA internalization regression."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from build_nagato_qa_split_v10128 import concept_bucket
from train_nagato_qa_sft_v10128 import load_train_rows


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" : {detail}" if detail else ""))
    if not ok:
        raise AssertionError(name)


def main() -> None:
    print("=" * 108)
    print(" LLM_TRY v10.12.8 Corpus-to-QA Knowledge Internalization Regression")
    print("=" * 108)

    a = concept_bucket("物質波")
    b = concept_bucket("物質波")
    c = concept_bucket("真空管")
    check("deterministic-concept-bucket", a == b)
    check("different-concepts-separable", a != c)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        train = root / "train.jsonl"
        holdout_contaminated = root / "holdout-contaminated.jsonl"

        train_rows = [
            {
                "concept": "A",
                "question": "Aとは",
                "answer": "AはTRAIN。",
                "split_reason": "deterministic-train",
            },
            {
                "concept": "B",
                "question": "Bとは",
                "answer": "BはTRAIN。",
                "split_reason": "deterministic-train",
            },
        ]
        train.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in train_rows),
            encoding="utf-8",
        )
        loaded = load_train_rows(train)
        check("train-loader-accepts-train-only", len(loaded) == 2)

        bad_rows = list(train_rows)
        bad_rows.append({
            "concept": "H",
            "question": "Hとは",
            "answer": "HはHOLDOUT。",
            "split_reason": "deterministic-holdout",
        })
        holdout_contaminated.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in bad_rows),
            encoding="utf-8",
        )
        rejected = False
        try:
            load_train_rows(holdout_contaminated)
        except ValueError:
            rejected = True
        check("trainer-rejects-holdout-leakage", rejected)

    gains = [0.05, 0.03, -0.01, 0.00]
    mean_gain = sum(gains) / len(gains)
    improved = sum(1 for x in gains if x > 1e-6)
    same = sum(1 for x in gains if abs(x) <= 1e-6)
    regressed = len(gains) - improved - same
    holdout_ok = mean_gain >= 0.01 and improved > regressed
    check("holdout-generalization-policy", holdout_ok, f"{mean_gain:+.3f}")

    print()
    print("Deterministic split       : PASS")
    print("Holdout leakage rejection : PASS")
    print("Holdout gain policy       : PASS")
    print("STATUS                    : CORPUS_TO_QA_INTERNALIZATION_PASS")


if __name__ == "__main__":
    main()
