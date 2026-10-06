#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.12.16 Subject-Keyed Corpus Memory evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from build_nagato_gain_probes_v10127 import resolve_data_path, sentence_candidates
from semantic_role_generalization_v101210 import decompose_role
from subject_keyed_corpus_memory_v101216 import (
    load_memory,
    lookup_subject,
    subject_mapping,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/data-nagato.txt")
    p.add_argument("--memory", default="data/subject_keyed_corpus_memory_v101216.jsonl")
    p.add_argument("--min-answer-chars", type=int, default=12)
    p.add_argument("--max-answer-chars", type=int, default=160)
    return p.parse_args()


def main():
    args = parse_args()
    data_path = resolve_data_path(args.data)
    memory_path = Path(args.memory)

    class BuildArgs:
        min_answer_chars = args.min_answer_chars
        max_answer_chars = args.max_answer_chars
        data = str(data_path)

    candidates = sentence_candidates(
        data_path.read_text(encoding="utf-8"),
        BuildArgs(),
    )

    expected = {}
    for row in candidates:
        item = decompose_role(row["concept"], row["question"], row["answer"])
        expected.setdefault(item.subject, [])
        if item.answer not in expected[item.subject]:
            expected[item.subject].append(item.answer)

    memory = load_memory(memory_path)
    subjects = sorted(expected)
    hits = 0
    statement_hits = 0
    total_statements = sum(len(v) for v in expected.values())

    print("=" * 116)
    print(" LLM_TRY v10.12.16 Subject-Keyed Corpus Memory Evaluation")
    print("=" * 116)
    print("Expected subjects   :", len(subjects))
    print("Memory records      :", len(memory))

    for subject in subjects:
        rows = lookup_subject(memory_path, subject)
        if rows:
            hits += 1
        statements = {row.statement for row in rows}
        statement_hits += sum(1 for x in expected[subject] if x in statements)

    subject_hit_rate = hits / len(subjects) if subjects else 0.0
    statement_hit_rate = (
        statement_hits / total_statements if total_statements else 0.0
    )

    mapping_sample_ok = True
    for subject in subjects[:5]:
        maps = subject_mapping(memory_path, subject)
        if not maps or not all(" => " in x for x in maps):
            mapping_sample_ok = False
            break

    final = (
        subject_hit_rate == 1.0
        and statement_hit_rate == 1.0
        and mapping_sample_ok
    )

    print(f"Subject hit rate    : {subject_hit_rate:.2%}")
    print(f"Statement hit rate  : {statement_hit_rate:.2%}")
    print("Canonical mappings  :", "PASS" if mapping_sample_ok else "FAIL")
    print("STATUS              :", "PASS" if final else "FAIL")

    if not final:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
