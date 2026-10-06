#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.12.16 Subject-Keyed Corpus Memory builder."""

from __future__ import annotations

import argparse
from pathlib import Path

from build_nagato_gain_probes_v10127 import resolve_data_path, sentence_candidates
from semantic_role_generalization_v101210 import decompose_role
from subject_keyed_corpus_memory_v101216 import (
    CorpusMemoryRecord,
    save_memory,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/data-nagato.txt")
    p.add_argument("--output", default="data/subject_keyed_corpus_memory_v101216.jsonl")
    p.add_argument("--min-answer-chars", type=int, default=12)
    p.add_argument("--max-answer-chars", type=int, default=160)
    return p.parse_args()


def main():
    args = parse_args()
    data_path = resolve_data_path(args.data)
    text = data_path.read_text(encoding="utf-8")

    class BuildArgs:
        min_answer_chars = args.min_answer_chars
        max_answer_chars = args.max_answer_chars
        data = str(data_path)

    candidates = sentence_candidates(text, BuildArgs())

    # Preserve all unique subject + full proposition pairs. Unlike the semantic
    # TRAIN/HOLDOUT dataset, corpus memory is a runtime knowledge store and is
    # therefore built from the complete source corpus.
    records = []
    seen = set()
    for row in candidates:
        item = decompose_role(row["concept"], row["question"], row["answer"])
        key = (item.subject, item.answer)
        if key in seen:
            continue
        seen.add(key)
        records.append(CorpusMemoryRecord(
            subject=item.subject,
            statement=item.answer,
            relation=item.relation,
            object_description=item.object_description,
            source_text=row["source_text"],
            source=str(data_path),
        ))

    save_memory(Path(args.output), records)

    subjects = {row.subject for row in records}
    print("=" * 116)
    print(" LLM_TRY v10.12.16 Subject-Keyed Corpus Memory Builder")
    print("=" * 116)
    print("Corpus             :", data_path)
    print("Source candidates  :", len(candidates))
    print("Memory records     :", len(records))
    print("Unique subjects    :", len(subjects))
    print("Output             :", args.output)
    print("Canonical form     : subject => full proposition")
    print("STATUS             : SUBJECT_KEYED_CORPUS_MEMORY_BUILD_PASS")


if __name__ == "__main__":
    main()
