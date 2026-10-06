#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.12.15 Subject-to-Proposition + Corpus-to-Semantic dataset builder.

Converts source-grounded Xは... sentences in data-nagato.txt into explicit
subject / relation / object_description propositions.

The split is concept-disjoint.  comparison is reserved as an unseen role;
other roles use deterministic subject-level TRAIN/HOLDOUT partitioning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from build_nagato_gain_probes_v10127 import resolve_data_path, sentence_candidates
from semantic_role_generalization_v101210 import decompose_role


DEFAULT_DATA = "data/data-nagato.txt"
DEFAULT_TRAIN = "data/nagato_corpus_semantic_train_v101212.jsonl"
DEFAULT_SEEN = "data/nagato_corpus_semantic_holdout_seen_v101212.jsonl"
DEFAULT_UNSEEN = "data/nagato_corpus_semantic_holdout_unseen_v101212.jsonl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build concept-disjoint corpus-to-semantic TRAIN/HOLDOUT data."
    )
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--train-output", default=DEFAULT_TRAIN)
    p.add_argument("--seen-holdout-output", default=DEFAULT_SEEN)
    p.add_argument("--unseen-holdout-output", default=DEFAULT_UNSEEN)
    p.add_argument("--holdout-ratio", type=float, default=0.25)
    p.add_argument("--unseen-relation", default="comparison")
    p.add_argument("--min-answer-chars", type=int, default=12)
    p.add_argument("--max-answer-chars", type=int, default=160)
    return p.parse_args()


def bucket(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 10000


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if not 0.0 < args.holdout_ratio < 1.0:
        raise ValueError("--holdout-ratio must be between 0 and 1")

    data_path = resolve_data_path(args.data)
    text = data_path.read_text(encoding="utf-8")

    class BuildArgs:
        min_answer_chars = args.min_answer_chars
        max_answer_chars = args.max_answer_chars
        data = str(data_path)

    candidates = sentence_candidates(text, BuildArgs())
    # Keep one source-grounded proposition per concept.  sentence_candidates
    # preserves corpus order; first occurrence is deterministic.
    first_by_subject: dict[str, dict] = {}
    for row in candidates:
        first_by_subject.setdefault(str(row["concept"]), row)

    propositions: list[dict] = []
    for index, row in enumerate(first_by_subject.values(), 1):
        item = decompose_role(row["concept"], row["question"], row["answer"])
        propositions.append({
            "version": "v10.12.15",
            "semantic_id": f"nagato-sem-{index:03d}",
            "subject": item.subject,
            "relation": item.relation,
            "object_description": item.object_description,
            "question": item.question,
            "answer": item.answer,
            "subject_mapping": f"{item.subject} => {item.answer}",
            "source_text": row["source_text"],
            "source": str(data_path),
        })

    threshold = int(args.holdout_ratio * 10000)
    train: list[dict] = []
    provisional_seen: list[dict] = []
    unseen: list[dict] = []

    for row in propositions:
        if row["relation"] == args.unseen_relation:
            row = dict(row)
            row["split_reason"] = "unseen-relation-holdout"
            unseen.append(row)
        elif bucket(row["subject"]) < threshold:
            row = dict(row)
            row["split_reason"] = "provisional-seen-role-holdout"
            provisional_seen.append(row)
        else:
            row = dict(row)
            row["split_reason"] = "corpus-semantic-train"
            train.append(row)

    train_relations = {row["relation"] for row in train}
    seen: list[dict] = []
    for row in provisional_seen:
        row = dict(row)
        if row["relation"] in train_relations:
            row["split_reason"] = "seen-role-unseen-concept"
            seen.append(row)
        else:
            row["split_reason"] = "unseen-relation-holdout"
            unseen.append(row)

    train_subjects = {row["subject"] for row in train}
    seen_subjects = {row["subject"] for row in seen}
    unseen_subjects = {row["subject"] for row in unseen}
    overlap = (train_subjects & seen_subjects) | (train_subjects & unseen_subjects)

    if overlap:
        raise RuntimeError(f"Concept leakage detected: {sorted(overlap)}")
    if not train or not seen or not unseen:
        raise RuntimeError("Degenerate semantic split; adjust holdout ratio.")

    write_jsonl(Path(args.train_output), train)
    write_jsonl(Path(args.seen_holdout_output), seen)
    write_jsonl(Path(args.unseen_holdout_output), unseen)

    before = Counter(row["relation"] for row in propositions)
    train_counts = Counter(row["relation"] for row in train)

    print("=" * 116)
    print(" LLM_TRY v10.12.15 Subject-to-Proposition + Corpus-to-Semantic Dataset")
    print("=" * 116)
    print("Corpus                 :", data_path)
    print("Source candidates      :", len(candidates))
    print("Unique concepts        :", len(propositions))
    print("Subject mappings       :", sum(1 for x in propositions if x.get("subject_mapping")))
    print("All relation counts    :", dict(sorted(before.items())))
    print("TRAIN rows             :", len(train))
    print("TRAIN relation counts  :", dict(sorted(train_counts.items())))
    print("Seen-role HOLDOUT      :", len(seen))
    print("Unseen-role HOLDOUT    :", len(unseen))
    print("Seen HOLDOUT relations :", sorted({x["relation"] for x in seen}))
    print("Unseen HOLDOUT roles   :", sorted({x["relation"] for x in unseen}))
    print("Concept overlap        :", len(overlap))
    print("STATUS                 : CORPUS_TO_SEMANTIC_SPLIT_PASS")


if __name__ == "__main__":
    main()
