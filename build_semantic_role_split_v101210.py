#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.12.10 Semantic Role dataset builder.

Builds source-grounded role propositions from data-nagato.txt and creates:
- TRAIN: multiple seen relations
- HOLDOUT seen-role: unseen concepts whose relation is present in TRAIN
- HOLDOUT unseen-role: all examples of one relation type excluded from TRAIN
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from build_nagato_gain_probes_v10127 import (
    resolve_data_path,
    sentence_candidates,
    rank_candidates,
)
from semantic_role_generalization_v101210 import decompose_role


DEFAULT_DATA = "data/data-nagato.txt"
DEFAULT_TRAIN = "data/nagato_role_train_v101210.jsonl"
DEFAULT_SEEN = "data/nagato_role_holdout_seen_v101210.jsonl"
DEFAULT_UNSEEN = "data/nagato_role_holdout_unseen_v101210.jsonl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build semantic-role TRAIN and two HOLDOUT sets."
    )
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--train-output", default=DEFAULT_TRAIN)
    p.add_argument("--seen-holdout-output", default=DEFAULT_SEEN)
    p.add_argument("--unseen-holdout-output", default=DEFAULT_UNSEEN)
    p.add_argument("--max-pairs", type=int, default=60)
    p.add_argument("--seen-holdout-ratio", type=float, default=0.25)
    p.add_argument("--unseen-relation", default="comparison")
    p.add_argument("--min-answer-chars", type=int, default=12)
    p.add_argument("--max-answer-chars", type=int, default=160)
    return p.parse_args()


def bucket(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if not 0.0 < args.seen_holdout_ratio < 1.0:
        raise ValueError("--seen-holdout-ratio must be between 0 and 1")

    data_path = resolve_data_path(args.data)
    text = data_path.read_text(encoding="utf-8")

    class BuildArgs:
        min_answer_chars = args.min_answer_chars
        max_answer_chars = args.max_answer_chars
        data = str(data_path)

    candidates = sentence_candidates(text, BuildArgs())
    selected = rank_candidates(candidates, max(10, args.max_pairs))

    train_rows: list[dict] = []
    seen_holdout: list[dict] = []
    unseen_holdout: list[dict] = []

    threshold = int(args.seen_holdout_ratio * 10000)

    for index, row in enumerate(selected, 1):
        item = decompose_role(row["concept"], row["question"], row["answer"])
        payload = {
            "version": "v10.12.10",
            "role_id": f"nagato-role-{index:03d}",
            "subject": item.subject,
            "relation": item.relation,
            "object_description": item.object_description,
            "question": item.question,
            "answer": item.answer,
            "source_text": row["source_text"],
            "source": str(data_path),
        }

        if item.relation == args.unseen_relation:
            payload["split_reason"] = "unseen-relation-holdout"
            unseen_holdout.append(payload)
            continue

        if bucket(item.subject) % 10000 < threshold:
            payload["split_reason"] = "seen-relation-unseen-concept"
            seen_holdout.append(payload)
        else:
            payload["split_reason"] = "semantic-role-train"
            train_rows.append(payload)

    train_subjects = {x["subject"] for x in train_rows}
    seen_subjects = {x["subject"] for x in seen_holdout}
    unseen_subjects = {x["subject"] for x in unseen_holdout}
    subject_overlap = (
        (train_subjects & seen_subjects)
        | (train_subjects & unseen_subjects)
    )

    train_relations = {x["relation"] for x in train_rows}
    seen_relations = {x["relation"] for x in seen_holdout}
    unseen_relations = {x["relation"] for x in unseen_holdout}

    if subject_overlap:
        raise RuntimeError(f"Subject leakage detected: {sorted(subject_overlap)}")
    if args.unseen_relation in train_relations:
        raise RuntimeError("Unseen relation leaked into TRAIN")
    if not train_rows or not seen_holdout or not unseen_holdout:
        raise RuntimeError(
            "Degenerate split. Increase --max-pairs or choose another --unseen-relation."
        )

    write_jsonl(Path(args.train_output), train_rows)
    write_jsonl(Path(args.seen_holdout_output), seen_holdout)
    write_jsonl(Path(args.unseen_holdout_output), unseen_holdout)

    print("=" * 112)
    print(" LLM_TRY v10.12.10 Semantic Role Dataset Builder")
    print("=" * 112)
    print("Corpus                 :", data_path)
    print("Candidates             :", len(candidates))
    print("Selected               :", len(selected))
    print("TRAIN rows             :", len(train_rows))
    print("Seen-role HOLDOUT      :", len(seen_holdout))
    print("Unseen-role HOLDOUT    :", len(unseen_holdout))
    print("TRAIN relations        :", sorted(train_relations))
    print("Seen HOLDOUT relations :", sorted(seen_relations))
    print("Unseen relation        :", args.unseen_relation)
    print("Unseen HOLDOUT roles   :", sorted(unseen_relations))
    print("Subject overlap        :", len(subject_overlap))
    print()
    print("STATUS                 : SEMANTIC_ROLE_SPLIT_PASS")


if __name__ == "__main__":
    main()
