#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.12.8 Corpus-to-QA dataset builder.

Builds source-grounded QA pairs from data-nagato.txt and deterministically
splits them into TRAIN and HOLDOUT sets by concept. The holdout set is never
written into the training file.
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


DEFAULT_DATA = "data/data-nagato.txt"
DEFAULT_TRAIN = "data/nagato_qa_train_v10128.jsonl"
DEFAULT_HOLDOUT = "data/nagato_qa_holdout_v10128.jsonl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build deterministic train/holdout QA sets from data-nagato.txt."
    )
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--train-output", default=DEFAULT_TRAIN)
    p.add_argument("--holdout-output", default=DEFAULT_HOLDOUT)
    p.add_argument("--max-pairs", type=int, default=30)
    p.add_argument("--holdout-ratio", type=float, default=0.30)
    p.add_argument("--min-answer-chars", type=int, default=12)
    p.add_argument("--max-answer-chars", type=int, default=120)
    return p.parse_args()


def concept_bucket(concept: str) -> int:
    digest = hashlib.sha256(concept.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def main() -> None:
    args = parse_args()
    if not 0.0 < args.holdout_ratio < 1.0:
        raise ValueError("--holdout-ratio must be between 0 and 1.")

    data_path = resolve_data_path(args.data)
    text = data_path.read_text(encoding="utf-8")

    # Reuse v10.12.7 source-grounded extraction.
    class BuildArgs:
        min_answer_chars = args.min_answer_chars
        max_answer_chars = args.max_answer_chars
        data = str(data_path)

    candidates = sentence_candidates(text, BuildArgs())
    selected = rank_candidates(candidates, max(2, args.max_pairs))

    # Keep v10.12.7 evaluation concepts out of TRAIN to prevent benchmark leakage.
    reserved_holdout = {
        "物質波", "シングルコアCPU", "多項式", "洪水伝説", "行列計算",
        "代数拡大", "真空管", "有限個の状態を持つ閉じた状態遷移マシン",
        "サーバ仮想化に対応したCPU", "遺伝子の情報処理",
    }

    train_rows: list[dict] = []
    holdout_rows: list[dict] = []

    threshold = int(args.holdout_ratio * 10000)
    for index, row in enumerate(selected, 1):
        payload = {
            "version": "v10.12.8",
            "qa_id": f"nagato-qa-{index:03d}",
            "concept": row["concept"],
            "question": row["question"],
            "answer": row["answer"],
            "source_text": row["source_text"],
            "source": str(data_path),
        }

        is_reserved = row["concept"] in reserved_holdout
        bucket = concept_bucket(row["concept"]) % 10000
        if is_reserved or bucket < threshold:
            payload["split_reason"] = (
                "reserved-v10.12.7-eval"
                if is_reserved
                else "deterministic-holdout"
            )
            holdout_rows.append(payload)
        else:
            payload["split_reason"] = "deterministic-train"
            train_rows.append(payload)

    # Fail closed if the corpus/extractor combination yields a degenerate split.
    if len(train_rows) < 2:
        raise RuntimeError(
            f"Too few TRAIN rows ({len(train_rows)}). Increase --max-pairs."
        )
    if len(holdout_rows) < 2:
        raise RuntimeError(
            f"Too few HOLDOUT rows ({len(holdout_rows)}). Increase --max-pairs."
        )

    train_path = Path(args.train_output)
    holdout_path = Path(args.holdout_output)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    holdout_path.parent.mkdir(parents=True, exist_ok=True)

    with train_path.open("w", encoding="utf-8") as handle:
        for row in train_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    with holdout_path.open("w", encoding="utf-8") as handle:
        for row in holdout_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    train_concepts = {row["concept"] for row in train_rows}
    holdout_concepts = {row["concept"] for row in holdout_rows}
    overlap = train_concepts & holdout_concepts

    print("=" * 104)
    print(" LLM_TRY v10.12.8 Corpus-to-QA Train/Holdout Builder")
    print("=" * 104)
    print("Corpus          :", data_path)
    print("Candidates      :", len(candidates))
    print("Selected        :", len(selected))
    print("TRAIN rows      :", len(train_rows))
    print("HOLDOUT rows    :", len(holdout_rows))
    print("Concept overlap :", len(overlap))
    print("TRAIN output    :", train_path)
    print("HOLDOUT output  :", holdout_path)
    print()
    if overlap:
        raise RuntimeError(f"Train/holdout concept leakage detected: {sorted(overlap)}")
    print("STATUS          : CORPUS_TO_QA_SPLIT_PASS")


if __name__ == "__main__":
    main()
