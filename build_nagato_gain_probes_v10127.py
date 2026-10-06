#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.12.7 corpus-grounded QA probe builder.

Extracts simple definitional/descriptive Japanese sentences from data-nagato.txt.
Every generated probe retains its exact source sentence so that the QA benchmark
can be traced back to the corpus rather than invented externally.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


DEFAULT_DATA = "data/data-nagato.txt"
DEFAULT_OUTPUT = "data/nagato_gain_probes_v10127.jsonl"

SUBJECT_RE = re.compile(
    r"^\s*([一-龯々ァ-ヶーA-Za-z0-9・]{2,20})は、?(.{8,120})[。！？!?]?\s*$"
)
STOP_SUBJECTS = {
    "これは", "それは", "あれは", "私は", "僕は", "彼は", "彼女は",
    "ここは", "そこは", "今日は", "今回は", "場合は",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build source-grounded QA probes from data-nagato.txt."
    )
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--max-probes", type=int, default=20)
    p.add_argument("--min-answer-chars", type=int, default=12)
    p.add_argument("--max-answer-chars", type=int, default=120)
    return p.parse_args()


def resolve_data_path(value: str) -> Path:
    path = Path(value)
    if path.exists():
        return path
    if value == DEFAULT_DATA:
        fallback = Path("data-nagato.txt")
        if fallback.exists():
            return fallback
    raise FileNotFoundError(f"Nagato corpus not found: {path}")


def sentence_candidates(text: str, args: argparse.Namespace) -> list[dict]:
    raw_sentences = [
        s.strip()
        for s in re.split(r"(?<=[。！？!?])\s*|\r?\n+", text)
        if s.strip()
    ]
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for sentence in raw_sentences:
        if len(sentence) < args.min_answer_chars:
            continue
        if len(sentence) > args.max_answer_chars:
            continue
        if sentence.startswith(("「", "『", "(", "（")):
            continue

        match = SUBJECT_RE.match(sentence)
        if not match:
            continue

        subject = match.group(1).strip()
        predicate = match.group(2).strip()
        if f"{subject}は" in STOP_SUBJECTS or subject in STOP_SUBJECTS:
            continue
        if subject in {"これ", "それ", "あれ", "ここ", "そこ", "場合", "今回"}:
            continue
        if len(predicate) < 8:
            continue

        answer = sentence
        if answer[-1] not in "。！？!?":
            answer += "。"
        question = f"{subject}とは"
        key = (question, answer)
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "concept": subject,
            "question": question,
            "answer": answer,
            "source_text": sentence,
            "source": str(resolve_data_path(args.data)),
        })

    return rows


def rank_candidates(rows: list[dict], max_probes: int) -> list[dict]:
    frequency = Counter(row["concept"] for row in rows)
    # Prefer repeated corpus concepts, then compact source-supported answers.
    ranked = sorted(
        rows,
        key=lambda row: (
            -frequency[row["concept"]],
            len(row["answer"]),
            row["concept"],
        ),
    )

    selected: list[dict] = []
    used_concepts: set[str] = set()
    for row in ranked:
        if row["concept"] in used_concepts:
            continue
        selected.append(row)
        used_concepts.add(row["concept"])
        if len(selected) >= max_probes:
            break
    return selected


def main() -> None:
    args = parse_args()
    data_path = resolve_data_path(args.data)
    output_path = Path(args.output)
    text = data_path.read_text(encoding="utf-8")

    candidates = sentence_candidates(text, args)
    selected = rank_candidates(candidates, max(1, args.max_probes))

    if not selected:
        raise RuntimeError(
            "No source-grounded 'Xは...' probe candidates were found. "
            "Inspect data-nagato.txt or relax the probe-builder limits."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(selected, 1):
            payload = {
                "version": "v10.12.7",
                "probe_id": f"nagato-{index:03d}",
                **row,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    print("=" * 96)
    print(" LLM_TRY v10.12.7 Corpus-Grounded QA Probe Builder")
    print("=" * 96)
    print("Corpus          :", data_path)
    print("Candidates      :", len(candidates))
    print("Selected probes :", len(selected))
    print("Output          :", output_path)
    print()
    for index, row in enumerate(selected, 1):
        print(
            f"{index:02d}. {row['question']} => "
            f"{row['answer'][:72]}"
        )


if __name__ == "__main__":
    main()
