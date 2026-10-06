#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.12.16.1 Retrieval-First Runtime evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from build_nagato_gain_probes_v10127 import resolve_data_path, sentence_candidates
from semantic_role_generalization_v101210 import decompose_role
from retrieval_first_runtime_v1012161 import resolve_subject


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
    relation_by_subject = {}
    for row in candidates:
        item = decompose_role(row["concept"], row["question"], row["answer"])
        expected.setdefault(item.subject, [])
        if item.answer not in expected[item.subject]:
            expected[item.subject].append(item.answer)
        relation_by_subject.setdefault(item.subject, item.relation)

    hit = 0
    proposition_covered = 0
    total_props = sum(len(v) for v in expected.values())
    function_subjects = 0
    informative_function = 0

    print("=" * 116)
    print(" LLM_TRY v10.12.16.1 Retrieval-First Runtime Evaluation")
    print("=" * 116)
    print("Subjects             :", len(expected))
    print("Source propositions  :", total_props)

    for subject, statements in expected.items():
        result = resolve_subject(memory_path, subject)
        if result.hit:
            hit += 1
        proposition_covered += sum(
            1 for statement in statements if statement in result.answer
        )

        if relation_by_subject[subject] == "function":
            function_subjects += 1
            fs = result.function_structure
            if fs is not None and (
                fs.action != "function"
                or fs.target != "unspecified"
                or fs.purpose != "unspecified"
            ):
                informative_function += 1

    subject_hit_rate = hit / len(expected) if expected else 0.0
    proposition_coverage = proposition_covered / total_props if total_props else 0.0
    function_slot_rate = (
        informative_function / function_subjects if function_subjects else 1.0
    )

    unknown = resolve_subject(memory_path, "__NOT_IN_CORPUS__")
    miss_ok = not unknown.hit and unknown.route == "FALLBACK"

    final = (
        subject_hit_rate == 1.0
        and proposition_coverage == 1.0
        and function_slot_rate >= 0.50
        and miss_ok
    )

    print(f"Subject HIT rate     : {subject_hit_rate:.2%}")
    print(f"Proposition coverage : {proposition_coverage:.2%}")
    print(f"Function slot rate   : {function_slot_rate:.2%}")
    print("Unknown fallback     :", "PASS" if miss_ok else "FAIL")
    print("Generation required  : NO for memory HIT")
    print("Model retraining     : NONE")
    print("STATUS               :", "PASS" if final else "FAIL")

    if not final:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
